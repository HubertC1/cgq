import glob, tqdm, wandb, os, json, random, time, jax, flax, importlib, sys
from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, get_wandb_video,CsvLogger

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.noise_wrapper import make_noise_cfg
try:
    import envs.spiral_maze  # noqa: registers pointmaze-spiral{R}-v0 envs
except ModuleNotFoundError:
    # envs/spiral_maze.py was never committed to git and is currently missing from disk (unrelated
    # to any env family other than the paused pointmaze-spiral{R} horizon study) -- degrade
    # gracefully rather than blocking every other env family's training on a missing file.
    print('WARNING: envs.spiral_maze not found -- pointmaze-spiral{R}-v0 envs will not be registered.')

from utils.flax_utils import save_agent, restore_agent_with_file
from utils.datasets import Dataset, ReplayBuffer
from utils.run_registry import flatten, upsert_run

from evaluation import evaluate_multi_seed, render_value_replay
from agents import agents

import numpy as np

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', '/tmp2/hubertchang/stochasticity/exp/', 'Save directory. Was '
                     '\'exp/\' (under the repo, on the home filesystem) -- moved to /tmp2 after that '
                     'filesystem filled to 100%, matching the earlier dataset migration.')
flags.DEFINE_string('exp_name', None, 'Experiment name (overrides automatic generation).')
flags.DEFINE_boolean('resume', False, 'Resume from checkpoint if found.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', -1, 'Save interval.')
flags.DEFINE_integer('max_checkpoints', 20, 'Maximum number of checkpoints to keep. Default 20 so '
                      'a full 1M-step run at the default 100k eval/save interval (10 checkpoints) '
                      'keeps every eval-aligned checkpoint instead of pruning to just the latest.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

flags.DEFINE_float('discount', 0.99, 'discount factor')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('num_eval_seeds', 5, 'Number of independent eval seeds per eval checkpoint. Each repeat '
                      'reseeds JAX policy sampling, the env actuator-noise RNG, and OGBench\'s init-position '
                      'jitter, so results are a genuinely independent draw, not the same rollout. Trades eval '
                      'wall-clock time (roughly linear in this) for a mean + std/min/max spread per metric at '
                      'each checkpoint, logged under the same key plus _std/_min/_max suffixes.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

# It is related to the function get_config
config_flags.DEFINE_config_file('agent', default='agents/bb.py', lock_config=False)

# Single-file experiment config (see configs/base_experiment.py). When given, this overrides
# --env_name/--ogbench_dataset_dir/--agent/etc. below wholesale; when omitted, every flag below
# behaves exactly as before (kept for backward compatibility with already-queued CLI invocations).
config_flags.DEFINE_config_file('exp_config', default=None, lock_config=False)
flags.DEFINE_string('run_registry_path', 'exp/run_registry.csv', 'CSV ledger of run configs + results.')

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_integer('max_episode_steps', None, 'Override the registered env max_episode_steps (default 1000 '
                      'for every pointmaze size). Needed for larger mazes / longer tasks whose datasets were '
                      'collected with a longer budget (e.g. giant navigate uses 2001) than the registered '
                      'default supports -- leave unset to use the registered default.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

flags.DEFINE_enum('noise_type', 'none', ['none', 'constant'],
                   'Action-space noise process for the env wrapper. Only "constant" is wired to flags today; '
                   'radial_from_goal/near_walls presets exist in envs/noise_wrapper.py but must be constructed '
                   'programmatically via make_noise_cfg.')
flags.DEFINE_float('noise_sigma_0', 0.0, 'Std for the "constant" noise preset (isotropic, per-action-dim).')
flags.DEFINE_integer('noise_seed', 12345, 'Seed for the training/rollout env noise process. Independent of '
                      '--seed (which seeds env/data/model).')
flags.DEFINE_integer('eval_noise_seed', 67890, 'Seed for the eval env noise process. Must differ from '
                      '--noise_seed so eval rollouts are not correlated with training/data noise realizations.')

flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")



class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)

def main(_):
    exp_config_path = None
    if FLAGS.exp_config is not None:
        ec = FLAGS.exp_config
        for arg in sys.argv:
            if arg.startswith('--exp_config='):
                exp_config_path = arg.split('=', 1)[1].split(':', 1)[0]
                break

        FLAGS.run_group = ec.run_group
        FLAGS.seed = ec.seed
        FLAGS.env_name = ec.task.env_name
        FLAGS.ogbench_dataset_dir = ec.task.dataset_dir
        FLAGS.max_episode_steps = ec.task.max_episode_steps
        # noise_type/noise_sigma_0 are kept in sync purely for flags.json/registry display; the
        # actual noise_cfg construction below reads ec.task.noise.{type,params} directly, so any
        # preset name (not just the legacy 'none'/'constant' the raw --noise_type flag validates
        # against on the CLI) is safe here -- direct attribute assignment bypasses enum validation.
        FLAGS.noise_type = ec.task.noise.type
        FLAGS.noise_sigma_0 = ec.task.noise.params.get('sigma_0', 0.0)
        FLAGS.noise_seed = ec.task.noise_seed
        FLAGS.eval_noise_seed = ec.task.eval_noise_seed
        FLAGS.horizon_length = ec.value.horizon_length
        FLAGS.discount = ec.value.discount
        FLAGS.offline_steps = ec.train.offline_steps
        FLAGS.eval_interval = ec.train.eval_interval
        FLAGS.save_interval = ec.train.save_interval
        FLAGS.dataset_replace_interval = ec.train.dataset_replace_interval

        # value.method selects agents/{method}.py; value/policy fields overlay that agent's
        # get_config() wherever key names match (both are named to match 1:1 with agent config
        # fields, so this is a plain "set these specific keys" merge, not a schema translation).
        agent_module = importlib.import_module(f'agents.{ec.value.method}')
        agent_config = agent_module.get_config()
        for k, v in ec.value.items():
            if k != 'method' and k in agent_config:
                agent_config[k] = v
        # config.policy.awr_alpha maps onto agent_config['alpha'] explicitly (not via the generic
        # name-match loop below) because 'alpha' means two unrelated things across agent families:
        # iql/aciql's own agent config field 'alpha' is the AWR inverse temperature (this is what
        # awr_alpha feeds), while acfql/cgq's agent config field 'alpha' is a BC/distillation
        # coefficient set directly by config.value.alpha above (see acfql.py/cgq.py -- deliberately
        # left named 'alpha' there, matching the actor-critic papers' own notation).
        #
        # NOTE: both config.policy.awr_alpha and config.value.alpha are declared unconditionally in
        # base_experiment.py's template (with real defaults), so 'awr_alpha' in ec.policy is ALWAYS
        # true regardless of ec.value.method -- an `in` membership check cannot tell iql/aciql runs
        # apart from acfql/cgq runs the way it could when awr_alpha was merely "set or not." Route
        # explicitly by agent family instead.
        AWR_ALPHA_METHODS = ('iql', 'aciql')  # agents whose own 'alpha' field is the AWR temperature
        if ec.value.method in AWR_ALPHA_METHODS and 'alpha' in agent_config:
            agent_config['alpha'] = ec.policy.awr_alpha
        for k, v in ec.policy.items():
            if k not in ('method', 'awr_alpha') and k in agent_config:
                agent_config[k] = v
        # ec.policy.method itself is deliberately excluded from the loop above (agents have no
        # 'method' field of their own -- that name is reserved for ec.value.method, which selects
        # the agent module). It maps onto agent_config['policy_method'] instead, which is a
        # distinct field some agents (iql, aciql, bc) expose to choose between AWR/gaussian and
        # flow-matching-based policy extraction. Guarded so agents without this field (sarsa,
        # acfql, fql) are unaffected.
        if 'policy_method' in agent_config:
            agent_config['policy_method'] = ec.policy.method
        FLAGS.agent = agent_config

        if ec.exp_name is not None:
            FLAGS.exp_name = ec.exp_name
        else:
            noise_tag = ec.task.noise.type
            if 'sigma_0' in ec.task.noise.params:
                noise_tag += f"{ec.task.noise.params['sigma_0']:g}"
            seed_tag = f'_seed{ec.seed}' if ec.seed != 0 else ''
            # agent_config['alpha'] (not ec.policy.alpha, which no longer exists -- see
            # config.policy.awr_alpha's comment in base_experiment.py) is whichever alpha is
            # actually live for this agent: BC/distillation coefficient for acfql/cgq, AWR
            # inverse temperature for iql/aciql, or absent (dqc/sarsa/bc) -> tag omitted.
            alpha_tag = f"_alpha{agent_config['alpha']:g}" if 'alpha' in agent_config else ''
            FLAGS.exp_name = (
                f'{ec.value.method}_h{ec.value.horizon_length}_disc{ec.value.discount}'
                f'{alpha_tag}_{noise_tag}{seed_tag}'
            )

    exp_name = FLAGS.exp_name if FLAGS.exp_name else get_exp_name(FLAGS.seed)

    project_name = FLAGS.project_name if hasattr(FLAGS, 'project_name') and FLAGS.project_name else 'cgq'
    save_dir_base = os.path.join(FLAGS.save_dir, project_name, FLAGS.run_group, FLAGS.env_name, exp_name)
    
    wandb_id = None
    wandb_id_path = os.path.join(save_dir_base, 'wandb_id.txt')
    if FLAGS.resume and os.path.exists(wandb_id_path):
        with open(wandb_id_path, 'r') as f:
            wandb_id = f.read().strip()
            print(f"Resuming WandB run ID: {wandb_id}")

    run = setup_wandb(project=project_name, group=FLAGS.run_group, name=exp_name, resume_id=wandb_id)
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    
    # Save run ID if we didn't resume or if we just created a new one
    if not wandb_id:
        with open(os.path.join(FLAGS.save_dir, 'wandb_id.txt'), 'w') as f:
            f.write(run.id)
    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    if exp_config_path is not None:
        saved_config_path = os.path.join(FLAGS.save_dir, os.path.basename(exp_config_path))
        with open(exp_config_path, 'r') as src, open(saved_config_path, 'w') as dst:
            dst.write(src.read())
        wandb.save(saved_config_path, base_path=FLAGS.save_dir)

    registry_keys = [
        'seed', 'run_group', 'env_name', 'exp_name', 'resume',
        'offline_steps', 'online_steps', 'eval_interval', 'save_interval',
        'discount', 'horizon_length', 'max_episode_steps',
        'ogbench_dataset_dir', 'dataset_replace_interval', 'dataset_proportion', 'sparse',
        'noise_type', 'noise_sigma_0', 'noise_seed', 'eval_noise_seed',
        'eval_episodes', 'num_eval_seeds', 'agent',
    ]
    registry_row = flatten({k: flag_dict[k] for k in registry_keys if k in flag_dict})
    if FLAGS.exp_config is not None:
        registry_row['policy_method'] = ec.policy.method
        # Full preset params (e.g. near_sigma/far_sigma/wall_distance_threshold for near_walls) --
        # noise_sigma_0 above only ever captures the 'constant' preset's single scalar.
        registry_row.update(flatten({'noise': dict(ec.task.noise)}))
    registry_row.update(
        run_group=FLAGS.run_group,
        exp_name=exp_name,
        wandb_run_id=run.id,
        wandb_url=run.url,
        status='running',
        start_time=time.strftime('%Y-%m-%d %H:%M:%S'),
    )
    upsert_run(FLAGS.run_registry_path, key_cols=['run_group', 'exp_name'], row=registry_row)

    config = FLAGS.agent

    if FLAGS.exp_config is not None:
        # Full preset support (near_walls/region/composite/...): params come straight from the
        # config file's task.noise.params, not squeezed through the single-scalar legacy flags.
        noise_cfg = make_noise_cfg(ec.task.noise.type, **dict(ec.task.noise.params))
    elif FLAGS.noise_type == 'none':
        noise_cfg = None
    elif FLAGS.noise_type == 'constant':
        noise_cfg = make_noise_cfg('constant', sigma_0=FLAGS.noise_sigma_0)
    else:
        raise NotImplementedError(FLAGS.noise_type)

    # data loading
    if FLAGS.ogbench_dataset_dir is not None:
        # custom ogbench dataset
        assert FLAGS.dataset_replace_interval != 0
        assert FLAGS.dataset_proportion == 1.0
        dataset_idx = 0
        dataset_paths = [
            file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")) if '-val.npz' not in file
        ]
        env_kwargs = {} if FLAGS.max_episode_steps is None else dict(max_episode_steps=FLAGS.max_episode_steps)
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
            noise_cfg=noise_cfg,
            noise_seed=FLAGS.noise_seed,
            eval_noise_seed=FLAGS.eval_noise_seed,
            **env_kwargs,
        )
    else:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
            FLAGS.env_name,
            noise_cfg=noise_cfg,
            noise_seed=FLAGS.noise_seed,
            eval_noise_seed=FLAGS.eval_noise_seed,
        )

    if FLAGS.exp_config is not None and FLAGS.ogbench_dataset_dir is not None:
        # Guard against training on a dataset generated at a different p_slip than eval_env is
        # about to replay. generate_cube_slip.py stores no p_slip metadata in the .npz itself --
        # the only record of what a dataset was generated with is its directory name -- so a typo'd
        # or stale config.task.slip.p_slip would otherwise silently train on one stochasticity level
        # and eval on another. Only fires for dataset dirs that actually follow the cube-slip naming
        # convention ('native' or 'pslipX.Y'); other families (pointmaze/antmaze) don't use this
        # mechanism and are left alone.
        dataset_dirname = os.path.basename(os.path.normpath(FLAGS.ogbench_dataset_dir))
        if dataset_dirname == 'native':
            dataset_p_slip = 0.0
        elif dataset_dirname.startswith('pslip'):
            dataset_p_slip = float(dataset_dirname[len('pslip'):])
        else:
            dataset_p_slip = None
        if dataset_p_slip is not None:
            assert abs(dataset_p_slip - ec.task.slip.p_slip) < 1e-9, (
                f"config.task.slip.p_slip={ec.task.slip.p_slip} does not match "
                f"config.task.dataset_dir={FLAGS.ogbench_dataset_dir!r} (p_slip={dataset_p_slip} "
                f"by directory-naming convention). Training-data and eval-time stochasticity must "
                f"match -- fix whichever one is stale."
            )

    if FLAGS.exp_config is not None and ec.task.slip.p_slip > 0:
        # cube-family grasp-slip stochasticity (separate mechanism from noise_cfg above). Only
        # eval_env is wrapped -- this project's cube work is offline-only, so the training rollout
        # env is never stepped. Matches whichever p_slip the dataset itself was generated with (set
        # this to the same value in the exp config), with its own eval seed so the realized slip
        # sequence isn't correlated with training data generation.
        from envs.slip_wrapper import SlipGripperWrapper

        eval_env = SlipGripperWrapper(
            eval_env,
            p_slip=ec.task.slip.p_slip,
            slip_seed=FLAGS.eval_noise_seed,
            gripper_closed_threshold=ec.task.slip.gripper_closed_threshold,
            slip_open_target=ec.task.slip.slip_open_target,
            slip_duration=ec.task.slip.slip_duration,
        )

    # house keeping
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    log_step = 0
    
    discount = FLAGS.discount
    config["horizon_length"] = FLAGS.horizon_length
    if config['agent_name'] == 'fql':
        config['horizon_length'] = 1    # FQL does not use action chunking
    # handle dataset
    def process_dataset(ds):
        """
        Process a dataset (train or val) to
            - handle dataset proportion
            - handle sparse reward
            - convert to action chunked dataset
        """

        ds = Dataset.create(**ds)
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(
                **{k: v[:new_size] for k, v in ds.items()}
            )
        
        if FLAGS.sparse:
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(**ds_dict)


        return ds
    
    train_dataset = process_dataset(train_dataset)
    val_dataset = process_dataset(val_dataset)
    example_batch = train_dataset.sample(())
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    start_step = 0
    if FLAGS.resume:
        chkpts = glob.glob(os.path.join(FLAGS.save_dir, 'params_*.pkl'))
        if chkpts:
            epochs = []
            for c in chkpts:
                try:
                    e = int(c.split('params_')[-1].split('.pkl')[0])
                    epochs.append(e)
                except:
                    pass
            if epochs:
                start_step = max(epochs)
                resume_file = os.path.join(FLAGS.save_dir, f'params_{start_step}.pkl')
                print(f"Resuming from {resume_file}, step {start_step}")
                agent = restore_agent_with_file(agent, resume_file)
                
                if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0:
                    dataset_idx = (start_step // FLAGS.dataset_replace_interval) % len(dataset_paths)
                    print(f"Fast-forwarding dataset to index {dataset_idx}")
                    train_dataset, val_dataset = make_ogbench_env_and_datasets(
                        FLAGS.env_name,
                        dataset_path=dataset_paths[dataset_idx],
                        compact_dataset=False,
                        dataset_only=True,
                        cur_env=env,
                    )
                    train_dataset = process_dataset(train_dataset)
                    val_dataset = process_dataset(val_dataset)

    # Setup logging.
    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
        prefixes.append("offline_agent_val")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")

    log_mode = 'a' if (FLAGS.resume and start_step > 0) else 'w'
    logger = LoggingHelper(
        csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv"), mode=log_mode) 
                    for prefix in prefixes},
        wandb_logger=wandb,
    )

    offline_init_time = time.time()
    log_step = start_step
    last_eval_info = {}

    # Offline RL
    if log_step < FLAGS.offline_steps:
        offline_loop_start = log_step + 1
        for i in tqdm.tqdm(range(offline_loop_start, FLAGS.offline_steps + 1)):
            log_step += 1

            if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0:
                dataset_idx = (dataset_idx + 1) % len(dataset_paths)
                print(f"Using new dataset: {dataset_paths[dataset_idx]}", flush=True)
                train_dataset, val_dataset = make_ogbench_env_and_datasets(
                    FLAGS.env_name,
                    dataset_path=dataset_paths[dataset_idx],
                    compact_dataset=False,
                    dataset_only=True,
                    cur_env=env,
                )
                train_dataset = process_dataset(train_dataset)
                val_dataset = process_dataset(val_dataset)

            batch = train_dataset.sample_sequence(
                config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount,
                truncate_reward_at_success=config.get('truncate_reward_at_success', False),
            )
            agent, offline_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                logger.log(offline_info, "offline_agent", step=log_step)

                # Held-out loss: same total_loss computation as training, forward-only (grad_params
                # = the agent's current, already-trained params; no apply_loss_fn, no target-network
                # update), on a batch drawn from val_dataset instead of train_dataset. Uses a
                # throwaway rng split off agent.rng -- not persisted back into the agent, so this
                # doesn't perturb the training rng stream agent.update() advances on its own.
                val_batch = val_dataset.sample_sequence(
                    config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount,
                    truncate_reward_at_success=config.get('truncate_reward_at_success', False),
                )
                _, val_rng = jax.random.split(agent.rng)
                _, val_info = agent.total_loss(val_batch, agent.network.params, rng=val_rng)
                logger.log(val_info, "offline_agent_val", step=log_step)

            # saving
            if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
                save_agent(agent, FLAGS.save_dir, log_step, max_to_keep=FLAGS.max_checkpoints)

            # eval
            if (FLAGS.eval_interval != 0 and (i % FLAGS.eval_interval == 0 or i == 1)):
                # during eval, the action chunk is executed fully
                want_video = FLAGS.video_episodes > 0
                # Cube envs have no 2D xy grid to sweep for a spatial value-map heatmap (that was
                # scripts/render_value_map.py, pointmaze-only, now retired) -- agents that expose
                # get_value instead get one rollout rendered as [env frame | running V(s) curve],
                # shared implementation with scripts/render_cube_value_replay.py. Agents without
                # get_value (dqc/cgq don't expose one yet) fall back to a plain rollout video.
                use_value_replay = want_video and hasattr(agent, 'get_value')
                eval_info, trajs, cur_renders = evaluate_multi_seed(
                    agent=agent,
                    env=eval_env,
                    action_dim=example_batch["actions"].shape[-1],
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=0 if use_value_replay else FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    num_seeds=FLAGS.num_eval_seeds,
                    base_seed=FLAGS.eval_noise_seed + log_step * FLAGS.num_eval_seeds,
                )
                logger.log(eval_info, "eval", step=log_step)
                last_eval_info = eval_info
                if want_video:
                    if use_value_replay:
                        composite_frames, _, _ = render_value_replay(
                            agent, eval_env, seed=FLAGS.eval_noise_seed + log_step,
                        )
                        video = get_wandb_video([composite_frames])
                    else:
                        video = get_wandb_video(cur_renders)
                    wandb.log({"video": video}, step=log_step)

        
    # transition from offline to online
    replay_buffer = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
    )
        
    ob, _ = env.reset()
    
    action_queue = []
    action_dim = example_batch["actions"].shape[-1]

    # Online RL
    update_info = {}

    from collections import defaultdict
    data = defaultdict(list)
    online_init_time = time.time()
    
    online_start_i = 1
    if log_step >= FLAGS.offline_steps:
        online_start_i = (log_step - FLAGS.offline_steps) + 1

    for i in tqdm.tqdm(range(online_start_i, FLAGS.online_steps + 1)):
        log_step += 1
        online_rng, key = jax.random.split(online_rng)
        
        # during online rl, the action chunk is executed fully
        if len(action_queue) == 0:
            action = agent.sample_actions(observations=ob, rng=key)

            action_chunk = np.array(action).reshape(-1, action_dim)
            for action in action_chunk:
                action_queue.append(action)
        action = action_queue.pop(0)
        
        next_ob, int_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if FLAGS.save_all_online_states:
            state = env.get_state()
            data["steps"].append(i)
            data["obs"].append(np.copy(next_ob))
            data["qpos"].append(np.copy(state["qpos"]))
            data["qvel"].append(np.copy(state["qvel"]))
            if "button_states" in state:
                data["button_states"].append(np.copy(state["button_states"]))
        
        # logging useful metrics from info dict
        env_info = {}
        for key, value in info.items():
            if key.startswith("distance"):
                env_info[key] = value
        # always log this at every step
        logger.log(env_info, "env", step=log_step)

        if 'antmaze' in FLAGS.env_name and (
            'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
        ):
            # Adjust reward for D4RL antmaze.
            int_reward = int_reward - 1.0

        if FLAGS.sparse:
            assert int_reward <= 0.0
            int_reward = (int_reward != 0.0) * -1.0

        transition = dict(
            observations=ob,
            actions=action,
            rewards=int_reward,
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_ob,
        )
        replay_buffer.add_transition(transition)
        
        # done
        if done:
            ob, _ = env.reset()
            action_queue = []  # reset the action queue
        else:
            ob = next_ob

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample_sequence(config['batch_size'] * FLAGS.utd_ratio,
                        sequence_length=FLAGS.horizon_length, discount=discount,
                        truncate_reward_at_success=config.get('truncate_reward_at_success', False))
            batch = jax.tree.map(lambda x: x.reshape((
                FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), batch)

            agent, update_info["online_agent"] = agent.batch_update(batch)
            
        if i % FLAGS.log_interval == 0:
            for key, info in update_info.items():
                logger.log(info, key, step=log_step)
            update_info = {}

            if FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0:
                want_video = FLAGS.video_episodes > 0
                use_value_replay = want_video and hasattr(agent, 'get_value')
                eval_info, trajs, cur_renders = evaluate_multi_seed(
                    agent=agent,
                    env=eval_env,
                    action_dim=action_dim,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=0 if use_value_replay else FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    num_seeds=FLAGS.num_eval_seeds,
                    base_seed=FLAGS.eval_noise_seed + log_step * FLAGS.num_eval_seeds,
                )

                logger.log(eval_info, "eval", step=log_step)
                last_eval_info = eval_info
                if want_video:
                    if use_value_replay:
                        composite_frames, _, _ = render_value_replay(
                            agent, eval_env, seed=FLAGS.eval_noise_seed + log_step,
                        )
                        video = get_wandb_video([composite_frames])
                    else:
                        video = get_wandb_video(cur_renders)
                    wandb.log({"video": video}, step=log_step)
 
        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step, max_to_keep=FLAGS.max_checkpoints)

    end_time = time.time()
    
    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    if FLAGS.save_all_online_states:
        c_data = {"steps": np.array(data["steps"]),
                 "qpos": np.stack(data["qpos"], axis=0), 
                 "qvel": np.stack(data["qvel"], axis=0), 
                 "obs": np.stack(data["obs"], axis=0), 
                 "offline_time": online_init_time - offline_init_time,
                 "online_time": end_time - online_init_time,
        }
        if len(data["button_states"]) != 0:
            c_data["button_states"] = np.stack(data["button_states"], axis=0)
        np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url)

    final_row = dict(
        run_group=FLAGS.run_group,
        exp_name=exp_name,
        status='complete',
        end_time=time.strftime('%Y-%m-%d %H:%M:%S'),
        duration_sec=round(end_time - offline_init_time, 1),
    )
    final_row.update({f'result.{k}': v for k, v in last_eval_info.items()})
    upsert_run(FLAGS.run_registry_path, key_cols=['run_group', 'exp_name'], row=final_row)

if __name__ == '__main__':
    app.run(main)
