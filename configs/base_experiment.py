"""Fixed experiment-config template.

Every experiment file under configs/exp/ imports get_config() from here and
overrides only the fields that differ for that run. The shape (seed/run_group/
exp_name/task/value/policy/train) never changes across method or task -- only
the values inside do. See main.py's `if FLAGS.exp_config is not None:` block
for how this gets mapped onto the agent + the rest of the training loop.

value vs. policy split follows CLAUDE.md's Phase 1 methodology: `value` is the
critic objective under study (the independent variable), `policy` is the
extraction method, which we're holding fixed (AWR today) while we study value
objectives, and which we'll swap for rejection sampling / a flow policy later
without touching this schema's shape.

Run with:  python main.py --exp_config=configs/exp/<your_file>.py
"""
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_group = 'Debug'
    config.exp_name = None  # None -> auto-derived from value/task fields (see main.py)

    # --- task: env, dataset, dynamics noise ---
    config.task = ml_collections.ConfigDict()
    config.task.env_name = 'pointmaze-large-navigate-singletask-task2-v0'
    config.task.dataset_dir = None  # required; path passed to --ogbench_dataset_dir
    config.task.max_episode_steps = None  # None -> registered env default (1000)
    config.task.noise = ml_collections.ConfigDict()
    config.task.noise.type = 'none'  # 'none' | 'constant' | 'near_walls' | 'region' | 'radial_from_goal'
    # | 'composite' -- see envs/noise_wrapper.NOISE_FN_REGISTRY. `params` is intentionally an open
    # dict: its keys are preset-specific (e.g. {'sigma_0': 0.2} for 'constant', {'near_sigma':...,
    # 'far_sigma':..., 'wall_distance_threshold':...} for 'near_walls').
    config.task.noise.params = ml_collections.ConfigDict()
    config.task.noise_seed = 12345  # training/rollout env noise RNG seed
    config.task.eval_noise_seed = 67890  # must differ from noise_seed

    # cube-family grasp-slip stochasticity (envs/slip_wrapper.SlipGripperWrapper) -- a separate
    # mechanism from config.task.noise above (that's the locomotion actuator-noise wrapper).
    # p_slip=0.0 (default) means no wrapper is attached at all. Only wraps eval_env, not the
    # (unused, offline-only) training rollout env -- matches whatever p_slip the dataset named in
    # config.task.dataset_dir was generated with, via data_gen_scripts/cube_slip_oracle.py, so
    # eval-time stochasticity matches training-time stochasticity.
    config.task.slip = ml_collections.ConfigDict()
    config.task.slip.p_slip = 0.0
    config.task.slip.gripper_closed_threshold = 0.5
    config.task.slip.slip_open_target = 0.15
    config.task.slip.slip_duration = 5

    # --- value: the critic objective under study (independent variable) ---
    config.value = ml_collections.ConfigDict()
    config.value.method = 'iql'  # selects agents/{method}.py -- 'iql' | 'aciql' | ...
    config.value.horizon_length = 1  # chunk length AND n-step return length
    config.value.discount = 0.99  # kept identical to top-level FLAGS.discount by construction
    config.value.expectile = 0.9
    config.value.q_agg = 'mean'
    config.value.tau = 0.005
    config.value.num_qs = 2
    config.value.value_hidden_dims = (512, 512, 512, 512)
    config.value.layer_norm = True
    config.value.lr = 3e-4
    config.value.weight_decay = 0.0
    config.value.batch_size = 256  # was missing from this template despite being a real field on
    # every agent's own get_config() (e.g. agents/iql.py) -- added so exp configs can override it
    # without ml_collections.ConfigDict raising AttributeError on an undeclared key.
    config.value.truncate_reward_at_success = False  # Only matters when horizon_length > 1: once a
    # step within the sampled n-step/chunk window reaches success (mask==0), freeze the reward sum
    # there instead of continuing to accumulate whatever the raw (non-goal-directed) play data does
    # afterward. See utils/datasets.py's sample_sequence. No-op at horizon_length==1.
    config.value.action_chunking = True  # acfql/cgq/dqc only: True -> critic conditions on the
    # full h-step action chunk (the "Q-Chunking" baseline). False -> critic conditions on only the
    # first action of the window with an h-step target, i.e. genuine n-step TD (the "n-step TD"
    # baseline). Same horizon_length either way -- this flag picks which CLAUDE.md baseline
    # acfql.py implements. No-op at horizon_length==1 (a 1-chunk == a 1-step n-step return).
    config.value.alpha = 100.0  # acfql/cgq only: BC/distillation coefficient in the actor-critic's
    # distill-ddpg loss (higher = more conservative / more BC-like). This is the actor-critic's own
    # 'alpha' (FQL/CGQ paper notation, left as-is) -- NOT the same knob as config.policy.awr_alpha
    # below (AWR inverse temperature, iql/aciql only). The two are unrelated concepts that used to
    # collide under one name; see config.policy.awr_alpha's comment and main.py's merge logic.
    config.value.actor_type = 'distill-ddpg'  # acfql/cgq only: 'distill-ddpg' | 'best-of-n'. Selects
    # the internal joint-actor mechanics used while training the critic. Irrelevant to eval --
    # CLAUDE.md's shared eval-time policy (flow rejection sampling) is what acts at eval, not this.

    # --- policy: extraction method, held fixed while `value` varies ---
    config.policy = ml_collections.ConfigDict()
    config.policy.method = 'awr'  # 'awr' | 'flow_rejection' (iql/aciql only -- see those agents'
    # get_config() for the flow_rejection-specific fields below). 'flow_rejection' is CLAUDE.md's
    # Phase 1 target extraction: best-of-N rejection sampling against the agent's own critic,
    # using a *pretrained, frozen* flow-matching BC policy (config.policy.bc_checkpoint) -- never
    # trained jointly with the critic here, so the exact same policy checkpoint can be shared
    # across every critic/seed that trains on the same dataset/horizon_length. Doesn't touch
    # value_loss/critic_loss.
    config.policy.awr_alpha = 10.0  # AWR inverse temperature (higher = more greedy toward Q).
    # Unused when policy.method='flow_rejection'. Renamed from config.policy.alpha (no longer
    # declared here): as a default declared on every config regardless of value.method, that name
    # collided with acfql/cgq's own agent-config field 'alpha' (BC/distillation coefficient, see
    # config.value.alpha above) -- main.py's merge logic now maps awr_alpha onto agent_config['alpha']
    # explicitly, only for iql/aciql, so it can never clobber the actor-critic's own 'alpha'.
    # Existing iql/aciql exp configs under configs/exp/ that still write the old
    # `config.policy.alpha = ...` keep working unchanged: since it's no longer predeclared here,
    # that line just adds a same-named orphan field on that one config's ConfigDict, which still
    # reaches agent_config['alpha'] via main.py's plain name-match loop (harmless coincidence, not
    # the awr_alpha routing) -- no file needs updating for this to work.
    config.policy.actor_hidden_dims = (512, 512, 512, 512)  # AWR actor only.
    config.policy.actor_layer_norm = False
    config.policy.const_std = True
    config.policy.bc_checkpoint = ml_collections.config_dict.placeholder(str)  # Path to a
    # pretrained agents/bc.py flow-BC params_*.pkl (see configs/exp/cube_double_pslip*_flowbc_h*.py
    # for how those are trained). Required when policy.method='flow_rejection'; must have been
    # trained on this same dataset/env with horizon_length == this config's value.horizon_length.
    config.policy.bc_actor_hidden_dims = (256, 256, 256, 256)  # Architecture of the checkpoint at
    # bc_checkpoint -- must match exactly what it was trained with (flow_rejection only).
    config.policy.flow_steps = 10  # Euler integration steps for the flow BC policy (flow_rejection only).
    config.policy.actor_num_samples = 16  # N candidates for rejection sampling (flow_rejection only).

    # --- train: schedule ---
    config.train = ml_collections.ConfigDict()
    config.train.offline_steps = 1_000_000
    config.train.eval_episodes = None  # None -> main.py's --eval_episodes default (50). Worth
    # lowering on envs with a long max_episode_steps: a failed episode runs the full cap, so on
    # e.g. pointmaze-umazelarge (cap 6000) 50 episodes cost 300k env steps *per eval*.
    config.train.video_episodes = None  # None -> main.py's --video_episodes default (1).
    config.train.num_eval_seeds = None  # None -> main.py's --num_eval_seeds default (5).
    config.train.value_map_interval = None  # None -> main.py's --value_map_interval (0/off).
    # Set to eval_interval to push the learned-vs-oracle value/action diagnostic figure to
    # wandb at every eval (2-D pointmaze only; see utils/maze_oracle.diagnostic_figure).
    # Eval cost is linear in this: it re-runs the whole eval_episodes batch per seed.
    config.train.eval_interval = 100_000
    config.train.save_interval = 100_000  # tied to eval_interval -- one checkpoint saved per eval
    # (see main.py's FLAGS.max_checkpoints, raised from 1 to 20 so these don't get pruned mid-run)
    config.train.dataset_replace_interval = 1000

    return config
