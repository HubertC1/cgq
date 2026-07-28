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

    # --- policy: extraction method, held fixed while `value` varies ---
    config.policy = ml_collections.ConfigDict()
    config.policy.method = 'awr'  # only 'awr' is implemented today; 'rejection_sampling' /
    # 'flow' are placeholders for later -- switching this field alone won't do anything until
    # main.py grows a branch for it.
    config.policy.alpha = 10.0  # AWR inverse temperature (higher = more greedy toward Q)
    config.policy.actor_hidden_dims = (512, 512, 512, 512)
    config.policy.actor_layer_norm = False
    config.policy.const_std = True

    # --- train: schedule ---
    config.train = ml_collections.ConfigDict()
    config.train.offline_steps = 1_000_000
    config.train.eval_interval = 100_000
    config.train.save_interval = 200_000
    config.train.dataset_replace_interval = 1000

    return config
