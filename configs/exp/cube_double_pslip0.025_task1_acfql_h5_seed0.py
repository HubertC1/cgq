"""ACFQL (Q-Chunking, h=5) on cube-double, our own generate_cube_slip.py output (per-step geometric
hazard rate on proprio/gripper_opening, p_slip=0.025 -- E[steps to trigger]=40, matching the
empirical grip-window duration for cube-double). config.task.slip.p_slip is set to the same value
so eval_env gets wrapped with the matching SlipGripperWrapper, matching eval-time stochasticity to
training-time stochasticity. task1, seed=0.

Identical to acfql_h1_seed0.py in this directory except horizon_length=5: action_chunking=True
here means the critic conditions on the full 5-step action chunk (CLAUDE.md baseline #3,
Q-Chunking), NOT n-step TD (baseline #2, which would be action_chunking=False -- the critic
conditioning on only the first action of a 5-step target window). Confirmed intentional as a
second Q-Chunking data point (h=1 vs h=5), not the separate n-step TD baseline.

See acfql_h1_seed0.py's docstring for the FQL-paper hyperparameter provenance and the eval-policy
caveat (acfql's own internal distill-ddpg actor is used at eval, not a shared flow_rejection
policy) and the p_slip=0.025/task1 assumption -- both apply identically here.
"""
from configs.base_experiment import get_config as get_base_config


def get_config():
    config = get_base_config()
    config.run_group = 'cube_double_slip_sweep'
    config.seed = 0

    config.task.env_name = 'cube-double-singletask-task1-v0'
    config.task.dataset_dir = '/tmp2/hubertchang/stochasticity/data/cube_double/pslip0.025'
    config.task.slip.p_slip = 0.025

    config.value.method = 'acfql'
    config.value.horizon_length = 5
    config.value.action_chunking = True
    config.value.discount = 0.99
    config.value.q_agg = 'mean'
    config.value.num_qs = 2
    config.value.value_hidden_dims = (512, 512, 512, 512)
    config.value.alpha = 300.0
    config.value.actor_type = 'distill-ddpg'
    config.value.batch_size = 256

    config.train.offline_steps = 1_000_000

    config.exp_name = 'acfql_h5_disc0.99_alpha300_hidden512_pslip0.025_seed0'

    return config
