"""ACFQL (Q-Chunking, h=5) on cube-double, native cube-double-play-v0 (p_slip=0, no slip mechanic
involved -- this is the real native dataset, not our generator's p_slip=0.0 no-op reproduction of
it), task1, seed=0.

Deterministic-dynamics counterpart to acfql_h5_seed0.py's pslip0.025 config in this directory --
same hyperparameters, only config.task.slip.p_slip / dataset_dir / exp_name differ.

action_chunking=True here means the critic conditions on the full 5-step action chunk (CLAUDE.md
baseline #3, Q-Chunking), NOT n-step TD (baseline #2, which would be action_chunking=False). See
acfql_h1_seed0.py (pslip0 sibling) and acfql_h5_seed0.py (pslip0.025 sibling) in this directory for
the FQL-paper hyperparameter provenance and the eval-time-policy caveat; both apply identically here.
"""
from configs.base_experiment import get_config as get_base_config


def get_config():
    config = get_base_config()
    config.run_group = 'cube_double_slip_sweep'
    config.seed = 0

    config.task.env_name = 'cube-double-singletask-task1-v0'
    config.task.dataset_dir = '/tmp2/hubertchang/stochasticity/data/cube_double/native'
    config.task.slip.p_slip = 0.0

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

    config.exp_name = 'acfql_h5_disc0.99_alpha300_hidden512_pslip0_seed0'

    return config
