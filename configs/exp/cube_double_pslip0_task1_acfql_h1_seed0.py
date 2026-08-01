"""ACFQL on cube-double, native cube-double-play-v0 (p_slip=0, no slip mechanic involved -- this is
the real native dataset, not our generator's p_slip=0.0 no-op reproduction of it), task1, seed=0.

Deterministic-dynamics counterpart to acfql_h1_seed0.py's pslip0.025 config in this directory --
same hyperparameters, only config.task.slip.p_slip / dataset_dir / exp_name differ. Pairing the two
is what lets a pslip0 vs pslip0.025 comparison actually isolate dynamics stochasticity as the
independent variable (CLAUDE.md's "OLC-violation bias... under uniform noise" motivation).

horizon_length=1 with action_chunking=True is a no-op chunk (chunk of 1 == single-step Bellman
backup) -- this is the control point for the acfql_h5 config in this same directory.

Hyperparameters are the FQL paper's own tuned cube-double config (Table 5/6): alpha=300.0,
hidden=512x4, actor_type='distill-ddpg' -- see acfql_h1_seed0.py (pslip0.025 sibling) for the
alpha-field-naming note and the eval-time-policy caveat (acfql's own internal distill-ddpg actor is
used at eval, not a shared flow_rejection policy); both apply identically here.

ASSUMPTION: task1 was picked to match every other existing cube_double baseline in this run_group.
Adjust config.task.env_name / dataset_dir if a different task index was actually intended.
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
    config.value.horizon_length = 1
    config.value.action_chunking = True  # no-op at h=1
    config.value.discount = 0.99
    config.value.q_agg = 'mean'
    config.value.num_qs = 2
    config.value.value_hidden_dims = (512, 512, 512, 512)
    config.value.alpha = 300.0  # FQL's tuned cube-double alpha
    config.value.actor_type = 'distill-ddpg'
    config.value.batch_size = 256

    config.train.offline_steps = 1_000_000

    config.exp_name = 'acfql_h1_disc0.99_alpha300_hidden512_pslip0_seed0'

    return config
