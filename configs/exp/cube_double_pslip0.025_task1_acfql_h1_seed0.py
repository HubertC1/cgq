"""ACFQL on cube-double, our own generate_cube_slip.py output (per-step geometric hazard rate on
proprio/gripper_opening, p_slip=0.025 -- E[steps to trigger]=40, matching the empirical grip-window
duration for cube-double). config.task.slip.p_slip is set to the same value so eval_env gets
wrapped with the matching SlipGripperWrapper, matching eval-time stochasticity to training-time
stochasticity. task1, seed=0.

horizon_length=1 with action_chunking=True is a no-op chunk (chunk of 1 == single-step Bellman
backup) -- this is the control point for the acfql_h5 config in this same directory, which is
otherwise identical except horizon_length=5 (Q-Chunking baseline, CLAUDE.md baseline #3).

Hyperparameters are the FQL paper's own tuned cube-double config (Table 5/6): alpha=300.0,
hidden=512x4, actor_type='distill-ddpg' -- NOT the IQL-baseline hyperparameters used by the
cube_double_*_iql1step_* / *_aciql*step_* configs in this directory (those use hidden=256x4,
alpha in the AWR-temperature convention, an unrelated number). See config.value.alpha's comment in
configs/base_experiment.py for why this alpha is a distinct field from config.policy.awr_alpha.

Eval-time policy: acfql's own internally-trained distill-ddpg actor (actor_onestep_flow) is used at
eval, NOT a shared flow_rejection policy -- config.policy.method='flow_rejection' isn't wired up for
acfql yet (only iql/aciql expose a policy_method field; see main.py's guard). This is a known
departure from CLAUDE.md's "shared eval-time policy" methodology for this run; revisit once
flow_rejection support lands for acfql.

ASSUMPTION: p_slip=0.025 (the stochastic grasp preset) and task1 were picked to match every other
existing cube_double baseline in this run_group, since the request that produced this file didn't
pin either. Adjust config.task.slip.p_slip / config.task.env_name / dataset_dir if pslip0 (native,
deterministic) or a different task index was actually intended.
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

    config.exp_name = 'acfql_h1_disc0.99_alpha300_hidden512_pslip0.025_seed0'

    return config
