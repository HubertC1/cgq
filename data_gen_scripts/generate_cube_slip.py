"""Generate a cube-manipulation dataset with a reactive grasp-slip stress test for action chunking.

Ported from ogbench/data_gen_scripts/generate_manipspace.py's 'play'/cube path (imported pieces,
not edited in place). Only the cube task is supported here.

At p_slip=0.0 this is an exact no-op relative to native generation (see cube_slip_oracle.py's
docstring for why) -- same seed should reproduce the native dataset bit-for-bit.

The oracle (data_gen_scripts/cube_slip_oracle.SlipCubePlanOracle) decides, per pick-and-place
sub-trajectory, whether to slip (triggered by observed cube z-motion, not a plan-internal signal);
when it does, it returns an ACTUATED action with the gripper channel forced open for a few steps,
while `agent.last_clean_action` holds the oracle's true intended ("closed"/holding) action.
Mirroring envs/noise_wrapper.py's actuate-loose/log-tight convention: env.step() gets the actuated
action, but the dataset records the clean one.

Run with MUJOCO_GL=egl (headless rendering; the cube env renders internally even off-screen).
"""
import os
import pathlib
import sys
from collections import defaultdict

import gymnasium
import numpy as np
from absl import app, flags
from tqdm import trange

import ogbench.manipspace  # noqa: registers the cube-* envs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # cgq/, for data_gen_scripts.*
from data_gen_scripts.cube_slip_oracle import ClonePeekSlipCubePlanOracle, SlipCubePlanOracle

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-v0', 'Environment name (base, non-singletask).')
flags.DEFINE_string('save_path', None, 'Save path (a train .npz; a sibling -val.npz is also written).')
flags.DEFINE_float('noise', 0.1, "Plan-following action noise level (PlanOracle's own motion noise, "
                    'independent of the slip mechanic below).')
flags.DEFINE_float('noise_smoothing', 0.5, 'Action noise smoothing level for PlanOracle.')
flags.DEFINE_integer('num_episodes', 1000, 'Number of training episodes (10% more are generated for validation). '
                      'Default matches native cube-double-play-v0 exactly (1000 x 1001 = 1,001,000 steps).')
flags.DEFINE_integer('max_episode_steps', 1001, 'Maximum number of steps in an episode. Default matches native '
                      'cube-double-play-v0 exactly.')

flags.DEFINE_float('p_slip', 0.0, 'Per-step probability of triggering a slip while the gripper is '
                    'closed (memoryless -- E[steps to trigger] = 1/p_slip). p_slip=0.0 is an exact '
                    'no-op vs. native generation. Applies continuously through cascading regrabs too, '
                    'up to --max_slips_per_manipulation. Calibrate 1/p_slip against the empirical '
                    'grip-window duration for the task (measured for cube-double: ~40-47 steps).')
flags.DEFINE_float('gripper_closed_threshold', 0.5, 'proprio/gripper_opening value above which the '
                    'gripper counts as closed and the hazard check is active.')
flags.DEFINE_float('slip_open_target', 0.15, 'Gripper closedness (0=open, 1=closed) actuated during '
                    "a slip, overriding the current closed target.")
flags.DEFINE_integer('slip_duration', 5, 'Consecutive control steps the loosened gripper target is actuated.')
flags.DEFINE_integer('settle_steps', 10, 'Extra steps after the loosened window, before snapshotting the '
                      "cube's position for the regrab plan (lets it finish falling). Ignored if "
                      '--clone_peek is set.')
flags.DEFINE_bool('clone_peek', False, 'Use ClonePeekSlipCubePlanOracle instead of SlipCubePlanOracle: '
                   'zero real-env settle wait, regrab target found by forking a disposable clone of the '
                   'env forward --peek_settle_steps instead. See cube_slip_oracle.py for the known '
                   'privileged-information caveat this carries.')
flags.DEFINE_integer('peek_settle_steps', 40, 'Only used if --clone_peek. Control steps the clone is run '
                      'forward to let the drop finish before reading back the landing pose.')
flags.DEFINE_float('max_slips_per_manipulation', float('inf'), 'Cap on cascading regrab depth per '
                    'manipulation sub-trajectory. Defaults to unbounded, matching '
                    'SlipGripperWrapper\'s eval-time behavior (no cap); max_episode_steps is what '
                    'actually bounds worst-case episode length. Pass a finite value only to '
                    'deliberately give up on a manipulation after N cascading slips.')
flags.DEFINE_float('p_random_action', 0, 'Probability of selecting a random action instead of the '
                    "oracle's, matching generate_manipspace.py's own flag. Kept only so the "
                    'np.random.rand() draw happens in the same place every step as vanilla -- '
                    'required for the p_slip=0.0 no-op guarantee, not because we expect to use it.')


def main(_):
    np.random.seed(FLAGS.seed)

    env = gymnasium.make(
        FLAGS.env_name,
        terminate_at_goal=False,
        mode='data_collection',
        max_episode_steps=FLAGS.max_episode_steps,
    )

    if FLAGS.clone_peek:
        agent = ClonePeekSlipCubePlanOracle(
            env=env,
            noise=FLAGS.noise,
            noise_smoothing=FLAGS.noise_smoothing,
            p_slip=FLAGS.p_slip,
            gripper_closed_threshold=FLAGS.gripper_closed_threshold,
            slip_open_target=FLAGS.slip_open_target,
            slip_duration=FLAGS.slip_duration,
            max_slips_per_manipulation=FLAGS.max_slips_per_manipulation,
            env_name=FLAGS.env_name,
            peek_settle_steps=FLAGS.peek_settle_steps,
        )
    else:
        agent = SlipCubePlanOracle(
            env=env,
            noise=FLAGS.noise,
            noise_smoothing=FLAGS.noise_smoothing,
            p_slip=FLAGS.p_slip,
            gripper_closed_threshold=FLAGS.gripper_closed_threshold,
            slip_open_target=FLAGS.slip_open_target,
            slip_duration=FLAGS.slip_duration,
            settle_steps=FLAGS.settle_steps,
            max_slips_per_manipulation=FLAGS.max_slips_per_manipulation,
        )

    if 'single' in FLAGS.env_name:
        p_stack_range = (0.0, 0.0)
    elif 'double' in FLAGS.env_name:
        p_stack_range = (0.0, 0.25)
    elif 'triple' in FLAGS.env_name:
        p_stack_range = (0.05, 0.35)
    elif 'quadruple' in FLAGS.env_name:
        p_stack_range = (0.1, 0.5)
    elif 'octuple' in FLAGS.env_name:
        p_stack_range = (0.0, 0.35)
    else:
        p_stack_range = (0.5, 0.5)

    dataset = defaultdict(list)
    total_steps = 0
    total_train_steps = 0
    num_train_episodes = FLAGS.num_episodes
    num_val_episodes = FLAGS.num_episodes // 10

    num_slip_steps = 0
    num_manipulations = 0

    for ep_idx in trange(num_train_episodes + num_val_episodes):
        ob, info = env.reset()
        p_stack = np.random.uniform(*p_stack_range)

        agent.reset(ob, info)
        num_manipulations += 1

        done = False
        step = 0

        while not done:
            if np.random.rand() < FLAGS.p_random_action:
                # Matches generate_manipspace.py:124-127 exactly, including when p_random_action=0
                # -- the draw itself (not just the branch) must happen every step in the same
                # place for the global np.random stream to stay aligned with vanilla generation.
                # Vanilla never calls select_action() on this branch either, so neither do we.
                actuated_action = np.clip(np.array(env.action_space.sample()), -1, 1)
                clean_action = actuated_action
            else:
                actuated_action = np.clip(np.array(agent.select_action(ob, info)), -1, 1)
                clean_action = np.clip(np.array(agent.last_clean_action), -1, 1)

            next_ob, reward, terminated, truncated, info = env.step(actuated_action)
            done = terminated or truncated

            if agent.done:
                agent_ob, agent_info = env.unwrapped.set_new_target(p_stack=p_stack)
                agent.reset(agent_ob, agent_info)
                num_manipulations += 1
                info = agent_info

            is_slip_step = not np.allclose(actuated_action, clean_action)
            if is_slip_step:
                num_slip_steps += 1

            dataset['observations'].append(ob)
            dataset['actions'].append(clean_action)
            dataset['terminals'].append(done)
            dataset['qpos'].append(info['prev_qpos'])
            dataset['qvel'].append(info['prev_qvel'])
            dataset['slip'].append(is_slip_step)

            ob = next_ob
            step += 1

        total_steps += step
        if ep_idx < num_train_episodes:
            total_train_steps += step

    print('Total steps:', total_steps)
    print('Total manipulation sub-trajectories:', num_manipulations)
    print('Total slip-actuated steps:', num_slip_steps)

    train_path = FLAGS.save_path
    val_path = FLAGS.save_path.replace('.npz', '-val.npz')
    pathlib.Path(train_path).parent.mkdir(parents=True, exist_ok=True)

    train_dataset = {}
    val_dataset = {}
    for k, v in dataset.items():
        if 'observations' in k and v[0].dtype == np.uint8:
            dtype = np.uint8
        elif k in ('terminals', 'slip'):
            dtype = bool
        else:
            dtype = np.float32
        train_dataset[k] = np.array(v[:total_train_steps], dtype=dtype)
        val_dataset[k] = np.array(v[total_train_steps:], dtype=dtype)

    for path, split_dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **split_dataset)


if __name__ == '__main__':
    app.run(main)
