"""Render a trained agent's V(s) alongside the actual eval rollout frame it was computed on, for
cube envs where a spatial value-map heatmap doesn't apply -- there's no 2D xy grid to sweep over a
high-dim manipulation observation. Diagnostic tool, standalone from a checkpoint; the same rollout
logic (evaluation.render_value_replay) is also wired live into main.py's training-loop eval.

Produces a side-by-side GIF: rendered env frame | running V(s_0..t) line plot with a marker at the
current step. If the exp config has task.slip.p_slip > 0, eval_env is wrapped with the same
SlipGripperWrapper used at real eval time (matching main.py's wiring exactly), and slip-actuated
steps are shaded on the value plot so drops are visible next to whatever V(s) does in response.

Usage:
  python scripts/render_cube_value_replay.py \
    --exp_config=configs/exp/cube_double_pslip0.025_task1_acfql_h1_seed0.py \
    --checkpoint=/tmp2/.../params_1000000.pkl \
    --save_path=renders/cube_value_replay.gif
"""
import glob
import importlib
import os
import sys

import imageio
from absl import app, flags
from ml_collections import config_flags

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # cgq/, for envs./utils./agents.
import ogbench.manipspace  # noqa: registers cube-* envs

from agents import agents
from envs.ogbench_utils import make_ogbench_env_and_datasets
from evaluation import render_value_replay
from utils.datasets import Dataset
from utils.flax_utils import restore_agent_with_file

FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint', None, 'Path to a params_*.pkl checkpoint.')
flags.DEFINE_integer('seed', 0, 'Seed for policy sampling and env reset.')
flags.DEFINE_float('eval_temperature', 0.0, 'Action sampling temperature (0 = greedy, matches real eval).')
flags.DEFINE_integer('max_steps', 400, 'Cap on rollout length (a full episode can be 1000+ steps).')
flags.DEFINE_string('save_path', 'renders/cube_value_replay.gif', 'Output GIF path.')
flags.DEFINE_integer('fps', 15, 'Output GIF frame rate.')
config_flags.DEFINE_config_file('exp_config', default=None, lock_config=False)


def main(_):
    ec = FLAGS.exp_config
    env_name = ec.task.env_name
    dataset_dir = ec.task.dataset_dir
    horizon_length = ec.value.horizon_length

    agent_module = importlib.import_module(f'agents.{ec.value.method}')
    agent_config = agent_module.get_config()
    for k, v in ec.value.items():
        if k != 'method' and k in agent_config:
            agent_config[k] = v
    for k, v in ec.policy.items():
        if k != 'method' and k in agent_config:
            agent_config[k] = v
    agent_config['horizon_length'] = horizon_length

    dataset_paths = [f for f in sorted(glob.glob(f'{dataset_dir}/*.npz')) if '-val.npz' not in f]
    env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
        env_name,
        dataset_path=dataset_paths[0],
        compact_dataset=False,
    )

    if ec.task.slip.p_slip > 0:
        from envs.slip_wrapper import SlipGripperWrapper
        eval_env = SlipGripperWrapper(
            eval_env,
            p_slip=ec.task.slip.p_slip,
            slip_seed=ec.task.eval_noise_seed,
            gripper_closed_threshold=ec.task.slip.gripper_closed_threshold,
            slip_open_target=ec.task.slip.slip_open_target,
            slip_duration=ec.task.slip.slip_duration,
        )
        print(f'eval_env wrapped with SlipGripperWrapper(p_slip={ec.task.slip.p_slip})')

    train_dataset = Dataset.create(**train_dataset)
    example_batch = train_dataset.sample(())

    agent_class = agents[agent_config['agent_name']]
    agent = agent_class.create(FLAGS.seed, example_batch['observations'], example_batch['actions'], agent_config)
    agent = restore_agent_with_file(agent, FLAGS.checkpoint)

    composite_frames, values, slip_flags = render_value_replay(
        agent, eval_env, seed=FLAGS.seed, max_steps=FLAGS.max_steps, temperature=FLAGS.eval_temperature,
    )

    print(f'Rollout: {len(composite_frames)} steps, V(s) range=[{min(values):.3f}, {max(values):.3f}], '
          f'slip steps={sum(slip_flags)}')

    os.makedirs(os.path.dirname(FLAGS.save_path) or '.', exist_ok=True)
    imageio.mimsave(FLAGS.save_path, composite_frames, fps=FLAGS.fps)
    print(f'Saved {len(composite_frames)} frames to {FLAGS.save_path}')


if __name__ == '__main__':
    app.run(main)
