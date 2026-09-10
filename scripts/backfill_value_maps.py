"""Render the value/action diagnostic figure for EVERY saved checkpoint of a finished run and
upload the series to that run's existing wandb entry.

main.py now logs this figure during training (--value_map_interval), but runs that finished before
that existed only have checkpoints on disk. This backfills them without retraining.

Two implementation details that matter:

* The dataset is loaded once per run, not once per checkpoint. Restoring params into an
  already-constructed agent is nearly free, while building the env + a 10M-transition dataset is
  ~30s -- so looping checkpoints inside one process turns ~35s/checkpoint into ~10s.
* wandb rejects logging at a step earlier than a resumed run's current step, so the checkpoint
  number cannot be used as the wandb step. Instead the series is logged against a custom step
  metric (`diag/ckpt_step`) via wandb.define_metric, which lets the media panel and the scalar
  panels be viewed against training step regardless of internal wandb step ordering.

Run with:
  python scripts/backfill_value_maps.py --exp_config=configs/exp/<name>.py --save_dir=<run dir>
"""
import glob
import os
import re
import sys

import numpy as np
from absl import app, flags
from ml_collections import config_flags

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb

import envs  # noqa: registers the custom pointmaze ids
from utils.exp_config import load_agent
from utils.flax_utils import restore_agent_with_file
from utils.maze_oracle import (make_value_probe, value_error_stats, action_agreement_stats,
                                diagnostic_figure)

FLAGS = flags.FLAGS

flags.DEFINE_string('save_dir', None, 'Run directory containing params_*.pkl and wandb_id.txt.')
flags.DEFINE_string('wandb_project', 'cgq', 'wandb project.')
flags.DEFINE_string('wandb_entity', 'bb-chang-nation-taiwan-university', 'wandb entity.')
flags.DEFINE_bool('upload', True, 'Upload to wandb. False = compute and save PNGs locally only.')
flags.DEFINE_string('png_dir', None, 'Also write each figure here as a PNG (optional).')
flags.DEFINE_float('resolution', 0.5, 'Probe grid resolution (maze units).')
flags.DEFINE_integer('seed', 0, 'Seed for probe subsampling and policy sampling.')
config_flags.DEFINE_config_file('exp_config', default=None, lock_config=False)


def checkpoints(save_dir):
    out = []
    for f in glob.glob(os.path.join(save_dir, 'params_*.pkl')):
        m = re.search(r'params_(\d+)\.pkl$', f)
        if m:
            out.append((int(m.group(1)), f))
    return sorted(out)


def main(_):
    ec = FLAGS.exp_config
    ckpts = checkpoints(FLAGS.save_dir)
    if not ckpts:
        print(f'no checkpoints in {FLAGS.save_dir}')
        return
    print(f'{len(ckpts)} checkpoints: {[s for s, _ in ckpts]}')

    # Build env/dataset/agent once; later checkpoints just swap params in.
    agent, agent_config, _, eval_env = load_agent(ec, ckpts[0][1], seed=FLAGS.seed)
    eval_env.reset(seed=FLAGS.seed)
    probe = make_value_probe(eval_env, float(ec.value.discount), resolution=FLAGS.resolution,
                             seed=FLAGS.seed)
    if probe is None:
        print('no oracle probe available for this env; nothing to do')
        return
    print(f"probe: {len(probe['observations']):,} states")

    run = None
    if FLAGS.upload:
        id_path = os.path.join(FLAGS.save_dir, 'wandb_id.txt')
        if not os.path.exists(id_path):
            print(f'no wandb_id.txt in {FLAGS.save_dir}; rerun with --noupload')
            return
        run_id = open(id_path).read().strip()
        run = wandb.init(project=FLAGS.wandb_project, entity=FLAGS.wandb_entity, id=run_id,
                         resume='must')
        wandb.define_metric('diag/ckpt_step')
        wandb.define_metric('diag/*', step_metric='diag/ckpt_step')
        print(f'resumed wandb run {run_id}')

    if FLAGS.png_dir:
        os.makedirs(FLAGS.png_dir, exist_ok=True)

    for step, path in ckpts:
        agent = restore_agent_with_file(agent, path)
        stats = value_error_stats(agent, agent_config, probe, seed=FLAGS.seed)
        agree, maps = action_agreement_stats(agent, agent_config, probe, seed=FLAGS.seed,
                                             return_maps=True)
        img = diagnostic_figure(agent, agent_config, probe, eval_env, maps=maps)

        print(f'  {step:>7d}  MAE {stats["value_mae"]:8.1f}  corr {stats["value_corr"]:.3f}  '
              f'greedy_cos {agree["greedy_action_cos"]:+.3f}  policy_cos {agree["policy_action_cos"]:+.3f}  '
              f'policy_vs_greedy {agree["policy_vs_greedy_cos"]:+.3f}', flush=True)

        if FLAGS.png_dir is not None and img is not None:
            import imageio.v2 as imageio
            imageio.imwrite(os.path.join(FLAGS.png_dir, f'valuemap_{step}.png'), img)

        if run is not None:
            payload = {'diag/ckpt_step': step}
            payload.update({f'diag/{k}': v for k, v in stats.items()})
            payload.update({f'diag/{k}': v for k, v in agree.items()})
            if img is not None:
                payload['diag/value_map'] = wandb.Image(img, caption=f'step {step}')
            wandb.log(payload)

    if run is not None:
        run.finish()
    print('done')


if __name__ == '__main__':
    app.run(main)
