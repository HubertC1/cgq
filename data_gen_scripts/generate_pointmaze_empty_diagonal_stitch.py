"""Generate a "1-step-stitch, chunk-hostile" dataset on the walls-free open room
(pointmaze-empty-v0, envs/custom_empty_maze.py).

Construction (per the design discussion, not ported from anywhere -- this dataset type does not
exist upstream):

1. Draw the near-optimal line from start to finish: the diagonal from the room's bottom-left free
   cell to its top-right free cell (same corners as pointmaze-large's own task1 pair). Segment it
   into steps of length --step_size (default 0.2, the exact xy distance PointEnv's oracle-style
   unit-vector action covers in one MDP step -- see PointEnv.step: `qpos += 0.2 * action`, and
   action here is a unit vector). This gives points p_0, p_1, ..., p_n along the diagonal, n =
   floor(diagonal_length / step_size).

2. For every consecutive pair (p_k, p_{k+1}), generate --repeats_per_segment independent episodes
   that each: start on a random side of the diagonal, wander a little (optional, cosmetic), walk
   straight to p_k, take exactly ONE step straight to p_{k+1} (exact because |p_{k+1} - p_k| ==
   step_size by construction -- this is the one recorded transition per episode that must look
   like a genuine step of the optimal diagonal path), then cross to the *other* side of the
   diagonal and wander a little more before the episode ends.

No single episode ever demonstrates two consecutive optimal-path transitions -- the full
p_0 -> p_1 -> ... -> p_n path only exists by chaining transitions learned from *different*
episodes. A 1-step TD backup can still recover it (each (p_k, a, p_{k+1}) transition is a real,
individually-experienced Bellman backup that chains transitively). A method that backs up
multi-step chunks *within one trajectory* cannot recover it from any single episode, since the
chunk starting at p_k always diverges off the diagonal after exactly one step in every episode
that ever visits p_k -- the true 2+-step optimal continuation from p_k is categorically absent
from the data, on purpose.

All waypoints (start, pre/post-wander, crossing points) are continuous xy positions, not grid
cells -- reachable via CustomEmptyPointMazeEnv.set_xy / the per-step direction-to-target actor
implemented in `move_to` below, bypassing the grid-cell/BFS oracle machinery entirely (moot here
since the room has no interior walls to route around).

Output schema matches every other locomaze dataset script exactly (observations, actions,
terminals, qpos, qvel).
"""
import pathlib
import sys
import os
from collections import defaultdict

import gymnasium
import numpy as np
from absl import app, flags

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # cgq/, for envs.*
import envs.custom_empty_maze  # noqa: registers pointmaze-empty-v0

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'pointmaze-empty-v0', 'Walls-free point-mass room env name.')
flags.DEFINE_string('save_path', None, 'Save path (a train .npz; a sibling -val.npz is also written).')
flags.DEFINE_integer('seed', 0, 'Seed for numpy global RNG (waypoint sampling, side draws).')
flags.DEFINE_integer('max_episode_steps', 500, 'TimeLimit passed to gymnasium.make -- a generous '
                      'safety cap; real episodes here are a few dozen steps at most.')

flags.DEFINE_float('step_size', 0.2, 'MDP-step length: the xy distance PointEnv covers in one step '
                    'under a unit-vector action (qpos += 0.2 * action). The diagonal is segmented '
                    'into steps of exactly this length, and it is also the max per-step travel '
                    'distance used everywhere else in this script (see `move_to`).')
flags.DEFINE_integer('repeats_per_segment', 4, 'Number of independent training episodes generated '
                      'per consecutive diagonal pair (p_k, p_{k+1}).')
flags.DEFINE_integer('val_repeats_per_segment', 1, 'Same, for the validation split.')

flags.DEFINE_float('wander_min_radius', 2.0, 'Min distance (maze units) from the crossing point '
                    'for a random pre/post-wander waypoint.')
flags.DEFINE_float('wander_max_radius', 6.0, 'Max distance (maze units) from the crossing point '
                    'for a random pre/post-wander waypoint.')
flags.DEFINE_integer('min_wander_waypoints', 1, 'Min number of random waypoints before entering '
                      'the line and after leaving it (each drawn independently).')
flags.DEFINE_integer('max_wander_waypoints', 3, 'Max number of random waypoints before entering '
                      'the line and after leaving it.')
flags.DEFINE_float('wall_margin', 1.0, 'Keep sampled waypoints at least this far (maze units) '
                    'inside the open interior, away from the border wall surface.')


def move_to(env, dataset, target_xy, step_size, max_steps=2000):
    """Drive the point mass in a straight line to target_xy, recording every real env.step.

    Each step moves at the max per-step speed (a unit-vector action, covering exactly step_size)
    except the final step, which is scaled down to land exactly on target_xy -- exact because a
    vector's per-component magnitude never exceeds its norm, so `remaining / step_size` is always
    within [-1, 1] once `|remaining| <= step_size`.
    """
    for _ in range(max_steps):
        cur_xy = env.unwrapped.get_xy().copy()
        delta = np.asarray(target_xy, dtype=np.float64) - cur_xy
        dist = np.linalg.norm(delta)
        if dist < 1e-9:
            return
        if dist > step_size:
            action = delta / dist
        else:
            action = delta / step_size
        action = np.clip(action, -1, 1).astype(np.float32)

        ob = env.unwrapped.get_ob()
        next_ob, reward, terminated, truncated, info = env.step(action)
        dataset['observations'].append(ob)
        dataset['actions'].append(action)
        dataset['terminals'].append(False)
        dataset['qpos'].append(info['prev_qpos'])
        dataset['qvel'].append(info['prev_qvel'])

        if dist <= step_size:
            return
    raise RuntimeError(f'move_to did not converge to {target_xy} within {max_steps} steps.')


def sample_point_on_side(center, side_sign, p0, normal, bounds_lo, bounds_hi, min_r, max_r, max_tries=200):
    """Sample a random point within [min_r, max_r] of `center`, on the given side of the diagonal
    (side_sign * dot(point - p0, normal) > 0), clipped inside [bounds_lo, bounds_hi]."""
    for _ in range(max_tries):
        r = np.random.uniform(min_r, max_r)
        theta = np.random.uniform(0, 2 * np.pi)
        cand = center + r * np.array([np.cos(theta), np.sin(theta)])
        if not (bounds_lo[0] <= cand[0] <= bounds_hi[0] and bounds_lo[1] <= cand[1] <= bounds_hi[1]):
            continue
        if side_sign * np.dot(cand - p0, normal) <= 1e-6:
            continue
        return cand
    # Fallback: nudge directly off the line by min_r, then clip into bounds.
    fallback = center + side_sign * min_r * normal
    return np.clip(fallback, bounds_lo, bounds_hi)


def run_episode(env, dataset, p_k, p_kp1, side_sign, p0, normal, bounds_lo, bounds_hi):
    """One episode: wander on `side_sign`'s side, cross at p_k -> p_kp1, wander on the other side."""
    start_xy = sample_point_on_side(
        p_k, side_sign, p0, normal, bounds_lo, bounds_hi, FLAGS.wander_min_radius, FLAGS.wander_max_radius
    )
    env.unwrapped.set_xy(start_xy)

    num_pre = np.random.randint(FLAGS.min_wander_waypoints, FLAGS.max_wander_waypoints + 1)
    cur_center = start_xy
    for _ in range(num_pre):
        wp = sample_point_on_side(
            cur_center, side_sign, p0, normal, bounds_lo, bounds_hi,
            FLAGS.wander_min_radius, FLAGS.wander_max_radius,
        )
        move_to(env, dataset, wp, FLAGS.step_size)
        cur_center = wp

    # Approach and cross exactly at p_k, then the one guaranteed optimal-path transition.
    move_to(env, dataset, p_k, FLAGS.step_size)
    move_to(env, dataset, p_kp1, FLAGS.step_size)

    num_post = np.random.randint(FLAGS.min_wander_waypoints, FLAGS.max_wander_waypoints + 1)
    cur_center = p_kp1
    for _ in range(num_post):
        wp = sample_point_on_side(
            cur_center, -side_sign, p0, normal, bounds_lo, bounds_hi,
            FLAGS.wander_min_radius, FLAGS.wander_max_radius,
        )
        move_to(env, dataset, wp, FLAGS.step_size)
        cur_center = wp

    dataset['terminals'][-1] = True


def main(_):
    np.random.seed(FLAGS.seed)

    env = gymnasium.make(FLAGS.env_name, terminate_at_goal=False, max_episode_steps=FLAGS.max_episode_steps)
    maze_map = env.unwrapped.maze_map
    rows, cols = maze_map.shape

    p0 = np.array(env.unwrapped.ij_to_xy((1, 1)), dtype=np.float64)
    p_end = np.array(env.unwrapped.ij_to_xy((rows - 2, cols - 2)), dtype=np.float64)
    diag = p_end - p0
    length = np.linalg.norm(diag)
    unit_dir = diag / length
    normal = np.array([-unit_dir[1], unit_dir[0]])  # perpendicular, defines the two "sides"

    maze_unit = env.unwrapped._maze_unit
    half_free = maze_unit / 2 - FLAGS.wall_margin
    bounds_lo = np.minimum(p0, p_end) - half_free
    bounds_hi = np.maximum(p0, p_end) + half_free

    n = int(length // FLAGS.step_size)
    points = [p0 + i * FLAGS.step_size * unit_dir for i in range(n + 1)]
    print(f'Diagonal from {tuple(p0)} to {tuple(p_end)}, length={length:.3f}, '
          f'{n} segments of length {FLAGS.step_size}.')

    # (segment_idx, side_sign, is_val) for every episode to generate, train first then val, so the
    # existing total_train_steps-based split (below) works unchanged.
    episode_specs = []
    for k in range(n):
        for _ in range(FLAGS.repeats_per_segment):
            episode_specs.append((k, np.random.choice([-1, 1]), False))
    for k in range(n):
        for _ in range(FLAGS.val_repeats_per_segment):
            episode_specs.append((k, np.random.choice([-1, 1]), True))

    dataset = defaultdict(list)
    total_steps = 0
    total_train_steps = 0

    env.reset(seed=FLAGS.seed)
    for k, side_sign, is_val in episode_specs:
        step_before = len(dataset['terminals'])
        run_episode(env, dataset, points[k], points[k + 1], side_sign, p0, normal, bounds_lo, bounds_hi)
        step_after = len(dataset['terminals'])
        total_steps += step_after - step_before
        if not is_val:
            total_train_steps = step_after

    print('Total steps:', total_steps)
    print('Total episodes:', len(episode_specs))

    train_path = FLAGS.save_path
    val_path = FLAGS.save_path.replace('.npz', '-val.npz')
    pathlib.Path(train_path).parent.mkdir(parents=True, exist_ok=True)

    train_dataset = {}
    val_dataset = {}
    for k, v in dataset.items():
        dtype = bool if k == 'terminals' else np.float32
        train_dataset[k] = np.array(v[:total_train_steps], dtype=dtype)
        val_dataset[k] = np.array(v[total_train_steps:], dtype=dtype)

    for path, split_dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **split_dataset)


if __name__ == '__main__':
    app.run(main)
