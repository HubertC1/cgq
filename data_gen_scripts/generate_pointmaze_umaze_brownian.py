"""Generate a pure Brownian-motion dataset on the U-maze (pointmaze-umaze-v0,
envs/custom_umaze.py): every episode is an unbiased random walk from a random free starting
point, with the direction resampled independently at *every* MDP step.

Why direction-per-step rather than the waypoint-following wander of
generate_pointmaze_empty_diagonal_stitch.py: that script's `move_to` drives in a *straight line*
to each wander waypoint, so every wander leg is 10-30 consecutive identical unit-vector actions.
That fills the dataset with long, coherent, straight multi-step action chunks in every direction
-- exactly the supervision an h-step chunk actor needs, which quietly defeats the "chunk-hostile"
intent. Resampling the direction every single step removes *all* coherent multi-step motion from
the data: an h-step chunk of consistent actions is itself out of distribution, so any method that
backs up multi-step chunks has nothing to imitate, while 1-step TD still sees a complete, densely
covered transition graph to chain through. Combined with the U-maze's detour geometry (where the
locally greedy "point at the goal" action is a wall, unlike the walls-free room), this is the
setting where 1-step stitching should be *necessary* rather than merely sufficient.

Walk mechanics: each step samples a direction uniformly on the circle and takes a full-size step
(a unit-vector action, covering exactly --step_size, matching PointEnv's `qpos += 0.2 * action`
convention used by every other generator here). Candidate steps that would put the point mass
within --wall_clearance of a wall cell are rejected and the direction resampled, so the walk
stays strictly inside the free corridors and mujoco never resolves a penetration -- which keeps
the recorded qpos exactly equal to the intended position.

Output schema matches every other locomaze dataset script exactly (observations, actions,
terminals, qpos, qvel).
"""
import pathlib
import sys
import os
import gymnasium
import numpy as np
from absl import app, flags

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # cgq/, for envs.*
import envs.custom_umaze  # noqa: registers pointmaze-umaze-v0

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'pointmaze-umaze-v0', 'U-maze point-mass env name.')
flags.DEFINE_string('save_path', None, 'Save path (a train .npz; a sibling -val.npz is also written).')
flags.DEFINE_integer('seed', 0, 'Seed for numpy global RNG (start positions, step directions).')

flags.DEFINE_integer('episode_length', 1000, 'MDP steps per episode (every episode is exactly this '
                      'long -- a random walk has no notion of finishing).')
flags.DEFINE_integer('num_train_episodes', 1000, 'Train episodes. Default 1000 x 1000 steps = 1M '
                      'train transitions, matching pointmaze-large-navigate-v0\'s size.')
flags.DEFINE_integer('num_val_episodes', 100, 'Validation episodes -- num_train_episodes // 10, '
                      'the ratio OGBench\'s own generate_locomaze.py uses.')

flags.DEFINE_float('step_size', 0.2, 'MDP-step length: the xy distance PointEnv covers in one step '
                    'under a unit-vector action (qpos += 0.2 * action).')
flags.DEFINE_float('wall_clearance', 0.8, 'Keep the point mass at least this far (maze units) from '
                    'any wall cell. Must exceed the pointbody sphere radius (0.7 in '
                    'ogbench/locomaze/assets/point.xml) so the walk never touches a wall geom and '
                    'mujoco never has to push the body back out of a penetration.')
flags.DEFINE_integer('max_direction_tries', 64, 'Directions to try per step before giving up and '
                      'recording a zero action (staying put) -- effectively never hit in open '
                      'corridors, but a hard bound keeps a bad geometry from hanging the script.')


def is_free_xy(env, xy, clearance):
    """True iff every corner of the [xy +- clearance] box lands in an open (non-wall) maze cell."""
    maze_map = env.unwrapped.maze_map
    for dx in (-clearance, clearance):
        for dy in (-clearance, clearance):
            i, j = env.unwrapped.xy_to_ij((xy[0] + dx, xy[1] + dy))
            if i < 0 or j < 0 or i >= maze_map.shape[0] or j >= maze_map.shape[1]:
                return False
            if maze_map[i, j] != 0:
                return False
    return True


def sample_free_xy(env, clearance, max_tries=1000):
    """Uniformly sample a start position over the maze's free cells (rejection-sampled so it also
    respects `clearance`)."""
    maze_map = env.unwrapped.maze_map
    free_cells = np.argwhere(maze_map == 0)
    half = env.unwrapped._maze_unit / 2
    for _ in range(max_tries):
        i, j = free_cells[np.random.randint(len(free_cells))]
        cx, cy = env.unwrapped.ij_to_xy((i, j))
        cand = np.array([cx + np.random.uniform(-half, half), cy + np.random.uniform(-half, half)])
        if is_free_xy(env, cand, clearance):
            return cand
    raise RuntimeError('sample_free_xy failed -- is wall_clearance larger than half a maze cell?')


class Buffer:
    """Preallocated transition storage. Every episode here is exactly --episode_length steps, so
    the total size is known up front -- worth preallocating rather than appending to lists, since
    at 10M+ transitions the list-of-small-arrays approach costs several GB in per-object overhead
    alone and makes the final np.array() conversion the slowest part of the script.
    """

    KEYS = {'observations': 2, 'actions': 2, 'qpos': 2, 'qvel': 2}

    def __init__(self, total_steps):
        self.n = 0
        self.data = {k: np.zeros((total_steps, dim), dtype=np.float32) for k, dim in self.KEYS.items()}
        self.data['terminals'] = np.zeros(total_steps, dtype=bool)

    def append(self, observation, action, qpos, qvel):
        i = self.n
        self.data['observations'][i] = observation
        self.data['actions'][i] = action
        self.data['qpos'][i] = qpos
        self.data['qvel'][i] = qvel
        self.n += 1

    def mark_terminal(self):
        self.data['terminals'][self.n - 1] = True

    def split(self, cut):
        """(train, val) dicts, split at transition index `cut`."""
        return ({k: v[:cut] for k, v in self.data.items()},
                {k: v[cut:self.n] for k, v in self.data.items()})


def run_brownian_episode(env, dataset, episode_length, step_size, clearance):
    """One episode: a random walk whose direction is resampled independently at every step."""
    start_xy = sample_free_xy(env, clearance)
    env.unwrapped.set_xy(start_xy)

    for _ in range(episode_length):
        cur_xy = env.unwrapped.get_xy().copy()

        action = np.zeros(2, dtype=np.float32)
        for _ in range(FLAGS.max_direction_tries):
            theta = np.random.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(theta), np.sin(theta)])
            if is_free_xy(env, cur_xy + step_size * direction, clearance):
                action = direction.astype(np.float32)
                break

        ob = env.unwrapped.get_ob()
        next_ob, reward, terminated, truncated, info = env.step(action)
        dataset.append(observation=ob, action=action, qpos=info['prev_qpos'], qvel=info['prev_qvel'])

    dataset.mark_terminal()


def main(_):
    np.random.seed(FLAGS.seed)

    env = gymnasium.make(FLAGS.env_name, terminate_at_goal=False,
                          max_episode_steps=FLAGS.episode_length + 1)
    env.reset(seed=FLAGS.seed)

    print(f'Maze:\n{env.unwrapped.maze_map}')
    print(f'{FLAGS.num_train_episodes} train + {FLAGS.num_val_episodes} val episodes '
          f'x {FLAGS.episode_length} steps.')

    total_train_steps = FLAGS.num_train_episodes * FLAGS.episode_length
    total_steps = total_train_steps + FLAGS.num_val_episodes * FLAGS.episode_length
    dataset = Buffer(total_steps)

    report_every = max(FLAGS.num_train_episodes // 20, 1)
    for k in range(FLAGS.num_train_episodes):
        run_brownian_episode(env, dataset, FLAGS.episode_length, FLAGS.step_size,
                             FLAGS.wall_clearance)
        if (k + 1) % report_every == 0:
            print(f'  train episode {k + 1}/{FLAGS.num_train_episodes} '
                  f'({dataset.n:,}/{total_steps:,} steps)', flush=True)
    assert dataset.n == total_train_steps, (dataset.n, total_train_steps)

    for _ in range(FLAGS.num_val_episodes):
        run_brownian_episode(env, dataset, FLAGS.episode_length, FLAGS.step_size,
                             FLAGS.wall_clearance)
    assert dataset.n == total_steps, (dataset.n, total_steps)

    print(f'Train: {FLAGS.num_train_episodes} episodes, {total_train_steps} steps.')
    print(f'Val: {FLAGS.num_val_episodes} episodes, {total_steps - total_train_steps} steps.')
    print('Total steps:', total_steps)

    train_path = FLAGS.save_path
    val_path = FLAGS.save_path.replace('.npz', '-val.npz')
    pathlib.Path(train_path).parent.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset = dataset.split(total_train_steps)

    for path, split_dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **split_dataset)
    print(f'Saved {train_path} and {val_path}')


if __name__ == '__main__':
    app.run(main)
