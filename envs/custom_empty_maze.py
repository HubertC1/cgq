"""Point-mass "maze" with no interior walls at all -- just an open room bounded by the outer
border wall (border walls are unavoidable: without them the point mass has nothing to stay
inside of, and update_tree's floor-sizing math below assumes a closed border like every
upstream maze_map). The grid is the exact same shape as pointmaze-large-v0's maze_map (9x12,
see ogbench/ogbench/locomaze/maze.py:112-123), so the free interior spans the identical xy
footprint (x in [0, 36], y in [0, 24] at cell centers) -- only the internal walls are removed.

Ported from ogbench.locomaze.maze.make_maze_env's MazeEnv ('point' branch) the same way
envs/custom_teleport_maze.py is: that file is reference-only (do not modify in place), so the
relevant logic (wall placement, floor sizing, camera, task/goal handling, BFS oracle subgoal)
is duplicated here rather than imported, with the maze_map/tasks made constructor args instead
of hardcoded per-maze_type blocks. States-only, point-only, no portals -- this is deliberately
the simplest possible instance of the pattern.

Registers 'pointmaze-empty-v0' at import time (see bottom of file), so any script that does
`import envs.custom_empty_maze` before `gymnasium.make('pointmaze-empty-v0', ...)` picks it up,
mirroring how `import ogbench.locomaze` registers the upstream maze envs.
"""
import tempfile
import xml.etree.ElementTree as ET

import gymnasium
import mujoco
import numpy as np
from gymnasium.envs.registration import register
from gymnasium.spaces import Box

from ogbench.locomaze.point import PointEnv

# Same shape as pointmaze-large-v0's maze_map (ogbench/ogbench/locomaze/maze.py:112-123):
# 9 rows x 12 cols, border walls only, everything else open.
EMPTY_LARGE_MAZE_MAP = np.ones((9, 12), dtype=int)
EMPTY_LARGE_MAZE_MAP[1:-1, 1:-1] = 0

# Same 5 (init, goal) pairs as pointmaze-large's own tasks (ogbench/ogbench/locomaze/maze.py:
# 322-329) -- valid here too since the grid is the same shape and every one of those cells is
# free (no interior walls). Task 1 is the bottom-left-to-top-right diagonal corner pair used as
# the "near-optimal line from start to finish" by generate_pointmaze_empty_diagonal_stitch.py.
# Having more than one task also matters for generate_pointmaze_noisy.py's 'navigate'/'path'
# logic: single-task mazes pin every episode's *first* goal to that one task's goal (see that
# script's `fixed_goal_ij`), which for an otherwise-diverse open room would make every navigate
# episode beeline to the same corner before ever resampling. >1 task disables that pinning, so
# the first goal is drawn from vertex_cells like every other multi-task maze.
DEFAULT_TASKS = [
    ((1, 1), (7, 10)),
    ((5, 4), (7, 1)),
    ((7, 4), (1, 10)),
    ((3, 8), (5, 4)),
    ((1, 1), (5, 4)),
]


class CustomEmptyPointMazeEnv(PointEnv):
    """Point-mass in an open room (no interior walls) with a caller-defined maze_map/tasks.

    Args:
        maze_map: 2D array/list of 0 (open) / 1 (wall) cells. Border cells should be walls (the
            geom-placement math in `update_tree` assumes a closed border, matching every
            upstream maze_map). Defaults to EMPTY_LARGE_MAZE_MAP.
        tasks: list of (init_ij, goal_ij) tuples. task_infos[k] gets task_name f'task{k+1}',
            matching upstream's 1-indexed task_id convention. Defaults to DEFAULT_TASKS.
    """

    def __init__(
        self,
        maze_map=None,
        tasks=None,
        maze_unit=4.0,
        maze_height=0.5,
        terminate_at_goal=True,
        success_timing='post',
        add_noise_to_goal=True,
        reward_task_id=None,
        *args,
        **kwargs,
    ):
        self.maze_map = np.array(EMPTY_LARGE_MAZE_MAP if maze_map is None else maze_map)
        self._tasks = list(DEFAULT_TASKS if tasks is None else tasks)
        self._maze_unit = maze_unit
        self._maze_height = maze_height
        self._terminate_at_goal = terminate_at_goal
        self._success_timing = success_timing
        self._add_noise_to_goal = add_noise_to_goal
        self._reward_task_id = reward_task_id

        assert success_timing in ['pre', 'post']
        assert self.maze_map.ndim == 2
        assert np.all(self.maze_map[0, :] == 1) and np.all(self.maze_map[-1, :] == 1), (
            'maze_map must have a walled top/bottom border.'
        )
        assert np.all(self.maze_map[:, 0] == 1) and np.all(self.maze_map[:, -1] == 1), (
            'maze_map must have a walled left/right border.'
        )

        # Constants matching ogbench/ogbench/locomaze/maze.py.
        self._offset_x = 4
        self._offset_y = 4
        self._noise = 1
        self._goal_tol = 1.0  # Point-mass goal tolerance (upstream: 1.0 if loco_env_type == 'point').

        # Update XML file.
        xml_file = self.xml_file
        tree = ET.parse(xml_file)
        self.update_tree(tree)
        _, maze_xml_file = tempfile.mkstemp(text=True, suffix='.xml')
        tree.write(maze_xml_file)

        super().__init__(xml_file=maze_xml_file, *args, **kwargs)

        # Custom top-down camera (matches upstream's default view).
        if self.camera_id is None and self.camera_name is None:
            camera = mujoco.MjvCamera()
            camera.lookat[0] = 2 * (self.maze_map.shape[1] - 3)
            camera.lookat[1] = 2 * (self.maze_map.shape[0] - 3)
            camera.distance = 5 * (self.maze_map.shape[1] - 2)
            camera.elevation = -90
            self.custom_camera = camera
        else:
            self.custom_camera = self.camera_id or self.camera_name

        self.task_infos = []
        self.cur_task_id = None
        self.cur_task_info = None
        self.set_tasks()
        self.num_tasks = len(self.task_infos)
        self.cur_goal_xy = np.zeros(2)

        self.custom_renderer = None
        ex_ob = self.get_ob()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=ex_ob.shape, dtype=ex_ob.dtype)

    def update_tree(self, tree):
        """Add the border wall, floor, and the goal target to the XML tree. No interior walls."""
        worldbody = tree.find('.//worldbody')

        for i in range(self.maze_map.shape[0]):
            for j in range(self.maze_map.shape[1]):
                if self.maze_map[i, j] == 1:
                    ET.SubElement(
                        worldbody,
                        'geom',
                        name=f'block_{i}_{j}',
                        pos=f'{j * self._maze_unit - self._offset_x} {i * self._maze_unit - self._offset_y} '
                            f'{self._maze_height / 2 * self._maze_unit}',
                        size=f'{self._maze_unit / 2} {self._maze_unit / 2} {self._maze_height / 2 * self._maze_unit}',
                        type='box',
                        contype='1',
                        conaffinity='1',
                        material='wall',
                    )

        center_x, center_y = 2 * (self.maze_map.shape[1] - 3), 2 * (self.maze_map.shape[0] - 3)
        size_x, size_y = 2 * self.maze_map.shape[1], 2 * self.maze_map.shape[0]
        floor = tree.find('.//geom[@name="floor"]')
        floor.set('pos', f'{center_x} {center_y} 0')
        floor.set('size', f'{size_x} {size_y} 0.2')

        ET.SubElement(
            worldbody,
            'geom',
            name='target',
            type='cylinder',
            size='.5 .05',
            pos='0 0 .05',
            material='target',
            contype='0',
            conaffinity='0',
        )

    def set_tasks(self):
        self.task_infos = []
        for i, (init_ij, goal_ij) in enumerate(self._tasks):
            self.task_infos.append(
                dict(
                    task_name=f'task{i + 1}',
                    init_ij=init_ij,
                    init_xy=self.ij_to_xy(init_ij),
                    goal_ij=goal_ij,
                    goal_xy=self.ij_to_xy(goal_ij),
                )
            )
        if self._reward_task_id == 0:
            self._reward_task_id = 1  # Default task.

    def initialize_renderer(self):
        self.custom_renderer = mujoco.Renderer(self.model, width=self.width, height=self.height)
        self.render()

    def reset(self, options=None, *args, **kwargs):
        if options is None:
            options = {}
        if self._reward_task_id is not None:
            assert 1 <= self._reward_task_id <= self.num_tasks, f'Task ID must be in [1, {self.num_tasks}].'
            self.cur_task_id = self._reward_task_id
            self.cur_task_info = self.task_infos[self.cur_task_id - 1]
        elif 'task_id' in options:
            assert 1 <= options['task_id'] <= self.num_tasks, f'Task ID must be in [1, {self.num_tasks}].'
            self.cur_task_id = options['task_id']
            self.cur_task_info = self.task_infos[self.cur_task_id - 1]
        elif 'task_info' in options:
            self.cur_task_id = None
            self.cur_task_info = options['task_info']
        else:
            self.cur_task_id = np.random.randint(1, self.num_tasks + 1)
            self.cur_task_info = self.task_infos[self.cur_task_id - 1]

        render_goal = options.get('render_goal', False)

        init_xy = self.add_noise(self.ij_to_xy(self.cur_task_info['init_ij']))
        goal_xy = self.ij_to_xy(self.cur_task_info['goal_ij'])
        if self._add_noise_to_goal:
            goal_xy = self.add_noise(goal_xy)

        # First, force set the position to the goal position to obtain the goal observation.
        super().reset(*args, **kwargs)
        for _ in range(5):
            super().step(self.action_space.sample())

        self.set_goal(goal_xy=goal_xy)
        self.set_xy(goal_xy)
        goal_ob = self.get_ob()
        if render_goal:
            goal_rendered = self.render()

        ob, info = super().reset(*args, **kwargs)
        self.set_goal(goal_xy=goal_xy)
        self.set_xy(init_xy)
        ob = self.get_ob()
        info['goal'] = goal_ob
        if render_goal:
            info['goal_rendered'] = goal_rendered

        return ob, info

    def step(self, action):
        if self._success_timing == 'pre':
            success = self.compute_success()

        ob, reward, terminated, truncated, info = super().step(action)

        if self._success_timing == 'post':
            success = self.compute_success()

        if success:
            if self._terminate_at_goal:
                terminated = True
            info['success'] = 1.0
            reward = 1.0
        else:
            info['success'] = 0.0
            reward = 0.0

        if self._reward_task_id is not None:
            reward = reward - 1.0

        return ob, reward, terminated, truncated, info

    def render(self):
        if self.custom_renderer is None:
            self.initialize_renderer()
        self.custom_renderer.update_scene(self.data, camera=self.custom_camera)
        return self.custom_renderer.render()

    def get_ob(self, ob_type=None):
        # ob_type is accepted (and ignored, always 'states') for interface parity with upstream
        # MazeEnv.get_ob -- generate_pointmaze_noisy.py calls this with ob_type='states'.
        return super().get_ob()

    def compute_success(self):
        return np.linalg.norm(self.get_xy() - self.cur_goal_xy) <= self._goal_tol

    def set_goal(self, goal_ij=None, goal_xy=None):
        if goal_xy is None:
            self.cur_goal_xy = self.ij_to_xy(goal_ij)
            if self._add_noise_to_goal:
                self.cur_goal_xy = self.add_noise(self.cur_goal_xy)
        else:
            self.cur_goal_xy = goal_xy
        self.model.geom('target').pos[:2] = self.cur_goal_xy

    def get_oracle_subgoal(self, start_xy, goal_xy):
        """BFS shortest-path subgoal. With no interior walls, this always reduces to stepping
        one grid cell directly toward the goal -- kept for interface parity with upstream/
        custom_teleport_maze.py so generate_pointmaze_noisy.py's 'path'/'navigate'/'stitch'
        logic (which calls this unconditionally) works unmodified against this env."""
        start_ij = self.xy_to_ij(start_xy)
        goal_ij = self.xy_to_ij(goal_xy)

        bfs_map = self.maze_map.copy()
        for i in range(self.maze_map.shape[0]):
            for j in range(self.maze_map.shape[1]):
                bfs_map[i][j] = -1

        bfs_map[goal_ij[0], goal_ij[1]] = 0
        queue = [goal_ij]
        while len(queue) > 0:
            i, j = queue.pop(0)
            for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                ni, nj = i + di, j + dj
                if (
                    0 <= ni < self.maze_map.shape[0]
                    and 0 <= nj < self.maze_map.shape[1]
                    and self.maze_map[ni, nj] == 0
                    and bfs_map[ni, nj] == -1
                ):
                    bfs_map[ni][nj] = bfs_map[i][j] + 1
                    queue.append((ni, nj))

        subgoal_ij = start_ij
        for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            ni, nj = start_ij[0] + di, start_ij[1] + dj
            if (
                0 <= ni < self.maze_map.shape[0]
                and 0 <= nj < self.maze_map.shape[1]
                and self.maze_map[ni, nj] == 0
                and bfs_map[ni, nj] < bfs_map[subgoal_ij[0], subgoal_ij[1]]
            ):
                subgoal_ij = (ni, nj)
        subgoal_xy = self.ij_to_xy(subgoal_ij)
        return np.array(subgoal_xy), bfs_map

    def xy_to_ij(self, xy):
        maze_unit = self._maze_unit
        i = int((xy[1] + self._offset_y + 0.5 * maze_unit) / maze_unit)
        j = int((xy[0] + self._offset_x + 0.5 * maze_unit) / maze_unit)
        return i, j

    def ij_to_xy(self, ij):
        i, j = ij
        x = j * self._maze_unit - self._offset_x
        y = i * self._maze_unit - self._offset_y
        return x, y

    def add_noise(self, xy):
        random_x = np.random.uniform(low=-self._noise, high=self._noise) * self._maze_unit / 4
        random_y = np.random.uniform(low=-self._noise, high=self._noise) * self._maze_unit / 4
        return xy[0] + random_x, xy[1] + random_y


def register_custom_empty_maze(name='empty', maze_map=None, tasks=None, max_episode_steps=1000, **env_kwargs):
    """Register `pointmaze-{name}-v0` plus one singletask variant per task.

    Mirrors envs/custom_teleport_maze.py's register_custom_teleport_maze (and, further back,
    ogbench/ogbench/locomaze/__init__.py's registration pattern) so
    envs/ogbench_utils.py:make_ogbench_env_and_datasets and ogbench/relabel_utils.py work with
    no further changes. Safe to call more than once (e.g. from repeated module imports within
    the same process) -- re-registration under an id gymnasium already knows about is a no-op.
    """
    base_id = f'pointmaze-{name}-v0'
    if base_id not in gymnasium.envs.registry:
        register(
            id=base_id,
            entry_point='envs.custom_empty_maze:CustomEmptyPointMazeEnv',
            max_episode_steps=max_episode_steps,
            kwargs=dict(maze_map=maze_map, tasks=tasks, **env_kwargs),
        )
    resolved_tasks = DEFAULT_TASKS if tasks is None else tasks
    # task_id in [None, 1, 2, ...]: None (reward_task_id=0) is the bare '-singletask-v0' id,
    # matching upstream's own registration loop (ogbench/ogbench/locomaze/__init__.py:249-253) --
    # MazeEnv.set_tasks treats reward_task_id=0 as "use the default (first) task".
    for task_id in [None] + list(range(1, len(resolved_tasks) + 1)):
        task_suffix = '' if task_id is None else f'-task{task_id}'
        reward_task_id = 0 if task_id is None else task_id
        singletask_id = f'pointmaze-{name}-singletask{task_suffix}-v0'
        if singletask_id not in gymnasium.envs.registry:
            register(
                id=singletask_id,
                entry_point='envs.custom_empty_maze:CustomEmptyPointMazeEnv',
                max_episode_steps=max_episode_steps,
                kwargs=dict(
                    maze_map=maze_map,
                    tasks=tasks,
                    reward_task_id=reward_task_id,
                    add_noise_to_goal=False,
                    success_timing='pre',
                    **env_kwargs,
                ),
            )
    return base_id


# Register 'pointmaze-empty-v0' (+ 'pointmaze-empty-singletask-task1-v0') at import time, the
# same way `import ogbench.locomaze` registers the upstream maze envs -- so any script need
# only `import envs.custom_empty_maze` before `gymnasium.make('pointmaze-empty-v0', ...)`.
register_custom_empty_maze()
