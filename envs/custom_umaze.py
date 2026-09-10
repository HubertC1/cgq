"""Point-mass U-maze: the smallest layout where the optimal path is a *detour* -- reaching the
goal requires first moving away from it in Euclidean terms.

Motivation (this is the whole point of the env, don't "simplify" the layout away): in the
walls-free room of envs/custom_empty_maze.py the locally greedy action ("point at the goal") is
also the globally optimal one from every state, so no amount of fragmenting the demonstrations
can create a task that *requires* stitching -- a chunk policy can just emit h straight-line
actions toward the goal and win. A U-maze breaks exactly that property: from the start cell the
goal is 8 units away in a straight line, but the only route is 24 units the long way around the
centre wall, so straight-to-goal is not merely suboptimal, it's a wall.

    row 0  # # # # #
    row 1  # . . . #      free cells (i, j): (1,1) (1,2) (1,3)
    row 2  # # # . #                         (2,3)
    row 3  # . . . #                         (3,1) (3,2) (3,3)
    row 4  # # # # #

task1 is (1,1) -> (3,1): the two ends of the U, Euclidean distance 8, path length 24 (detour
ratio 3.0).

No new env class is needed: envs/custom_empty_maze.CustomEmptyPointMazeEnv already takes
maze_map/tasks as constructor args and its update_tree/BFS-oracle logic is fully general over
walls (only its *default* map is wall-free, hence the "empty" in the name). This module just
supplies a walled map plus tasks and registers `pointmaze-umaze-v0` at import time, mirroring how
custom_empty_maze.py registers its own.
"""
import numpy as np

from envs.custom_empty_maze import register_custom_empty_maze

# Classic U-maze (same shape as d4rl/antmaze umaze): two horizontal corridors joined at the right.
UMAZE_MAP = np.array(
    [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ],
    dtype=int,
)

# (init_ij, goal_ij). task1 is the detour pair described in the module docstring. task2 is its
# reverse and task3 a shorter in-corridor pair -- both kept only so num_tasks > 1, matching why
# custom_empty_maze.py's DEFAULT_TASKS keeps more than one (see that file's comment).
UMAZE_TASKS = [
    ((1, 1), (3, 1)),
    ((3, 1), (1, 1)),
    ((1, 1), (3, 3)),
]


def is_free_cell(maze_map, ij):
    """True iff `ij` is inside the map and is an open (non-wall) cell."""
    i, j = ij
    if i < 0 or j < 0 or i >= maze_map.shape[0] or j >= maze_map.shape[1]:
        return False
    return maze_map[i, j] == 0


# Registers 'pointmaze-umaze-v0' (+ '-singletask-task{1,2,3}-v0') at import time. max_episode_steps
# is raised from the 1000 default to 1001 so a 1000-step data-generation episode isn't truncated
# one step early by the TimeLimit wrapper.
register_custom_empty_maze(name='umaze', maze_map=UMAZE_MAP, tasks=UMAZE_TASKS,
                            max_episode_steps=1001)


def make_long_umaze_map(cols):
    """Build the same U topology as UMAZE_MAP but `cols` wide instead of 5: two 1-cell-tall
    corridors joined by a single opening at the far right.

        # # # # # ... # #
        # . . . . ... . #     <- top corridor    (row 1)
        # # # # # ... . #     <- divider, gap at the second-to-last column (row 2)
        # . . . . ... . #     <- bottom corridor (row 3)
        # # # # # ... # #

    Path from (1,1) to (3,1) is 2*cols - 4 cells; every other ratio (corridor width, maze_unit,
    divider thickness) is left identical to the 5-column UMAZE_MAP so that scale is the only
    variable between the two.
    """
    m = np.ones((5, cols), dtype=int)
    m[1, 1:-1] = 0            # top corridor
    m[3, 1:-1] = 0            # bottom corridor
    m[2, cols - 2] = 0        # the single opening joining them, at the far right
    return m


# A deliberately enormous U: 77 columns gives a 150-cell path = 600 maze units = exactly 3000 MDP
# steps at the standard step_size of 0.2, versus 120 steps for the 5-column UMAZE_MAP. The
# straight-line distance between the two ends stays 8 units, so the detour ratio is ~75x (vs 3x).
#
# Two consequences worth knowing before training on this:
#   * discount: gamma=0.99 has an effective horizon of 100 steps and gamma^3000 = 8e-14, so the
#     goal is invisible from the start cell and the value function is flat at -100 over nearly the
#     whole maze. Anything trained here needs gamma >= 0.999 (horizon 1000), ideally 0.9995.
#   * max_episode_steps is set to 4000, giving a 1.33x budget over the 3000-step optimum. Failed
#     eval episodes therefore cost 4000 env steps each.
UMAZE_LARGE_COLS = 77
UMAZE_LARGE_MAP = make_long_umaze_map(UMAZE_LARGE_COLS)
UMAZE_LARGE_TASKS = [
    ((1, 1), (3, 1)),
    ((3, 1), (1, 1)),
    ((1, 1), (3, UMAZE_LARGE_COLS - 2)),
]

register_custom_empty_maze(name='umazelarge', maze_map=UMAZE_LARGE_MAP, tasks=UMAZE_LARGE_TASKS,
                            max_episode_steps=4000)
