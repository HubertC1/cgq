"""Exact optimal value function V*(s) for the point-mass maze envs.

These envs are deterministic, the reward is -1 per step until the goal (see
CustomEmptyPointMazeEnv.step: reward 1.0 on success, then `reward - 1.0` in single-task mode) and
bootstrapping stops at the goal (mask = 1 - success). So the optimal value of a state is fully
determined by the number of MDP steps needed to reach the goal region:

    V*(s) = -(1 - gamma^d(s)) / (1 - gamma),      V*(s) = 0 inside the goal tolerance

with d(s) = geodesic_distance(s -> goal region) / step_size. The geodesic is computed by a
Dijkstra sweep over a fine grid of free positions (8-connected, orthogonal cost 1 and diagonal
cost sqrt(2), scaled by the grid resolution), which routes around walls exactly the way the agent
must.

Having V* in closed form is what makes the umazelarge experiment measurable rather than
anecdotal: instead of only asking "did the policy reach the goal", we can ask how far the learned
critic's value landscape is from the truth, and *where* it goes wrong -- e.g. whether it smooths
across the thin divider wall separating two corridors whose true values differ by ~1400.

Caveat on d(s): the walk convention everywhere in this repo is a unit-norm action covering exactly
step_size per step, so d = geodesic / step_size. The action space is actually the box [-1, 1]^2,
which permits up to step_size*sqrt(2) per step on a perfect diagonal, so a policy exploiting the
box corner could beat this bound. In an axis-aligned corridor maze that only helps at corners, so
the approximation is tight here -- but it is an *upper* bound on the true optimal cost, not an
exact optimum, if the corridors are ever diagonal.
"""
import heapq

import numpy as np


def free_mask(env, resolution, clearance=0.0):
    """Boolean grid over the maze bounding box: True where the point mass may be.

    `clearance` mirrors the data generator's wall_clearance -- pass 0 to get every position inside
    a free maze cell (the right choice for evaluating a critic, which can be queried anywhere),
    or the generator's value to restrict to positions the data actually covers.
    """
    u = env.unwrapped
    maze_map = u.maze_map
    unit = u._maze_unit
    rows, cols = maze_map.shape

    lo = np.array(u.ij_to_xy((0, 0)), dtype=np.float64) - unit / 2
    hi = np.array(u.ij_to_xy((rows - 1, cols - 1)), dtype=np.float64) + unit / 2

    xs = np.arange(lo[0] + resolution / 2, hi[0], resolution)
    ys = np.arange(lo[1] + resolution / 2, hi[1], resolution)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')

    def cells_free(px, py):
        i = np.floor((py + u._offset_y + 0.5 * unit) / unit).astype(int)
        j = np.floor((px + u._offset_x + 0.5 * unit) / unit).astype(int)
        ok = (i >= 0) & (j >= 0) & (i < rows) & (j < cols)
        out = np.zeros_like(ok)
        ii, jj = np.clip(i, 0, rows - 1), np.clip(j, 0, cols - 1)
        out[ok] = maze_map[ii[ok], jj[ok]] == 0
        return out

    if clearance <= 0:
        mask = cells_free(gx, gy)
    else:
        mask = np.ones_like(gx, dtype=bool)
        for dx in (-clearance, clearance):
            for dy in (-clearance, clearance):
                mask &= cells_free(gx + dx, gy + dy)
    return mask, xs, ys


def geodesic_steps(env, goal_xy, step_size=0.2, resolution=0.2, clearance=0.0, goal_tol=None):
    """Grid of MDP steps-to-goal. Returns (steps, xs, ys, mask); steps is +inf outside the maze."""
    u = env.unwrapped
    mask, xs, ys = free_mask(env, resolution, clearance)
    goal_tol = u._goal_tol if goal_tol is None else goal_tol

    nx, ny = len(xs), len(ys)
    dist = np.full((nx, ny), np.inf)

    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    at_goal = mask & (np.hypot(gx - goal_xy[0], gy - goal_xy[1]) <= goal_tol)

    heap = []
    for i, j in zip(*np.where(at_goal)):
        dist[i, j] = 0.0
        heap.append((0.0, int(i), int(j)))
    heapq.heapify(heap)

    nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))]
    while heap:
        d, i, j = heapq.heappop(heap)
        if d > dist[i, j]:
            continue
        for di, dj, w in nbrs:
            ni, nj = i + di, j + dj
            if ni < 0 or nj < 0 or ni >= nx or nj >= ny or not mask[ni, nj]:
                continue
            nd = d + w * resolution
            if nd < dist[ni, nj] - 1e-12:
                dist[ni, nj] = nd
                heapq.heappush(heap, (nd, ni, nj))

    return dist / step_size, xs, ys, mask


def oracle_value(steps, gamma):
    """V*(s) = -(1 - gamma^d) / (1 - gamma), elementwise; +inf steps -> nan."""
    with np.errstate(over='ignore', invalid='ignore'):
        v = -(1.0 - np.power(gamma, steps)) / (1.0 - gamma)
    v[~np.isfinite(steps)] = np.nan
    return v


def oracle_value_field(env, goal_xy, gamma, step_size=0.2, resolution=0.2, clearance=0.0):
    """Convenience wrapper: (V*, steps, xs, ys, mask)."""
    steps, xs, ys, mask = geodesic_steps(env, goal_xy, step_size, resolution, clearance)
    return oracle_value(steps, gamma), steps, xs, ys, mask


def optimal_direction_field(steps):
    """Unit vector per grid cell pointing along the geodesic toward the goal.

    Taken as the direction of the 8-neighbour with the smallest steps-to-go, rather than a
    finite-difference gradient of the distance field: the neighbour rule is exact for the grid
    geodesic and stays well-defined right against a wall, where a numerical gradient is garbage --
    and next to walls is precisely where a critic's greedy action is most likely to be wrong.
    """
    nx, ny = steps.shape
    finite = np.isfinite(steps)
    padded = np.where(finite, steps, np.inf)

    best_val = np.full((nx, ny), np.inf)
    best_dir = np.full((nx, ny, 2), np.nan)
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        shifted = np.full((nx, ny), np.inf)
        i0, i1 = max(0, -di), nx - max(0, di)
        j0, j1 = max(0, -dj), ny - max(0, dj)
        shifted[i0:i1, j0:j1] = padded[i0 + di:i1 + di, j0 + dj:j1 + dj]
        better = shifted < best_val
        best_val = np.where(better, shifted, best_val)
        v = np.array([di, dj], dtype=np.float64)
        best_dir[better] = v / np.linalg.norm(v)

    best_dir[~finite] = np.nan
    return best_dir


def _unit(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.divide(v, np.where(n < 1e-9, 1.0, n))


def greedy_action_directions(agent, agent_config, observations, n_dirs=64, batch_size=4096):
    """For each state, the unit action direction maximizing the critic.

    Sweeps `n_dirs` directions on the unit circle (dataset actions are unit-norm by construction,
    so this is the right comparison set) and takes the argmax of Q. For a chunk critic with h > 1
    the candidate is that direction held for all h sub-actions -- i.e. the best *constant-heading*
    chunk, which is the natural chunk analogue of a single greedy step and avoids searching an
    exponentially large chunk space.
    """
    observations = np.asarray(observations)
    critic_key = 'chunk_critic' if agent_config.get('eval_policy', 'step') == 'chunk' else 'step_critic'
    h = int(agent_config.get('horizon_length', 1) or 1)
    if critic_key == 'step_critic':
        h = 1
    adim = int(agent_config['action_dim'])

    thetas = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=-1)  # (K, 2)

    qs_all = np.empty((n_dirs, len(observations)), dtype=np.float32)
    for k, d in enumerate(dirs):
        cand = np.tile(np.asarray(d, dtype=np.float32), h)[None, :]
        out = []
        for s in range(0, len(observations), batch_size):
            chunk = observations[s:s + batch_size]
            a = np.repeat(cand, len(chunk), axis=0)
            q = np.asarray(agent.network.select(critic_key)(chunk, actions=a))
            out.append(q.min(axis=0) if agent_config.get('q_agg') == 'min' else q.mean(axis=0))
        qs_all[k] = np.concatenate(out)

    return dirs[np.argmax(qs_all, axis=0)][:, :2]


def policy_action_directions(agent, agent_config, observations, batch_size=4096, seed=0):
    """For each state, the unit direction of the policy's first action."""
    import jax

    observations = np.asarray(observations)
    adim = int(agent_config['action_dim'])
    rng = jax.random.PRNGKey(seed)
    out = []
    for s in range(0, len(observations), batch_size):
        chunk = observations[s:s + batch_size]
        rng, key = jax.random.split(rng)
        a = np.asarray(agent.sample_actions(observations=chunk, temperature=0.0, rng=key))
        a = a.reshape(len(chunk), -1)[:, :adim]   # first sub-action of the chunk
        out.append(a)
    return _unit(np.concatenate(out))


def action_agreement_stats(agent, agent_config, probe, n_dirs=64, batch_size=4096, seed=0,
                            return_maps=False):
    """Cosine agreement of (a) the critic's greedy action and (b) the policy's action with the
    geodesic-optimal direction.

    The pair is what separates "the critic is wrong" from "the critic is right and the actor
    cannot use it" -- a distinction success rate alone can never make.
    """
    obs = probe['observations']
    opt = probe['optimal_direction']

    greedy = greedy_action_directions(agent, agent_config, obs, n_dirs=n_dirs, batch_size=batch_size)
    policy = policy_action_directions(agent, agent_config, obs, batch_size=batch_size, seed=seed)

    cos_greedy = np.sum(greedy * opt, axis=-1)
    cos_policy = np.sum(policy * opt, axis=-1)
    cos_gp = np.sum(greedy * policy, axis=-1)

    stats = {
        'greedy_action_cos': float(np.mean(cos_greedy)),
        'greedy_action_frac_correct': float(np.mean(cos_greedy > 0)),
        'policy_action_cos': float(np.mean(cos_policy)),
        'policy_action_frac_correct': float(np.mean(cos_policy > 0)),
        'policy_vs_greedy_cos': float(np.mean(cos_gp)),
    }
    if return_maps:
        return stats, {'cos_greedy': cos_greedy, 'cos_policy': cos_policy,
                       'greedy': greedy, 'policy': policy}
    return stats


def make_value_probe(env, gamma, resolution=0.5, max_states=20000, seed=0):
    """A fixed set of states with known V*, for logging critic value error during training.

    Returns {'observations', 'v_star'} or None if `env` isn't a 2-D maze this oracle understands
    (no maze_map / no 2-D goal), so callers can guard with a simple `is None` check.
    """
    u = getattr(env, 'unwrapped', env)
    if not hasattr(u, 'maze_map') or not hasattr(u, 'cur_goal_xy'):
        return None
    goal_xy = np.asarray(u.cur_goal_xy, dtype=np.float64)
    if goal_xy.shape != (2,):
        return None

    v_star, steps, xs, ys, _ = oracle_value_field(env, goal_xy, gamma, resolution=resolution)
    opt_dir = optimal_direction_field(steps)
    reachable = np.isfinite(steps)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    obs = np.stack([gx[reachable], gy[reachable]], axis=-1).astype(np.float32)
    vs = v_star[reachable].astype(np.float32)
    od = opt_dir[reachable].astype(np.float32)

    keep = np.arange(len(obs))
    if len(obs) > max_states:
        keep = np.random.RandomState(seed).choice(len(obs), size=max_states, replace=False)
    return {'observations': obs[keep], 'v_star': vs[keep], 'optimal_direction': od[keep],
            'reachable': reachable, 'keep': keep, 'xs': xs, 'ys': ys}


def agent_values(agent, agent_config, observations, batch_size=4096, seed=0):
    """V(s) = Q(s, pi(s)) for the critic matching the agent's acting policy.

    Prefers the agent's own get_value when it has one (iql/aciql/acfql/sarsa); otherwise queries
    the cgq-family critic that pairs with `eval_policy`.
    """
    import jax  # local: keeps this module importable from plain-numpy contexts

    observations = np.asarray(observations)
    if hasattr(agent, 'get_value'):
        return np.concatenate([np.asarray(agent.get_value(observations[k:k + batch_size]))
                               for k in range(0, len(observations), batch_size)])

    critic_key = 'chunk_critic' if agent_config.get('eval_policy', 'step') == 'chunk' else 'step_critic'
    rng = jax.random.PRNGKey(seed)
    out = []
    for k in range(0, len(observations), batch_size):
        chunk = observations[k:k + batch_size]
        rng, key = jax.random.split(rng)
        actions = np.asarray(agent.sample_actions(observations=chunk, temperature=0.0, rng=key))
        actions = actions.reshape(len(chunk), -1)
        qs = np.asarray(agent.network.select(critic_key)(chunk, actions=actions))
        out.append(qs.min(axis=0) if agent_config.get('q_agg') == 'min' else qs.mean(axis=0))
    return np.concatenate(out)


def value_error_stats(agent, agent_config, probe, batch_size=4096, seed=0):
    """{'value_mae', 'value_rmse', 'value_corr', 'value_bias', 'value_pred_mean'} against V*."""
    v = agent_values(agent, agent_config, probe['observations'], batch_size=batch_size, seed=seed)
    v_star = probe['v_star']
    err = v - v_star
    corr = float(np.corrcoef(v, v_star)[0, 1]) if np.std(v) > 1e-8 else 0.0
    return {
        'value_mae': float(np.mean(np.abs(err))),
        'value_rmse': float(np.sqrt(np.mean(err ** 2))),
        'value_bias': float(np.mean(err)),
        'value_corr': corr,
        'value_pred_mean': float(np.mean(v)),
        'value_star_mean': float(np.mean(v_star)),
    }


def lookup(field, xs, ys, query_xy):
    """Nearest-grid-cell lookup of `field` at each row of query_xy (N, 2)."""
    i = np.clip(np.rint((query_xy[:, 0] - xs[0]) / (xs[1] - xs[0])).astype(int), 0, len(xs) - 1)
    j = np.clip(np.rint((query_xy[:, 1] - ys[0]) / (ys[1] - ys[0])).astype(int), 0, len(ys) - 1)
    return field[i, j]


def diagnostic_figure(agent, agent_config, probe, env, maps=None, n_dirs=64, batch_size=4096,
                      seed=0, dpi=110):
    """The learned-vs-oracle value + action-agreement figure as an RGB array, for wandb.Image.

    Same content as scripts/render_value_map.py but in-process during training, so every eval
    checkpoint gets a picture rather than only the final checkpoint. Returns None if the probe
    lacks the grid metadata needed to place values back on the maze.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if probe is None or 'reachable' not in probe:
        return None
    reachable = probe['reachable']
    if len(probe['keep']) != int(reachable.sum()):
        return None  # subsampled probe can't be scattered back onto the full grid

    xs, ys = probe['xs'], probe['ys']
    obs = probe['observations']

    v = agent_values(agent, agent_config, obs, batch_size=batch_size, seed=seed)
    if maps is None:
        _, maps = action_agreement_stats(agent, agent_config, probe, n_dirs=n_dirs,
                                         batch_size=batch_size, seed=seed, return_maps=True)

    def scatter(flat):
        f = np.full(reachable.shape, np.nan)
        f[reachable] = flat
        return f

    V_learned, V_star = scatter(v), scatter(probe['v_star'])
    err = V_learned - V_star
    COS_G, COS_P = scatter(maps['cos_greedy']), scatter(maps['cos_policy'])

    u = env.unwrapped
    unit = u._maze_unit
    lo = np.array([xs[0], ys[0]]); hi = np.array([xs[-1], ys[-1]])
    span = hi - lo
    wide = span[0] / span[1] > 3.0
    panel_h = (16.0 * span[1] / span[0] + 1.3) if wide else 5.0
    fig, axes = plt.subplots(5, 1, figsize=(16, 5 * panel_h), dpi=dpi) if wide else \
        plt.subplots(1, 5, figsize=(30, 5.5), dpi=dpi)

    vmin, vmax = np.nanmin(V_star), np.nanmax(V_star)
    emax = np.nanmax(np.abs(err)) or 1.0
    panels = [
        (V_learned, 'learned V', 'viridis', vmin, vmax),
        (V_star, 'oracle V*', 'viridis', vmin, vmax),
        (err, f'error  MAE {np.nanmean(np.abs(err)):.0f}', 'coolwarm', -emax, emax),
        (COS_G, f'cos(greedy Q, optimal)  {np.nanmean(COS_G):.3f}', 'RdYlGn', -1, 1),
        (COS_P, f'cos(policy, optimal)  {np.nanmean(COS_P):.3f}', 'RdYlGn', -1, 1),
    ]
    for ax, (field, title, cmap, c0, c1) in zip(np.atleast_1d(axes).ravel(), panels):
        im = ax.imshow(np.ma.masked_invalid(field.T), origin='lower',
                       extent=[lo[0], hi[0], lo[1], hi[1]], cmap=cmap, vmin=c0, vmax=c1,
                       interpolation='nearest', zorder=1)
        for i in range(u.maze_map.shape[0]):
            for j in range(u.maze_map.shape[1]):
                if u.maze_map[i, j] == 1:
                    cx, cy = u.ij_to_xy((i, j))
                    ax.add_patch(plt.Rectangle((cx - unit / 2, cy - unit / 2), unit, unit,
                                                facecolor='0.8', edgecolor='none', zorder=3))
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, orientation='horizontal' if wide else 'vertical',
                     fraction=0.05, pad=0.1)
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return arr
