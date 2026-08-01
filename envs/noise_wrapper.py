"""Action-space noise wrapper for stress-testing critic robustness to stochastic dynamics.

The wrapper adds state-dependent Gaussian noise eps ~ N(0, sigma(s)^2 I) to the action
before stepping the underlying env: eps is isotropic (a single scalar sigma(s) reused
across all action dims), and sigma(s) is evaluated at the pre-transition agent xy (the
state the agent occupies when the action is chosen, before this step's transition).

The wrapper is pure: it never mutates the caller's action array in place, and never
writes to the underlying env's state outside of the normal step()/reset() calls.
"""

import gymnasium as gym
import numpy as np


def _constant_sigma_fn(cfg):
    sigma_0 = float(cfg['sigma_0'])

    def sigma_fn(env, xy):
        return sigma_0

    return sigma_fn


def _radial_from_goal_sigma_fn(cfg):
    inner_sigma = float(cfg['inner_sigma'])
    outer_sigma = float(cfg['outer_sigma'])
    transition_radius = float(cfg['transition_radius'])
    center = cfg.get('center', None)

    def sigma_fn(env, xy):
        c = np.asarray(center) if center is not None else np.asarray(env.unwrapped.cur_goal_xy)
        dist = np.linalg.norm(np.asarray(xy) - c)
        return inner_sigma if dist <= transition_radius else outer_sigma

    return sigma_fn


def _near_walls_sigma_fn(cfg):
    wall_distance_threshold = float(cfg['wall_distance_threshold'])
    near_sigma = float(cfg['near_sigma'])
    far_sigma = float(cfg['far_sigma'])
    wall_xy_cache = {}

    def wall_xys(env):
        maze_map = env.unwrapped.maze_map
        key = id(maze_map)
        if key not in wall_xy_cache:
            ijs = np.argwhere(maze_map == 1)
            wall_xy_cache[key] = np.array([env.unwrapped.ij_to_xy((i, j)) for i, j in ijs])
        return wall_xy_cache[key]

    def sigma_fn(env, xy):
        xys = wall_xys(env)
        dist_to_wall_center = np.linalg.norm(xys - np.asarray(xy), axis=1).min()
        dist_to_wall_face = max(0.0, dist_to_wall_center - env.unwrapped._maze_unit / 2.0)
        return near_sigma if dist_to_wall_face <= wall_distance_threshold else far_sigma

    return sigma_fn


def _region_sigma_fn(cfg):
    """Elevated sigma inside axis-aligned boxes (e.g. slippery-ice patches), default elsewhere.

    cfg['regions']: list of (xmin, xmax, ymin, ymax, sigma). First matching box wins; cfg is
    checked in list order, so overlapping boxes should be ordered highest-priority first.
    cfg['default_sigma']: sigma outside every box.
    """
    regions = [tuple(r) for r in cfg['regions']]
    default_sigma = float(cfg['default_sigma'])

    def sigma_fn(env, xy):
        x, y = xy
        for xmin, xmax, ymin, ymax, sigma in regions:
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return float(sigma)
        return default_sigma

    return sigma_fn


def _composite_sigma_fn(cfg):
    """Combines several sub-presets (each a full noise_cfg dict) into one sigma_fn.

    cfg['components']: list of sub-cfgs, each dict(type=..., **preset_kwargs) -- built the same
    way a top-level noise_cfg is, just without a wrapper of its own.
    cfg['combine']: 'max' (worst local hazard wins, default) or 'sum' (hazards stack).
    """
    combine = cfg.get('combine', 'max')
    if combine not in ('max', 'sum'):
        raise ValueError(f"combine must be 'max' or 'sum', got {combine!r}")
    sub_fns = [NOISE_FN_REGISTRY[sub_cfg['type']](sub_cfg) for sub_cfg in cfg['components']]
    reducer = max if combine == 'max' else sum

    def sigma_fn(env, xy):
        return reducer(fn(env, xy) for fn in sub_fns)

    return sigma_fn


NOISE_FN_REGISTRY = {
    'constant': _constant_sigma_fn,
    'radial_from_goal': _radial_from_goal_sigma_fn,
    'near_walls': _near_walls_sigma_fn,
    'region': _region_sigma_fn,
    'composite': _composite_sigma_fn,
}


def make_noise_cfg(noise_type, **kwargs):
    """Build a plain-dict noise_cfg for NoisyActionWrapper.

    Args:
        noise_type: One of 'none', 'constant', 'radial_from_goal', 'near_walls', 'region',
            'composite'.
        **kwargs: Preset-specific parameters (e.g. sigma_0 for 'constant'; regions/default_sigma
            for 'region'; components/combine for 'composite').

    Returns:
        A dict with a 'type' key, or None if noise_type is 'none'/None (no wrapper applied).
    """
    if noise_type is None or noise_type == 'none':
        return None
    if noise_type not in NOISE_FN_REGISTRY:
        raise ValueError(f'Unknown noise_type: {noise_type!r}. Valid: {list(NOISE_FN_REGISTRY)}')
    return dict(type=noise_type, **kwargs)


class NoisyActionWrapper(gym.Wrapper):
    """Adds eps ~ N(0, sigma(s)^2 I) to the action before stepping the underlying env."""

    def __init__(self, env, noise_cfg, noise_seed):
        super().__init__(env)
        self.noise_cfg = noise_cfg
        self.noise_seed = noise_seed
        self._rng = np.random.default_rng(noise_seed)
        self._sigma_fn = NOISE_FN_REGISTRY[noise_cfg['type']](noise_cfg)
        self._episode_sigmas = []

    def reset(self, **kwargs):
        self._episode_sigmas = []
        return self.env.reset(**kwargs)

    def set_noise_seed(self, seed):
        """Reseed the noise RNG in place (e.g. for independently-seeded eval repeats)."""
        self._rng = np.random.default_rng(seed)

    def step(self, action):
        action = np.asarray(action)
        xy = self.unwrapped.get_xy()
        sigma = self._sigma_fn(self, xy)
        self._episode_sigmas.append(sigma)
        eps = self._rng.normal(0.0, sigma, size=action.shape)
        return self.env.step(action + eps)

    def get_noise_stats(self, threshold=None):
        """Diagnostic stats over the noise realized so far in the current episode."""
        if len(self._episode_sigmas) == 0:
            stats = {'noise_sigma_mean': 0.0, 'noise_sigma_max': 0.0}
        else:
            sigmas = np.array(self._episode_sigmas)
            stats = {'noise_sigma_mean': float(sigmas.mean()), 'noise_sigma_max': float(sigmas.max())}
        if threshold is not None:
            sigmas = np.array(self._episode_sigmas) if self._episode_sigmas else np.zeros(1)
            stats['noise_sigma_frac_above_threshold'] = float((sigmas > threshold).mean())
        return stats
