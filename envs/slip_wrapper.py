"""Env-side wrapper applying the same per-step geometric-hazard grasp slip as
data_gen_scripts/cube_slip_oracle.py, but usable with ANY policy (not just the scripted oracle).

Why this has to be a separate mechanism from the data-gen oracle: the oracle's regrab logic can
consult its own precomputed plan, but a trained policy has no such plan -- there's nothing to
"replan" from its point of view. So this wrapper only implements the slip-injection half (actuate
a loosened gripper target while the real action is what got recorded/would be recorded), not the
regrab half; the policy itself is responsible for reacting to the dropped object (that's the whole
point of matching train/eval stochasticity -- a policy trained on regrab demonstrations should
learn to re-approach on its own).

Matches envs/noise_wrapper.py's NoisyActionWrapper interface and conventions exactly (own
np.random.default_rng instance, not global np.random -- this runs interleaved with training via
evaluate_multi_seed(), and global np.random usage here would perturb the training batch-sampling
sequence, see evaluation.py's docstring for why that matters).
"""

import gymnasium as gym
import numpy as np


class SlipGripperWrapper(gym.Wrapper):
    """While proprio/gripper_opening > gripper_closed_threshold, every step independently has
    probability p_slip of forcing the gripper channel of the action toward slip_open_target for
    slip_duration steps (memoryless -- E[steps to trigger] = 1/p_slip). Only meaningful for cube-*
    envs (5-dim action, dim 4 = gripper delta); do not attach to other env families.
    """

    def __init__(self, env, p_slip, slip_seed, gripper_closed_threshold=0.5, slip_open_target=0.15, slip_duration=5):
        super().__init__(env)
        self.p_slip = p_slip
        self.slip_seed = slip_seed
        self.gripper_closed_threshold = gripper_closed_threshold
        self.slip_open_target = slip_open_target
        self.slip_duration = slip_duration
        self._rng = np.random.default_rng(slip_seed)
        self._slip_steps_remaining = 0
        self._episode_slip_steps = 0
        self._episode_steps = 0

    def reset(self, **kwargs):
        self._slip_steps_remaining = 0
        self._episode_slip_steps = 0
        self._episode_steps = 0
        return self.env.reset(**kwargs)

    def set_noise_seed(self, seed):
        """Reseed the slip RNG in place (matches NoisyActionWrapper's convention, e.g. for
        independently-seeded eval repeats -- called from the same evaluate_multi_seed() site)."""
        self._rng = np.random.default_rng(seed)

    def step(self, action):
        action = np.asarray(action)
        self._episode_steps += 1
        info = self.env.unwrapped.compute_ob_info()
        gripper_closed = info['proprio/gripper_opening'][0] > self.gripper_closed_threshold

        if self._slip_steps_remaining == 0 and gripper_closed and self.p_slip > 0 and self._rng.random() < self.p_slip:
            self._slip_steps_remaining = self.slip_duration

        if self._slip_steps_remaining > 0:
            self._slip_steps_remaining -= 1
            self._episode_slip_steps += 1
            cur_closedness = info['proprio/gripper_opening'][0]
            delta_raw = self.slip_open_target - cur_closedness
            raw_action = np.zeros(5)
            raw_action[4] = delta_raw
            actuated = np.array(action, dtype=np.float32)
            actuated[4] = self.env.unwrapped.normalize_action(raw_action)[4]
            actuated = np.clip(actuated, -1, 1)
            return self.env.step(actuated)

        return self.env.step(action)

    def get_slip_stats(self):
        """Diagnostic stats over the slip realized so far in the current episode (mirrors
        NoisyActionWrapper.get_noise_stats())."""
        frac = self._episode_slip_steps / self._episode_steps if self._episode_steps > 0 else 0.0
        return {'slip_steps': float(self._episode_slip_steps), 'slip_step_frac': float(frac)}
