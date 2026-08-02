"""A CubePlanOracle that stress-tests action chunking with a reactive slip-and-regrab mechanic.

Subclasses ogbench.manipspace.oracles.plan.cube_plan.CubePlanOracle (imported, not edited in
place). Design (v4, per user direction -- per-step geometric hazard rate, replacing v3's
plan-window-uniform-sample timing):

- p_slip=0.0 is an exact no-op relative to vanilla CubePlanOracle: reset() calls super().reset()
  unmodified, and the only extra randomness this class ever draws (np.random.rand() for the
  per-step hazard check) is short-circuited behind `self._p_slip > 0`, so at p_slip=0 zero extra
  random draws happen anywhere and select_action() always returns the unmodified clean action.

- Slip timing: while proprio/gripper_opening indicates the gripper is currently closed (> 0.5,
  matching CubeMarkovOracle's own open/closed heuristic), every step independently has probability
  p_slip of triggering a slip (memoryless/geometric -- E[steps to trigger] = 1/p_slip). This is
  deliberately observation-based, not plan-based: gripper_opening is part of `ob`, so the identical
  rule can drive an env-side wrapper around ANY policy (including a trained IQL/ACIQL agent) at
  eval time, with no precomputed trajectory to consult -- unlike a trained policy, only the
  scripted oracle has a "plan" to sample a window from, so this is what makes train/eval slip
  statistics matchable at all. p_slip should be calibrated so 1/p_slip falls within the empirical
  range of grip-window durations for the task (measured for cube-double: ~40-47 steps).

- No "did it actually drop" detection is needed for the regrab trigger, because we are the ones
  causing the slip: once the hazard check fires, we deterministically actuate a loosened gripper
  target for slip_duration steps (letting MuJoCo's own contact/gravity physics drop the cube --
  see envs/noise_wrapper.py's docstring for the same actuate-loose/log-tight asymmetry), wait
  settle_steps more for it to finish falling, then unconditionally replan a fresh pick-and-place
  from the cube's new resting pose to the SAME original target. Cascading is allowed and, by
  default, unbounded: the fresh attempt is itself eligible for further hazard-rate slips, with no
  cap on regrab depth other than max_slips_per_manipulation if the caller explicitly sets one
  (default float('inf') -- matches envs/slip_wrapper.py's SlipGripperWrapper, which has never had a
  cap). The only thing that bounds worst-case episode length is max_episode_steps itself.
"""

import gymnasium
import mujoco
import numpy as np

from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle


class SlipCubePlanOracle(CubePlanOracle):
    def __init__(
        self,
        *args,
        p_slip=0.0,
        gripper_closed_threshold=0.5,
        slip_open_target=0.15,
        slip_duration=5,
        settle_steps=10,
        max_slips_per_manipulation=float('inf'),
        **kwargs,
    ):
        """Args (see module docstring for what each trades off):
            p_slip: per-step probability of triggering a slip while the gripper is currently
                closed (memoryless -- E[steps to trigger] = 1/p_slip). Calibrate 1/p_slip to sit
                within the task's typical grip-window duration for a "healthy" trigger rate.
            gripper_closed_threshold: proprio/gripper_opening value above which the gripper counts
                as "closed" and the hazard check is active.
            slip_open_target: physical gripper closedness (0=open, 1=closed) actuated during a
                slip, overriding the current closed target.
            slip_duration: consecutive control steps the loosened target is actuated.
            settle_steps: extra steps after the loosened window ends, before snapshotting the
                cube's position for the regrab plan -- gives physics time to finish the fall.
            max_slips_per_manipulation: cap on cascading regrab depth per manipulation
                sub-trajectory. Defaults to unbounded (matches SlipGripperWrapper's eval-time
                behavior, which has no such cap either) -- worst-case episode length is bounded by
                max_episode_steps regardless. Set to a finite value only if you deliberately want
                data-gen to give up on a manipulation after N cascading slips instead of letting
                physics/max_episode_steps decide.
        """
        super().__init__(*args, **kwargs)
        self._p_slip = p_slip
        self._gripper_closed_threshold = gripper_closed_threshold
        self._slip_open_target = slip_open_target
        self._slip_duration = slip_duration
        self._settle_steps = settle_steps
        self._max_slips = max_slips_per_manipulation

        self._target_block = None
        self._plan_input = None  # stashed so a regrab can reuse the same block_goal

        self._slip_count = 0
        self._slip_steps_remaining = 0
        self._settle_steps_remaining = 0
        self._pending_regrab = False

        # Set by select_action() each call; read by the generation script for the actuated env.step().
        self.last_clean_action = None

    def reset(self, ob, info):
        self._target_block = info['privileged/target_block']
        self._slip_count = 0
        self._plan_input = None  # cleared so the base reset()'s own block_goal is used fresh
        super().reset(ob, info)  # unmodified call -- preserves RNG-stream parity at p_slip=0

        # Recover block_goal for later regrabs (base class doesn't stash it). Rebuilt from `info`
        # the same way CubePlanOracle.reset() does -- no extra randomness drawn here.
        self._plan_input = {
            'block_goal': self.to_pose(
                pos=info['privileged/target_block_pos'],
                yaw=info['privileged/target_block_yaw'][0],
            ),
        }

    def _regrab(self, info):
        """Replan a fresh pick-and-place from the cube's current (post-slip) pose to the original
        target. Called unconditionally once the post-slip settle window elapses -- no detection
        needed, since we caused the slip ourselves."""
        block_pos = info[f'privileged/block_{self._target_block}_pos']
        block_yaw = info[f'privileged/block_{self._target_block}_yaw'][0]
        effector_pose = self.to_pose(pos=info['proprio/effector_pos'], yaw=info['proprio/effector_yaw'][0])

        plan_input = {
            'effector_initial': effector_pose,
            'effector_goal': self.to_pose(
                pos=np.random.uniform(*self._env.unwrapped._arm_sampling_bounds),
                yaw=np.random.uniform(-np.pi, np.pi),
            ),
            'block_initial': self.to_pose(pos=block_pos, yaw=block_yaw),
            'block_goal': self._plan_input['block_goal'],
        }
        times, poses, grasps = self.compute_keyframes(plan_input)
        poses = [poses[name] for name in times.keys()]
        grasps = [grasps[name] for name in times.keys()]
        times = list(times.values())

        self._t_init = info['time'][0]
        self._t_max = times[-1]
        self._done = False
        self._plan = self.compute_plan(times, poses, grasps)

        self._slip_count += 1

    def select_action(self, ob, info):
        if self._pending_regrab:
            self._regrab(info)
            self._pending_regrab = False

        clean_action = super().select_action(ob, info)
        self.last_clean_action = np.array(clean_action)

        if self._slip_steps_remaining == 0 and self._settle_steps_remaining == 0 and self._slip_count < self._max_slips:
            gripper_closed = info['proprio/gripper_opening'][0] > self._gripper_closed_threshold
            # Short-circuit: self._p_slip > 0 must be checked first, or np.random.rand() would be
            # called (and a random draw consumed) even at p_slip=0, breaking the no-op guarantee.
            if gripper_closed and self._p_slip > 0 and np.random.rand() < self._p_slip:
                self._slip_steps_remaining = self._slip_duration

        if self._slip_steps_remaining > 0:
            self._slip_steps_remaining -= 1
            cur_closedness = info['proprio/gripper_opening'][0]
            delta_raw = self._slip_open_target - cur_closedness
            raw_action = np.zeros(5)
            raw_action[4] = delta_raw
            actuated = np.array(clean_action)
            actuated[4] = self._env.unwrapped.normalize_action(raw_action)[4]
            if self._slip_steps_remaining == 0:
                self._settle_steps_remaining = self._settle_steps
            return np.clip(actuated, -1, 1)

        if self._settle_steps_remaining > 0:
            self._settle_steps_remaining -= 1
            if self._settle_steps_remaining == 0:
                self._pending_regrab = True
            return clean_action

        return clean_action


class ClonePeekSlipCubePlanOracle(SlipCubePlanOracle):
    """SlipCubePlanOracle variant with a zero-reaction-time regrab.

    Instead of settle_steps (idle real-env wait), this forks a disposable clone of the real env at
    the moment a slip's actuation window ends, runs ONLY the clone forward peek_settle_steps
    (continuing the same "push toward slip_open_target" actuation the real slip step used) so
    gravity finishes the drop, reads the block's landing pose back from the clone, and starts the
    real regrab immediately using that peeked pose. The real (recorded) env sees zero idle frames
    between the last slip-actuated step and the start of a correctly-targeted regrab -- every
    transition the real env records is still a genuine single env.step() apart, only the choice of
    where to regrab uses the clone.

    Known property: the very first regrab action is computed from the clone's peeked FUTURE
    landing pose, not from anything present in the real env's observation at that step -- a
    reactive policy trained only on real-time observations cannot reproduce this.

    peek_settle_steps should be large enough that the block has essentially stopped moving in the
    clone by the end of the window. Measured for cube-double: ~40 steps gets the vast majority of
    drops to zero measured motion in the last 10 clone steps; a small minority (~15%) still show
    sub-cm residual motion at 40 steps, shrinking further (to ~3mm worst case) by 80 steps but
    never exactly reaching zero -- this is asymptotic contact-solver settling, not a bug, and is
    well within grasp tolerance either way.
    """

    def __init__(self, *args, env_name, peek_settle_steps=40, **kwargs):
        """Args (in addition to SlipCubePlanOracle's, minus settle_steps which this class ignores):
            env_name: registered env id to construct the peek clone with (must match `env`'s).
            peek_settle_steps: control steps the clone is run forward to let the drop finish. See
                class docstring for the cube-double calibration.
        """
        super().__init__(*args, **kwargs)
        self._peek_env_name = env_name
        self._peek_settle_steps = peek_settle_steps
        self._clone_env = None

    def _peek_settled_block_pose(self, info):
        """Fork a throwaway clone of the real env at its current physics state, run the clone
        forward peek_settle_steps (continuing the real slip's own "push toward slip_open_target"
        actuation) so gravity finishes the drop, and read the block's resulting pose back -- never
        stepping or recording anything in the real env. The clone is created once and reused
        (qpos/qvel/ctrl/time resynced from the real env before every peek)."""
        if self._clone_env is None:
            self._clone_env = gymnasium.make(
                self._peek_env_name,
                terminate_at_goal=False,
                mode='data_collection',
                max_episode_steps=self._peek_settle_steps + 5,
            )
            self._clone_env.reset()

        real_data = self._env.unwrapped.data
        clone_data = self._clone_env.unwrapped.data
        clone_data.qpos[:] = real_data.qpos[:]
        clone_data.qvel[:] = real_data.qvel[:]
        clone_data.ctrl[:] = real_data.ctrl[:]
        clone_data.time = real_data.time
        mujoco.mj_forward(self._clone_env.unwrapped.model, clone_data)

        hold_info = info
        for _ in range(self._peek_settle_steps):
            cur_closedness = hold_info['proprio/gripper_opening'][0]
            raw_action = np.zeros(5)
            raw_action[4] = self._slip_open_target - cur_closedness
            hold_action = self._clone_env.unwrapped.normalize_action(raw_action)
            _, _, _, _, hold_info = self._clone_env.step(np.clip(hold_action, -1, 1))

        return (
            hold_info[f'privileged/block_{self._target_block}_pos'],
            hold_info[f'privileged/block_{self._target_block}_yaw'],
        )

    def select_action(self, ob, info):
        if self._pending_regrab:
            peeked_pos, peeked_yaw = self._peek_settled_block_pose(info)
            peeked_info = dict(info)
            peeked_info[f'privileged/block_{self._target_block}_pos'] = peeked_pos
            peeked_info[f'privileged/block_{self._target_block}_yaw'] = peeked_yaw
            self._regrab(peeked_info)
            self._pending_regrab = False

        # Bypasses SlipCubePlanOracle.select_action entirely (its settle_steps machinery) -- reuses
        # only the base plan-tracking select_action, then reimplements the hazard check + slip
        # actuation, ending in an immediate pending_regrab (no idle settle at all; the clone peek
        # above already knows the landing pose by the time the next call consumes it).
        clean_action = super(SlipCubePlanOracle, self).select_action(ob, info)
        self.last_clean_action = np.array(clean_action)

        if self._slip_steps_remaining == 0 and self._slip_count < self._max_slips:
            gripper_closed = info['proprio/gripper_opening'][0] > self._gripper_closed_threshold
            if gripper_closed and self._p_slip > 0 and np.random.rand() < self._p_slip:
                self._slip_steps_remaining = self._slip_duration

        if self._slip_steps_remaining > 0:
            self._slip_steps_remaining -= 1
            cur_closedness = info['proprio/gripper_opening'][0]
            delta_raw = self._slip_open_target - cur_closedness
            raw_action = np.zeros(5)
            raw_action[4] = delta_raw
            actuated = np.array(clean_action)
            actuated[4] = self._env.unwrapped.normalize_action(raw_action)[4]
            if self._slip_steps_remaining == 0:
                self._pending_regrab = True  # zero real settle -- peeked pose already known
            return np.clip(actuated, -1, 1)

        return clean_action
