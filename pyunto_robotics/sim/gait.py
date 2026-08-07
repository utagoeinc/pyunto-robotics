"""Gait controllers behind a single velocity-command interface.

Everything above this file - navigation, perception, the planner - only ever says
"move at (vx, vy, wz)". How the legs actually achieve that is a detail that lives here.

Two backends:

  KinematicGait  Drives the floating base directly and plays a leg cycle for looks. It cannot
                 fall over, which is what makes a live demo survivable. Physics still applies
                 to everything else: the arms, the doors, and any object the robot touches.

  PolicyGait     Reserved for a trained RL policy (phase 6). Same interface, so swapping it in
                 changes nothing upstream.

The seam matters because on this machine RL training is viable (measured ~72k env-steps/s for
rollouts, and MPS gradient updates ~24x faster than CPU) but not something to block the demo on.
"""

from __future__ import annotations

import math
from typing import Protocol

import mujoco
import numpy as np


class Gait(Protocol):
    """Turns a velocity command into joint targets."""

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Prepare for a new episode."""
        ...

    def apply(
        self, model: mujoco.MjModel, data: mujoco.MjData, vx: float, vy: float, wz: float, dt: float
    ) -> None:
        """Advance one control step toward the requested body-frame velocity."""
        ...


def _quat_yaw(quat: np.ndarray) -> float:
    """Yaw angle from a wxyz quaternion."""
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def _slerp_to(current: np.ndarray, target: np.ndarray, t: float) -> np.ndarray:
    """Move `current` a fraction `t` of the way toward `target`, normalised.

    A plain lerp is fine here because the two orientations are always close: this only ever
    corrects a small lean back to upright.
    """
    t = float(np.clip(t, 0.0, 1.0))
    if np.dot(current, target) < 0.0:  # take the short way round
        target = -target
    blended = current * (1.0 - t) + target * t
    norm = np.linalg.norm(blended)
    return target if norm < 1e-9 else blended / norm


class KinematicGait:
    """Moves the base kinematically and animates the legs to match.

    The base is a free joint, so we write its position and velocity directly each step rather
    than hoping leg contacts produce the requested motion. The leg cycle is driven by the
    commanded speed, so the feet visually keep pace with the body instead of sliding.

    Arms are left alone: whoever is doing manipulation owns those joints.
    """

    # Joint targets for a neutral stance, applied to whichever of these joints exist.
    STANCE = {
        "hip_yaw_r": 0.0, "hip_roll_r": 0.0, "hip_pitch_r": -0.25,
        "knee_r": 0.50, "ank_pitch_r": -0.25, "ank_roll_r": 0.0,
        "hip_yaw_l": 0.0, "hip_roll_l": 0.0, "hip_pitch_l": -0.25,
        "knee_l": 0.50, "ank_pitch_l": -0.25, "ank_roll_l": 0.0,
        "waist_yaw": 0.0,
    }

    def __init__(
        self,
        step_freq: float = 1.6,      # leg cycles per second at full speed
        stride_gain: float = 0.45,   # hip swing per m/s of commanded speed
        lift_gain: float = 0.55,     # knee lift per m/s
        accel: float = 2.5,          # m/s^2 ramp, so commands do not snap
        yaw_accel: float = 6.0,      # rad/s^2
        height_gain: float = 12.0,   # how hard to hold the torso at its standing height
        upright_gain: float = 14.0,  # how hard to damp out pitch/roll
    ):
        self.step_freq = step_freq
        self.stride_gain = stride_gain
        self.lift_gain = lift_gain
        self.accel = accel
        self.yaw_accel = yaw_accel
        self.height_gain = height_gain
        self.upright_gain = upright_gain

        self._phase = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._wz = 0.0
        self._act: dict[str, int] = {}
        self._base_z = 0.81

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._phase = 0.0
        self._vx = self._vy = self._wz = 0.0
        self._act = {}
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self._act[name] = i
        # Remember the height the robot was placed at; the base is held there.
        self._base_z = float(data.qpos[2])
        self._write_stance(model, data, 0.0, 0.0)

    def _set(self, model: mujoco.MjModel, data: mujoco.MjData, joint: str, value: float) -> None:
        idx = self._act.get(joint)
        if idx is None:
            return
        lo, hi = model.actuator_ctrlrange[idx]
        data.ctrl[idx] = float(np.clip(value, lo, hi))

    def _write_stance(
        self, model: mujoco.MjModel, data: mujoco.MjData, swing: float, lift: float
    ) -> None:
        """Write leg targets for the current phase.

        `swing` is the hip amplitude and `lift` the knee amplitude; the two legs run half a
        cycle apart so one swings while the other supports.
        """
        s = math.sin(self._phase)
        c = math.cos(self._phase)

        for joint, base in self.STANCE.items():
            self._set(model, data, joint, base)

        # Right leg leads, left leg trails by pi.
        self._set(model, data, "hip_pitch_r", self.STANCE["hip_pitch_r"] + swing * s)
        self._set(model, data, "hip_pitch_l", self.STANCE["hip_pitch_l"] - swing * s)
        # Knee lifts only while the leg is swinging forward (positive half of the cycle).
        self._set(model, data, "knee_r", self.STANCE["knee_r"] + lift * max(c, 0.0))
        self._set(model, data, "knee_l", self.STANCE["knee_l"] + lift * max(-c, 0.0))
        # Ankles counter-rotate so the foot stays roughly flat.
        self._set(model, data, "ank_pitch_r", self.STANCE["ank_pitch_r"] - 0.4 * swing * s)
        self._set(model, data, "ank_pitch_l", self.STANCE["ank_pitch_l"] + 0.4 * swing * s)

    def apply(
        self, model: mujoco.MjModel, data: mujoco.MjData, vx: float, vy: float, wz: float, dt: float
    ) -> None:
        if not self._act:
            self.reset(model, data)

        # Ramp toward the command instead of jumping, so the animation does not pop.
        self._vx += float(np.clip(vx - self._vx, -self.accel * dt, self.accel * dt))
        self._vy += float(np.clip(vy - self._vy, -self.accel * dt, self.accel * dt))
        self._wz += float(np.clip(wz - self._wz, -self.yaw_accel * dt, self.yaw_accel * dt))

        speed = math.hypot(self._vx, self._vy)

        # Advance the leg cycle in proportion to speed; hold the phase when standing still.
        if speed > 1e-3:
            self._phase = (self._phase + 2 * math.pi * self.step_freq * speed * dt) % (2 * math.pi)
        swing = self.stride_gain * min(speed, 1.2)
        lift = self.lift_gain * min(speed, 1.2)
        self._write_stance(model, data, swing, lift)

        # Drive the base by writing VELOCITY, never position.
        #
        # Setting qpos directly teleports the body one step at a time, which skips collision
        # response entirely -- the robot walked through walls with 31 contacts active, because
        # the solver's correction was overwritten the instant it was computed. Writing qvel
        # instead lets MuJoCo integrate the motion, so contacts actually stop the robot while
        # the gait keeps it upright.
        yaw = _quat_yaw(data.qpos[3:7])
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        # Body-frame command -> world frame.
        world_vx = self._vx * cos_y - self._vy * sin_y
        world_vy = self._vx * sin_y + self._vy * cos_y

        # Horizontal motion goes through qvel so the solver can stop it at a wall.
        data.qvel[0] = world_vx
        data.qvel[1] = world_vy

        # Hold the torso upright and at a constant height. This is what "cannot fall over"
        # buys: vertical drift and lean are corrected every step, while horizontal motion stays
        # under the solver's control so obstacles still matter.
        data.qvel[2] += (self._base_z - data.qpos[2]) * self.height_gain
        data.qvel[3] = 0.0
        data.qvel[4] = 0.0

        # Orientation is set outright: yaw follows the command exactly, pitch and roll are zero.
        #
        # Two reasons this is not left to the solver. The feet are planted by a scripted stance
        # rather than stepping, so foot friction cancels a commanded spin almost entirely (0.8
        # rad/s came out as 0.23). And blending toward the target only applies a fraction of it
        # per step, which throttled turning to a quarter of what was asked for. Heading is also
        # the one degree of freedom a wall has no business resisting, so overriding it costs
        # nothing that collisions care about -- position, which is what walls act on, still goes
        # through qvel above.
        data.qpos[3:7] = _yaw_quat(yaw + self._wz * dt)
        data.qvel[5] = self._wz


class PolicyGait:
    """Placeholder for a trained RL locomotion policy (phase 6).

    Deliberately not implemented yet: the demo runs on KinematicGait, and dropping a policy in
    here later requires no change anywhere above the Gait interface.
    """

    def __init__(self, policy_path: str):
        self.policy_path = policy_path
        raise NotImplementedError(
            "RL locomotion is phase 6. Train a policy, then load it here; the Gait interface "
            "stays the same so navigation code is unaffected."
        )

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:  # pragma: no cover
        raise NotImplementedError

    def apply(  # pragma: no cover
        self, model: mujoco.MjModel, data: mujoco.MjData, vx: float, vy: float, wz: float, dt: float
    ) -> None:
        raise NotImplementedError
