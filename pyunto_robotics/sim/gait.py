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

    The amplitudes are modest on purpose. Raising them to 0.9/1.6 lifts the feet 9-12 cm
    instead of 3, which looks far more like walking -- and rocks the body enough that Asimov
    stopped fitting through a 1.1 m doorway, taking the errand from 4/4 to 0/4 with 63% of
    control steps in contact. Legs that look right are worth less than a robot that gets
    through the door.
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
        # Which way each leg joint bends, read off the model rather than assumed.
        #
        # STANCE is written for pyunto_h1, whose knee flexes positive (range 0..2.2). Asimov 1
        # flexes the other way (-1.5..0), so the same numbers land outside the limit and the
        # servo holds a straight leg through the whole gait cycle -- the robot slid along on a
        # frozen pose. A joint that cannot reach the stance value in the sign STANCE assumes,
        # but can reach its mirror, is simply the other convention.
        self._flip = {}
        for joint, base in self.STANCE.items():
            index = self._act.get(joint)
            if index is None or base == 0.0:
                continue
            low, high = model.jnt_range[model.actuator_trnid[index, 0]]
            if low == high:  # unlimited
                continue
            # Which way this joint bends, from where its travel lies rather than from whether
            # a particular value fits. Asimov's legs are mirrored -- right knee [-1.5, 0],
            # left knee [0, 1.5] -- so a stance of +0.5 is inside the left one's range and
            # outside the right's. Testing only for "does it fit" therefore flipped the right
            # knee and left the left alone, and the two legs drove in phase instead of
            # opposed: measured the right leg travelling 0.246 rad against the left's 0.504,
            # which is a robot dragging one foot. A joint whose range runs mostly negative
            # wants a negative stance, whatever the sign STANCE happens to use.
            self._flip[joint] = (low + high < 0.0) != (base < 0.0)

        # Remember the height the robot was placed at; the base is held there.
        self._base_z = float(data.qpos[2])
        self._write_stance(model, data, 0.0, 0.0)

    def _stance(self, joint: str) -> float:
        """The stance target for a joint, in this model's own sign convention."""
        return -self.STANCE[joint] if self._flip.get(joint) else self.STANCE[joint]

    def _room(self, model: mujoco.MjModel, joint: str, want: float) -> float:
        """Nudge a stance target inward if the swing around it would hit the joint's limit.

        A stance sitting on a limit has travel in one direction only, and half the cycle is
        clipped away. Asimov's left knee stops at 0 and its stance lands there, so that leg
        lifted 0.055 m where the right, whose stance sits mid-range, lifted 0.123 m -- one
        straight leg and one bent one, which is not a walk.
        """
        index = self._act.get(joint)
        if index is None:
            return want
        low, high = model.jnt_range[model.actuator_trnid[index, 0]]
        if low >= high:
            return want
        margin = min(abs(want), (high - low) * 0.25)
        return float(np.clip(want, low + margin, high - margin))

    def _set(self, model: mujoco.MjModel, data: mujoco.MjData, joint: str, value: float) -> None:
        idx = self._act.get(joint)
        if idx is None:
            return
        lo, hi = model.actuator_ctrlrange[idx]
        # Clip to the JOINT's limit too, not just the actuator's. A command outside the joint
        # range is not refused, it is simply not reached, and the half of the cycle that lies
        # outside is silently flattened: Asimov's left knee runs [0, 1.5] and the swing took
        # the command to -0.14, so that leg lifted 0.046 m where the right lifted 0.094 and
        # the robot walked with one straight leg. Clipping here makes the loss visible to the
        # caller instead of leaving it to the physics.
        joint_lo, joint_hi = model.jnt_range[model.actuator_trnid[idx, 0]]
        if joint_lo < joint_hi:
            lo, hi = max(lo, joint_lo), min(hi, joint_hi)
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

        # Each target goes through _stance, so a model whose joints bend the other way gets
        # the mirrored value -- the swing amplitudes flip with it, or a flipped knee would
        # lift by straightening.
        for joint in self.STANCE:
            self._set(model, data, joint, self._room(model, joint, self._stance(joint)))

        def sign(joint: str) -> float:
            return -1.0 if self._flip.get(joint) else 1.0

        # Right leg leads, left leg trails by pi.
        #
        # The half-cycle offset is the gait's own, and must survive whatever sign convention
        # the model uses: on a mirrored pair of legs -- Asimov's right knee runs [-1.5, 0] and
        # its left [0, 1.5] -- flipping each side independently negates the left leg twice,
        # once for the mirror and once for the phase, and the two legs drive together.
        # Measured a hip correlation of +0.88 that way, against -0.94 on a robot that walks.
        # So the phase is applied first, in the gait's own frame, and the model's sign is put
        # on the result.
        # One sign for the pair, not one per joint. A mirrored robot has hip_pitch_r flipped
        # and hip_pitch_l not, so signing each separately cancels the half-cycle offset
        # between them and both legs swing together -- measured a command correlation of
        # +1.00, where a robot that walks reads -1.00. The gait's own left/right opposition
        # has to survive whatever convention the model uses, so take the sign from one side
        # and apply it to the pair.
        leg = sign("hip_pitch_r")
        self._set(model, data, "hip_pitch_r",
                  self._room(model, "hip_pitch_r", self._stance("hip_pitch_r")) + leg * swing * s)
        self._set(model, data, "hip_pitch_l",
                  self._room(model, "hip_pitch_l", self._stance("hip_pitch_l")) - leg * swing * s)
        # Knee lifts only while the leg is swinging forward (positive half of the cycle).
        # A knee lifts by FLEXING, and which way that is depends on the joint, not on the
        # gait's convention -- so each knee takes its own sign here rather than sharing the
        # right one's. Sharing it drove Asimov's left knee from its stance at +0.50 down to 0,
        # its own limit, straightening the leg on the half-cycle it was meant to lift:
        # measured that foot rising 0.058 m against the right's 0.124.
        self._set(model, data, "knee_r",
                  self._room(model, "knee_r", self._stance("knee_r"))
                  + sign("knee_r") * lift * max(c, 0.0))
        self._set(model, data, "knee_l",
                  self._room(model, "knee_l", self._stance("knee_l"))
                  + sign("knee_l") * lift * max(-c, 0.0))
        # Ankles counter-rotate so the foot stays roughly flat.
        ankle = sign("ank_pitch_r")
        self._set(model, data, "ank_pitch_r",
                  self._room(model, "ank_pitch_r", self._stance("ank_pitch_r")) - ankle * 0.4 * swing * s)
        self._set(model, data, "ank_pitch_l",
                  self._room(model, "ank_pitch_l", self._stance("ank_pitch_l")) + ankle * 0.4 * swing * s)

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
