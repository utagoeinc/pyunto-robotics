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
        # Height of the pelvis above the surface under the feet, not above the world.
        self._base_z = float(data.qpos[2]) - self._ground_height(model, data)
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

    def _ground_height(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        """Top of whatever the feet are resting on, in world z.

        Read from the feet themselves rather than assumed to be zero. A robot standing in a
        lift is on a surface several metres up, and one walking up a ramp is on a surface that
        changes continuously; both are the same question.
        """
        lowest = None
        for name in ("foot_r_g", "foot_l_g", "foot_r", "foot_l"):
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom < 0:
                continue
            # The sole, not the foot's centre.
            sole = float(data.geom_xpos[geom][2] - model.geom_size[geom][2])
            lowest = sole if lowest is None else min(lowest, sole)
        return lowest if lowest is not None else 0.0

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

        # Hold the torso upright and at a constant height ABOVE WHATEVER IT IS STANDING ON.
        # This is what "cannot fall over" buys: vertical drift and lean are corrected every
        # step, while horizontal motion stays under the solver's control so obstacles matter.
        #
        # "Above whatever it is standing on", not "at a fixed world height", because a robot
        # in a lift is standing on a floor that moves. Pinned to a world height the gait
        # teleported the robot back down every step while the car rose out from under it: the
        # lift reached the first floor and the robot was still on the ground, having walked
        # off the platform it was fighting. Ground height is taken from the feet, which is
        # where the question is actually answered.
        ground = self._ground_height(model, data)
        data.qvel[2] += (ground + self._base_z - data.qpos[2]) * self.height_gain
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


class DynamicGait:
    """A walking controller that pushes the robot along with its legs.

    KinematicGait writes the base velocity directly and animates the legs to match, which
    cannot fall over and cannot walk either: the feet slide because nothing connects them to
    the motion. This one commands only joints. Everything the body does -- forward motion,
    turning, staying upright -- has to come out of the feet pressing on the floor, which is
    what makes a leg that does not take its share immediately visible.

    Three loops, in the order they matter:

      Balance   The ankles hold the torso upright, reading pitch and roll from the base and
                pushing back through whichever foot is loaded. Without this the robot tips
                inside a second: measured every stance the servos can hold statically ending
                up past 0.2 rad of lean.

      Weight    A walk is a controlled fall from one foot to the other. The hips roll the
                body over the stance leg before the swing leg lifts, so there is something to
                stand on when it does. Skipping this is what leaves one foot glued down --
                measured the left foot in contact for 300 of 300 control steps.

      Swing     The unloaded leg lifts, reaches, and plants. Hip pitch sets the step length,
                knee flexion the ground clearance, ankle pitch keeps the sole flat so it
                lands on the whole foot rather than a toe or a heel.

    Signs come from the model. Asimov's legs are mirrored -- right knee [-1.5, 0], left knee
    [0, 1.5] -- so every target is expressed as "flex" or "extend" and turned into a number
    per joint from its own range.

    STATUS: the gait cycle works, the stiffness does not. Foot alternation is exactly right --
    measured 200 of 400 control steps on each foot, against the kinematic gait's 300 and 66 --
    which is the shuffle this was written to fix. What is unresolved is holding 32 kg up while
    doing it. The knees need about 46 Nm in single support, which a position servo only
    delivers after sagging: at kp 500 the robot folds to a pelvis height of 0.16 m, and at the
    2000 that holds it upright the legs diverge instead, hitting 19740 rad/s within three
    control steps. Refining the timestep to 1 ms removes the divergence and leaves the sag.

    The gap is that a position servo is the wrong actuator for this. Holding a pose against
    gravity wants feedforward -- gravity compensation from the model's own inverse dynamics,
    with the servo correcting only the residual -- or torque actuators under a policy trained
    for it, which is what PolicyGait below is reserved for. Both are real work rather than a
    gain to be found, which is why KinematicGait remains the default.

    Not wired to any scene: pass it explicitly, `Robot(scene, gait=DynamicGait())`, to
    continue this.
    """

    # Nominal stance, as flexion magnitudes rather than signed angles.
    HIP_FLEX = 0.25
    KNEE_FLEX = 0.50
    ANKLE_FLEX = 0.25

    def __init__(
        self,
        step_freq: float = 1.3,        # gait cycles per second
        step_length: float = 0.28,     # hip swing, radians
        step_height: float = 0.35,     # knee flexion during swing, radians
        hip_roll: float = 0.08,        # weight shift onto the stance leg, radians
        balance_gain: float = 1.8,     # ankle response to torso lean
        balance_damp: float = 0.25,
        turn_gain: float = 0.5,        # hip yaw per rad/s of commanded turn
        accel: float = 2.0,
    ):
        self.step_freq = step_freq
        self.step_length = step_length
        self.step_height = step_height
        self.hip_roll = hip_roll
        self.balance_gain = balance_gain
        self.balance_damp = balance_damp
        self.turn_gain = turn_gain
        self.accel = accel

        self._phase = 0.0
        self._vx = self._vy = self._wz = 0.0
        self._act: dict[str, int] = {}
        self._flex: dict[str, float] = {}

    # -- setup --------------------------------------------------------------------

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._phase = 0.0
        self._vx = self._vy = self._wz = 0.0
        self._act = {}
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self._act[name] = i

        # Which direction is "flex" for each leg joint, from its own travel. A joint whose
        # range lies mostly below zero flexes negative; mostly above, positive. This is what
        # lets one controller drive a mirrored pair without a table of per-robot signs.
        #
        # The stance is then written into qpos as well as ctrl, not just commanded. A keyframe
        # saved for the kinematic gait puts the legs somewhere this controller does not want
        # them, and at the stiffness needed to hold 32 kg upright that error is an impulse:
        # measured qacc of 2.4 million on the second control step, which is the solver being
        # handed a spring compressed by half a radian and asked to integrate it at 5 ms.
        self._flex = {}
        for joint in ("hip_pitch_r", "hip_pitch_l", "hip_roll_r", "hip_roll_l",
                      "hip_yaw_r", "hip_yaw_l", "knee_r", "knee_l",
                      "ank_pitch_r", "ank_pitch_l", "ank_roll_r", "ank_roll_l"):
            index = self._act.get(joint)
            if index is None:
                continue
            low, high = model.jnt_range[model.actuator_trnid[index, 0]]
            self._flex[joint] = -1.0 if (low + high) < 0.0 else 1.0

        # Put the body in the stance this controller holds, so step one has nothing to correct.
        for joint, amount in (("hip_pitch_r", self.HIP_FLEX), ("hip_pitch_l", self.HIP_FLEX),
                              ("knee_r", self.KNEE_FLEX), ("knee_l", self.KNEE_FLEX),
                              ("ank_pitch_r", self.ANKLE_FLEX), ("ank_pitch_l", self.ANKLE_FLEX)):
            index = self._act.get(joint)
            if index is None:
                continue
            target = self._flex.get(joint, 1.0) * amount
            data.qpos[model.jnt_qposadr[model.actuator_trnid[index, 0]]] = target
            data.ctrl[index] = target
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    # -- helpers ------------------------------------------------------------------

    def _set(self, model: mujoco.MjModel, data: mujoco.MjData, joint: str, value: float) -> None:
        index = self._act.get(joint)
        if index is None:
            return
        low, high = model.actuator_ctrlrange[index]
        joint_low, joint_high = model.jnt_range[model.actuator_trnid[index, 0]]
        if joint_low < joint_high:
            low, high = max(low, joint_low), min(high, joint_high)
        data.ctrl[index] = float(np.clip(value, low, high))

    def _flexed(self, joint: str, amount: float) -> float:
        """`amount` of flexion at `joint`, in that joint's own sign."""
        return self._flex.get(joint, 1.0) * amount

    # -- the loop -----------------------------------------------------------------

    def apply(
        self, model: mujoco.MjModel, data: mujoco.MjData, vx: float, vy: float, wz: float, dt: float
    ) -> None:
        if not self._act:
            self.reset(model, data)

        self._vx += float(np.clip(vx - self._vx, -self.accel * dt, self.accel * dt))
        self._vy += float(np.clip(vy - self._vy, -self.accel * dt, self.accel * dt))
        self._wz += float(np.clip(wz - self._wz, -self.accel * 3 * dt, self.accel * 3 * dt))

        speed = math.hypot(self._vx, self._vy)
        moving = speed > 0.02 or abs(self._wz) > 0.05
        if moving:
            self._phase = (self._phase + 2 * math.pi * self.step_freq * dt) % (2 * math.pi)

        # Torso attitude, for the balance term. Pitch is lean fore-aft, roll side to side.
        quat = data.qpos[3:7]
        w, x, y, z = quat
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch_rate, roll_rate = float(data.qvel[4]), float(data.qvel[3])

        # Ankles resist the lean. This is the whole of the balance controller: the feet are
        # the only things touching the ground, so it is the only place a correction can act.
        ankle_pitch = self.balance_gain * pitch + self.balance_damp * pitch_rate
        ankle_roll = self.balance_gain * roll + self.balance_damp * roll_rate

        cycle = math.sin(self._phase)
        # Right leg swings on the first half of the cycle, left on the second.
        swing_r = max(math.sin(self._phase), 0.0)
        swing_l = max(-math.sin(self._phase), 0.0)

        # Weight rolls toward whichever leg is about to take the load, a quarter cycle early
        # so the body is already over the foot when the other one lifts.
        shift = self.hip_roll * math.sin(self._phase - math.pi / 2) if moving else 0.0

        stride = self.step_length * float(np.clip(self._vx / 0.5, -1.0, 1.0))
        yaw_offset = self.turn_gain * self._wz

        for side, swing, lead in (("r", swing_r, 1.0), ("l", swing_l, -1.0)):
            hip_pitch = f"hip_pitch_{side}"
            knee = f"knee_{side}"
            ank_pitch = f"ank_pitch_{side}"
            hip_roll = f"hip_roll_{side}"
            hip_yaw = f"hip_yaw_{side}"
            ank_roll = f"ank_roll_{side}"

            # Hip: nominal crouch, plus the stride, plus the balance correction. The swinging
            # leg reaches forward while the stance leg drives back -- that push is what moves
            # the robot, since nothing else does.
            reach = stride * lead * cycle
            self._set(model, data, hip_pitch,
                      self._flexed(hip_pitch, self.HIP_FLEX) - reach - pitch * 0.6)

            # Knee: crouch plus lift while swinging, so the foot clears the floor.
            self._set(model, data, knee,
                      self._flexed(knee, self.KNEE_FLEX + self.step_height * swing))

            # Ankle: hold the sole flat against the hip's motion, and take the balance term.
            self._set(model, data, ank_pitch,
                      self._flexed(ank_pitch, self.ANKLE_FLEX) + reach * 0.5 - ankle_pitch)

            # Roll: shift the weight, and keep the foot flat sideways.
            self._set(model, data, hip_roll, self._flexed(hip_roll, 0.0) + shift * lead)
            self._set(model, data, ank_roll, -ankle_roll)
            self._set(model, data, hip_yaw, yaw_offset * lead)

        self._set(model, data, "waist_yaw", 0.0)


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
