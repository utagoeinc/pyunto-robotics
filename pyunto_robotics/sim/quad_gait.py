"""A trot for the four-legged patrol robot.

Implements the same `Gait` protocol as the humanoid's KinematicGait, so everything above the
simulator -- navigation, perception, skills -- says `step(vx, vy, wz)` and never learns that
this robot has four legs. That seam is the whole point of the interface.

Two backends, mirroring the humanoid:

  TrotGait       Kinematic. The base is driven directly and the legs play a diagonal trot for
                 looks and for ground contact. It cannot fall over, which is what makes a live
                 demo survivable, and obstacles still stop it because horizontal motion goes
                 through qvel rather than qpos.

  QuadPolicyGait Reserved for a trained RL policy, exactly as PolicyGait is on the biped.

Why a trot and not a walk: a trot moves diagonal pairs together, so there are always two feet
down and the support line runs through the middle of the body. It is the gait that looks right
at patrol speed and the one that is least upset by a step in the ground.

The leg IK here is closed-form, not iterative. Each leg is a two-link chain in its own sagittal
plane -- thigh 0.200 m, shin 0.205 m to the foot centre -- so putting a foot at (x, z) relative
to the hip is the standard two-link solution, and there is no reason to run a solver for it.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

# Link lengths, from assets/pyunto_q1.xml. Kept here rather than read from the model because
# they are geometry the gait is written around, and a silent change to either should be a
# visible edit to this file too.
THIGH_M = 0.200
SHIN_M = 0.205

# Nominal standing foot position relative to the hip: straight down, slightly tucked.
STANCE_HEIGHT_M = 0.325
STANCE_OFFSET_M = 0.0

# Leg names in the order their actuators are declared.
LEGS = ("fl", "fr", "hl", "hr")

# Trot pairs: diagonal legs move together. Front-left with hind-right, front-right with
# hind-left. This is what keeps the support polygon under the centre of mass.
PHASE_OFFSET = {"fl": 0.0, "hr": 0.0, "fr": math.pi, "hl": math.pi}


def leg_ik(reach: float, drop: float) -> tuple[float, float]:
    """Hip and knee angles that put a foot at (reach, -drop) in the leg's sagittal plane.

    `reach` is forward of the hip, `drop` is below it, both in metres. Returns (hip, knee) in
    radians, with knee positive because the joint is one-sided in the model -- a knee that can
    invert finds inverted solutions and the robot walks itself inside out.

    Targets beyond the leg's span are clamped to just inside it rather than raising: a gait
    that asks for an impossible foot position should straighten the leg toward it, which is
    what a real leg does, not stop the robot.
    """
    distance = math.hypot(reach, drop)
    span = THIGH_M + SHIN_M
    distance = min(distance, span * 0.995)
    distance = max(distance, abs(THIGH_M - SHIN_M) + 1e-4)

    # Law of cosines for the knee's interior angle, then convert to joint angle.
    cos_knee = (THIGH_M**2 + SHIN_M**2 - distance**2) / (2 * THIGH_M * SHIN_M)
    knee_interior = math.acos(float(np.clip(cos_knee, -1.0, 1.0)))
    knee = math.pi - knee_interior

    # Angle from straight-down to the foot, plus the thigh's offset from the hip-foot line.
    #
    # Note the sign on `reach`: the hip joint's +y axis rotates the leg BACKWARDS in this
    # model, so a positive hip angle swings the foot aft. Verified against MuJoCo rather than
    # assumed -- the first version had the height exactly right and the fore-aft component
    # mirrored, which produces a gait that walks the robot backwards while the base is driven
    # forwards, feet skating the whole way.
    cos_thigh = (THIGH_M**2 + distance**2 - SHIN_M**2) / (2 * THIGH_M * distance)
    thigh_offset = math.acos(float(np.clip(cos_thigh, -1.0, 1.0)))
    to_foot = math.atan2(-reach, drop)
    hip = to_foot - thigh_offset

    return hip, knee


class TrotGait:
    """Kinematic trot. Same interface as the humanoid's gait, four legs instead of two."""

    def __init__(
        self,
        step_freq: float = 2.2,       # trot cycles per second at full speed
        stride_m: float = 0.16,       # how far a foot travels fore-aft per cycle, at 1 m/s
        lift_m: float = 0.075,        # peak foot clearance during swing
        accel: float = 2.0,           # m/s^2 ramp on the commanded velocity
        yaw_accel: float = 5.0,       # rad/s^2
        height_gain: float = 12.0,    # how hard to hold the trunk at its standing height
        turn_stride_m: float = 0.10,  # extra fore-aft travel per rad/s of yaw command
    ):
        self.step_freq = step_freq
        self.stride_m = stride_m
        self.lift_m = lift_m
        self.accel = accel
        self.yaw_accel = yaw_accel
        self.height_gain = height_gain
        self.turn_stride_m = turn_stride_m

        self._phase = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._wz = 0.0
        self._act: dict[str, int] = {}
        self._base_z = 0.36

    # -- Gait protocol -------------------------------------------------------------

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._phase = 0.0
        self._vx = self._vy = self._wz = 0.0
        self._act = {}
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self._act[name] = i
        self._base_z = float(data.qpos[2])
        self._write_legs(model, data, 0.0, 0.0, 0.0)

    def apply(
        self, model: mujoco.MjModel, data: mujoco.MjData, vx: float, vy: float, wz: float, dt: float
    ) -> None:
        if not self._act:
            self.reset(model, data)

        # Ramp toward the command rather than jumping, so the legs do not snap between poses.
        self._vx += float(np.clip(vx - self._vx, -self.accel * dt, self.accel * dt))
        self._vy += float(np.clip(vy - self._vy, -self.accel * dt, self.accel * dt))
        self._wz += float(np.clip(wz - self._wz, -self.yaw_accel * dt, self.yaw_accel * dt))

        speed = math.hypot(self._vx, self._vy)
        # Turning on the spot still has to cycle the legs, or the robot pivots on planted feet
        # and the trot stops looking like locomotion.
        activity = max(speed, abs(self._wz) * 0.35)
        if activity > 1e-3:
            self._phase = (
                self._phase + 2 * math.pi * self.step_freq * min(activity, 1.2) * dt
            ) % (2 * math.pi)

        self._write_legs(model, data, self._vx, self._vy, self._wz)
        self._drive_base(model, data, dt)

    # -- internals -----------------------------------------------------------------

    def _write_legs(
        self, model: mujoco.MjModel, data: mujoco.MjData, vx: float, vy: float, wz: float
    ) -> None:
        """Place all four feet for the current phase."""
        for leg in LEGS:
            phase = (self._phase + PHASE_OFFSET[leg]) % (2 * math.pi)

            # Fore-aft travel scales with commanded speed. Legs on the outside of a turn take
            # longer steps than those on the inside, which is what actually turns the body.
            side = 1.0 if leg in ("fl", "hl") else -1.0
            reach_amplitude = self.stride_m * min(abs(vx), 1.2) * (1.0 if vx >= 0 else -1.0)
            reach_amplitude -= self.turn_stride_m * wz * side

            # Stance is the half of the cycle with the foot on the ground, sweeping backwards;
            # swing is the other half, lifting and returning.
            reach = reach_amplitude * math.cos(phase)
            lift = self.lift_m * max(math.sin(phase), 0.0) * min(
                max(abs(vx), abs(vy), abs(wz) * 0.35), 1.2
            )

            hip, knee = leg_ik(STANCE_OFFSET_M + reach, STANCE_HEIGHT_M - lift)
            self._set(model, data, f"hip_{leg}", hip)
            self._set(model, data, f"knee_{leg}", knee)
            # Abduction leans the legs into a sideways command, so strafing does not just drag
            # the feet across the ground.
            self._set(model, data, f"abd_{leg}", float(np.clip(vy * 0.35, -0.5, 0.5)))

    def _set(self, model: mujoco.MjModel, data: mujoco.MjData, joint: str, value: float) -> None:
        index = self._act.get(joint)
        if index is None:
            return
        lo, hi = model.actuator_ctrlrange[index]
        data.ctrl[index] = float(np.clip(value, lo, hi))

    def _drive_base(self, model: mujoco.MjModel, data: mujoco.MjData, dt: float) -> None:
        """Move the trunk, exactly as the humanoid's kinematic gait moves its pelvis.

        Horizontal motion is written as VELOCITY, never position: setting qpos teleports the
        body a step at a time and skips collision response entirely, so the robot walks through
        walls. Writing qvel lets MuJoCo integrate it, and contacts still stop the robot.

        Height and attitude ARE overridden, which is what "cannot fall over" buys. On this robot
        that matters more than on the biped: the patrol route has steps and a slope, and the
        purpose here is to exercise navigation over terrain rather than to solve legged balance.
        """
        quat = data.qpos[3:7]
        w, x, y, z = quat
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        data.qvel[0] = self._vx * cos_y - self._vy * sin_y
        data.qvel[1] = self._vx * sin_y + self._vy * cos_y

        # Hold the trunk at its standing height ABOVE THE GROUND IT IS ON, not at a fixed world
        # height -- otherwise the robot refuses to climb: it would fight its way back down to
        # the height of the lawn while standing on a step.
        support = self._ground_height(model, data)
        data.qvel[2] += (support + self._base_z - data.qpos[2]) * self.height_gain
        data.qvel[3] = 0.0
        data.qvel[4] = 0.0

        # Set the heading through qpos ONLY, and leave the yaw velocity at zero.
        #
        # Writing both the quaternion and qvel[5] makes MuJoCo apply the rotation twice: the
        # explicit quaternion sets it, then the integrator adds the velocity on top, and the
        # combination overshoots so far it comes back round the other way. Measured a command
        # of +0.8 rad/s for 3 s producing -2.63 rad where +2.40 was wanted -- a turn that looks
        # like a sign error and is actually double integration. With qvel[5] left at zero the
        # same command gives exactly +2.400.
        heading = yaw + self._wz * dt
        data.qpos[3:7] = np.array(
            [math.cos(heading / 2.0), 0.0, 0.0, math.sin(heading / 2.0)]
        )
        data.qvel[5] = 0.0

    @staticmethod
    def _ground_height(model: mujoco.MjModel, data: mujoco.MjData) -> float:
        """Height of whatever the feet are standing on, in world z.

        Taken as the lowest foot, which is the one bearing weight on a slope or a step. Using a
        fixed world height instead makes the trunk controller fight the terrain: on a 0.16 m
        step it would haul the robot back down to lawn height while its feet were on the stair.
        """
        lowest = None
        for leg in LEGS:
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{leg}")
            if geom < 0:
                continue
            foot_z = float(data.geom_xpos[geom][2] - model.geom_size[geom][0])
            lowest = foot_z if lowest is None else min(lowest, foot_z)
        return lowest if lowest is not None else 0.0


class QuadPolicyGait:
    """Placeholder for a trained RL trot, matching PolicyGait on the biped.

    Deliberately not implemented: the demo runs on TrotGait, and dropping a policy in here
    later requires no change anywhere above the Gait interface.
    """

    def __init__(self, policy_path: str):
        self.policy_path = policy_path
        raise NotImplementedError(
            "RL locomotion for the quadruped is future work. Train a policy, then load it "
            "here; the Gait interface stays the same so navigation code is unaffected."
        )

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:  # pragma: no cover
        raise NotImplementedError

    def apply(  # pragma: no cover
        self, model: mujoco.MjModel, data: mujoco.MjData, vx: float, vy: float, wz: float, dt: float
    ) -> None:
        raise NotImplementedError
