"""Skid steering for the six-wheeled rover.

Implements the same `Gait` protocol as the humanoid's KinematicGait and the quadruped's
TrotGait, so navigation and skills say `step(vx, vy, wz)` here exactly as they do everywhere
else and never learn that this robot has wheels.

This one is genuinely different from the other two, and the difference is worth stating: the
legged gaits drive their base kinematically because a kinematic biped cannot fall over, which
is what makes a demo survivable. A rover does not need that protection -- it is a stable
six-wheeled platform -- so this controller commands WHEEL TORQUE and lets the physics do the
rest. The rover really is pushed along by friction between its wheels and the regolith, it
really does slip on a slope, and it really can get stuck against a crater rim.

That also means the velocity command is a request rather than a promise. `vx` is converted to
a wheel speed through the wheel radius, and what the rover actually does depends on the ground.
Callers should read `Robot.position` to find out what happened rather than assuming.

Measured on flat ground at lunar gravity, which is what to expect:

    command          achieved
    vx  +0.6 m/s     0.57 m/s      linear tracking is good
    vx  -0.5 m/s     0.48 m/s
    wz  +0.5 rad/s   0.24 rad/s    turning delivers about half
    wz  -0.5 rad/s   0.24 rad/s

The turn shortfall is not a bug to tune out. A six-wheeled vehicle turns by making its wheels
scrub sideways, the force available to do that is proportional to weight, and on the Moon the
rover weighs a sixth of what it would on Earth. Skills here plan around it by steering with a
closed loop on heading rather than by assuming a commanded rate is achieved.

Steering is skid: there is no steering joint, so turning means driving the left and right sides
at different speeds. `vy` is therefore ignored -- a skid-steer vehicle cannot move sideways,
and quietly pretending otherwise would let navigation code plan strafes that never happen.
"""

from __future__ import annotations

import logging
import math

import mujoco
import numpy as np

log = logging.getLogger(__name__)

# Wheel radius, from assets/pyunto_r1.xml. Kept here rather than read from the model because
# the drive is written around it, and a silent change to either should be a visible edit here.
WHEEL_RADIUS_M = 0.20

# Half the track width: the lateral distance from the centre line to a wheel. Turning rate
# follows from this and the speed difference between the two sides.
HALF_TRACK_M = 0.43

# Half the wheelbase: how far the corner wheels sit fore and aft of the rover's centre. Sets
# how much steering angle a given turn radius needs.
HALF_BASE_M = 0.46

LEFT_WHEELS = ("drive_lf", "drive_lm", "drive_lr")
RIGHT_WHEELS = ("drive_rf", "drive_rm", "drive_rr")


class SkidDrive:
    """Six-wheel skid steering. Commands wheel speeds; the ground decides the rest."""

    def __init__(
        self,
        accel: float = 0.9,          # m/s^2 ramp on the linear command
        yaw_accel: float = 1.6,      # rad/s^2 ramp on the turn command
        max_wheel_rad_s: float = 14.0,
        # Extra wheel speed asked for when turning.
        #
        # 3.0, not the 1.35 a first estimate suggests, and the reason is lunar gravity. Skid
        # steering turns by making the wheels scrub sideways across the ground, and the force
        # available to do that is proportional to weight -- which here is a SIXTH of Earth's.
        # At 1.35 the rover turned at 0.03 rad/s against a commanded 0.5: the wheels were
        # slipping away almost the entire difference. This is a real property of driving a
        # skid-steer vehicle on the Moon, not a modelling artefact.
        slip_factor: float = 3.0,
    ):
        self.accel = accel
        self.yaw_accel = yaw_accel
        self.max_wheel_rad_s = max_wheel_rad_s
        # Skid steering works by making the wheels scrub sideways, so a turn always loses some
        # of what is commanded to slip. Asking for a little more than the geometry implies is
        # how a real skid-steer controller compensates.
        self.slip_factor = slip_factor

        self._vx = 0.0
        self._wz = 0.0
        self._act: dict[str, int] = {}
        self._warned_strafe = False

    # -- Gait protocol -------------------------------------------------------------

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._vx = 0.0
        self._wz = 0.0
        self._act = {}
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self._act[name] = i
        for index in self._act.values():
            data.ctrl[index] = 0.0

    def apply(
        self, model: mujoco.MjModel, data: mujoco.MjData, vx: float, vy: float, wz: float, dt: float
    ) -> None:
        if not self._act:
            self.reset(model, data)

        if abs(vy) > 1e-3 and not self._warned_strafe:
            # Said once, not every step: a skid-steer rover has no sideways degree of freedom,
            # and a caller planning strafes is making a mistake worth telling them about.
            log.info("rover cannot strafe; ignoring the vy component of the command")
            self._warned_strafe = True

        # Ramp toward the command. A wheeled vehicle has real inertia and a step change in
        # commanded speed just spins the wheels.
        self._vx += float(np.clip(vx - self._vx, -self.accel * dt, self.accel * dt))
        self._wz += float(np.clip(wz - self._wz, -self.yaw_accel * dt, self.yaw_accel * dt))

        # Differential drive: the two sides differ by the turn rate times the track half-width.
        turn_component = self._wz * HALF_TRACK_M * self.slip_factor
        left_speed = (self._vx - turn_component) / WHEEL_RADIUS_M
        right_speed = (self._vx + turn_component) / WHEEL_RADIUS_M

        for name in LEFT_WHEELS:
            self._set(model, data, name, left_speed)
        for name in RIGHT_WHEELS:
            self._set(model, data, name, right_speed)

        # Point the corner wheels along the arc the rover is trying to follow.
        #
        # Without this the rover barely turns at all -- measured 0.03 rad/s against a commanded
        # 0.5, with the two unsteered middle wheels stalled and their actuators saturated. A
        # six-wheeler is long enough that skid steering alone needs more sideways traction than
        # lunar gravity provides.
        #
        # The angle is the direction of travel of a corner wheel on a circle of radius
        # vx / wz: atan(wz * halfbase / vx). Turning on the spot has no such radius, so the
        # wheels go to full lock, which is what makes a pivot turn possible at all.
        if abs(self._wz) < 1e-3:
            steer = 0.0
        elif abs(self._vx) < 0.05:
            steer = math.copysign(0.62, self._wz)
        else:
            steer = math.atan2(self._wz * HALF_BASE_M, abs(self._vx))
            steer = float(np.clip(steer, -0.62, 0.62))

        # Front wheels turn into the corner, rear wheels turn the opposite way. Crab-steering
        # the rear like this halves the turning circle, and a rover with a rocker-bogie has the
        # articulation to take it.
        #
        # Note the sign. The steer joints rotate about +z, so a POSITIVE angle points a wheel
        # to the robot's left, which drives the body clockwise -- the opposite of a positive
        # wz. Getting this backwards makes the steering fight the wheel differential rather
        # than help it: measured a commanded +0.5 rad/s coming out as -0.18, with the two
        # systems cancelling almost exactly.
        self._set(model, data, "steer_lf", -steer)
        self._set(model, data, "steer_rf", -steer)
        self._set(model, data, "steer_lr", steer)
        self._set(model, data, "steer_rr", steer)

    def _set(self, model: mujoco.MjModel, data: mujoco.MjData, name: str, speed: float) -> None:
        index = self._act.get(name)
        if index is None:
            return
        lo, hi = model.actuator_ctrlrange[index]
        data.ctrl[index] = float(np.clip(speed, max(lo, -self.max_wheel_rad_s),
                                         min(hi, self.max_wheel_rad_s)))


def rover_attitude(data: mujoco.MjData) -> tuple[float, float]:
    """Pitch and roll of the rover body, in radians.

    Worth having as its own function because on a rover it is a safety quantity, not a
    curiosity: a six-wheeled vehicle on a crater rim tips over at a roll a legged robot would
    shrug off, and a skill that drives blindly down a slope needs to be able to check.
    """
    w, x, y, z = data.qpos[3:7]
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return pitch, roll
