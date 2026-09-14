"""How much power the robot's panel is actually making, and what it has stored.

This is the part of the energy demonstration that has to be honest. It would be easy to make
charging a timer -- sit still for ten seconds, gain a hundred watt-hours -- and the robot
would look exactly the same on screen. But then "go and find the sun" would be theatre: the
robot could park in the carport and do just as well, and there would be nothing for it to be
right or wrong about.

So generation is computed from two things the robot can change by moving:

  * How much light actually reaches the panel, measured from the camera. Standing in shadow
    genuinely collects less, because shadow is genuinely darker.
  * How squarely the panel faces the sun, from the panel's own surface normal. Aiming it is
    worth something, which is why the panel is on a hinge.

Both come out of the simulation rather than from a script, so the robot's energy at the end
of an errand is a measurement of what it did.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import mujoco
import numpy as np

log = logging.getLogger(__name__)

# Full sunlight on a clear day. The figure everything is scaled against.
PEAK_IRRADIANCE_W_M2 = 1000.0

# Panel efficiency. 20% is what a good consumer silicon panel does; quoting 45% would make the
# demo faster and the number a lie.
PANEL_EFFICIENCY = 0.20

# What the robot spends just being awake -- computer, sensors, idling motors. It matters that
# this is not zero: it means sitting in the shade slowly LOSES charge, so a robot that fails
# to find the sun does not merely stall, it runs down.
IDLE_DRAW_W = 3.5

# Battery capacity. Small enough that a few minutes of good sun is a visible fraction of it,
# which is what lets the diary report say something meaningful about one errand.
BATTERY_CAPACITY_WH = 120.0

# What the house lights need to run for the demonstration's purposes.
LIGHTS_DRAW_W = 18.0


@dataclass
class SolarReading:
    """One measurement of the panel's situation."""

    irradiance_w_m2: float
    """Light reaching the panel, after shadow and after the angle to the sun."""

    power_w: float
    """What the panel is generating right now, net of what the robot is spending."""

    in_sun: bool
    """Whether this counts as sunlight rather than shade."""

    tilt_rad: float
    """The panel's current tilt, so a caller can report what it did to improve things."""


class SolarPanel:
    """Reads the panel's situation out of the simulation and keeps the battery's state.

    Attaches to a robot whose model contains a geom named `solar_panel`; the area and the
    surface normal are taken from that geom, so changing the panel in the XML changes the
    physics here without touching this file.
    """

    # Below this fraction of peak, the robot is in shade.
    #
    # Measured in the scene rather than guessed, because a guess put the shaded street on the
    # wrong side of the line: the carport reads 0.48 of peak, the shaded street 0.65, the
    # sunlit park 1.0. At 0.55 the street counted as sunlight, so a robot could stop halfway
    # and declare success while generating half what it should. 0.80 puts both shaded places
    # below the line and leaves the park well clear of it.
    SUN_THRESHOLD = 0.80

    def __init__(self, robot, capacity_wh: float = BATTERY_CAPACITY_WH,
                 charge_wh: float = 0.0):  # noqa: ANN001
        self.robot = robot
        self.capacity_wh = capacity_wh
        self.charge_wh = charge_wh
        self.generated_wh = 0.0
        """Everything collected this run, which is what the errand reports."""

        self._panel_geom = mujoco.mj_name2id(
            robot.model, mujoco.mjtObj.mjOBJ_GEOM, "solar_panel"
        )
        if self._panel_geom < 0:
            raise ValueError("this robot has no geom named 'solar_panel'")
        size = robot.model.geom_size[self._panel_geom]
        # MuJoCo box sizes are half-extents, so the face is (2a x 2b).
        self.area_m2 = float(4.0 * size[0] * size[1])

        self._sun_dir = self._find_sun_direction()
        self._tilt_act = self._actuator("panel_tilt")
        log.info("solar panel: %.3f m2, sun from %s", self.area_m2, np.round(self._sun_dir, 2))

    # -- reading the situation ------------------------------------------------------

    def read(self) -> SolarReading:
        """Measure what the panel is receiving right now."""
        # Brightness of the scene as the robot sees it, normalised to 0..1. This is the
        # shadow term: standing under the carport roof genuinely darkens the frame.
        frame = self.robot.look().rgb.astype(np.float32)
        # The upper half of the frame is mostly sky, which is bright everywhere and would
        # hide the shadow the robot is standing in. The ground is what carries the signal.
        ground = frame[frame.shape[0] // 2:, :, :]
        brightness = float(ground.mean()) / 255.0

        # How squarely the panel faces the sun. The panel's +z in world coordinates is its
        # normal; the cosine against the sun direction is the standard incidence term.
        normal = self.robot.data.geom_xmat[self._panel_geom].reshape(3, 3)[:, 2]
        cosine = float(np.dot(normal, -self._sun_dir))
        cosine = max(cosine, 0.0)  # a panel facing away collects nothing, not negative light

        # Scale brightness so the sunlit park reaches roughly full sun. The park measures
        # about 0.33 of 255 with the ground filling the lower frame; anchoring on that keeps
        # the model tied to the scene rather than to an arbitrary constant.
        lit_fraction = min(brightness / 0.33, 1.0)

        irradiance = PEAK_IRRADIANCE_W_M2 * lit_fraction * cosine
        generated = irradiance * self.area_m2 * PANEL_EFFICIENCY
        net = generated - IDLE_DRAW_W

        return SolarReading(
            irradiance_w_m2=irradiance,
            power_w=net,
            in_sun=lit_fraction >= self.SUN_THRESHOLD,
            tilt_rad=float(self.robot.data.qpos[self._tilt_qpos()]) if self._tilt_act else 0.0,
        )

    def collect(self, seconds: float) -> SolarReading:
        """Charge (or drain) for a stretch of simulated time, and return what happened."""
        reading = self.read()
        delta_wh = reading.power_w * seconds / 3600.0
        before = self.charge_wh
        self.charge_wh = max(0.0, min(self.capacity_wh, self.charge_wh + delta_wh))
        if delta_wh > 0:
            # Count only what actually entered the battery; a full battery collects nothing
            # more, and reporting otherwise would overstate the errand.
            self.generated_wh += self.charge_wh - before
        return reading

    # -- acting on it ---------------------------------------------------------------

    def aim_at_sun(self) -> float:
        """Tilt the panel toward the sun. Returns the angle commanded, in radians.

        The sun here is fixed, so this is a one-off calculation rather than tracking -- but it
        is computed from the sun's actual direction and the robot's actual heading, so turning
        the robot around changes the answer, as it must.
        """
        if self._tilt_act is None:
            return 0.0
        # Sun elevation, and its bearing relative to the robot's forward axis.
        elevation = math.asin(max(-1.0, min(1.0, -self._sun_dir[2])))
        sun_bearing = math.atan2(-self._sun_dir[1], -self._sun_dir[0])
        relative = (sun_bearing - self.robot.yaw + math.pi) % (2 * math.pi) - math.pi

        # The hinge tilts about the robot's y axis, so it can only lean fore and aft. Facing
        # the sun, lean back by (90 - elevation); facing away, lean the other way. Anything
        # sideways is out of this joint's reach, which is why the skill turns the body first.
        want = (math.pi / 2 - elevation) * math.cos(relative)
        lo, hi = self.robot.model.actuator_ctrlrange[self._tilt_act]
        want = float(np.clip(want, lo, hi))
        self.robot.data.ctrl[self._tilt_act] = want
        return want

    def level_panel(self) -> None:
        """Lay the panel flat, for travelling."""
        if self._tilt_act is not None:
            self.robot.data.ctrl[self._tilt_act] = 0.0

    # -- state ----------------------------------------------------------------------

    @property
    def percent(self) -> float:
        return 100.0 * self.charge_wh / self.capacity_wh if self.capacity_wh else 0.0

    def lighting_hours(self, draw_w: float = LIGHTS_DRAW_W) -> float:
        """How long the stored charge would run the house lights."""
        return self.charge_wh / draw_w if draw_w > 0 else 0.0

    def spend(self, wh: float) -> float:
        """Take energy out of the battery. Returns what was actually available."""
        taken = min(wh, self.charge_wh)
        self.charge_wh -= taken
        return taken

    # -- internals ------------------------------------------------------------------

    def _find_sun_direction(self) -> np.ndarray:
        """The scene's directional light, normalised. Falls back to straight down."""
        for i in range(self.robot.model.nlight):
            name = mujoco.mj_id2name(self.robot.model, mujoco.mjtObj.mjOBJ_LIGHT, i)
            if name == "sun":
                d = np.array(self.robot.model.light_dir[i], dtype=float)
                norm = np.linalg.norm(d)
                if norm > 1e-6:
                    return d / norm
        log.warning("no light named 'sun'; assuming it is overhead")
        return np.array([0.0, 0.0, -1.0])

    def _actuator(self, name: str) -> int | None:
        index = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        return index if index >= 0 else None

    def _tilt_qpos(self) -> int:
        joint = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_JOINT, "panel_tilt")
        return int(self.robot.model.jnt_qposadr[joint]) if joint >= 0 else 0
