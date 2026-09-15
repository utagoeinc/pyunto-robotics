"""Going out, finding sunlight, charging, coming home, and turning the lights on.

This is the errand Utagoe cares most about: a robot that fetches its own energy. Everything
here is written so that the robot has to actually do it rather than appear to.

The temptation in a demonstration like this is to script the route -- drive to (20, 0), wait,
drive back -- and it would look identical on screen. It would also be worthless, because the
robot would not be finding anything. So `find_sun` searches by measuring what the panel is
receiving as the robot moves, and stops when the measurement says it is in sunlight. If
somebody moves the park, the robot still finds the sun; if somebody parks it under the
carport roof, it correctly reports that it cannot charge there.

The payoff is deliberately physical. Energy that stays a number in a report has not
demonstrated anything; energy that turns the house lights on has.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import mujoco
import numpy as np

from ..nav.explore import MaplessNavigator
from ..perception.grounding import Grounder
from ..sim.robot import Robot
from ..sim.solar import LIGHTS_DRAW_W, SolarPanel
from .result import SkillResult

log = logging.getLogger(__name__)

# How long one charging session runs, in simulated seconds.
#
# 45 minutes of simulated sun, which the robot collects in about a second of wall clock
# because charging advances simulated time in slices rather than in real time. Four minutes
# was the first choice and it produced a truthful but dismal result -- 1.7 Wh, a 1% battery,
# "about 6 minutes of light" -- which reads as a robot that went out and achieved nothing.
# Three quarters of an hour in good sun is both a realistic errand and enough to light the
# house for a couple of hours, so the number at the end is worth the trip.
CHARGE_SECONDS = 2700.0

# How far the robot will wander east looking for light before giving up. The park is 20 m
# away; 34 m lets it overshoot and still be honest about failing rather than driving forever.
SEARCH_LIMIT_M = 34.0

# What it costs to run the house lights for an hour, used to express the collected energy in
# terms a person cares about ("that is 40 minutes of light") rather than watt-hours.
LIGHTS_W = LIGHTS_DRAW_W


class SolarErrandSkills:
    """The skills for a robot that goes out and fetches power."""

    actions = (
        "fetch_power", "find_sun", "charge", "power_lights",
        "battery", "goto", "home", "describe", "where", "report",
    )

    def __init__(self, robot: Robot, grounder: Grounder):
        self.robot = robot
        self.grounder = grounder
        self.nav = MaplessNavigator(robot, grounder)
        self.panel = SolarPanel(robot)
        # Where the errand began, so "home" means the place the robot actually started rather
        # than a coordinate written down here.
        self.home_position = robot.position[:2].copy()
        self._lights_on = False

    # -- the whole errand ----------------------------------------------------------

    def fetch_power(self) -> SkillResult:
        """Leave home, find sunlight, charge, come back, and light the house.

        One skill rather than four, because "go and get power" is one intention. Splitting it
        across a planned sequence would make the robot stop and await instructions in the
        middle of an errand it was already told to complete.
        """
        found = self.find_sun()
        if not found.ok:
            return found

        charged = self.charge()
        if not charged.ok:
            return charged

        returned = self.go_home()
        if not returned.ok:
            # The energy is real even if the robot is not home yet, so say both.
            return SkillResult(
                False,
                f"{charged.message} {returned.message}",
                {**charged.data, **returned.data},
            )

        lit = self.power_lights()
        return SkillResult(
            lit.ok,
            f"{charged.message} {returned.message} {lit.message}",
            {**charged.data, **returned.data, **lit.data},
        )

    # -- the parts -----------------------------------------------------------------

    def find_sun(self) -> SkillResult:
        """Drive east until the panel is actually in sunlight.

        The test is the panel's own reading, not arrival at a place. That is the difference
        between a robot that finds the sun and a robot that drives to a coordinate where the
        sun happened to be when the scene was written.
        """
        if self.panel.read().in_sun:
            return SkillResult(
                True, "I am already in sunlight.",
                {"travelled_m": 0.0, "irradiance_w_m2": round(self.panel.read().irradiance_w_m2)},
            )

        self.panel.level_panel()
        start = self.robot.position[:2].copy()
        best = 0.0
        steps = 0

        while steps < 4000:
            reading = self.panel.read()
            if reading.irradiance_w_m2 > best:
                best = reading.irradiance_w_m2
            if reading.in_sun:
                travelled = float(np.linalg.norm(self.robot.position[:2] - start))
                return SkillResult(
                    True,
                    f"I found sunlight {travelled:.0f} m from where I started.",
                    {"travelled_m": round(travelled, 1),
                     "irradiance_w_m2": round(reading.irradiance_w_m2)},
                )

            travelled = float(np.linalg.norm(self.robot.position[:2] - start))
            if travelled > SEARCH_LIMIT_M:
                break

            # Head east, where the ground is open. Steering is a gentle correction toward the
            # +x axis rather than a planned path: the robot is looking for light, not a place.
            heading_error = (0.0 - self.robot.yaw + np.pi) % (2 * np.pi) - np.pi
            self.robot.step(0.55, 0.0, float(np.clip(heading_error * 1.2, -0.8, 0.8)))
            steps += 1

        travelled = float(np.linalg.norm(self.robot.position[:2] - start))
        return SkillResult(
            False,
            f"I went {travelled:.0f} m and could not find anywhere the sun reaches. "
            f"The best I measured was {best:.0f} W/m².",
            {"travelled_m": round(travelled, 1), "best_irradiance_w_m2": round(best)},
        )

    def charge(self) -> SkillResult:
        """Aim the panel and collect power where the robot is standing."""
        reading = self.panel.read()
        if not reading.in_sun:
            # Refuse rather than sit in the shade producing nothing. A robot that reports
            # "charged for four minutes" after collecting 0.1 Wh has told the truth in a way
            # that misleads.
            return SkillResult(
                False,
                f"I am in shade here — only {reading.irradiance_w_m2:.0f} W/m² reaches the "
                f"panel. I need to find sunlight before charging.",
                {"irradiance_w_m2": round(reading.irradiance_w_m2)},
            )

        tilt = self.panel.aim_at_sun()
        for _ in range(150):
            self.robot.step()

        before = self.panel.charge_wh
        aimed = self.panel.read()
        # Collect in one-minute slices so the reading is re-measured as the simulation runs,
        # rather than extrapolating a single instant across the whole session.
        parked = self.robot.position[:2].copy()
        for _ in range(int(CHARGE_SECONDS // 60)):
            self.panel.collect(60.0)
            for _ in range(20):
                # Commanded to a standstill, not merely left alone. A bare step() lets the
                # wheels coast, and over the 900 steps this session takes the robot wandered
                # far enough that the journey home then failed -- it had quietly charged
                # somewhere other than where it stopped.
                self.robot.step(0.0, 0.0, 0.0)
        drift = float(np.linalg.norm(self.robot.position[:2] - parked))
        if drift > 1.0:
            log.warning("robot drifted %.1f m while charging", drift)

        gained = self.panel.charge_wh - before
        minutes = CHARGE_SECONDS / 60.0
        return SkillResult(
            True,
            f"I charged for {minutes:.0f} minutes in {aimed.irradiance_w_m2:.0f} W/m² of sun "
            f"and collected {gained:.1f} Wh. The battery is at {self.panel.percent:.0f}%.",
            {"collected_wh": round(gained, 2),
             "battery_percent": round(self.panel.percent),
             "irradiance_w_m2": round(aimed.irradiance_w_m2),
             "panel_tilt_deg": round(np.degrees(tilt))},
        )

    def go_home(self) -> SkillResult:
        """Drive back to where the errand started."""
        self.panel.level_panel()
        start = self.robot.position[:2].copy()

        # Steering straight at home is not enough: the hedges lining the street sit between
        # the park and the carport, and a robot that only corrects its heading drives into one
        # and grinds there. Measured: it stopped 21.6 m short, pinned against a hedge, with
        # the heading controller still faithfully asking for west.
        stuck_for = 0
        detour_side = 0.0

        # Go back the way the street runs, not in a straight line to the door.
        #
        # The park is wide and the robot can finish charging anywhere in it, including well
        # north of the road. Steering straight at home from there cuts the corner across the
        # hedges and the robot grinds along them -- measured stopping 19 m short even after
        # the hedge was shortened. A person would walk out to the road first and follow it,
        # so the return has two legs: reach the road's centre line, then run along it home.
        # The waypoint sits on the road beside where the robot is now, not at the far end of
        # it. Aiming at the road's western end gave almost the same heading as aiming at the
        # house, so the robot cut the corner into the hedges exactly as before and stopped
        # 19 m short. Going sideways onto the road first is the whole point of the detour.
        via = np.array([self.robot.position[0], 0.0])
        leg = 0  # 0 = out to the road, 1 = along it

        # 12000 steps, not 6000. At this robot's cruise the return from the park is about
        # 6000 on its own, and the first budget ran out mid-journey and reported failure for a
        # trip that was simply still in progress -- the worst kind of wrong answer, because the
        # robot was doing everything right.
        for _ in range(12000):
            # Follow the road home rather than aiming at the house.
            #
            # The road runs along y=0, and the hedges sit at y=+-3.4. Steering at the house
            # from anywhere off the centre line puts the robot on a diagonal that meets a
            # hedge side-on -- which is what left it stuck at (5.5, 3.0) however the budget or
            # the waypoint was adjusted. So while it is still east of home, the target is a
            # point on the centre line ahead of it; only once it is level with the house does
            # it turn off toward the carport.
            position = self.robot.position[:2]
            if position[0] > self.home_position[0] + 1.5:
                waypoint = np.array([max(position[0] - 4.0, self.home_position[0]), 0.0])
            else:
                waypoint = self.home_position
            delta = waypoint - position
            distance = float(np.linalg.norm(self.home_position - self.robot.position[:2]))
            if distance < 1.5:
                travelled = float(np.linalg.norm(self.robot.position[:2] - start))
                return SkillResult(
                    True, f"I am home, {travelled:.0f} m back.",
                    {"returned_m": round(travelled, 1)},
                )

            # Steer back to the road's centre line, always.
            #
            # The corridor home is the road; the hedges either side of it are what the robot
            # keeps getting caught on. Correcting only the heading toward a waypoint let a
            # detour push it onto the verge, where it then ran along a hedge, backed off, hit
            # it again, and repeated -- traced sitting at (-1.0, -3.8) for 6000 steps doing
            # exactly that. Folding the lateral error into the steering means the robot is
            # always being pulled back toward the middle, so a nudge off line corrects itself
            # instead of becoming the new course.
            lateral_pull = float(np.clip(-position[1] * 0.35, -0.5, 0.5))

            touching = self.robot.wall_contact_side()
            if touching is not None:
                # Back off and commit to going around the side away from the contact. Choosing
                # a side once and holding it matters: alternating on every frame leaves the
                # robot rocking against the obstacle instead of clearing it.
                if detour_side == 0.0:
                    detour_side = -float(touching)
                self.robot.step(-0.4, 0.0, detour_side * 0.8)
                stuck_for = 0
                continue

            if detour_side != 0.0:
                # Clear of the obstacle. Drive on briefly before re-aiming, so the robot gets
                # past it rather than turning straight back into it -- but briefly really does
                # mean briefly. At 240 steps the detour outlasted the obstacle by a wide
                # margin and steered the robot 5 m off the road to (3.5, -4.5), where it then
                # sat with nothing touching it at all: not blocked, just driving the wrong way
                # under its own instructions. 60 steps is about a metre, which is enough to
                # clear a hedge end and little enough to be a nudge rather than a route.
                stuck_for += 1
                self.robot.step(0.5, 0.0, detour_side * 0.25 + lateral_pull * 0.5)
                if stuck_for > 60:
                    detour_side = 0.0
                    stuck_for = 0
                continue

            desired = float(np.arctan2(delta[1], delta[0]))
            error = (desired - self.robot.yaw + np.pi) % (2 * np.pi) - np.pi
            self.robot.step(0.55, 0.0, float(np.clip(error * 1.4 + lateral_pull, -0.9, 0.9)))

        distance = float(np.linalg.norm(self.home_position - self.robot.position[:2]))
        return SkillResult(
            False, f"I could not get all the way home — I am still {distance:.0f} m away.",
            {"distance_home_m": round(distance, 1)},
        )

    def power_lights(self) -> SkillResult:
        """Put the stored charge into the house lights.

        This is the point of the errand. Until the lights come on, the robot has collected a
        number; afterwards it has done something.
        """
        if self.panel.charge_wh <= 0.5:
            return SkillResult(
                False,
                "I do not have enough charge to light the house.",
                {"battery_percent": round(self.panel.percent)},
            )

        hours = self.panel.lighting_hours(LIGHTS_W)
        self._set_lights(True)
        return SkillResult(
            True,
            f"The house lights are on. There is {self.panel.charge_wh:.1f} Wh stored — "
            f"about {hours * 60:.0f} minutes of light.",
            {"battery_wh": round(self.panel.charge_wh, 1),
             "lighting_minutes": round(hours * 60),
             "lights_on": True},
        )

    def battery(self) -> SkillResult:
        """State of charge, and when it would be full at the present rate.

        "How long until it is full?" is the question people actually ask, and a percentage
        alone does not answer it. The estimate is the honest arithmetic -- what is missing
        divided by what the panel is making right now -- which means it is only true while
        the robot stays where it is. In shade it is making less than it spends, and the right
        answer is that it will never fill here, not a number.
        """
        reading = self.panel.read()
        power = reading.power_w
        percent = self.panel.percent
        missing_wh = max(self.panel.capacity_wh - self.panel.charge_wh, 0.0)

        message = (
            f"The battery is at {percent:.0f}% ({self.panel.charge_wh:.1f} of "
            f"{self.panel.capacity_wh:.0f} Wh). The panel is making {max(power, 0):.0f} W."
        )
        data = {
            "battery_percent": round(percent),
            "battery_wh": round(self.panel.charge_wh, 1),
            "capacity_wh": round(self.panel.capacity_wh),
            "power_w": round(max(power, 0)),
        }

        if missing_wh <= 0.1:
            message += " It is full."
        elif power <= 0.5:
            # Idle draw exceeds what the panel collects: it is going down, not up. Saying
            # "3 hours" here would be arithmetic on a number with the wrong sign.
            message += (
                " Here in the shade it is not charging at all — I would need to find"
                " sunlight before it fills."
            )
            data["charging"] = False
        else:
            hours = missing_wh / power
            eta = datetime.now() + timedelta(hours=hours)
            if hours < 1:
                when = f"{hours * 60:.0f} minutes"
            else:
                when = f"{hours:.1f} hours"
            message += f" At this rate it would be full in about {when}, around {eta:%H:%M}."
            data["charging"] = True
            data["hours_to_full"] = round(hours, 2)
            data["full_at"] = eta.strftime("%H:%M")

        return SkillResult(True, message, data)

    def describe(self) -> SkillResult:
        reading = self.panel.read()
        where = "in sunlight" if reading.in_sun else "in shade"
        return SkillResult(
            True,
            f"I am {where}. The panel is receiving {reading.irradiance_w_m2:.0f} W/m².",
            {"in_sun": reading.in_sun, "irradiance_w_m2": round(reading.irradiance_w_m2)},
        )

    def where(self) -> SkillResult:
        position = self.robot.position[:2]
        from_home = float(np.linalg.norm(position - self.home_position))
        return SkillResult(
            True, f"I am {from_home:.0f} m from home.",
            {"distance_home_m": round(from_home, 1)},
        )

    def goto(self, target: str | None) -> SkillResult:
        """Drive to a named place. `home` is handled here; anything else goes to the navigator."""
        if target in (None, "home"):
            return self.go_home()
        result = self.nav.goto(target)
        return SkillResult(result.success, result.describe())

    # -- dispatch ------------------------------------------------------------------

    def run(
        self,
        action: str,
        argument: str | None = None,
        where: str | None = None,
        expect: int | None = None,
    ) -> SkillResult:
        handlers = {
            "fetch_power": lambda: self.fetch_power(),
            "find_sun": lambda: self.find_sun(),
            "charge": lambda: self.charge(),
            "power_lights": lambda: self.power_lights(),
            "battery": lambda: self.battery(),
            "goto": lambda: self.goto(argument),
            "home": lambda: self.go_home(),
            "describe": lambda: self.describe(),
            "where": lambda: self.where(),
            "report": lambda: SkillResult(True, argument or "Done."),
        }
        handler = handlers.get(action)
        if handler is None:
            return SkillResult(False, f"I do not know how to '{action}'.")
        log.info("skill: %s(%s)", action, argument or "")
        return handler()

    # -- the lights ----------------------------------------------------------------

    def _set_lights(self, on: bool) -> None:
        """Switch the house lamps by swapping their material.

        Materials rather than a light source: MuJoCo's lights are fixed at compile time, and
        an emissive material is both cheaper and, for a window seen from outside, what the
        effect actually looks like.
        """
        model = self.robot.model
        target = "lamp_on" if on else "lamp_off"
        material = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, target)
        if material < 0:
            log.warning("this scene has no '%s' material; lights cannot be switched", target)
            return
        for name in ("lamp_a", "lamp_b", "lamp_c"):
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom >= 0:
                model.geom_matid[geom] = material
        self._lights_on = on
