"""What the lunar rover can do.

Same shape as the other skill modules: one method per verb the planner may emit, each returning
a SkillResult with a sentence fit to send back over Pyunto.

Three things about this domain shape the code, and none of them apply to the other robots.

**Turning is expensive and imprecise.** A six-wheeled rover turns by scrubbing its wheels
sideways, and the force available to do that is proportional to weight -- which at 1.62 m/s^2
is a sixth of Earth's. Measured, a commanded 0.5 rad/s delivers about 0.24. So nothing here
assumes a commanded turn rate is achieved; heading is closed-loop, always.

**The terrain can stop the rover, and can tip it over.** Crater rims are the interesting
hazard: shallow enough to look crossable, steep enough to strand a vehicle sideways. So
`drive_to` watches roll and abandons a heading that is tilting the rover rather than pressing
on, and every skill reports the attitude it ended at.

**Half the world is in shadow.** The Sun is a few degrees above the horizon and there is no
atmosphere to scatter light into a crater floor, so a camera genuinely cannot see into one.
That is a real constraint on a camera-driven rover at the pole, and `describe_view` reports
darkness as a finding rather than as an absence of objects.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..perception.depth import free_space
from ..perception.grounding import Grounder
from ..sim.robot import Robot
from ..sim.wheel_drive import rover_attitude
from .skills import SkillResult

log = logging.getLogger(__name__)

# How close counts as having arrived. Generous: these are places to get to, not spots to park
# on, and a rover that fusses over the last half metre wastes power for nothing.
#
# 3.0 rather than 2.0 because the rover is 1.6 m long and its targets are things it works
# beside, not points it stands on. At 2.0 a drive that ended 2 m from the beacon -- close
# enough to photograph it -- was reported as a failure.
ARRIVE_TOLERANCE_M = 3.0

# Cruise speed. Real polar rovers crawl; this is quick by comparison but slow enough that the
# suspension has time to work over a crater rim.
CRUISE_SPEED = 0.55

# Roll beyond which the rover is treated as being in trouble. A six-wheeler on a crater rim
# tips at an angle a legged robot would shrug off, and the recovery is to back off, not to
# press on and hope.
ROLL_LIMIT_RAD = math.radians(22.0)

# Clearance below which the way ahead counts as blocked -- a boulder, or the far wall of a
# crater the rover has driven into.
#
# 1.2 m, not the 2.2 a first guess suggests. The mast camera sits 0.9 m up on rolling terrain
# and therefore always has GROUND in the lower half of its view: on open, obstacle-free plain
# the forward clearance reads about 2.0 m, so a 2.2 m threshold declared the rover permanently
# blocked and it spent every step turning aside from the surface it was driving on. What the
# threshold has to be below is the reading from empty ground, not from a real obstacle.
OBSTACLE_CLEARANCE_M = 1.2

# Mean brightness below which a view counts as being in shadow. Regolith in sunlight renders
# around 90-110; a permanently shadowed floor comes out under 25.
SHADOW_LEVEL = 32.0


@dataclass
class LunarSkills:
    """The rover's action repertoire."""

    robot: Robot
    grounder: Grounder
    _home: np.ndarray = field(init=False)
    _home_heading: float = field(init=False)

    def __post_init__(self) -> None:
        self._home = self.robot.position[:2].copy()
        self._home_heading = self.robot.yaw

    # -- scene lookups ------------------------------------------------------------

    def targets(self) -> dict[str, np.ndarray]:
        """Every named target on the surface, read from the scene.

        Sites called `target_<name>`, so the mission is defined by the XML rather than by a
        list here -- move the ice deposit in the scene and "go to the ice" follows it.
        """
        found: dict[str, np.ndarray] = {}
        for site in range(self.robot.model.nsite):
            name = mujoco.mj_id2name(self.robot.model, mujoco.mjtObj.mjOBJ_SITE, site) or ""
            if name.startswith("target_"):
                found[name[len("target_"):]] = self.robot.data.site_xpos[site].copy()
        return found

    # -- driving ------------------------------------------------------------------

    def _clearance_ahead(self) -> float:
        observation = self.robot.look()
        return free_space(observation.depth, self.robot.camera_fovy()).clearance_ahead()

    def _turn_to(self, heading: float, max_steps: int = 600) -> float:
        """Turn to a world heading. Returns the error left over.

        Closed-loop on heading rather than open-loop on time, because a commanded turn rate is
        not what the rover achieves: 0.5 rad/s asked for comes out around 0.24 on regolith, and
        that ratio changes with slope and with how much weight is on which wheels.
        """
        for _ in range(max_steps):
            error = (heading - self.robot.yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(error) < 0.08:
                break
            # Full commanded rate until close, since only about half of it arrives anyway.
            self.robot.step(0.0, 0.0, float(np.clip(error * 2.0, -0.8, 0.8)))
        self.robot.stand(0.2)
        return abs((heading - self.robot.yaw + math.pi) % (2 * math.pi) - math.pi)

    def drive_to(
        self, point: np.ndarray, max_steps: int = 6000, tolerance: float = ARRIVE_TOLERANCE_M
    ) -> tuple[float, str]:
        """Drive to a point on the surface. Returns (distance remaining, why it stopped).

        Watches three things while it drives, in order of how badly they end:

          * ROLL. Past 22 degrees the rover is on the way to lying on its side, and the answer
            is to stop and back off, not to keep driving and hope the far wheel comes down.
          * CLEARANCE. A boulder or a crater's far wall means turning aside.
          * PROGRESS. A rover whose wheels are turning while it goes nowhere is bogged, which
            on regolith is a real outcome and worth reporting as itself rather than as a
            timeout.
        """
        target = np.asarray(point, dtype=float)[:2]
        detour = 0
        detour_sign = 1.0
        last_position = self.robot.position[:2].copy()
        stalled = 0
        clearance = 99.0

        for index in range(max_steps):
            remaining = float(np.linalg.norm(target - self.robot.position[:2]))
            if remaining < tolerance:
                return remaining, "arrived"

            _, roll = rover_attitude(self.robot.data)
            if abs(roll) > ROLL_LIMIT_RAD:
                # Back straight out of trouble, then let the caller decide what to do.
                for _ in range(60):
                    self.robot.step(vx=-0.4)
                self.robot.stand(0.3)
                return (
                    float(np.linalg.norm(target - self.robot.position[:2])),
                    "tilted",
                )

            if index % 6 == 0:
                clearance = self._clearance_ahead()

            # Progress check, every couple of seconds of simulated time.
            if index % 100 == 99:
                moved = float(np.linalg.norm(self.robot.position[:2] - last_position))
                last_position = self.robot.position[:2].copy()
                # Not making progress only counts as bogged when there is still somewhere to
                # go. Close to the target the rover is slowing down on purpose, and calling
                # that "my wheels are turning but I am not moving" is both wrong and alarming.
                if moved < 0.15 and remaining > tolerance * 1.5:
                    stalled += 1
                    if stalled >= 3:
                        return (
                            float(np.linalg.norm(target - self.robot.position[:2])),
                            "bogged",
                        )
                else:
                    stalled = 0

            delta = target - self.robot.position[:2]
            desired = math.atan2(delta[1], delta[0])
            error = (desired - self.robot.yaw + math.pi) % (2 * math.pi) - math.pi

            if clearance < OBSTACLE_CLEARANCE_M:
                # Commit to one side for a while: re-deciding every step just weaves.
                if detour <= 0:
                    detour = 90
                    left, right = self.robot.side_clearance()
                    detour_sign = 1.0 if left > right else -1.0
                self.robot.step(0.25, 0.0, 0.7 * detour_sign)
                detour -= 1
                continue

            detour = max(0, detour - 1)
            turn = float(np.clip(error * 1.6, -0.8, 0.8))
            self.robot.step(CRUISE_SPEED * max(0.35, 1.0 - abs(turn)), 0.0, turn)

        return float(np.linalg.norm(target - self.robot.position[:2])), "out of range"

    def goto_target(self, name: str | None = None) -> SkillResult:
        """Drive to one of the named places on the surface."""
        places = self.targets()
        if not places:
            return SkillResult(False, "There is nothing marked on this surface to drive to.")

        key = _canonical_target(name, places)
        if key is None:
            known = ", ".join(sorted(places))
            return SkillResult(
                False, f"I do not know where '{name}' is. I know: {known}."
            )

        remaining, reason = self.drive_to(places[key])

        # A tilt or a bog is a reason to try a different line, not to give up. The rover backed
        # itself out when it hit trouble, so turning aside and setting off again is a route
        # around the obstacle -- which is what an operator would tell it to do, and what makes
        # the difference between "the crater rim is in the way" and "I cannot get there".
        for attempt in range(2):
            if reason == "arrived":
                break
            log.info("re-routing to %s after %s (attempt %d)", key, reason, attempt + 1)
            delta = places[key][:2] - self.robot.position[:2]
            direct = math.atan2(delta[1], delta[0])
            # Strike out well off the direct line, alternating sides, then head in again.
            self._turn_to(direct + (0.9 if attempt % 2 == 0 else -0.9))
            for _ in range(220):
                self.robot.step(vx=CRUISE_SPEED)
            remaining, reason = self.drive_to(places[key])

        _, roll = rover_attitude(self.robot.data)
        data = {
            "target": key,
            "remaining": remaining,
            "reason": reason,
            "roll_degrees": math.degrees(roll),
        }
        if reason == "arrived":
            return SkillResult(True, f"I am at the {key}.", data)
        if reason == "tilted":
            return SkillResult(
                False,
                f"I had to back off on the way to the {key} -- the ground tilted me "
                f"{abs(math.degrees(roll)):.0f} degrees. I am {remaining:.0f} m short.",
                data,
            )
        if reason == "bogged":
            return SkillResult(
                False,
                f"My wheels are turning but I am not moving; I am {remaining:.0f} m from "
                f"the {key}.",
                data,
            )
        return SkillResult(
            False, f"I could not reach the {key} ({remaining:.0f} m short).", data
        )

    def survey(self) -> SkillResult:
        """Turn a full circle, reporting what is visible and where the light is.

        The lighting report is the part worth having here. At the pole the Sun is a few degrees
        up, so which way the rover is facing decides whether it can see anything at all, and
        "it is dark that way" is a finding rather than a failure.
        """
        seen: dict[str, int] = {}
        dark_headings = 0
        samples = 0

        for _ in range(24):
            self._turn_to((self.robot.yaw + math.pi / 12) % (2 * math.pi))
            observation = self.robot.look()
            samples += 1
            if float(observation.rgb.mean()) < SHADOW_LEVEL:
                dark_headings += 1
            for name in ("lander", "beacon", "panel"):
                if self.grounder.find(observation.rgb, name):
                    seen[name] = seen.get(name, 0) + 1

        parts = []
        if seen:
            parts.append("I can see: " + ", ".join(sorted(seen, key=lambda k: -seen[k])))
        else:
            parts.append("I did not recognise anything")
        if dark_headings:
            parts.append(
                f"{dark_headings} of {samples} directions are in deep shadow"
            )
        return SkillResult(
            True, ". ".join(parts) + ".", {"seen": list(seen), "dark": dark_headings}
        )

    def return_home(self) -> SkillResult:
        """Drive back to where the rover started, which is beside the lander."""
        remaining, reason = self.drive_to(self._home, tolerance=2.5)
        self._turn_to(self._home_heading)
        if reason == "arrived":
            return SkillResult(True, "I am back where I started, beside the lander.")
        return SkillResult(
            False,
            f"I could not get back to the lander ({remaining:.0f} m short, {reason}).",
            {"remaining": remaining, "reason": reason},
        )

    # -- reporting ----------------------------------------------------------------

    def describe_view(self) -> SkillResult:
        """Say what is in front of the rover, including whether it can see at all."""
        observation = self.robot.look()
        brightness = float(observation.rgb.mean())
        ahead = free_space(observation.depth, self.robot.camera_fovy()).clearance_ahead()

        found = [
            name
            for name in ("lander", "beacon", "panel")
            if self.grounder.find(observation.rgb, name)
        ]

        if brightness < SHADOW_LEVEL and not found:
            return SkillResult(
                True,
                f"It is in shadow ahead and I cannot make anything out. "
                f"The ground is clear for {ahead:.0f} m.",
                {"brightness": brightness, "clearance": ahead, "shadowed": True},
            )
        if not found:
            return SkillResult(
                True,
                f"Just regolith ahead, clear for {ahead:.0f} m.",
                {"brightness": brightness, "clearance": ahead},
            )
        return SkillResult(
            True,
            f"I can see: {', '.join(found)}. Clear for {ahead:.0f} m.",
            {"objects": found, "brightness": brightness, "clearance": ahead},
        )

    def report_position(self) -> SkillResult:
        """Where the rover is, relative to the things that have names."""
        here = self.robot.position[:2]
        pitch, roll = rover_attitude(self.robot.data)

        places = self.targets()
        nearest, distance = None, float("inf")
        for name, position in places.items():
            offset = float(np.linalg.norm(position[:2] - here))
            if offset < distance:
                nearest, distance = name, offset

        where = (
            f"{distance:.0f} m from the {nearest}"
            if nearest and distance > ARRIVE_TOLERANCE_M
            else f"at the {nearest}"
            if nearest
            else "out on the surface"
        )
        return SkillResult(
            True,
            f"I am {where}, tilted {abs(math.degrees(roll)):.0f} degrees.",
            {
                "x": float(here[0]),
                "y": float(here[1]),
                "nearest": nearest,
                "distance": distance,
                "roll_degrees": math.degrees(roll),
                "pitch_degrees": math.degrees(pitch),
            },
        )

    def report_attitude(self) -> SkillResult:
        """Pitch and roll, which on a rover is a safety question rather than a curiosity."""
        pitch, roll = rover_attitude(self.robot.data)
        safe = abs(roll) < ROLL_LIMIT_RAD
        message = (
            f"I am pitched {math.degrees(pitch):+.0f} degrees and rolled "
            f"{math.degrees(roll):+.0f} degrees."
        )
        if not safe:
            message += " That is steeper than I am happy with."
        return SkillResult(
            True,
            message,
            {"pitch_degrees": math.degrees(pitch), "roll_degrees": math.degrees(roll),
             "safe": safe},
        )

    # -- dispatch -----------------------------------------------------------------

    def run(
        self,
        action: str,
        argument: str | None = None,
        where: str | None = None,
        expect: int | None = None,
    ) -> SkillResult:
        """Execute one planner-issued action."""
        handlers = {
            "goto": lambda: self.goto_target(argument or where),
            "survey": lambda: self.survey(),
            "look_around": lambda: self.survey(),
            "describe": lambda: self.describe_view(),
            "where": lambda: self.report_position(),
            "attitude": lambda: self.report_attitude(),
            "home": lambda: self.return_home(),
            "report": lambda: SkillResult(True, argument or "Done."),
        }
        handler = handlers.get(action)
        if handler is None:
            return SkillResult(False, f"I do not know how to '{action}'.")
        log.info("skill: %s(%s)", action, argument or "")
        return handler()


# What a person might call each target, mapped onto the site name in the scene.
_TARGET_WORDS: dict[str, str] = {
    "lander": "lander", "base": "lander", "home": "lander",
    "着陸船": "lander", "着陸機": "lander", "ランダー": "lander", "基地": "lander",
    "ice": "ice", "water": "ice", "deposit": "ice",
    "氷": "ice", "水": "ice", "氷床": "ice",
    "beacon": "beacon", "marker": "beacon", "mast": "beacon",
    "ビーコン": "beacon", "目印": "beacon", "標識": "beacon",
    "crater": "crater", "rim": "crater", "edge": "crater",
    "クレーター": "crater", "クレータ": "crater", "縁": "crater", "ふち": "crater",
}


def _canonical_target(description: str | None, known: dict[str, np.ndarray]) -> str | None:
    """Map free text onto one of the scene's target names, longest match first."""
    if not description:
        return None
    text = description.lower().strip()
    if text in known:
        return text
    matches = [(len(word), name) for word, name in _TARGET_WORDS.items() if word in text]
    for _, name in sorted(matches, reverse=True):
        if name in known:
            return name
    return None
