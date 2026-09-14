"""A planetary rover that is told where to go in a diary entry, and goes.

Thin on purpose. Everything hard here -- seeing a target, keeping track of which one it is
while the view changes, steering around what is in the way -- already lives in
`nav.explore.MaplessNavigator`, and a rover's skills are mostly a matter of naming the places
a mission cares about and reporting back in terms a person can check.

What is specific to a rover rather than a humanoid is the reporting. On a planet nobody can
see what the machine is doing, so "I arrived" is not enough: how far it drove, how steeply it
is tilted, and how far it now is from base are the numbers that tell an operator whether the
machine is in trouble. They come out of the simulation, so they are measurements.
"""

from __future__ import annotations

import logging

import numpy as np

from ..nav.explore import MaplessNavigator
from ..perception.grounding import Grounder
from ..sim.robot import Robot
from .result import SkillResult

log = logging.getLogger(__name__)

# Beyond this the rover is on ground steep enough to be worth mentioning unprompted. Real
# rovers have a tilt limit and it is one of the few numbers that will actually end a mission.
TILT_WARNING_DEG = 18.0


class RoverSkills:
    """Drive to named places, and report in numbers an operator can check."""

    actions = ("goto", "home", "survey", "attitude", "describe", "where", "report")

    def __init__(self, robot: Robot, grounder: Grounder):
        self.robot = robot
        self.grounder = grounder
        # Outdoor settings, not the indoor defaults.
        #
        # On rolling terrain the ground itself is close ahead whenever the rover crests a
        # dune, and at the indoor safety distance of 0.55 m the navigator read that as an
        # obstacle and refused to drive into it -- measured moving 0.06 m in 400 steps on
        # ground the rover crosses easily when simply told to go straight. A planetary rover
        # drives over what a robot in a corridor must go round, so the clearance it insists on
        # is smaller and the distance at which it calls itself arrived is larger.
        self.nav = MaplessNavigator(
            robot, grounder, safety_distance=0.30, arrive_distance=3.2, cruise_speed=0.5,
            lost_frames_allowed=45
        )
        # Where the rover started, which is what "home" means. Taken from the simulation
        # rather than written down, so moving the lander in the scene moves home with it.
        self.base = robot.position[:2].copy()

    def goto(self, target: str | None, where: str | None = None) -> SkillResult:
        if not target:
            return SkillResult(False, "Where should I drive to?")
        if target == "lander":
            # The rover knows where the lander is without looking; see home().
            return self.home()
        before = self.robot.position[:2].copy()
        # A bigger step budget than the indoor default of 3000.
        #
        # Distances outdoors are simply larger: the beacon is 19 m off across dunes, and at
        # this rover's pace that is most of 3000 steps before any detour. Running out returns
        # "I saw it but lost sight of it", which describes a robot that was tracking the target
        # perfectly well and merely ran out of clock -- a misleading report of a good drive.
        # search_steps as well as max_steps. The head sweeps about 190 degrees, so a target
        # behind the rover is found only by turning the body, and 260 steps is not enough of
        # a turn out here: the lander sits 21 m off across a bank and was reported "not found
        # from here" while being plainly visible once the rover faced it.
        result = self.nav.goto(target, where=where, max_steps=9000, search_steps=900)
        drove = float(np.linalg.norm(self.robot.position[:2] - before))
        tilt = self._tilt_deg()
        message = result.describe()
        if result.success:
            message += f" I drove {drove:.0f} m to get here."
            if tilt > TILT_WARNING_DEG:
                # Say it without being asked. An operator who learns about a 20-degree slope
                # only when they think to ask has learned it too late.
                message += f" I am on a {tilt:.0f} degree slope."
        return SkillResult(
            result.success, message,
            {"drove_m": round(drove, 1), "tilt_deg": round(tilt, 1),
             "distance_from_base_m": round(self._from_base(), 1)},
        )

    def home(self) -> SkillResult:
        """Drive back to the lander.

        By dead reckoning, not by looking, and that is the right instrument for this one job.
        The lander is the one place on the planet whose position the rover knows without
        seeing it -- it started there. Searching for it by camera from down in the channel
        means catching glimpses of a bright object over a bank, losing them as the terrain
        rolls, and re-searching from the same spot: measured going nowhere at all over 9000
        steps while faithfully reporting "24.8 m away" each time.

        Every real rover does this. Visual navigation finds things it was not told about;
        odometry gets you back to where you came from.
        """
        return self._drive_to(self.base, "the lander")

    def survey(self) -> SkillResult:
        """Turn a full circle and report what is visible."""
        seen: dict[str, int] = {}
        for _ in range(36):
            for target in ("lander", "cache", "beacon"):
                count = self.nav.observe_landmarks(target)
                if count:
                    seen[target] = max(seen.get(target, 0), count)
            for _ in range(20):
                self.robot.step(0.0, 0.0, 0.35)
        if not seen:
            return SkillResult(
                True, "I turned all the way round and cannot see any of the hardware from here.",
                {"visible": 0},
            )
        listed = ", ".join(f"{name} ({count})" for name, count in sorted(seen.items()))
        return SkillResult(True, f"From here I can see: {listed}.", {"visible": len(seen)})

    def attitude(self) -> SkillResult:
        tilt = self._tilt_deg()
        state = "level" if tilt < 5 else ("tilted" if tilt < TILT_WARNING_DEG else "steeply tilted")
        return SkillResult(
            True, f"I am {state} — {tilt:.0f} degrees off vertical.",
            {"tilt_deg": round(tilt, 1)},
        )

    def describe(self) -> SkillResult:
        result = self.nav.goto  # noqa: F841 - kept out of the way; describing uses perception
        visible = [t for t in ("lander", "cache", "beacon") if self.nav.observe_landmarks(t)]
        if not visible:
            return SkillResult(True, "Nothing but regolith in view.", {"visible": 0})
        return SkillResult(
            True, f"I can see the {', the '.join(visible)}.", {"visible": len(visible)}
        )

    def where(self) -> SkillResult:
        distance = self._from_base()
        return SkillResult(
            True, f"I am {distance:.0f} m from the lander.",
            {"distance_from_base_m": round(distance, 1), "tilt_deg": round(self._tilt_deg(), 1)},
        )

    # -- dispatch ------------------------------------------------------------------

    def run(
        self,
        action: str,
        argument: str | None = None,
        where: str | None = None,
        expect: int | None = None,
    ) -> SkillResult:
        handlers = {
            "goto": lambda: self.goto(argument, where),
            "home": lambda: self.home(),
            "survey": lambda: self.survey(),
            "attitude": lambda: self.attitude(),
            "describe": lambda: self.describe(),
            "where": lambda: self.where(),
            "report": lambda: SkillResult(True, argument or "Done."),
        }
        handler = handlers.get(action)
        if handler is None:
            return SkillResult(False, f"I do not know how to '{action}'.")
        log.info("skill: %s(%s)", action, argument or "")
        return handler()

    # -- measurements ---------------------------------------------------------------

    def _tilt_deg(self) -> float:
        """How far off vertical the rover is standing, in degrees."""
        # The body's own up axis against the world's. On a heightfield this is the number that
        # says whether the machine is about to be in trouble.
        up = self.robot.data.xmat[self.robot.model.body_rootid[1]].reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))

    def _from_base(self) -> float:
        return float(np.linalg.norm(self.robot.position[:2] - self.base))

    def _drive_to(self, point: np.ndarray, name: str) -> SkillResult:
        """Dead reckoning to a remembered place, for when the camera cannot find it."""
        for _ in range(8000):
            delta = point - self.robot.position[:2]
            distance = float(np.linalg.norm(delta))
            if distance < 2.0:
                return SkillResult(
                    True, f"I am back at {name}.",
                    {"distance_from_base_m": round(self._from_base(), 1)},
                )
            desired = float(np.arctan2(delta[1], delta[0]))
            error = (desired - self.robot.yaw + np.pi) % (2 * np.pi) - np.pi
            self.robot.step(0.5, 0.0, float(np.clip(error * 1.3, -0.8, 0.8)))
        return SkillResult(
            False,
            f"I could not get back to {name} — I am still "
            f"{np.linalg.norm(point - self.robot.position[:2]):.0f} m away.",
            {"distance_from_base_m": round(self._from_base(), 1)},
        )
