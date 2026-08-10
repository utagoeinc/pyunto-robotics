"""What the four-legged patrol robot can do.

Same shape as brain/skills.py and brain/laundry.py: one method per verb the planner may emit,
each returning a SkillResult carrying a sentence fit to send back over Pyunto.

The domain differs from the other two in one way that shapes everything here: the robot is
outdoors, on a site tens of metres across, and its job is a ROUTE rather than a manipulation.
So the primitives are "go to that corner", "walk the loop", "climb the steps", and the thing
that can go wrong is being unable to get somewhere rather than being unable to grasp something.

Waypoints come from the SCENE, not from a list in this file. campus.xml declares `waypoint_1`
.. `waypoint_4` as sites, and the patrol reads them, so moving the building in the XML moves
the patrol route with it. Hard-coding coordinates here would make the two drift apart the first
time either changed.
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
from .skills import SkillResult

log = logging.getLogger(__name__)

# How close counts as having reached a waypoint. Generous, because a patrol corner is a region
# to pass through rather than a spot to stand on, and tightening it just makes the robot fuss.
WAYPOINT_TOLERANCE_M = 1.2

# Clearance below which the route ahead is treated as blocked and worth steering around.
OBSTACLE_CLEARANCE_M = 1.3

# Patrol speed. Slower than the gait's top speed: a patrol that sprints has no time to look at
# anything, and the camera is the point of the errand.
PATROL_SPEED = 0.55

# Speed on the stairs. Deliberately half of patrol speed -- a step taken fast enough that the
# swing foot arrives before the body has settled catches the nosing.
STAIR_SPEED = 0.28


@dataclass
class PatrolSkills:
    """The patrol robot's action repertoire."""

    robot: Robot
    grounder: Grounder
    _home: np.ndarray = field(init=False)
    _home_heading: float = field(init=False)
    # Anything noticed during a lap, so a report can say what was seen rather than just that
    # the lap finished.
    _sightings: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._home = self.robot.position[:2].copy()
        self._home_heading = self.robot.yaw

    # -- scene lookups ------------------------------------------------------------

    def waypoints(self) -> list[tuple[str, np.ndarray]]:
        """The patrol route, read from the scene in order.

        Returns (name, position) pairs for every site called `waypoint_<n>`, sorted by n. The
        route lives in the XML so that changing the building changes the patrol; a list of
        coordinates in Python would be a second source of truth and would go stale.
        """
        found: list[tuple[int, str, np.ndarray]] = []
        for site in range(self.robot.model.nsite):
            name = mujoco.mj_id2name(self.robot.model, mujoco.mjtObj.mjOBJ_SITE, site) or ""
            if not name.startswith("waypoint_"):
                continue
            suffix = name.rsplit("_", 1)[-1]
            if not suffix.isdigit():
                continue
            found.append((int(suffix), name, self.robot.data.site_xpos[site].copy()))
        found.sort()
        return [(name, position) for _, name, position in found]

    def _site(self, name: str) -> np.ndarray | None:
        site = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site < 0:
            return None
        return self.robot.data.site_xpos[site].copy()

    # -- movement -----------------------------------------------------------------

    def _clearance_ahead(self) -> float:
        """Metres of clear space in front, from the forward depth camera."""
        observation = self.robot.look()
        return free_space(observation.depth, self.robot.camera_fovy()).clearance_ahead()

    def _turn_to(self, heading: float, max_steps: int = 240) -> None:
        for _ in range(max_steps):
            error = (heading - self.robot.yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(error) < 0.06:
                break
            self.robot.step(0.0, 0.0, float(np.clip(error * 1.5, -1.1, 1.1)))
        self.robot.stand(0.2)

    def walk_to(
        self, point: np.ndarray, max_steps: int = 3000, tolerance: float = WAYPOINT_TOLERANCE_M
    ) -> float:
        """Walk to a point on the site, steering around whatever is in the way.

        Obstacle avoidance sits UNDERNEATH target-seeking, exactly as it does in the office
        navigator: a confident heading toward a waypoint still cannot drive the robot into a
        hedge. The difference outdoors is that there is almost always a way round, so a blocked
        route turns into a detour rather than a failure.

        Returns the distance still remaining, so a caller can tell arrival from giving up.
        """
        target = np.asarray(point, dtype=float)[:2]
        detour = 0
        detour_sign = 1.0

        for step_index in range(max_steps):
            delta = target - self.robot.position[:2]
            distance = float(np.linalg.norm(delta))
            if distance < tolerance:
                break

            desired = math.atan2(delta[1], delta[0])
            error = (desired - self.robot.yaw + math.pi) % (2 * math.pi) - math.pi

            # Look where the body is going, not every step: rendering is the expensive part of
            # this loop and the clearance does not change meaningfully in 20 ms.
            if step_index % 5 == 0:
                self._clear = self._clearance_ahead()

            if getattr(self, "_clear", 99.0) < OBSTACLE_CLEARANCE_M:
                # Something ahead. Commit to turning one way for a while rather than
                # re-deciding every step, which just oscillates in front of the obstacle.
                if detour <= 0:
                    detour = 60
                    # Turn toward whichever side has more room, and keep that choice.
                    left, right = self.robot.side_clearance()
                    detour_sign = 1.0 if left > right else -1.0
                self.robot.step(0.25, 0.0, 0.9 * detour_sign)
                detour -= 1
                continue

            detour = max(0, detour - 1)
            turn = float(np.clip(error * 1.3, -1.0, 1.0))
            self.robot.step(PATROL_SPEED * max(0.3, 1.0 - abs(turn)), 0.0, turn)

        self.robot.stand(0.2)
        return float(np.linalg.norm(target - self.robot.position[:2]))

    def goto_waypoint(self, which: str | int | None = None) -> SkillResult:
        """Walk to one numbered corner of the patrol route."""
        route = self.waypoints()
        if not route:
            return SkillResult(False, "This site has no patrol waypoints marked.")

        index = _waypoint_index(which, len(route))
        if index is None:
            return SkillResult(
                False, f"I do not know which waypoint '{which}' is; there are {len(route)}."
            )

        name, position = route[index]
        remaining = self.walk_to(position)
        if remaining < WAYPOINT_TOLERANCE_M * 1.5:
            return SkillResult(
                True,
                f"I am at waypoint {index + 1}.",
                {"waypoint": name, "remaining": remaining},
            )
        return SkillResult(
            False,
            f"I could not get to waypoint {index + 1} ({remaining:.1f} m short).",
            {"waypoint": name, "remaining": remaining},
        )

    def patrol_loop(self, laps: int = 1) -> SkillResult:
        """Walk the whole route once round the building, reporting what was seen.

        This is the errand the robot exists for. It visits every waypoint in order and comes
        back to the first, looking around at each corner -- a lap that never looks at anything
        is just a walk.
        """
        route = self.waypoints()
        if not route:
            return SkillResult(False, "This site has no patrol waypoints marked.")

        self._sightings = {}
        reached = 0
        missed: list[int] = []

        for lap in range(max(1, laps)):
            for index, (name, position) in enumerate(route):
                # Waypoint positions are re-read each visit rather than cached: on a
                # heightfield the site's world z depends on the terrain under it.
                remaining = self.walk_to(position)
                if remaining < WAYPOINT_TOLERANCE_M * 1.5:
                    reached += 1
                    self._note_what_is_visible()
                else:
                    missed.append(index + 1)
                    log.info("waypoint %d not reached (%.1f m short)", index + 1, remaining)

        total = len(route) * max(1, laps)
        seen = ", ".join(sorted(self._sightings, key=lambda k: -self._sightings[k]))
        if reached == total:
            message = f"I walked the whole patrol route ({total} waypoints)."
            if seen:
                message += f" Along the way I saw: {seen}."
            return SkillResult(True, message, {"reached": reached, "seen": list(self._sightings)})

        message = f"I reached {reached} of {total} waypoints"
        if missed:
            message += f"; I could not get to {', '.join(str(m) for m in sorted(set(missed)))}"
        return SkillResult(
            False, message + ".", {"reached": reached, "missed": missed}
        )

    def climb_steps(self) -> SkillResult:
        """Walk up the steps to the building entrance.

        Slower than a patrol walk on purpose. The gait holds the trunk at a fixed height above
        whatever the lowest foot is standing on, so climbing happens by itself as the front
        feet find the tread -- but only if the swing foot has time to land before the body has
        moved on. Measured: at 0.35 m/s the robot climbs all three 0.16 m rises and ends on the
        landing, trunk rising 0.365 -> 0.831 m.
        """
        entrance = self._site("entrance_site")
        if entrance is None:
            return SkillResult(False, "I cannot find the entrance.")

        start_height = float(self.robot.position[2])

        # Approach in two stages, and the second one does NOT avoid obstacles.
        #
        # walk_to steers around anything within 1.3 m, which is correct everywhere on this site
        # except here: the steps ARE an obstacle by that measure, so the avoidance slid the
        # robot sideways along the building and it arrived at x=-2.98 for a staircase at x=0,
        # having never touched a step. So the approach stops short, squares up, and the last
        # couple of metres are driven blind.
        # The standoff sits BETWEEN the bollards and the steps, not behind the bollards. At
        # 2.2 m back it landed on the far side of the bollard line at y=-9.0, and the blind
        # run-in spent its whole budget pushing against a post: 160 steps at 0.28 m/s advanced
        # 0.86 m and the robot never reached a step.
        standoff = np.array([entrance[0], entrance[1] - 1.1])
        self.walk_to(standoff, tolerance=0.5)
        self._turn_to(math.pi / 2)  # face the building, +y

        # Close the last of the gap on a straight line, correcting only sideways drift so the
        # robot stays lined up with the middle of the flight.
        for _ in range(90):
            drift = float(entrance[0] - self.robot.position[0])
            self.robot.step(vx=STAIR_SPEED, wz=float(np.clip(drift * 0.8, -0.35, 0.35)))

        # Then keep walking into them: the climb is a consequence of walking at a step, not a
        # separate manoeuvre. The gait holds the trunk above whatever the lowest foot is on, so
        # the body rises as the front feet find each tread.
        for _ in range(420):
            self.robot.step(vx=STAIR_SPEED)
        self.robot.stand(0.4)

        climbed = float(self.robot.position[2]) - start_height
        if climbed > 0.25:
            return SkillResult(
                True,
                f"I climbed the steps to the entrance (up {climbed:.2f} m).",
                {"climbed": climbed},
            )
        return SkillResult(
            False,
            f"I could not get up the steps (only {climbed:.2f} m higher).",
            {"climbed": climbed},
        )

    def return_home(self) -> SkillResult:
        """Walk back to where the robot was standing when it was given the task."""
        remaining = self.walk_to(self._home, tolerance=0.9)
        self._turn_to(self._home_heading)
        if remaining < 1.4:
            return SkillResult(True, "I am back where I started.")
        return SkillResult(
            False, f"I could not get back to where I started ({remaining:.1f} m short)."
        )

    def look_around(self, degrees: float = 360.0) -> SkillResult:
        """Turn on the spot, reporting what came into view."""
        seen: dict[str, int] = {}
        turn_rate = 0.7
        steps = int(abs(math.radians(degrees)) / (turn_rate * self.robot.control_dt))

        for index in range(steps):
            self.robot.step(0.0, 0.0, turn_rate)
            if index % 12 == 0:
                rgb = self.robot.look().rgb
                for name in ("tree", "hedge", "building", "entrance", "bollard"):
                    if self.grounder.find(rgb, name):
                        seen[name] = seen.get(name, 0) + 1
        self.robot.stand(0.3)

        if not seen:
            return SkillResult(True, "I looked around but did not recognise anything.")
        items = ", ".join(sorted(seen, key=lambda k: -seen[k]))
        return SkillResult(True, f"Looking around I can see: {items}.", {"seen": list(seen)})

    def _note_what_is_visible(self) -> None:
        """Fold whatever is in the current frame into the lap's running tally."""
        rgb = self.robot.look().rgb
        for name in ("tree", "hedge", "entrance", "bollard", "building"):
            if self.grounder.find(rgb, name):
                self._sightings[name] = self._sightings.get(name, 0) + 1

    # -- reporting ----------------------------------------------------------------

    def describe_view(self) -> SkillResult:
        """Say what is in front of the robot right now."""
        observation = self.robot.look()
        found = [
            name
            for name in ("tree", "hedge", "entrance", "bollard", "building")
            if self.grounder.find(observation.rgb, name)
        ]
        ahead = free_space(observation.depth, self.robot.camera_fovy()).clearance_ahead()
        if not found:
            return SkillResult(True, f"Nothing I recognise ahead. Clear for {ahead:.1f} m.")
        return SkillResult(
            True,
            f"I can see: {', '.join(found)}. Clear space ahead: {ahead:.1f} m.",
            {"objects": found, "clearance": ahead},
        )

    def report_position(self) -> SkillResult:
        """Where the robot is, relative to the building it is patrolling."""
        x, y = float(self.robot.position[0]), float(self.robot.position[1])

        route = self.waypoints()
        nearest, distance = None, float("inf")
        for index, (_, position) in enumerate(route):
            offset = float(np.linalg.norm(position[:2] - np.array([x, y])))
            if offset < distance:
                nearest, distance = index + 1, offset

        # The building sits at the origin, so the side is just the dominant axis.
        side = (
            ("north" if y > 0 else "south")
            if abs(y) > abs(x)
            else ("east" if x > 0 else "west")
        )
        where = f"on the {side} side of the building"
        if nearest is not None and distance < 3.0:
            where += f", at waypoint {nearest}"
        elif nearest is not None:
            where += f", {distance:.0f} m from waypoint {nearest}"

        return SkillResult(
            True,
            f"I am {where}, {float(self.robot.position[2]):.2f} m up.",
            {"x": x, "y": y, "side": side, "nearest_waypoint": nearest},
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
            "patrol": lambda: self.patrol_loop(_laps(argument)),
            "goto": lambda: self.goto_waypoint(argument or where),
            "climb": lambda: self.climb_steps(),
            "look_around": lambda: self.look_around(),
            "describe": lambda: self.describe_view(),
            "where": lambda: self.report_position(),
            "home": lambda: self.return_home(),
            "report": lambda: SkillResult(True, argument or "Done."),
        }
        handler = handlers.get(action)
        if handler is None:
            return SkillResult(False, f"I do not know how to '{action}'.")
        log.info("skill: %s(%s)", action, argument or "")
        return handler()


# Words for waypoints, in the forms a person writes them.
_WAYPOINT_WORDS: dict[str, int] = {
    "1": 0, "one": 0, "first": 0, "一": 0, "1つ目": 0, "最初": 0,
    "2": 1, "two": 1, "second": 1, "二": 1, "2つ目": 1,
    "3": 2, "three": 2, "third": 2, "三": 2, "3つ目": 2,
    "4": 3, "four": 3, "fourth": 3, "四": 3, "4つ目": 3,
}


def _waypoint_index(which: str | int | None, count: int) -> int | None:
    """Turn "the second corner" or 2 into a zero-based index."""
    if which is None:
        return 0
    if isinstance(which, int):
        return which - 1 if 0 < which <= count else None
    text = str(which).lower().strip()
    # Longest match first, so "1つ目" beats "1".
    for word in sorted(_WAYPOINT_WORDS, key=len, reverse=True):
        if word in text:
            index = _WAYPOINT_WORDS[word]
            return index if index < count else None
    return None


def _laps(argument: str | None) -> int:
    """How many laps an instruction asked for. Defaults to one."""
    if not argument:
        return 1
    digits = "".join(c for c in str(argument) if c.isdigit())
    try:
        return max(1, min(5, int(digits))) if digits else 1
    except ValueError:
        return 1
