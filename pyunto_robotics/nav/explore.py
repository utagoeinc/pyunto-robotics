"""Mapless navigation: go to a thing you can see, and look around when you cannot.

No SLAM, no occupancy grid, no prior map. Each control step re-reads the current camera frame
and decides what to do from that alone. The robot's memory of the world is exactly one frame
deep, which is the point: nothing to build, nothing to drift, nothing to invalidate when a
door swings open.

The loop is a small state machine:

    SEARCH   turn on the spot, sampling frames, until the target appears
    APPROACH walk toward it, steering by its bearing, re-checking every frame
    ARRIVED  close enough to act on
    LOST     target vanished mid-approach -- fall back to SEARCH
    BLOCKED  no route forward and nowhere open to turn

Obstacle avoidance is layered under the target-seeking: the depth-derived clearance can veto
or redirect the command, so a confident but wrong detection still cannot walk the robot into a
wall. That separation is what keeps a vision mistake from becoming a collision.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..perception.depth import FreeSpace, free_space, target_offset
from ..perception.grounding import Detection, Grounder
from ..sim.robot import Robot

log = logging.getLogger(__name__)


class NavState(Enum):
    SEARCH = "search"
    APPROACH = "approach"
    ARRIVED = "arrived"
    LOST = "lost"
    BLOCKED = "blocked"


@dataclass
class NavResult:
    """Outcome of a navigation attempt."""

    state: NavState
    target: str
    distance: float | None = None
    bearing: float | None = None
    steps: int = 0
    detections: list[Detection] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.state is NavState.ARRIVED

    def describe(self) -> str:
        """A sentence a human can read -- this ends up in the Pyunto reply."""
        if self.state is NavState.ARRIVED:
            return f"Reached the {self.target} ({self.distance:.1f} m away when I stopped)."
        if self.state is NavState.LOST:
            return f"I saw the {self.target} but lost sight of it before I got there."
        if self.state is NavState.BLOCKED:
            return f"I could not find a way through to the {self.target}."
        return f"I could not find a {self.target} from here."


class MaplessNavigator:
    """Drives the robot toward a described target using only what the camera sees."""

    def __init__(
        self,
        robot: Robot,
        grounder: Grounder,
        arrive_distance: float = 0.85,
        cruise_speed: float = 0.55,
        turn_gain: float = 1.4,
        safety_distance: float = 0.55,
        perception_every: int = 4,
    ):
        self.robot = robot
        self.grounder = grounder
        self.arrive_distance = arrive_distance
        self.cruise_speed = cruise_speed
        self.turn_gain = turn_gain
        self.safety_distance = safety_distance
        # Perception is cheap here (~2 ms/frame) but the vision model may not be, so the
        # cadence is configurable independently of the control rate.
        self.perception_every = perception_every

    # -- perception ---------------------------------------------------------------

    def _observe(
        self, target: str
    ) -> tuple[list[Detection], FreeSpace, float, tuple[int, int], np.ndarray]:
        """One camera frame, turned into everything the loop needs from it.

        Returns the depth image too, so callers never render a second frame just to measure a
        detection -- doing that would also risk pairing a detection with a *different* frame.
        """
        obs = self.robot.look()
        fovy = self.robot.camera_fovy()
        height, width = obs.depth.shape
        detections = self.grounder.find(obs.rgb, target)
        space = free_space(obs.depth, fovy)
        return detections, space, fovy, (width, height), obs.depth

    def _locate(
        self,
        detections: list[Detection],
        depth: np.ndarray,
        fovy: float,
        size: tuple[int, int],
        prefer_bearing: float = 0.0,
    ) -> tuple[Detection, float, float] | None:
        """Pick a detection to chase and measure it.

        Scoring on distance alone is unstable when several identical targets are equidistant:
        the office has three doors 3.8-3.9 m away, and "nearest" flip-flopped between the
        left and right one every frame, so the robot just oscillated. Bearing is part of the
        score, which both settles that and matches what "the door" usually means -- the one
        being looked at, not one 44 degrees off to the side.

        `prefer_bearing` biases the choice toward a target already being tracked, so an
        approach does not switch horses halfway.
        """
        best: tuple[Detection, float, float] | None = None
        best_score = -math.inf
        for det in detections:
            u, v = det.pixel(*size)
            offset = target_offset(depth, u, v, fovy, size)
            if offset is None:
                continue
            bearing, distance = offset
            # Nearer is better; straight ahead is better; already-tracked is better.
            score = -distance - 2.0 * abs(bearing) - 1.5 * abs(bearing - prefer_bearing)
            if score > best_score:
                best_score = score
                best = (det, bearing, distance)
        return best

    # -- steering -----------------------------------------------------------------

    def _avoid(self, space: FreeSpace, desired_turn: float) -> tuple[float, float]:
        """Blend the desired heading with what the depth image says is safe.

        Returns (speed_scale, turn). Speed is cut as obstacles close in, and if the way ahead
        is genuinely blocked the turn is overridden toward open space.
        """
        ahead = space.clearance_ahead(half_angle=0.35)

        if ahead > self.safety_distance * 2.5:
            return 1.0, desired_turn
        if ahead > self.safety_distance:
            # Slow down proportionally as the gap narrows.
            scale = (ahead - self.safety_distance) / (self.safety_distance * 1.5)
            return max(0.25, scale), desired_turn

        # Too close to keep going: turn toward whatever opening exists.
        escape = space.best_bearing(prefer=desired_turn, min_range=self.safety_distance * 2.0)
        if escape is None:
            return 0.0, 0.0
        return 0.0, float(np.clip(escape * self.turn_gain, -1.2, 1.2))

    # -- the loop -----------------------------------------------------------------

    def goto(self, target: str, max_steps: int = 900, search_steps: int = 260) -> NavResult:
        """Find `target` and walk to it.

        max_steps bounds the whole attempt; search_steps bounds how long the initial
        look-around lasts before giving up.
        """
        state = NavState.SEARCH
        steps = 0
        searched = 0
        last_seen: tuple[Detection, float, float] | None = None
        lost_frames = 0
        detections: list[Detection] = []

        log.info("navigating to %r", target)

        while steps < max_steps:
            steps += 1
            refresh = (steps % self.perception_every == 1) or state is NavState.SEARCH

            if refresh:
                detections, space, fovy, size, depth = self._observe(target)
                # Bias toward whatever we were already chasing so the choice does not
                # jump between identical targets mid-approach.
                prefer = last_seen[1] if last_seen is not None else 0.0
                located = self._locate(detections, depth, fovy, size, prefer_bearing=prefer)
            else:
                located = last_seen

            if state is NavState.SEARCH:
                searched += 1
                if located is not None:
                    _, bearing, distance = located
                    log.info(
                        "found %s at %.1f m, %.0f deg", target, distance, math.degrees(bearing)
                    )
                    state = NavState.APPROACH
                    last_seen = located
                    continue
                if searched > search_steps:
                    return NavResult(NavState.SEARCH, target, steps=steps)
                # Rotate to bring new parts of the room into view.
                self.robot.step(0.0, 0.0, 0.7)
                continue

            if state is NavState.APPROACH:
                if located is None:
                    lost_frames += 1
                    if lost_frames > 12:
                        # It may just be out of frame; sweep again rather than fail outright.
                        log.info("lost sight of %s, searching again", target)
                        state = NavState.SEARCH
                        searched = 0
                        lost_frames = 0
                        continue
                    located = last_seen
                    if located is None:
                        state = NavState.SEARCH
                        continue
                else:
                    lost_frames = 0
                    last_seen = located

                _, bearing, distance = located

                if distance <= self.arrive_distance:
                    log.info("arrived at %s (%.2f m)", target, distance)
                    self.robot.stand(0.3)
                    return NavResult(
                        NavState.ARRIVED, target, distance=distance, bearing=bearing,
                        steps=steps, detections=detections,
                    )

                turn = float(np.clip(bearing * self.turn_gain, -1.2, 1.2))
                scale, turn = self._avoid(space, turn)

                if scale == 0.0 and turn == 0.0:
                    return NavResult(
                        NavState.BLOCKED, target, distance=distance, bearing=bearing, steps=steps
                    )

                # Slow down on approach, and while turning hard, so it does not overshoot.
                closing = min(1.0, max(0.25, (distance - self.arrive_distance) / 1.5))
                straightness = max(0.3, 1.0 - abs(turn))
                speed = self.cruise_speed * scale * closing * straightness
                self.robot.step(speed, 0.0, turn)
                # Re-measure the distance we are closing on next refresh.
                last_seen = (located[0], bearing, max(0.0, distance - speed * self.robot.control_dt))

        return NavResult(NavState.LOST, target, steps=steps)

    def face(self, target: str, max_steps: int = 200) -> NavResult:
        """Turn to put the target dead ahead, without walking anywhere."""
        tracked = 0.0
        for step in range(max_steps):
            detections, _, fovy, size, depth = self._observe(target)
            located = self._locate(detections, depth, fovy, size, prefer_bearing=tracked)
            if located is None:
                self.robot.step(0.0, 0.0, 0.6)
                continue
            _, bearing, distance = located
            tracked = bearing
            if abs(bearing) < 0.06:
                self.robot.stand(0.2)
                return NavResult(
                    NavState.ARRIVED, target, distance=distance, bearing=bearing, steps=step
                )
            self.robot.step(0.0, 0.0, float(np.clip(bearing * self.turn_gain, -0.9, 0.9)))
        return NavResult(NavState.LOST, target, steps=max_steps)
