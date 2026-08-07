"""What the robot can actually do.

A skill is one verb the planner may emit: go somewhere, look at something, open a door. Each
returns a SkillResult carrying a sentence fit to send back over Pyunto, because the human on
the other end only ever sees that sentence.

Skills are written so that failing is normal and reported, not raised. "I could not find the
door" is a perfectly good answer to send someone; a traceback is not.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..nav.explore import MaplessNavigator, NavState
from ..perception.grounding import Grounder
from ..sim.robot import Robot

log = logging.getLogger(__name__)

# How close to stand before reaching for a door. The arm reaches ~0.43 m in front of the base
# at handle height, so anything beyond this leaves the hand short of the leaf.
PUSH_STANDOFF_M = 0.62


@dataclass
class SkillResult:
    """Outcome of one skill, in a form that can be messaged to a human."""

    ok: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


class Skills:
    """The robot's action repertoire."""

    def __init__(self, robot: Robot, grounder: Grounder, navigator: MaplessNavigator | None = None):
        self.robot = robot
        self.grounder = grounder
        self.nav = navigator or MaplessNavigator(robot, grounder)

    # -- navigation ---------------------------------------------------------------

    def goto(self, target: str) -> SkillResult:
        """Walk to something the camera can find."""
        result = self.nav.goto(target)
        return SkillResult(
            ok=result.success,
            message=result.describe(),
            data={"distance": result.distance, "state": result.state.value},
        )

    def face(self, target: str) -> SkillResult:
        """Turn to look straight at something."""
        result = self.nav.face(target)
        if result.success:
            return SkillResult(True, f"I am now facing the {target}.",
                               {"distance": result.distance})
        return SkillResult(False, f"I could not find a {target} to face.")

    def look_around(self, degrees: float = 360.0) -> SkillResult:
        """Turn on the spot, reporting what came into view."""
        seen: dict[str, int] = {}
        turn_rate = 0.8
        steps = int(abs(math.radians(degrees)) / (turn_rate * self.robot.control_dt))

        for i in range(steps):
            self.robot.step(0.0, 0.0, turn_rate)
            if i % 12 == 0:
                rgb = self.robot.look().rgb
                for name in ("door", "whiteboard", "desk", "plant", "monitor", "table"):
                    if self.grounder.find(rgb, name):
                        seen[name] = seen.get(name, 0) + 1
        self.robot.stand(0.3)

        if not seen:
            return SkillResult(True, "I looked around but did not recognise anything.")
        items = ", ".join(sorted(seen, key=lambda k: -seen[k]))
        return SkillResult(True, f"Looking around I can see: {items}.", {"seen": list(seen)})

    # -- manipulation -------------------------------------------------------------

    def open_door(self, target: str = "door", side: str = "r") -> SkillResult:
        """Walk up to a door and push it open.

        A push, not a handle turn. Pushing needs the hand somewhere on the leaf rather than
        precisely on a 3.6 cm handle, so it survives the pose error that navigation leaves
        behind - and it is what a person does to an unlatched office door anyway.

        The sequence: get close, square up, reach out at handle height, walk into the door so
        the arm loads it, then check the hinge actually moved.
        """
        # Stop within arm's length. The arm reaches ~0.43 m in front of the base at handle
        # height (measured by sweeping the shoulder/elbow range), so the default 0.85 m
        # stand-off leaves the hand half a metre short of the door.
        approach = self.nav.goto(target)
        if not approach.success:
            return SkillResult(False, approach.describe())

        facing = self.nav.face(target)
        if not facing.success:
            return SkillResult(False, f"I reached the {target} but could not square up to it.")

        # Close the remaining gap until the door is within reach.
        for _ in range(140):
            space = self._clearance_ahead()
            if space <= PUSH_STANDOFF_M:
                break
            self.robot.step(vx=0.3)
        self.robot.stand(0.3)

        angle_before = self._door_angle()

        # Best forward reach at handle height, found by sweeping the joint ranges:
        # shoulder pitch -1.10, elbow -0.20 puts the gripper 0.43 m ahead at z=1.03.
        self.robot.set_arm(side, shoulder_pitch=-1.10, shoulder_roll=0.0,
                           shoulder_yaw=0.0, elbow=-0.20)
        self.robot.grip(side, 0.35)
        self.robot.stand(0.6)

        # Push: keep walking forward so the extended arm loads the door.
        for _ in range(160):
            self.robot.step(vx=0.35)

        angle_after = self._door_angle()
        swing = abs(math.degrees(angle_after - angle_before))

        if swing < 5.0:
            self.robot.arm_home(side)
            self.robot.stand(0.3)
            return SkillResult(
                False,
                f"I pushed the {target} but it did not open - it may be locked.",
                {"swing_degrees": swing},
            )

        # Walk through while it is open, then let the arm relax.
        for _ in range(120):
            self.robot.step(vx=0.45)
        self.robot.arm_home(side)
        self.robot.stand(0.4)

        return SkillResult(
            True,
            f"I opened the {target} and went through (it swung {swing:.0f} degrees).",
            {"swing_degrees": swing},
        )

    def _clearance_ahead(self) -> float:
        """Distance to whatever is directly in front, from the current depth frame."""
        from ..perception.depth import free_space  # noqa: PLC0415 - avoids a circular import

        obs = self.robot.look()
        return free_space(obs.depth, self.robot.camera_fovy()).clearance_ahead(half_angle=0.25)

    def _door_angle(self) -> float:
        """Hinge angle of whichever door is nearest, or 0 if there is none.

        Reads the simulator directly. On a real robot this would come from watching the door
        move; here it is the ground truth that tells us whether the push worked.
        """
        import mujoco  # noqa: PLC0415 - only needed for this introspection

        best_angle = 0.0
        best_distance = math.inf
        for name in ("door_workspace", "door_meeting", "door_pantry"):
            bid = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                continue
            distance = float(np.linalg.norm(self.robot.data.xpos[bid][:2] - self.robot.position[:2]))
            if distance < best_distance:
                joint = self.robot.model.body_jntadr[bid]
                if joint >= 0:
                    best_distance = distance
                    best_angle = float(self.robot.data.qpos[self.robot.model.jnt_qposadr[joint]])
        return best_angle

    def point_at(self, target: str, side: str = "r") -> SkillResult:
        """Turn toward something and raise an arm at it."""
        facing = self.nav.face(target)
        if not facing.success:
            return SkillResult(False, f"I could not find a {target} to point at.")
        self.robot.set_arm(side, shoulder_pitch=-1.35, shoulder_roll=0.0,
                           shoulder_yaw=0.0, elbow=-0.1)
        self.robot.stand(1.0)
        self.robot.arm_home(side)
        return SkillResult(True, f"That is the {target}.")

    # -- reporting ----------------------------------------------------------------

    def describe_view(self) -> SkillResult:
        """Say what is in front of the robot right now."""
        obs = self.robot.look()
        found = []
        for name in ("door", "whiteboard", "desk", "monitor", "plant", "table", "fridge"):
            detections = self.grounder.find(obs.rgb, name)
            if detections:
                found.append(f"{name} ({len(detections)})" if len(detections) > 1 else name)

        from ..perception.depth import free_space  # noqa: PLC0415 - avoids a circular import

        ahead = free_space(obs.depth, self.robot.camera_fovy()).clearance_ahead()
        if not found:
            return SkillResult(True, f"Nothing I recognise ahead. Clear for {ahead:.1f} m.")
        return SkillResult(
            True,
            f"I can see: {', '.join(found)}. Clear space ahead: {ahead:.1f} m.",
            {"objects": found, "clearance": ahead},
        )

    def report_position(self) -> SkillResult:
        """Where the robot is, in terms a person can picture."""
        x, y = self.robot.position[0], self.robot.position[1]
        heading = math.degrees(self.robot.yaw) % 360

        if y > 1.2:
            if x < -2.0:
                where = "in the workspace"
            elif x > 2.0:
                where = "in the pantry"
            else:
                where = "in the meeting room"
        elif y > -1.0:
            where = "in the corridor"
        else:
            where = "in the lobby"

        return SkillResult(
            True,
            f"I am {where}, facing {heading:.0f} degrees.",
            {"x": float(x), "y": float(y), "heading": heading, "room": where},
        )

    # -- dispatch -----------------------------------------------------------------

    @staticmethod
    def _normalise_target(argument: str | None) -> str | None:
        """Reduce a planner's phrasing to something the grounder recognises.

        The LLM answers in natural language -- "the meeting room door", "会議室" -- while the
        grounder only knows a handful of object names. Mapping here keeps that vocabulary
        mismatch out of the perception layer, and means a room name still resolves to the door
        that leads to it, which is the useful interpretation of "go to the meeting room".
        """
        if not argument:
            return argument
        text = argument.lower().strip()
        # A room is reached through its door, so any room mention resolves to "door".
        for room in ("meeting", "会議", "workspace", "執務", "pantry", "給湯",
                     "office", "オフィス", "room", "部屋"):
            if room in text:
                return "door"
        for name in ("door", "whiteboard", "monitor", "desk", "table", "plant", "fridge"):
            if name in text:
                return name
        return argument

    def run(self, action: str, argument: str | None = None) -> SkillResult:
        """Execute one planner-issued action."""
        if action in ("goto", "face", "open", "open_door", "point_at"):
            argument = self._normalise_target(argument)
        handlers = {
            "goto": lambda: self.goto(argument or "door"),
            "face": lambda: self.face(argument or "door"),
            "open": lambda: self.open_door(argument or "door"),
            "open_door": lambda: self.open_door(argument or "door"),
            "point_at": lambda: self.point_at(argument or "door"),
            "look_around": lambda: self.look_around(),
            "describe": lambda: self.describe_view(),
            "where": lambda: self.report_position(),
            "report": lambda: SkillResult(True, argument or "Done."),
        }
        handler = handlers.get(action)
        if handler is None:
            return SkillResult(False, f"I do not know how to '{action}'.")
        log.info("skill: %s(%s)", action, argument or "")
        return handler()
