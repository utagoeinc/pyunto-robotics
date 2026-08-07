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
        # Where the robot stood just before pushing through a doorway, so it can get back out.
        # Navigation is otherwise memoryless, and a room is a dead end without this: from
        # inside, the open door shows only its edge and no other door is visible at all, so
        # there is nothing for a purely reactive search to home in on. One remembered pose is
        # a much smaller concession than building a map.
        self._doorway_return: np.ndarray | None = None

    # -- navigation ---------------------------------------------------------------

    def goto(self, target: str, where: str | None = None) -> SkillResult:
        """Walk to something the camera can find.

        `where` picks between identical candidates: "the door on the right".
        """
        result = self.nav.goto(target, where=where)
        return SkillResult(
            ok=result.success,
            message=result.describe(),
            data={"distance": result.distance, "state": result.state.value},
        )

    def face(self, target: str, where: str | None = None) -> SkillResult:
        """Turn to look straight at something."""
        result = self.nav.face(target, where=where)
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

    def open_door(
        self, target: str = "door", side: str = "r", where: str | None = None
    ) -> SkillResult:
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
        approach = self.nav.goto(target, where=where)
        if not approach.success:
            return SkillResult(False, approach.describe())

        # Square up using the bearing goto already measured, rather than calling face().
        # face() re-runs detection from scratch, and next to a door the neighbouring one is
        # often the better-looking candidate -- which is how "open the left door" ended up
        # walking back to the middle one after correctly arriving at the left.
        residual = approach.bearing or 0.0
        if abs(residual) > 0.05:
            turn = float(np.clip(residual * 1.2, -0.9, 0.9))
            for _ in range(int(abs(residual) / (abs(turn) * self.robot.control_dt)) + 1):
                self.robot.step(0.0, 0.0, turn)
        self.robot.stand(0.3)

        # Close the remaining gap until the door is within reach.
        for _ in range(140):
            space = self._clearance_ahead()
            if space <= PUSH_STANDOFF_M:
                break
            self.robot.step(vx=0.3)
        self.robot.stand(0.3)

        # Remember the doorway itself, not where we are standing. The push-off point is up to
        # 0.7 m short of the opening, and aiming leave_room at that put the exit heading well
        # away from the actual gap. The doorway is one arm's length ahead along the current
        # heading, which is where the hand is about to make contact.
        heading = np.array([math.cos(self.robot.yaw), math.sin(self.robot.yaw)])
        self._doorway_return = self.robot.position[:2] + heading * PUSH_STANDOFF_M

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
        # Judge on how far the door ends up open, not on how much THIS push added. Squeezing
        # past a door on the way to it can already have swung it (a detour nudged the pantry
        # door 35 degrees open before the arm ever touched it), and measuring only the delta
        # then reports a door standing wide open as "it did not open".
        swing = max(
            abs(math.degrees(angle_after)),
            abs(math.degrees(angle_after - angle_before)),
        )

        if swing < 5.0:
            self.robot.arm_home(side)
            self.robot.stand(0.3)
            return SkillResult(
                False,
                f"I pushed the {target} but it did not open - it may be locked.",
                {"swing_degrees": swing},
            )

        # Walk through while it is open -- but only as far as being through. A fixed 120 steps
        # at 0.45 m/s carried the robot over a metre past the doorway and wedged it into the
        # far corner of the pantry, with 0.31 m of clearance in every direction and no way to
        # manoeuvre back out.
        # Walk a fixed distance first to clear the doorway itself -- the swinging leaf keeps
        # the measured clearance low, so checking it too early stops the robot in the opening.
        for _ in range(90):
            self.robot.step(vx=0.45)
        # Then continue only while there is room, so it does not end up wedged in a corner.
        for _ in range(75):
            if self._clearance_ahead() < 0.9:
                break
            self.robot.step(vx=0.45)
        self.robot.arm_home(side)
        self.robot.stand(0.4)

        return SkillResult(
            True,
            f"I opened the {target} and went through (it swung {swing:.0f} degrees).",
            {"swing_degrees": swing},
        )

    def pull_door(self, target: str = "door", side: str = "r") -> SkillResult:
        """Grasp the handle and pull the door open, rather than pushing through it.

        Pushing was the original and only option, chosen because a push needs the hand
        somewhere on the leaf rather than precisely on a 3.6 cm handle. That was a reasonable
        simplification and a bad long-term choice: a door pushed into a room swings across the
        way back out, which is exactly why leaving a room turned out to be so hard.

        The robot has always had the hardware for this -- the gripper opens to 6.4 cm and
        closes to 4.2 cm, and the handle is 3.6 cm across, with high-friction fingers and
        condim=4 for torsional friction. It simply was never asked to grasp anything.

        Sequence: line up on the handle, reach, close the gripper, walk backwards to swing the
        leaf toward us, release, then step around it.
        """
        approach = self.nav.goto(target)
        if not approach.success:
            return SkillResult(False, approach.describe())

        residual = approach.bearing or 0.0
        if abs(residual) > 0.05:
            turn = float(np.clip(residual * 1.2, -0.9, 0.9))
            for _ in range(int(abs(residual) / (abs(turn) * self.robot.control_dt)) + 1):
                self.robot.step(0.0, 0.0, turn)
        self.robot.stand(0.3)

        for _ in range(140):
            if self._clearance_ahead() <= PUSH_STANDOFF_M:
                break
            self.robot.step(vx=0.3)
        self.robot.stand(0.3)

        angle_before = self._door_angle()

        # Line up on the handle, not on the middle of the door. The hinge is at one edge and
        # the handle at the other, so stopping square to the leaf leaves the hand about half a
        # metre off to the side -- measured 0.486 m, against a 0.064 m gripper opening.
        self.robot.grip(side, 0.0)
        self.robot.set_arm(side, shoulder_pitch=-1.10, shoulder_roll=0.0,
                           shoulder_yaw=0.0, elbow=-0.20)
        self.robot.stand(0.6)

        for _ in range(200):
            offset = self._handle_offset(side)
            if offset is None or abs(offset) < 0.04:
                break
            # Sidestep: strafing keeps the robot square to the door while it closes the gap.
            self.robot.step(0.0, float(np.clip(offset * 1.5, -0.3, 0.3)), 0.0)
        self.robot.stand(0.4)

        # Close the last of the gap. Sidestepping leaves the hand lined up but still short --
        # measured 0.27 m of reach and 0.12 m of height to make up -- so drop the arm to handle
        # height and edge forward until the fingers are around it.
        self.robot.set_arm(side, shoulder_pitch=-0.95, shoulder_roll=0.0,
                           shoulder_yaw=0.0, elbow=-0.15)
        self.robot.stand(0.4)
        for _ in range(90):
            gap = self._handle_gap(side)
            if gap is None or gap < 0.05:
                break
            self.robot.step(vx=0.12)
        self.robot.stand(0.3)

        # Close the fingers, then weld. Friction alone will not hold: the fingers slip off a
        # 3.6 cm handle long before the arm can move a 20 kg leaf, so the closed hand is
        # modelled as a rigid grip. This is standard practice for manipulation in MuJoCo.
        self.robot.grip(side, 1.0)
        self.robot.stand(0.5)

        door_body = self._nearest_door_body()
        if door_body is None or not self.robot.grasp(door_body):
            self.robot.grip(side, 0.0)
            self.robot.arm_home(side)
            return SkillResult(False, f"I could not get hold of the {target}.")

        # Pull: back away while holding on, so the leaf swings toward us.
        for _ in range(220):
            self.robot.step(vx=-0.25)

        angle_after = self._door_angle()
        swing = abs(math.degrees(angle_after - angle_before))

        self.robot.release()
        self.robot.grip(side, 0.0)
        self.robot.stand(0.4)
        self.robot.arm_home(side)
        self.robot.stand(0.3)

        if swing < 5.0:
            return SkillResult(
                False,
                f"I took hold of the {target} but could not pull it open.",
                {"swing_degrees": swing},
            )

        # Step around the leaf and through. The door now stands between the robot and the
        # opening, so this sidesteps clear of it before walking forward.
        for _ in range(60):
            self.robot.step(0.0, -0.3, 0.0)
        for _ in range(40):
            self.robot.step(0.0, 0.0, 0.5)
        for _ in range(180):
            if self._clearance_ahead() < 0.7:
                break
            self.robot.step(vx=0.4)
        self.robot.stand(0.3)

        return SkillResult(
            True,
            f"I pulled the {target} open (it swung {swing:.0f} degrees).",
            {"swing_degrees": swing},
        )

    def _handle_offset(self, side: str = "r") -> float | None:
        """Lateral distance from the gripper to the nearest door handle, in the robot's frame.

        Positive means the handle is to the robot's left. Reads the simulator directly; on a
        real robot this would come from the camera, but the geometry is what matters here.
        """
        import mujoco  # noqa: PLC0415 - only needed for this introspection

        hand = self.robot.hand_position(side)
        best: float | None = None
        best_distance = math.inf
        for name in ("door_1_handle", "door_2_handle", "door_3_handle"):
            gid = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                continue
            handle = self.robot.data.geom_xpos[gid]
            distance = float(np.linalg.norm(handle[:2] - self.robot.position[:2]))
            if distance >= best_distance:
                continue
            best_distance = distance
            delta = handle[:2] - hand[:2]
            # Project onto the robot's left axis.
            left = np.array([-math.sin(self.robot.yaw), math.cos(self.robot.yaw)])
            best = float(np.dot(delta, left))
        return best

    def _nearest_door_body(self) -> str | None:
        """Name of the door body closest to the robot."""
        import mujoco  # noqa: PLC0415 - only needed for this introspection

        best: str | None = None
        best_distance = math.inf
        for name in ("door_workspace", "door_meeting", "door_pantry"):
            bid = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                continue
            distance = float(
                np.linalg.norm(self.robot.data.xpos[bid][:2] - self.robot.position[:2])
            )
            if distance < best_distance:
                best_distance = distance
                best = name
        return best

    def _handle_gap(self, side: str = "r") -> float | None:
        """Straight-line distance from the gripper to the nearest handle."""
        import mujoco  # noqa: PLC0415 - only needed for this introspection

        hand = self.robot.hand_position(side)
        best: float | None = None
        for name in ("door_1_handle", "door_2_handle", "door_3_handle"):
            gid = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                continue
            distance = float(np.linalg.norm(self.robot.data.geom_xpos[gid] - hand))
            if best is None or distance < best:
                best = distance
        return best

    def leave_room(self) -> SkillResult:
        """Get back out to the corridor.

        The robot pushed its way in, which leaves the leaf swung across the way back out. So
        leaving is the same manoeuvre in reverse: take hold of the handle and pull, which swings
        the door clear instead of pressing against it, then walk through.

        This is why pull_door exists at all. The original design could only push, and a
        push-only robot has no way out of a room it pushed into -- the door it opened is now in
        the way. Adding the grasp was not an extra feature so much as finishing the first one.
        """
        started_in = self.report_position().data.get("room")

        # No pre-aiming: pull_door runs its own approach, and turning first only fought it.
        result = self.pull_door("door")
        self._doorway_return = None

        where = self.report_position()
        room = where.data.get("room", "")
        left = room != started_in
        if left:
            return SkillResult(True, f"I came back out. {where.message}", where.data)
        return SkillResult(
            False,
            f"I could not get back out of {room}: {result.message}",
            where.data,
        )

    def _find_exit_heading(self) -> float | None:
        """Turn on the spot and return the world heading that most looks like the way out.

        Scores each direction by how far the depth image sees: a doorway shows the corridor
        beyond it, while every wall of the room is close. When open_door left a remembered
        pose, directions pointing toward it are preferred, which settles the case where a room
        has more than one deep-looking direction.
        """
        from ..perception.depth import free_space  # noqa: PLC0415 - avoids a circular import

        best_heading: float | None = None
        best_score = -math.inf
        turn_rate = 0.9
        steps = int(2 * math.pi / (turn_rate * self.robot.control_dt))

        for i in range(steps):
            self.robot.step(0.0, 0.0, turn_rate)
            if i % 5:
                continue

            obs = self.robot.look()
            space = free_space(obs.depth, self.robot.camera_fovy())
            reach = space.clearance_ahead(half_angle=0.22)
            if reach < 1.0:
                continue  # a wall, not a way out

            score = reach
            if self._doorway_return is not None:
                delta = self._doorway_return - self.robot.position[:2]
                desired = math.atan2(delta[1], delta[0])
                error = abs((desired - self.robot.yaw + math.pi) % (2 * math.pi) - math.pi)
                # Strongly prefer the direction we came from; the room may have other openings.
                score += 3.0 * max(0.0, 1.0 - error / math.pi)

            if score > best_score:
                best_score = score
                best_heading = self.robot.yaw

        self.robot.stand(0.2)
        return best_heading

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

    def point_at(self, target: str, side: str = "r", where: str | None = None) -> SkillResult:
        """Turn toward something and raise an arm at it."""
        facing = self.nav.face(target, where=where)
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

        # The doorway plane is y=1.0, so that is where a room begins. A looser 1.2 reported the
        # robot as "in the corridor" while it was still standing inside a doorway, which made
        # leave_room think it had succeeded when it had not moved.
        if y > 1.0:
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

    def run(
        self, action: str, argument: str | None = None, where: str | None = None
    ) -> SkillResult:
        """Execute one planner-issued action."""
        if action in ("goto", "face", "open", "open_door", "point_at"):
            argument = self._normalise_target(argument)
        handlers = {
            "goto": lambda: self.goto(argument or "door", where),
            "face": lambda: self.face(argument or "door", where),
            "open": lambda: self.open_door(argument or "door", where=where),
            "pull": lambda: self.pull_door(argument or "door"),
            "open_door": lambda: self.open_door(argument or "door", where=where),
            "point_at": lambda: self.point_at(argument or "door", where=where),
            "leave": lambda: self.leave_room(),
            "look_around": lambda: self.look_around(),
            "describe": lambda: self.describe_view(),
            "where": lambda: self.report_position(),
            "report": lambda: SkillResult(True, argument or "Done."),
        }
        handler = handlers.get(action)
        if handler is None:
            return SkillResult(False, f"I do not know how to '{action}'.")
        log.info("skill: %s(%s%s)", action, f"{where} " if where else "", argument or "")
        return handler()
