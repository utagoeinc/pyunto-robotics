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
        # Heading the robot had when it went through, so leaving can line up on the reverse.
        self._doorway_heading: float | None = None

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

    def _get_a_view_of_the_doors(self, minimum: int = 3, max_steps: int = 500) -> int:
        """Move to somewhere the whole row of doors is visible, and face it.

        "left" and "right" are relative to what the robot can see, so they only mean what the
        user intended if the robot can see the doors it is choosing between. Straight after
        leaving a room it stands a metre from one doorway facing sideways, and asking for "the
        left door" there picks the one it is standing next to.

        Measured: from y=0.9, right at the thresholds, only one door fits in frame; from
        y=0.67 two do; all three need y=0 or the lobby. Seeing only two is worse than
        useless -- standing by the pantry, "left" then means the meeting room rather than
        the workspace -- so this insists on all three by default.

        Returns how many doors ended up visible.
        """
        best = self._count_doors()
        if best >= minimum:
            return best

        # Face the doors. They are all on the north wall, so turning to look that way is the
        # one piece of layout knowledge this needs.
        self._turn_to(math.pi / 2)
        best = max(best, self._count_doors())

        # Then go and stand where the whole row is visible: the middle of the lobby, looking
        # north. Two things rule out anywhere nearer. A door left standing open fills the view
        # from a metre away, and the robot has just opened one. And the corridor is long
        # enough that from one end the far door is outside a 75-degree field of view.
        #
        # This drives to a fixed spot rather than searching for one. Searching was tried: each
        # pass drifted 0.12 m sideways, and twenty passes later the robot was in the east
        # corner with no doors in sight at all.
        viewpoint = np.array([0.0, -2.2])
        for _ in range(max_steps):
            delta = viewpoint - self.robot.position[:2]
            if float(np.linalg.norm(delta)) < 0.4:
                break
            desired = math.atan2(delta[1], delta[0])
            error = (desired - self.robot.yaw + math.pi) % (2 * math.pi) - math.pi
            turn = float(np.clip(error * 1.5, -1.0, 1.0))
            self.robot.step(0.4 * max(0.35, 1.0 - abs(turn)), 0.0, turn)

        self._turn_to(math.pi / 2)
        self.robot.stand(0.2)
        return self._count_doors()

    def _count_doors(self, min_confidence: float = 0.25, min_separation: float = 0.08) -> int:
        """How many distinct doors are in view.

        Colour matching splits one door into several slivers when a frame edge catches the
        light, and a low-confidence sliver at the edge of frame is enough to make the robot
        think it can see the whole row when it cannot. Drop faint detections and merge ones
        that sit on top of each other.
        """
        detections = [
            d for d in self.grounder.find(self.robot.look().rgb, "door")
            if d.confidence >= min_confidence
        ]
        distinct: list[float] = []
        for det in sorted(detections, key=lambda d: d.x):
            if not distinct or det.x - distinct[-1] > min_separation:
                distinct.append(det.x)
        return len(distinct)

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
        # A spatial qualifier only means what the user intended if the robot can see the row
        # of doors it is choosing between. Straight after leaving a room it is next to one
        # doorway facing sideways, and "the left door" then picks whichever it is standing by.
        if where in ("left", "right", "middle"):
            self._get_a_view_of_the_doors()

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

        # Remember where we are STANDING, which is the corridor side of the threshold -- that
        # is where leaving has to get back to. Recording the doorway itself (one arm's length
        # ahead) put the target inside the room: measured (4.54, 1.27) for a doorway at y=1.0,
        # so "returning" to it never left the pantry.
        self._doorway_return = self.robot.position[:2].copy()
        self._doorway_heading = self.robot.yaw

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

        # Step around the leaf and through. A fixed sidestep-turn-walk left the robot facing
        # away from the doorway entirely (measured: ended up at +8 degrees, east, with the
        # opening behind it to the west), so this steers by where the opening actually is.
        self._walk_through_doorway()

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

    def _walk_through_doorway(self, steps: int = 320, until_room_changes: bool = True) -> None:
        """After pulling, sidestep clear of the leaf and walk out through the opening.

        Aims at the remembered doorway pose when there is one -- that is the corridor side of
        the opening, recorded on the way in -- and otherwise at whatever direction the depth
        image says is most open. Either way it steers each step rather than replaying a fixed
        sequence, because the robot's pose after a pull depends on how far the door swung.
        """
        from ..perception.depth import free_space  # noqa: PLC0415 - avoids a circular import

        # Back off just enough to unload the leaf. A longer reverse pushed the robot deeper
        # into the room than it started -- measured y=1.22 going to 1.55 in the workspace,
        # away from a doorway at y=1.0.
        for _ in range(8):
            self.robot.step(-0.3, 0.0, 0.0)
        self.robot.stand(0.2)

        # Stop on having actually changed rooms, not on proximity to the remembered pose.
        # Distance is a poor test here: the robot ends up hemmed in by the leaf right at the
        # threshold, so a tight tolerance never triggers and a loose one fires while still
        # inside.
        started_in = self.report_position().data.get("room")

        for _ in range(steps):
            if until_room_changes and self.report_position().data.get("room") != started_in:
                break

            obs = self.robot.look()
            space = free_space(obs.depth, self.robot.camera_fovy())

            if self._doorway_return is not None:
                delta = self._doorway_return - self.robot.position[:2]
                desired = math.atan2(delta[1], delta[0])
                bearing = (desired - self.robot.yaw + math.pi) % (2 * math.pi) - math.pi
            else:
                bearing = space.best_bearing(prefer=0.0, min_range=1.0) or 0.0

            ahead = space.clearance_ahead(half_angle=0.28)
            turn = float(np.clip(bearing * 1.5, -1.0, 1.0))

            if ahead < 0.55:
                # Something in the way -- most likely the leaf. Strafe rather than turn: in a
                # doorway there is no room to swing round, and the gap is usually just to one
                # side. Sidestep toward whichever side the target is on.
                self.robot.step(0.05, math.copysign(0.3, bearing or 1.0), 0.0)
                continue

            self.robot.step(0.4 * max(0.35, 1.0 - abs(turn)), 0.0, turn)
        self.robot.stand(0.3)

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
        """Get back out to the corridor by retracing the way in.

        This is the one manoeuvre in the system that is planned rather than reactive, and it has
        to be. In a doorway the robot has under 0.5 m of clearance in every direction, so the
        obstacle-avoiding controller that works everywhere else has no single-step move that
        improves anything -- forward is the leaf, back is the room. It just oscillates.

        So this executes a fixed sequence instead, using the pose open_door recorded on the way
        through: turn to face back the way we came, pull the door clear if it is in the way,
        then drive to the remembered spot without re-planning. Short, blind, and reliable,
        which is what a doorway needs.
        """
        started_in = self.report_position().data.get("room")
        if self._doorway_return is None:
            return SkillResult(False, "I do not remember how I came in.")

        target = self._doorway_return.copy()

        # 1. Face back toward the corridor.
        delta = target - self.robot.position[:2]
        heading = math.atan2(delta[1], delta[0])
        self._turn_to(heading)

        # 2. The leaf swung into the room when we pushed in, so it is now between us and the
        #    opening. Pull it clear if we are up against it.
        #
        #    Test contact, not forward clearance. In the pantry the robot ends up pressed
        #    against the leaf with chest, thigh and foot while the depth camera still reports
        #    1.8 m ahead -- the door is beside it, not in front -- so a clearance check misses
        #    exactly the case this exists for.
        if self._touching_door() or self._clearance_ahead() < 1.0:
            self._pull_leaf_clear()
            self._turn_to(heading)

        # 3. Drive to the remembered pose. Deliberately not re-planning: the reactive
        #    controller cannot navigate a doorway, and the route is only a metre or two.
        for _ in range(400):
            delta = target - self.robot.position[:2]
            distance = float(np.linalg.norm(delta))
            # Only stop early once we are actually out. Reaching the remembered pose is not the
            # same as having left: the robot got within 0.09 m of the threshold and stopped
            # there, still inside, with the spring closing the door on it.
            if distance < 0.25 and self.report_position().data.get("room") != started_in:
                break
            desired = math.atan2(delta[1], delta[0])
            error = (desired - self.robot.yaw + math.pi) % (2 * math.pi) - math.pi
            turn = float(np.clip(error * 1.4, -0.9, 0.9))
            self.robot.step(0.35 * max(0.4, 1.0 - abs(turn)), 0.0, turn)

            # Fouling the leaf. The spring is closing the door onto the robot -- measured
            # going from 21 to 13 degrees while it stood in the gap -- so the answer is to
            # hold the door open, not to squeeze past a shrinking opening. Put an arm out
            # against it and keep walking; the leaf gives way and the robot goes through.
            if self._touching_door():
                self._hold_door_open()

        self.robot.stand(0.3)
        self._doorway_return = None
        self._doorway_heading = None

        where = self.report_position()
        room = where.data.get("room", "")
        left = room != started_in
        return SkillResult(
            left,
            f"I came back out. {where.message}" if left else f"I could not get back out of {room}.",
            where.data,
        )

    def close_door(self, side: str = "r") -> SkillResult:
        """Pull the nearest door shut behind us.

        Worth doing for its own sake, and it keeps the corridor usable: a door left standing
        open fills the view from close by, and the robot has to see the row of doors to make
        sense of "the left one".

        The spring already pulls each door toward closed, so this only has to stand clear and
        wait -- there is no need to grasp and haul.

        Known limitation: this usually leaves the door 20-40 degrees open rather than shut. The
        leaf needs about a metre of clearance to swing through, and the robot cannot reliably
        get that far from a doorway it has just come through. Stiffening the closer so it shuts
        faster was tried and made things worse: the door then closes on the robot while it is
        still leaving, and getting out of a room dropped from 3/3 to 2/3.
        """
        import mujoco  # noqa: PLC0415 - only needed for this introspection

        door = self._nearest_door_body()
        if door is None:
            return SkillResult(False, "There is no door here to close.")

        bid = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_BODY, door)
        joint = self.robot.model.body_jntadr[bid]
        adr = self.robot.model.jnt_qposadr[joint]
        before = abs(math.degrees(float(self.robot.data.qpos[adr])))

        # Stand clear so the leaf has room to swing, then let the spring do the work.
        self.robot.arm_home(side)
        for _ in range(50):
            self.robot.step(-0.25, 0.35, 0.0)
        self.robot.stand(0.3)
        # Then wait. The doors swing both ways, so a leaf released from 80 degrees overshoots
        # through the closed position and comes back -- measured -87, -54, +19, then settling
        # at +1 after about eight seconds. Waiting only for the first crossing catches it
        # mid-swing and reports a door that is closing as one that failed to close.
        for _ in range(10):
            self.robot.stand(1.0)
            if abs(math.degrees(float(self.robot.data.qpos[adr]))) < 8.0:
                break

        after = abs(math.degrees(float(self.robot.data.qpos[adr])))
        if after < 12.0:
            return SkillResult(True, "I closed the door behind me.", {"angle": after})
        return SkillResult(
            False,
            f"The door did not swing shut ({after:.0f} degrees still open).",
            {"angle_before": before, "angle": after},
        )

    def _hold_door_open(self, side: str = "r", steps: int = 90) -> None:
        """Brace an arm against the leaf and push on through.

        The spring returns each door to closed, so a robot that stops in the opening gets
        squeezed: measured closing from 21 degrees to 13 while it stood there. Backing off and
        strafing only ever loses ground. Extending an arm turns the robot into a doorstop --
        the leaf presses against the forearm instead of the torso, and forward motion swings
        it back open.
        """
        self.robot.set_arm(side, shoulder_pitch=-1.15, shoulder_roll=0.0,
                           shoulder_yaw=0.0, elbow=-0.15)
        self.robot.grip(side, 0.2)
        self.robot.stand(0.3)
        # Keep pushing for a moment after contact breaks. Stopping the instant the leaf lets go
        # leaves the robot still in the opening, where the spring closes it again -- it reached
        # 0.11 m short of the threshold that way and got squeezed a second time.
        clear_for = 0
        for _ in range(steps):
            self.robot.step(0.3, 0.0, 0.0)
            clear_for = clear_for + 1 if not self._touching_door() else 0
            if clear_for > 25:
                break
        self.robot.arm_home(side)
        self.robot.stand(0.2)

    def _touching_door(self) -> bool:
        """True when any part of the robot is in contact with a door leaf or its frame."""
        import mujoco  # noqa: PLC0415 - only needed for this introspection

        robot_parts = (
            "torso", "pelvis", "uarm", "farm", "palm", "fing", "thigh", "shin", "foot",
            "head", "neck", "visor", "chest",
        )
        for c in range(self.robot.data.ncon):
            contact = self.robot.data.contact[c]
            n1 = str(mujoco.mj_id2name(self.robot.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1))
            n2 = str(mujoco.mj_id2name(self.robot.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2))
            door_side = "door" in n1 or "door" in n2 or "frame" in n1 or "frame" in n2
            robot_side = any(p in n1 for p in robot_parts) or any(p in n2 for p in robot_parts)
            if door_side and robot_side:
                return True
        return False

    def _turn_to(self, heading: float, max_steps: int = 220) -> None:
        """Rotate on the spot to a world heading."""
        for _ in range(max_steps):
            error = (heading - self.robot.yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(error) < 0.05:
                break
            self.robot.step(0.0, 0.0, float(np.clip(error * 1.6, -1.0, 1.0)))
        self.robot.stand(0.2)

    def _pull_leaf_clear(self, side: str = "r") -> bool:
        """Grasp whatever door is in front and pull it out of the way.

        Unlike pull_door this does no navigation -- the robot is already at the doorway and
        letting the navigator loose here is what causes it to wander back into the room.
        """
        door = self._nearest_door_body()
        if door is None:
            return False

        self.robot.grip(side, 0.0)
        self.robot.set_arm(side, shoulder_pitch=-1.05, shoulder_roll=0.0,
                           shoulder_yaw=0.0, elbow=-0.20)
        self.robot.stand(0.5)

        # Line up on the handle, then close the last of the gap.
        for _ in range(120):
            offset = self._handle_offset(side)
            if offset is None or abs(offset) < 0.05:
                break
            self.robot.step(0.0, float(np.clip(offset * 1.5, -0.25, 0.25)), 0.0)
        for _ in range(60):
            gap = self._handle_gap(side)
            if gap is None or gap < 0.12:
                break
            self.robot.step(vx=0.1)
        self.robot.stand(0.3)

        self.robot.grip(side, 1.0)
        self.robot.stand(0.4)
        if not self.robot.grasp(door):
            self.robot.grip(side, 0.0)
            self.robot.arm_home(side)
            return False

        # Drag it aside: reverse and turn at once so the leaf sweeps away from the opening.
        for _ in range(120):
            self.robot.step(-0.25, 0.0, 0.35)

        self.robot.release()
        self.robot.grip(side, 0.0)
        self.robot.arm_home(side)
        self.robot.stand(0.3)
        return True

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
            "close": lambda: self.close_door(),
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
