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
from ..perception.landmarks import LandmarkMap
from ..sim.robot import Robot

log = logging.getLogger(__name__)


def _canonical_label(target: str) -> str:
    """Normalise a described target to the name landmarks are filed under."""
    from ..perception.grounding import _canonical  # noqa: PLC0415 - avoids a circular import

    return _canonical(target) or target.lower().strip()

# How far a tracked target may appear to move between frames and still count as the same
# object, in metres. Tracking is done in world coordinates rather than by bearing: a target's
# bearing changes fast as the robot turns or crosses a room, so a bearing window either loses
# a near target or lets a far one hop to its neighbour. Its position does not move at all.
# The doors are 4 m apart, so a 2 m gate cannot confuse one for its neighbour. It has to be
# this wide because the estimate is noisy: depth samples the face of the leaf rather than the
# doorway, and the error swings with viewpoint -- measured the estimate for one door moving
# 0.72 m in a single approach step.
TRACK_GATE_M = 2.0

# The most a single new fix may MOVE the tracked position. Identity (gate above) and update
# (this) need different scales: a detection 1.5 m from the anchor is still plausibly the same
# door -- the anchor may simply be poor -- but letting it drag the anchor there in one step is
# how oblique under-ranged views walked the anchor a metre south of the wall, 0.7-1.0 m per
# frame. Genuine refinement arrives in decimetres.
TRACK_REFIX_M = 0.5

# How far either side the head sweeps while searching, and how fast it cycles. Wide enough to
# reach well past the forward camera's 37 degrees, slow enough that frames are not smeared.
SEARCH_SWEEP_RAD = 1.0
SEARCH_SWEEP_RATE = 0.09

# Head angles used when surveying the room before committing to a target, radians.
SURVEY_ANGLES = (-1.0, -0.5, 0.0, 0.5, 1.0)

# Two sightings closer than this in bearing are the same object.
SURVEY_SEPARATION_RAD = 0.35

# Roughly how wide a door is, used to work out how far apart two sightings must be before they
# can be different doors rather than two edges of one.
DOOR_WIDTH_M = 1.0

# A target counts as reached only if it is also roughly ahead. Brushing past a door on the way
# somewhere else puts it within arm's reach at 30-40 degrees off, which is not arriving at it.
# 0.30 rad was picked by measurement, not taste: at 0.45 the robot accepted the middle door
# seen edge-on at 27 degrees while walking to the right one, and at 0.22 it could never satisfy
# the gate at all and wandered off.
ARRIVE_BEARING_RAD = 0.30  # ~17 degrees

# Inside this range of the target the side look is skipped: the last stretch of any approach
# is a narrowing gap, because doors are set into walls.
SIDE_LOOK_DISABLE_M = 2.5

# How far the view may swing between perception frames before looking again early. About 6
# degrees: a fraction of the 75-degree field, so a target cannot cross the frame unseen.
REFRESH_SWING_RAD = 0.10

# Body turn rate above which the head stops tracking and holds still, so the two do not swing
# the camera through the sum of both at once.
TURNING_HARD_RAD = 0.45

# Clearance below which the head gives up watching the target and looks where the body is going.
HEAD_YIELD_M = 0.8

# Sideways steering on the approach, to keep a gap off the walls the forward camera cannot see.
#
# Room to want beside the body, on top of its own half-width; how hard to correct per metre
# missing; and how far the neck may be turned before the side cameras stop meaning anything
# (they are mounted on the head, so a neck on a door points them down the corridor and at the
# floor rather than at the flanks).
SIDE_STEER_MARGIN_M = 0.20
SIDE_STEER_GAIN = 1.2
SIDE_STEER_HEAD_RAD = 0.60

# Sideways push away from something the body is actually touching, m/s, folded into the same
# command the approach already issues. Touch is the only flank sense that stays true when the
# neck turns away, and this is where the scraping survived every camera-based correction.
SIDE_TOUCH_PUSH = 0.25


# Walk the corridor square-on, and turn in only when level with the target.
#
# A door is set into a wall, so steering by its bearing means crossing the corridor diagonally
# and arriving alongside that wall: measured 663 control steps of this errand with an arm
# against a door frame, all of it walking past walls rather than going through doorways.
# Steering away from the wall cannot fix it, because the target is *in* the wall -- every gain
# tried that way either did nothing or made the scraping worse.
#
# So take the two apart. The body follows the corridor's own axis -- the world axis most nearly
# at right angles to the target, which for a door in the north wall is the east-west line -- and
# the neck carries the looking. Once the head has had to swing past HEAD_COMMIT_RAD the robot is
# roughly level with the door, and turning in from there runs straight at it instead of along
# the wall. It is how a person walks a corridor: down the middle, then turn in at the door.
#
# The first cut walked the axis at whatever offset the robot started at, and wall contact
# fell from 663 control steps (8.3%) to 0-106 (0-1.1%) -- but a metre off the far wall the
# avoidance stack throttled the walk, and a fixed lean toward the target reintroduced the
# oblique arrival. The current shape centres in the corridor first (side cameras, by feel),
# walks the true middle, and steps away from anything the body actually touches.
CORRIDOR_TRAVEL = False

# Neck angle -- the target's bearing off the travel axis -- past which the body gives up the
# corridor line and turns in. The spec's example was 30 degrees, but from the lobby the door
# already sits 47 degrees off the corridor axis, so 30 fires before a single corridor step is
# taken and the whole thing degenerates to the old diagonal. 75 means properly level: for a
# door 4 m across the corridor the turn comes just 1.1 m before being square with it, so the
# final leg runs at the door nearly straight-on -- at 60 the leg still crossed at an angle,
# and arrivals landed far enough along the wall to push at the frame, or at the wrong door
# entirely. Still 25 degrees inside what the neck can hold (100).
HEAD_COMMIT_RAD = 1.31

# Only travel this way while the target is further off than this. Inside it the approach is the
# final run at the door, where holding a corridor line would just walk past it.
CORRIDOR_TRAVEL_MIN_M = 2.0

# Walk the MIDDLE of the corridor, found by feel rather than by map: strafe until the side
# cameras report the same room on both sides. Walking the axis at whatever offset the robot
# happens to start at was tried twice and failed twice -- a metre off the far wall the
# avoidance stack throttled the walk to 0.01 m/s and bent it into detours, and a fixed lean
# toward the target's side just traded that for arriving alongside the target's wall. The
# centre is the one line that is far from both.
CENTRE_TOLERANCE_M = 0.4

# Step budget for finding the middle, and how often to re-render the side cameras while
# strafing toward it (each look is two renders).
CENTRE_MAX_STEPS = 200
CENTRE_LOOK_EVERY = 5

# How long the wall-brush reflex sidesteps away from a touch before rejoining the approach.
WALL_REFLEX_STEPS = 20

# How hard the reflex reverses while it sidesteps, m/s. Pressed against a surface, a purely
# sideways command scrubs along it and friction wins: measured 60 steps of pure strafe failing
# to break contact at all, against 12 once the drive had a component off the surface. But the
# reverse is a component, not the move -- swept over a full errand, 0.15 halves both the
# contact (14.8% of control steps to 10.6%) and the length of the errand itself, while 0.25
# backs out of approaches faster than they can close and loses the last door entirely.
WALL_REFLEX_BACK = 0.15

# The slowest forward command the gait actually converts into walking, m/s. Below this the
# stance friction wins and the robot marches in place.
MIN_WALK_SPEED = 0.2

# The stuck watchdog: if an approach has moved less than PROGRESS_MIN_M in PROGRESS_WINDOW
# steps, stop pretending it is going anywhere -- back off a stride and search again. Every
# stall this session had a different proximate cause (a gait dead zone, an avoid knife-edge,
# a counter the contact check did not know by name) and the same signature: the position
# freezing while the loop kept issuing commands. A person who notices they have stopped
# making progress steps back and reassesses; they do not push on the same spot for two
# thousand steps.
PROGRESS_WINDOW = 400
PROGRESS_MIN_M = 0.25

# How many steps to keep walking toward a remembered position before admitting it is not
# working and going back to looking. Sized against the room, not against nerves: at blind
# walking speed 120 steps covers under a metre, and a door picked from home is 5.5 m away --
# the colour match flickers at that range, and a patience shorter than the gaps between its
# good frames turns every long approach into search-and-give-up. 300 covers about 2.5 m, and
# the walk is aimed at a fixed world point the whole time, not at a guess.
BLIND_PATIENCE = 300

# How far past the visible free space a sighting may sit before it is treated as a bad range
# rather than a real object. Generous, because the free-space columns are coarse and a door set
# into an alcove genuinely is slightly further than the wall beside it.
BEHIND_TOLERANCE_M = 0.6

# Only reject "behind the wall" sightings within this bearing of straight ahead.
BEHIND_CHECK_RAD = 0.5

# Sideways room to leave beyond the robot's own width, in metres. Small: this is "do not scrape"
# clearance, not a comfortable berth. Asking for a wide margin was measured to cost more than it
# saved -- it fights every doorway, which is barely wider than the robot -- whereas correcting
# only when the body genuinely will not fit leaves narrow gaps passable.
CLEARANCE_MARGIN_M = 0.08

# Inside this range of the target, stop correcting for width. The last stretch of any approach
# is a narrowing gap, because doors are set into walls; keeping clear there means never getting
# there.
CLEARANCE_DISABLE_M = 2.0


class NavState(Enum):
    SEARCH = "search"
    APPROACH = "approach"
    DETOUR = "detour"
    ARRIVED = "arrived"
    LOST = "lost"
    BLOCKED = "blocked"


# Detour tuning. A target can be plainly visible with no route to it -- doors are seen across a
# corridor long before the robot can walk to them -- and heading straight at one just presses it
# into the intervening wall.
# Wall-following detours are off.
#
# The manoeuvre exists to round an obstacle between the robot and a visible target, and it
# does that by sliding along the obstruction -- contact is its method, not a failure of it.
# Which makes it the source of the scraping: attributing every contact step on the approach
# to the code that issued it put 30 of 30 inside _wall_follow, and none in the approach
# itself. It also fires on the target: the trigger distance below is shorter than the
# distance an approach stops at, so a door filling the view reads as a wall to round.
#
# Without it the plain approach still avoids obstacles -- _avoid steers around what the depth
# frame sees and refuses commands that would not fit -- it simply does not seek walls out to
# hug them.
DETOUR_ENABLED = False

DETOUR_TRIGGER_M = 0.75  # clearance below which a straight approach is judged blocked
DETOUR_CLEAR_M = 1.6  # clearance above which the route ahead counts as open again
DETOUR_MIN_STEPS = 40  # commit for at least this long, or it oscillates in and out
# Long enough to get round a wall and back on course. Crossing a flat to the far wall with
# a door standing open on the way takes several detours, and at 320 the budget ran out mid-way.
DETOUR_MAX_STEPS = 500

# How much room to keep beside the body while following a wall, in metres. Just over the
# robot's own half-width, so it travels alongside rather than against.
WALL_STANDOFF_M = 0.55

# Sideways speed used to hold that gap, m/s.
WALL_PUSH_OFF = 0.10


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
        # How many frames the target may be missing before the robot gives up tracking it and
        # sweeps again. Twelve suits a corridor, where losing sight of a door means it really
        # has gone out of frame. Outdoors it is wrong: cresting a dune pitches the camera off
        # the target for longer than that every time, and the rover abandoned targets it was
        # tracking perfectly well -- "I saw the beacon but lost sight of it before I got
        # there", said about a drive that was going fine.
        lost_frames_allowed: int = 12,
    ):
        self.robot = robot
        self.grounder = grounder
        # Which door is which, accumulated as the robot looks around. Bearing alone cannot
        # express "that one, not its neighbour"; a landmark keeps its identity however the
        # view changes, and its averaged position is far steadier than any single frame.
        self.landmarks = LandmarkMap()
        # View direction when the last perception frame was taken, so the loop can tell how far
        # the world has moved across the camera since it last looked.
        self._view_yaw = robot.yaw
        self._view_head = robot.head_yaw
        self.arrive_distance = arrive_distance
        self.cruise_speed = cruise_speed
        self.turn_gain = turn_gain
        self.safety_distance = safety_distance
        # Perception is cheap here (~2 ms/frame) but the vision model may not be, so the
        # cadence is configurable independently of the control rate.
        self.perception_every = perception_every
        self.lost_frames_allowed = lost_frames_allowed

    # -- perception ---------------------------------------------------------------

    def observe_landmarks(self, target: str) -> int:
        """Take one frame and fold what it sees into the map. Returns the map's count.

        Exists so a caller that is looking around rather than travelling -- counting the doors
        the user said were there, say -- can build up the map without driving the whole
        navigation loop.
        """
        self._observe(target)
        return len(self.landmarks.of_label(_canonical_label(target)))

    def _observe(
        self, target: str
    ) -> tuple[list[Detection], FreeSpace, float, tuple[int, int], np.ndarray]:
        """One camera frame, turned into everything the loop needs from it.

        Returns the depth image too, so callers never render a second frame just to measure a
        detection -- doing that would also risk pairing a detection with a *different* frame.
        """
        obs = self.robot.look()
        fovy = self.robot.camera_fovy()
        # Everything downstream steers the body, so bearings have to be body-relative. The head
        # can be turned, and a bearing measured in the camera's frame is offset by exactly that.
        head = self.robot.head_yaw
        height, width = obs.depth.shape
        detections = self.grounder.find(obs.rgb, target)
        space = free_space(obs.depth, fovy)

        # Fold what is visible into the landmark map as we go, so the robot builds up an
        # identity for each door instead of re-deciding from scratch every frame.
        positions = []
        ranges = []
        for det in detections:
            u, v = det.pixel(width, height)
            offset = target_offset(obs.depth, u, v, fovy, (width, height), head)
            if offset is None:
                continue

            # Drop sightings that sit further away than the free space in that direction. A
            # door cannot be behind the nearest surface the robot can see through the same
            # pixels; when it reads that way the range is wrong, usually because the body was
            # mid-stride and the head camera swung. Measured 364 such sightings in one run,
            # putting doors up to 1.5 m beyond the wall they are set into, which is what filled
            # the map with landmarks that were nowhere near a real door.
            bearing, distance = offset
            column = int(np.argmin(np.abs(space.bearings - bearing)))
            behind = distance > space.ranges[column] + BEHIND_TOLERANCE_M
            # Only trust this test near the centre of the frame. The free-space columns are
            # coarse and the correction from axial to true range grows with angle, so out at
            # the edges a perfectly good sighting can read as being behind the wall -- applying
            # it everywhere threw away two of the three doors entirely.
            if behind and abs(bearing) < BEHIND_CHECK_RAD:
                continue

            positions.append(self._world_position(bearing, distance))
            ranges.append(distance)
        if positions:
            self.landmarks.observe_all(_canonical_label(target), positions, ranges)

        return detections, space, fovy, (width, height), obs.depth

    def survey(self, target: str) -> list[tuple[float, float]]:
        """Turn the head across its range and collect (bearing, range) for everything seen.

        Bearings are body-relative, so the caller can treat the result like one very wide
        camera frame. This is what makes a spatial qualifier mean what the user meant: a
        forward frame covers 75 degrees, and from beside a doorway "the left door" resolves
        against whatever single door happens to be in it. Sweeping the head covers about 190
        degrees and finds all three from the same spot.

        Duplicates are collapsed by direction rather than by position -- a bearing is enough to
        say two sightings are the same door, and it avoids the range errors that made an
        earlier position-based count inflate to four doors in a room with one.
        """
        found: list[tuple[float, float, float]] = []  # bearing, range, confidence
        for angle in SURVEY_ANGLES:
            self.robot.turn_head_to(angle)
            detections, _, fovy, size, depth = self._observe(target)
            for det in detections:
                u, v = det.pixel(*size)
                offset = target_offset(depth, u, v, fovy, size, self.robot.head_yaw)
                if offset is None:
                    continue
                bearing, distance = offset
                # A door subtends more of the view the closer it is, and colour matching
                # splits a wide one into its two edges. So the angle that separates "two doors"
                # from "two edges of one door" has to scale with range: from 1.6 m the middle
                # door came back as two sightings 32 degrees apart, while the real doors either
                # side sat 68 degrees away.
                separation = max(
                    SURVEY_SEPARATION_RAD,
                    math.atan2(DOOR_WIDTH_M, max(distance, 0.3)),
                )
                match = next(
                    (i for i, (b, _, _) in enumerate(found) if abs(bearing - b) < separation),
                    None,
                )
                if match is None:
                    found.append((bearing, distance, det.confidence))
                elif det.confidence > found[match][2]:
                    found[match] = (bearing, distance, det.confidence)
        self.robot.face_forward()
        return [(b, d) for b, d, _ in found]

    def _walk_to(self, point: np.ndarray, max_steps: int = 600) -> None:
        """Walk back to a remembered spot, steering round whatever is in the way."""
        for _ in range(max_steps):
            delta = point - self.robot.position[:2]
            if float(np.linalg.norm(delta)) < 0.4:
                break
            desired = math.atan2(delta[1], delta[0]) - self.robot.yaw
            desired = (desired + math.pi) % (2 * math.pi) - math.pi
            # Feel first, as the approach loop does: a walk-to that grazes a doorway edge
            # used to lean on it for the rest of its budget.
            touch = self.robot.wall_contact_side()
            if touch is not None:
                for _ in range(WALL_REFLEX_STEPS):
                    self.robot.step(-WALL_REFLEX_BACK, -touch * 0.25, 0.0)
                    if self.robot.wall_contact_side() is None:
                        break
                continue
            _, space, _, _, _ = self._observe("door")
            # Ease the head forward rather than face_forward(), which stands still for up to
            # 80 control steps while the neck settles -- called once per walking step, that
            # re-launched the gait from a standstill every stride: measured 0.16 m of progress
            # over 400 iterations against the 3.1 m the commanded speed should give. The head
            # converges to centre over a few strides at the bounded rate and the walk never
            # stops.
            self.robot.look_toward(0.0)
            scale, turn = self._avoid(space, float(np.clip(desired * self.turn_gain, -1.2, 1.2)))
            self.robot.step(self.cruise_speed * scale * 0.7, 0.0, turn)
        self.robot.stand(0.3)

    def _corridor_bearing(self, target_bearing: float) -> float | None:
        """Body-relative heading that runs along the corridor instead of at the target.

        No map is read and none exists. What is assumed is that the building is rectilinear --
        walls and corridors run along the world's own axes -- which is the same single piece of
        layout knowledge skills.py leans on. The corridor direction is then the world axis
        nearest to right angles with the line of sight, taken in whichever sense still makes
        progress toward the target.

        Steering at right angles to the line of sight itself, without snapping to an axis, is
        the tempting version of this and it is wrong: a heading perpendicular to the sight line
        never changes the distance to the target, so the robot orbits the door at constant
        range instead of walking the corridor past it.

        Returns None when the sight line already runs along an axis -- there the corridor and
        the direct approach are the same line, and the caller should just close in.
        """
        theta = self.robot.yaw + target_bearing
        quarter = math.pi / 2
        best = None
        best_progress = 0.2  # cos threshold: below this the axis makes no real progress
        for perpendicular in (theta + quarter, theta - quarter):
            axis = round(perpendicular / quarter) * quarter
            progress = math.cos(axis - theta)
            if progress > best_progress:
                best_progress = progress
                best = axis
        if best is None:
            return None
        return (best - self.robot.yaw + math.pi) % (2 * math.pi) - math.pi

    def _relative_to(self, position: np.ndarray) -> tuple[float, float]:
        """Bearing and range from where the robot is now to a fixed world point."""
        delta = position - self.robot.position[:2]
        bearing = math.atan2(delta[1], delta[0]) - self.robot.yaw
        return (bearing + math.pi) % (2 * math.pi) - math.pi, float(np.linalg.norm(delta))

    def _turn_body_to(self, bearing: float, max_steps: int = 200) -> None:
        """Rotate on the spot until `bearing` is straight ahead, stepping off walls first.

        A pivot with a shoulder loaded against a jamb grinds round it; step clear, then turn.
        """
        target = self.robot.yaw + bearing
        recoils = 0
        for _ in range(max_steps):
            touch = self.robot.wall_contact_side()
            if touch is not None and recoils < 3:
                recoils += 1
                for _ in range(WALL_REFLEX_STEPS):
                    self.robot.step(-WALL_REFLEX_BACK, -touch * 0.25, 0.0)
                    if self.robot.wall_contact_side() is None:
                        break
                continue
            error = (target - self.robot.yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(error) < 0.06:
                break
            self.robot.step(0.0, 0.0, float(np.clip(error * 1.6, -1.0, 1.0)))
        self.robot.stand(0.2)

    @staticmethod
    def _pick_bearing(
        seen: list[tuple[float, float]], where: str
    ) -> tuple[float, float] | None:
        """Choose among surveyed sightings using a spatial word. Bearing is + to the left."""
        if where == "left":
            return max(seen, key=lambda s: s[0])
        if where == "right":
            return min(seen, key=lambda s: s[0])
        if where == "middle":
            return sorted(seen, key=lambda s: s[0])[len(seen) // 2]
        return None

    def _world_position(self, bearing: float, distance: float) -> np.ndarray:
        """Where a detection sits in the world, from its bearing and range.

        This is what makes tracking stable. A door does not move; only the robot's view of it
        does, so comparing positions survives the viewpoint changing while comparing bearings
        does not.
        """
        heading = self.robot.yaw + bearing
        return self.robot.position[:2] + distance * np.array(
            [math.cos(heading), math.sin(heading)]
        )

    def _locate(
        self,
        detections: list[Detection],
        depth: np.ndarray,
        fovy: float,
        size: tuple[int, int],
        prefer_bearing: float = 0.0,
        where: str | None = None,
        tracking: np.ndarray | None = None,
    ) -> tuple[Detection, float, float] | None:
        """Pick a detection to chase and measure it.

        Scoring on distance alone is unstable when several identical targets are equidistant:
        a corridor may have three doors 3.8-3.9 m away, and "nearest" flip-flopped between the
        left and right one every frame, so the robot just oscillated. Bearing is part of the
        score, which both settles that and matches what "the door" usually means -- the one
        being looked at, not one 44 degrees off to the side.

        `tracking` is the world position of the target already being chased. When set, the
        choice is locked to whatever is nearest that spot rather than re-scored, which is what
        stops an approach switching horses halfway.

        `where` is a spatial qualifier from the instruction ("the door on the right"). When
        present it OVERRIDES the default preference for whatever is straight ahead -- the whole
        point of saying "the right one" is to pick something the robot would not have chosen.
        """
        measured: list[tuple[Detection, float, float]] = []
        for det in detections:
            u, v = det.pixel(*size)
            offset = target_offset(depth, u, v, fovy, size, self.robot.head_yaw)
            if offset is None:
                continue
            bearing, distance = offset
            measured.append((det, bearing, distance))

        if not measured:
            return None

        if where:
            return self._pick_by_qualifier(measured, where)

        if tracking is not None:
            # Already committed to one candidate: re-acquire the SAME one by where it is in the
            # world, and do not reconsider. Scoring afresh each frame let the straight-ahead
            # term outweigh the tracking term, so a door picked at 43 degrees was abandoned for
            # whatever sat in the middle of the frame -- which is exactly how "the door on the
            # left" ended up opening the centre door.
            best_match: tuple[Detection, float, float] | None = None
            best_error = math.inf
            for candidate in measured:
                _, bearing, distance = candidate
                error = float(
                    np.linalg.norm(self._world_position(bearing, distance) - tracking)
                )
                if error < best_error:
                    best_error = error
                    best_match = candidate
            # Only accept it if it plausibly is the same object; otherwise treat as lost.
            return best_match if best_error < TRACK_GATE_M else None

        best: tuple[Detection, float, float] | None = None
        best_score = -math.inf
        for candidate in measured:
            _, bearing, distance = candidate
            # Nearer is better; straight ahead is better.
            score = -distance - 2.0 * abs(bearing) - 1.5 * abs(bearing - prefer_bearing)
            if score > best_score:
                best_score = score
                best = candidate
        return best

    @staticmethod
    def _pick_by_qualifier(
        measured: list[tuple[Detection, float, float]], where: str
    ) -> tuple[Detection, float, float] | None:
        """Choose among several candidates using a spatial word from the instruction.

        Left/right are from the ROBOT's point of view, which is also the viewer's when looking
        at the camera feed. Bearing is positive to the left, so "left" is the largest bearing.
        """
        if where == "left":
            return max(measured, key=lambda m: m[1])
        if where == "right":
            return min(measured, key=lambda m: m[1])
        if where == "middle":
            return min(measured, key=lambda m: abs(m[1]))
        if where == "nearest":
            return min(measured, key=lambda m: m[2])
        if where == "far":
            return max(measured, key=lambda m: m[2])
        return None

    # -- detouring ----------------------------------------------------------------

    @staticmethod
    def _openest_side(space: FreeSpace) -> float:
        """Which flank has more room: positive for left, negative for right."""
        left = space.ranges[space.bearings > 0.25]
        right = space.ranges[space.bearings < -0.25]
        left_room = float(left.max(initial=0.0))
        right_room = float(right.max(initial=0.0))
        return left_room - right_room

    def _wall_follow(self, space: FreeSpace, side: float) -> tuple[float, float, float]:
        """One control command that slides along an obstacle toward `side`.

        Returns (vx, vy, wz). The robot turns toward the open flank while creeping forward,
        which walks it around a corner instead of stalling against it.
        """
        ahead = space.clearance_ahead(half_angle=0.30)

        # Aim at the most open bearing on the chosen side.
        mask = (space.bearings > 0.1) if side > 0 else (space.bearings < -0.1)
        if mask.any():
            candidates = space.bearings[mask]
            room = space.ranges[mask]
            aim = float(candidates[int(np.argmax(room))])
        else:
            aim = side * 0.8

        turn = float(np.clip(aim * self.turn_gain, -1.2, 1.2))

        # Back off if genuinely nose-first into something, otherwise keep edging forward.
        # Reverse AND slide off whatever the body is loaded against: this return fires before
        # the touch check below, and it was where every contact on the approach ended up --
        # 30 of 30, commanding vx=-0.15 with vy=0.00 while a flank was hard against a wall.
        # Pure reverse does not unload a surface the body is pressed along.
        if ahead < self.safety_distance * 0.8:
            loaded = self.robot.wall_contact_side()
            return -0.15, (-loaded * WALL_PUSH_OFF if loaded else 0.0), turn
        speed = 0.35 if abs(turn) < 0.8 else 0.18

        # Following a wall means travelling BESIDE it, and only the forward camera has a say
        # in that so far -- so a flank pressed against a door post reads as a clear road and
        # the robot grinds along it: measured a single unbroken 466-step scrape beside a
        # doorway, the largest source of contact left in the errand. Lean away from whatever
        # the body can actually feel while still creeping forward, so the follow keeps its
        # shape instead of turning into a recoil.
        touch = self.robot.wall_contact_side()
        if touch is not None:
            return speed * 0.5, -touch * 0.25, turn
        return speed, 0.0, turn

    # -- steering -----------------------------------------------------------------

    def _avoid(
        self,
        space: FreeSpace,
        desired_turn: float,
        target_bearing: float = 0.0,
        target_distance: float = math.inf,
    ) -> tuple[float, float]:
        """Blend the desired heading with what the depth image says is safe.

        Returns (speed_scale, turn). Speed is cut as obstacles close in, and if the way ahead
        is genuinely blocked the robot steers along the obstacle rather than stopping dead.

        The "steer along" part matters for anything not in a straight line from here. Heading
        directly at a door on the far side of a corridor walks the robot into the partition
        between them; it has to slide along the wall until the doorway opens up. Without this
        it just parks against the wall with the target in sight and never arrives.
        """
        ahead = space.clearance_ahead(half_angle=0.35)

        # Before anything else, check the body actually fits along the way it is being steered.
        # Range alone cannot tell: a wall a metre ahead and slightly off to one side reads as a
        # comfortable clearance in every direction, and the robot walks into it shoulder first
        # and scrapes along it. Asking "how much room is there beside the line I am walking"
        # catches that, and the answer only has meaning against a width, which the robot knows.
        # Not on the last stretch, though. A doorway is 1.1 m wide against a 0.67 m body, so
        # from close up the target itself reads as a gap the robot barely fits through -- which
        # is true, and steering away from it is exactly wrong. Ten tests failed that way:
        # contact fell, but the robot stopped arriving anywhere.
        if target_distance > CLEARANCE_DISABLE_M:
            half_width = self.robot.half_width
            gap = space.widest_gap(desired_turn, half_width)
            if gap < CLEARANCE_MARGIN_M:
                roomier = space.clearest_heading(desired_turn, half_width)
                if space.widest_gap(roomier, half_width) > gap:
                    desired_turn = roomier

        if ahead > self.safety_distance * 2.5:
            return 1.0, desired_turn
        if ahead > self.safety_distance:
            # Slow down proportionally as the gap narrows.
            scale = (ahead - self.safety_distance) / (self.safety_distance * 1.5)
            return max(0.25, scale), desired_turn

        # Blocked ahead. The target may be visible straight through a wall -- doors are seen
        # across a corridor long before there is a route to them -- so steering by the target's
        # bearing here just presses the robot into the obstruction. Commit to the open
        # direction instead, and only use the target to break ties between equally open sides.
        escape = space.best_bearing(prefer=0.0, min_range=self.safety_distance * 1.6)
        if escape is None:
            # Nothing open at all: rotate on the spot toward the target and try again.
            return 0.0, float(np.clip(math.copysign(0.8, target_bearing or 1.0), -1.2, 1.2))

        # A near-zero escape heading means "forward is the most open direction", which cannot
        # be true when we already know the way ahead is blocked. That happens when the opening
        # is off to one side but only slightly further than the wall in front. Force a real
        # commitment to whichever flank is genuinely clearer.
        if abs(escape) < 0.25:
            left = float(space.ranges[space.bearings > 0.3].max(initial=0.0))
            right = float(space.ranges[space.bearings < -0.3].max(initial=0.0))
            escape = 0.7 if left >= right else -0.7

        turn = float(np.clip(escape * self.turn_gain, -1.2, 1.2))
        # Creep forward while turning so the robot slides past the obstruction rather than
        # spinning in place next to it.
        crawl = 0.3 if abs(escape) < 0.9 else 0.12
        return crawl, turn

    # -- the loop -----------------------------------------------------------------

    def goto(
        self,
        target: str,
        max_steps: int = 3000,
        search_steps: int = 260,
        where: str | None = None,
    ) -> NavResult:
        """Find `target` and walk to it.

        max_steps bounds the whole attempt; search_steps bounds how long the initial
        look-around lasts before giving up. `where` picks between identical candidates
        ("the door on the right").

        The step budget has to cover detours, not just the straight-line walk: reaching a door
        on the far side of a building means following a wall around, which took ~1400 steps
        where the direct approach took 350. At the old 900 the robot ran out of budget
        mid-detour and reported the target lost after having correctly found it.
        """
        state = NavState.SEARCH
        steps = 0
        searched = 0
        last_seen: tuple[Detection, float, float] | None = None
        # World position of the door being chased, so a changing viewpoint cannot swap it.
        tracked_at: np.ndarray | None = None
        lost_frames = 0
        # Last turn command issued, so the head can hold still while the body swings.
        turn_last = 0.0
        # Steps spent walking toward a remembered position without seeing the target.
        blind_steps = 0
        # Where the errand started, to return to if the target cannot be found from here.
        home = self.robot.position[:2].copy()
        returned_home = False
        ever_seen = False
        detections: list[Detection] = []
        turn_last = 0.0
        # Whether the body is currently holding the corridor line rather than facing the target,
        # and whether it has already turned in for good.
        corridor_travelling = False
        committed = False
        # When the wall-brush reflex last fired, to keep its log line from repeating every
        # control step while the robot works itself clear.
        last_reflex_step = -10_000
        # The stuck watchdog's anchor: where the robot was when the current window opened.
        progress_step = 0
        progress_pos = self.robot.position[:2].copy()
        detour_side = 1.0
        detour_steps = 0

        # Set when a head sweep resolved the qualifier. That choice was made with far more of
        # the room in view than the forward camera has, so losing sight of the door is not a
        # reason to give it up -- the door has not moved.
        surveyed: np.ndarray | None = None

        log.info("navigating to %r", target)

        # A spatial qualifier has to be resolved against everything the robot can see, not just
        # what is straight ahead. From the start pose the forward camera finds one door and
        # "the left one" resolves against that; a head sweep finds all three, at -69, -16 and
        # +69 degrees against truth of -68.2, 0.0 and +68.2. Pick from the sweep, then hand the
        # chosen door to the approach as a world position so it is tracked like any other.
        # Only when the forward view is not enough. A sweep costs a couple of seconds of
        # standing still and, more importantly, resolving the qualifier against a much wider
        # field changes which door "the left one" names -- from the corridor the forward camera
        # already sees all three and gets it right, so sweeping there only adds ways to be
        # wrong. It earns its keep where the forward camera sees one door and the head finds
        # three, which is the case that used to open the middle door when told the left one.
        if where in ("left", "right", "middle"):
            forward, _, _, _, _ = self._observe(target)
            seen = self.survey(target) if len(forward) < 2 else []
            if len(seen) > 1:
                chosen = self._pick_bearing(seen, where)
                if chosen is not None:
                    tracked_at = self._world_position(*chosen)
                    log.info(
                        "%s %s of %d is at %.0f deg", where, target, len(seen),
                        math.degrees(chosen[0]),
                    )
                    # Set off toward it even though the forward camera cannot see it yet. The
                    # sweep found the left door at (-4.05, 0.91) -- truth (-4, 1) -- while the
                    # forward frame held only the middle one 4 m away, so re-acquiring on the
                    # first frame fails and the correctly-chosen door gets thrown away. Walking
                    # a few metres in its direction brings it into view, and from there the
                    # normal approach takes over.
                    state = NavState.APPROACH
                    surveyed = tracked_at.copy()
                    last_seen = (None, *self._relative_to(tracked_at))


        while steps < max_steps:
            steps += 1
            # Look again on a schedule -- but sooner if the view has moved much since the last
            # frame. A fixed one-in-four was fine while the head was bolted to the torso; once
            # both can turn, four steps of a brisk turn swing the view 15 degrees, and a target
            # can cross most of the frame between looks. Measured the estimate of a door going
            # from 0.43 m of error to 1.71 m across exactly one such gap, and never recovering.
            swung = abs(
                (self.robot.yaw - self._view_yaw + math.pi) % (2 * math.pi) - math.pi
            ) + abs(self.robot.head_yaw - self._view_head)
            refresh = (
                (steps % self.perception_every == 1)
                or state is NavState.SEARCH
                or swung > REFRESH_SWING_RAD
            )
            if refresh:
                self._view_yaw = self.robot.yaw
                self._view_head = self.robot.head_yaw

            if refresh:
                detections, space, fovy, size, depth = self._observe(target)
                # Bias toward whatever we were already chasing so the choice does not
                # jump between identical targets mid-approach.
                # Apply the qualifier only on the first sighting. After that the target is
                # tracked by continuity: it drifts toward the centre of the frame as the robot
                # turns to face it, so re-picking "the rightmost" every frame would keep
                # handing off to the next door along.
                # Once a door has been chosen, re-acquire it by where it is in the world.
                # The qualifier only applies to the first sighting: after that the robot may
                # have moved somewhere "leftmost" means a different door.
                if tracked_at is None:
                    located = self._locate(detections, depth, fovy, size, where=where)
                else:
                    located = self._locate(detections, depth, fovy, size, tracking=tracked_at)
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
                    tracked_at = self._world_position(bearing, distance)
                    continue
                if searched > search_steps:
                    # Sweeping from here has not found it. Go back to where the instruction was
                    # given and look again: that is the one place the target is known to have
                    # been visible, and it is what a person does when they lose their bearings.
                    # Only worth going back if the target was ever seen. Something that has
                    # never been in view is not there, and walking back to look again just
                    # turns "I could not find a door" into a much slower "I could not find a
                    # door".
                    if ever_seen and not returned_home:
                        log.info("cannot find %s; going back to look from the start", target)
                        returned_home = True
                        searched = 0
                        blind_steps = 0
                        tracked_at = None
                        self._walk_to(home)
                        continue
                    self.robot.face_forward()
                    return NavResult(NavState.SEARCH, target, steps=steps)
                # Sweep the head as the body turns, so each step of the search covers the
                # head's range as well as the body's. Holding the head still instead was
                # measured costing a step of the four-step errand, so the extra coverage is
                # worth more than the occasional frame taken mid-swing.
                sweep = math.sin(searched * SEARCH_SWEEP_RATE) * SEARCH_SWEEP_RAD
                self.robot.look_toward(sweep)
                self.robot.step(0.0, 0.0, 0.7)
                continue

            if state is NavState.APPROACH:
                # Notice going nowhere. Commands can keep flowing while the body is parked --
                # see PROGRESS_WINDOW -- and every long stall looks the same from here
                # whatever caused it. Back off a stride and look again, like a person who
                # realises they have stopped making progress.
                if steps - progress_step >= PROGRESS_WINDOW:
                    moved = float(
                        np.linalg.norm(self.robot.position[:2] - progress_pos)
                    )
                    progress_step = steps
                    progress_pos = self.robot.position[:2].copy()
                    if moved < PROGRESS_MIN_M:
                        log.info(
                            "moved %.2f m in %d steps; backing off to look again",
                            moved, PROGRESS_WINDOW,
                        )
                        for _ in range(40):
                            self.robot.step(-0.3, 0.0, 0.0)
                        self.robot.stand(0.3)
                        state = NavState.SEARCH
                        searched = 0
                        last_seen = None
                        corridor_travelling = False
                        tracked_at = surveyed.copy() if surveyed is not None else tracked_at
                        # Keep the last anchor when there is no survey to reset to.
                        # The door has not moved; forgetting where it was flips the
                        # loop onto the assumed-progress estimate, which once counted
                        # its way down from 6.9 m to an "arrival" 8.4 m from the door.
                        continue

                # Feel before looking. Nothing in this loop used to read contact, so the
                # robot walked whole corridors pressed against a wall without knowing --
                # the harness counted 663 control steps of it in one errand. A person whose
                # shoulder brushes a wall steps away from it before thinking about anything
                # else; do the same, and only then go back to steering.
                touch = self.robot.wall_contact_side()
                if touch is not None:
                    if steps - last_reflex_step > WALL_REFLEX_STEPS * 2:
                        log.info(
                            "brushed something on the %s; stepping away",
                            "left" if touch > 0 else "right",
                        )
                    last_reflex_step = steps
                    # Back off as well as sideways: pressed against a surface a purely
                    # sideways command scrubs along it and friction wins -- measured 60 steps
                    # of pure strafe failing to break contact at all, against 12 with a
                    # component off the surface.
                    for _ in range(WALL_REFLEX_STEPS):
                        self.robot.step(-WALL_REFLEX_BACK, -touch * 0.25, 0.0)
                        if self.robot.wall_contact_side() is None:
                            break
                    continue

                if located is None:
                    lost_frames += 1

                    # Out of sight is not lost while the robot knows where the thing is. A door
                    # does not move, so walking to a remembered position gets there whether or
                    # not the camera can see it on the way. The head hunts for it meanwhile,
                    # and _avoid below still runs on the live depth frame, so this supplies a
                    # heading, never permission to walk into something.
                    if tracked_at is not None and blind_steps < BLIND_PATIENCE:
                        blind_steps += 1
                        want, range_to = self._relative_to(tracked_at)
                        # LOOK AT IT. Turning the head costs nothing and is how the target is
                        # re-acquired -- it is the body that must not be pointed at obstacles.
                        #
                        # The head used to yield whenever clearance ahead fell below 0.8 m,
                        # which sounds prudent and is exactly backwards in a small room:
                        # approaching a washing machine the clearance IS the machine, measured
                        # at 0.18 m, so the head locked forward for the whole approach and the
                        # robot walked the last metre and a half blind. The log filled with
                        # hundreds of "lost sight of it" lines while the target sat a
                        # head-turn away.
                        #
                        # The head is now only forced forward when the target is behind the
                        # robot -- past the neck's own limit, where tracking would crane it
                        # round and see nothing anyway -- or while the body is turning hard
                        # enough that a moving head would smear every frame.
                        behind = abs(want) > self.robot.NECK_LIMIT_RAD
                        spinning = abs(float(self.robot.data.qvel[5])) > TURNING_HARD_RAD
                        if behind or spinning:
                            self.robot.face_forward()
                        else:
                            self.robot.look_toward(want)
                        if blind_steps == 1:
                            log.info(
                                "lost sight of %s; heading for where it was, %.1f m away",
                                target, range_to,
                            )
                        located = (last_seen[0] if last_seen else None, want, range_to)
                        lost_frames = 0
                    elif lost_frames > self.lost_frames_allowed:
                        # Out of frame. Sweep again, and re-apply the qualifier when we do:
                        # dropping it here would re-acquire whichever door is most convenient
                        # rather than the one that was asked for.
                        log.info("lost sight of %s, searching again", target)
                        state = NavState.SEARCH
                        searched = 0
                        lost_frames = 0
                        last_seen = None
                        tracked_at = surveyed.copy() if surveyed is not None else tracked_at
                        # Keep the last anchor when there is no survey to reset to.
                        # The door has not moved; forgetting where it was flips the
                        # loop onto the assumed-progress estimate, which once counted
                        # its way down from 6.9 m to an "arrival" 8.4 m from the door.
                        continue
                    located = last_seen
                    if located is None:
                        state = NavState.SEARCH
                        continue
                else:
                    lost_frames = 0
                    blind_steps = 0
                    ever_seen = True
                    last_seen = located
                    # Do not update the world fix while the body is swinging hard. The bearing
                    # was measured from a frame rendered a beat ago, and pairing it with the yaw
                    # the body has ALREADY turned to puts the target somewhere it never was --
                    # measured the tracked door drifting 0.7-1.0 m per frame during the swing
                    # onto the corridor axis, ending with a "door" 0.85 m away in the middle of
                    # open floor and a triumphant phantom arrival. The head learned this lesson
                    # long ago (turn head or body, not both); the tracker needs the same rule.
                    #
                    # And only let a new fix MOVE the anchor by a refinement, not a leap. The
                    # target is a door; it does not move. A genuine better fix lands within
                    # decimetres of the old one, while the failure mode this guards against
                    # moves in strides: viewed obliquely down the corridor the range reads
                    # short, and each under-ranged fix dragged the anchor 0.7-1.0 m per frame,
                    # from (-3.6, 0.8) to (-4.4, -0.2) -- a metre south of the wall the door
                    # is in -- after which every honest detection sat 6-10 m out and the gate
                    # rejected 356 in a row. Freezing the anchor outright was tried and traded
                    # one failure for another: it locks in the first fix's error too. Accept
                    # small corrections, refuse strides.
                    # The stride rule applies only while walking the corridor: that is where
                    # the oblique bias lives, and it is the only place the strides were
                    # measured. On the plain approach a big correction is usually the honest
                    # one -- the first fix from across the room is the worst one taken.
                    if abs(turn_last) < TURNING_HARD_RAD:
                        new_fix = self._world_position(located[1], located[2])
                        stride = (
                            tracked_at is not None
                            and float(np.linalg.norm(new_fix - tracked_at)) >= TRACK_REFIX_M
                        )
                        if not (corridor_travelling and stride):
                            tracked_at = new_fix

                _, bearing, distance = located

                # Arriving means the target is close AND roughly in front. Close-but-sideways
                # is what you get brushing past a door on the way to another one -- accepting
                # that stopped the robot next to the middle door while walking to the right one.
                #
                # But the bearing test is a proxy for "is this really the door I was sent to",
                # and when the robot is standing next to the door it was sent to it fails for
                # the wrong reason: measured circling at 0.4 m from the correct door because
                # the heading never came out tidy. Where the target's position is known, ask
                # the question directly instead -- close to *that* is arrival however the body
                # happens to be pointing.
                at_the_target = (
                    tracked_at is not None
                    and float(np.linalg.norm(tracked_at - self.robot.position[:2]))
                    <= self.arrive_distance
                )
                if distance <= self.arrive_distance and (
                    abs(bearing) <= ARRIVE_BEARING_RAD or at_the_target
                ):
                    # Look again before believing it. The range that satisfied the test came
                    # from a frame taken while walking, and the estimate it is measured against
                    # is only as good as the view that built it -- measured arriving at (2.61,
                    # 0.52) for a door at (4.00, 1.00), having reached an estimate that was
                    # itself 0.91 m out. A standstill frame settles that in a few tenths of a
                    # second: stop, look, and if the door is still further off than this,
                    # carry on toward where it actually is.
                    self.robot.stand(0.3)
                    detections, space, fovy, size, depth = self._observe(target)
                    # Re-apply the qualifier, not the tracking. Tracking re-acquires whatever
                    # is nearest the estimate, and the estimate is exactly what is in doubt:
                    # measured stopping at (2.61, 0.52) with two doors in view, the correct one
                    # 1.85 m off and a nearer sliver 0.85 m away that matched the drifted
                    # anchor. From a standstill in front of the row, "the right-hand one" is a
                    # question the frame can answer directly.
                    confirm = self._locate(detections, depth, fovy, size, where=where)
                    if confirm is None:
                        confirm = self._locate(detections, depth, fovy, size, tracking=tracked_at)
                    if confirm is not None and confirm[2] > self.arrive_distance * 1.3:
                        log.info(
                            "thought I was at the %s, but it is %.1f m off; closing in",
                            target, confirm[2],
                        )
                        last_seen = confirm
                        tracked_at = self._world_position(confirm[1], confirm[2])
                        # A confirmed sighting is a fresh start on the approach, not a
                        # continuation of the one that went astray: the blind budget spent
                        # walking to the wrong estimate must not count against walking to the
                        # right one, or the correction runs out of patience before it arrives.
                        blind_steps = 0
                        lost_frames = 0
                        continue
                    log.info("arrived at %s (%.2f m)", target, distance)
                    # Face front again before handing back. Everything after arrival -- lining
                    # up on a handle, pushing, walking through -- reads the forward camera, and
                    # a head still turned toward the target from the approach points it at a
                    # wall: measured arriving 1 m from the left door with the head at -33
                    # degrees and reporting it could not find a door.
                    self.robot.face_forward()
                    self.robot.stand(0.3)
                    return NavResult(
                        NavState.ARRIVED, target, distance=distance, bearing=bearing,
                        steps=steps, detections=detections,
                    )

                # A wall between here and a visible target means the straight line is not the
                # route. Switch to following the obstacle instead of grinding against it.
                if DETOUR_ENABLED and space.clearance_ahead(half_angle=0.30) < DETOUR_TRIGGER_M:
                    detour_side = 1.0 if self._openest_side(space) >= 0 else -1.0
                    detour_steps = 0
                    state = NavState.DETOUR
                    log.info(
                        "%s is behind an obstacle; following the wall to the %s",
                        target, "left" if detour_side > 0 else "right",
                    )
                    continue

                # Keep the head on the target while the body steers wherever it needs to.
                # This is the whole reason the neck exists: without it, avoiding a wall swings
                # the cameras off the door, and with three identical doors in a corridor the
                # robot re-acquires whichever is nearest. Measured the head holding a target
                # 51 degrees off the body's heading, well past the 37 degrees a fixed forward
                # camera can reach.
                if tracked_at is not None:
                    # Watch the target -- unless something is close enough ahead that the body
                    # needs the forward camera. A turned head aims the depth frame away from
                    # where the feet are going, and obstacle avoidance reads that frame:
                    # measured wall contact of 277 steps with the head always on the target
                    # against 16 when it gives way near obstacles. A person crossing a room
                    # looks at the door, and glances ahead when something is in the way.
                    if space.clearance_ahead(half_angle=0.35) < HEAD_YIELD_M:
                        self.robot.face_forward()
                    elif abs(turn_last) < TURNING_HARD_RAD or corridor_travelling:
                        # Corridor travel deliberately points the body away from the target, so
                        # the head has to keep tracking through the turn that sets that up.
                        # Letting it hold still there loses the door within a few steps.
                        self.robot.look_at(tracked_at)
                    # Else: hold the head still. Turning both at once swings the camera through
                    # the sum of the two, and the target crosses the frame faster than
                    # perception can follow. One at a time is also what a person does -- you
                    # stop moving your eyes while your body is swinging round.

                # Walk the corridor square-on rather than cutting the diagonal to the door.
                # The neck is already holding the target above, so giving the body a different
                # heading costs no sight of it -- that separation is the whole point of having
                # a neck. Turn in once the door has swung HEAD_COMMIT_RAD off the travel axis,
                # which is the robot telling itself it is level with the doorway.
                #
                # Every angle in this block comes from the remembered world position, not from
                # the frame's detection. The two big body turns this feature adds (onto the
                # axis, and back onto the door) are exactly when a detection's bearing pairs a
                # stale render with a yaw the body has already left -- measured a turn-in fired
                # by a phantom sight line 35 degrees off truth, mid-swing, before a single
                # corridor step was taken. The tracked position holds still through all of it.
                # And both turns happen on the spot, outside this loop, so no frames are taken
                # mid-swing in the first place: stop, face down the corridor, walk. Stop, face
                # the door, walk in. It is how a person does it.
                steer_to = bearing
                if CORRIDOR_TRAVEL and not committed and tracked_at is not None:
                    want, range_geo = self._relative_to(tracked_at)
                    along = (
                        self._corridor_bearing(want)
                        if range_geo > CORRIDOR_TRAVEL_MIN_M
                        else None
                    )
                    off_axis = None
                    if along is not None:
                        off_axis = abs(((want - along) + math.pi) % (2 * math.pi) - math.pi)
                    if off_axis is not None and off_axis < HEAD_COMMIT_RAD:
                        if not corridor_travelling:
                            corridor_travelling = True
                            log.info("walking the corridor, watching the %s", target)
                            # Face down the corridor, then find its middle by feel: strafe
                            # toward whichever side the cameras say has more room, until both
                            # report the same. The head faces forward for this, not the door
                            # -- the side cameras hang off the head, so a head turned toward
                            # the door points "left" down the corridor and "right" at the
                            # floor ahead -- and looking along the direction of travel while
                            # moving is no vice anyway. The door's position is anchored in the
                            # world; look_at re-pins the gaze the moment the middle is found.
                            self._turn_body_to(along)
                            self.robot.face_forward()
                            imbalance = 0.0
                            for i in range(CENTRE_MAX_STEPS):
                                if i % CENTRE_LOOK_EVERY == 0:
                                    left, right = self.robot.side_clearance(
                                        above_horizon=True
                                    )
                                    imbalance = left - right
                                    if abs(imbalance) < CENTRE_TOLERANCE_M:
                                        break
                                self.robot.step(0.0, math.copysign(0.25, imbalance), 0.0)
                            self.robot.stand(0.2)
                            self.robot.look_at(tracked_at)
                            log.info("centred in the corridor")
                            continue
                        steer_to = along
                    elif corridor_travelling:
                        # Level with the door now. Turn the body onto it before closing in: the
                        # last metre drops the forward clearance below HEAD_YIELD_M, which snaps
                        # the head square to the body, and a body still on the corridor line is
                        # pointing at the wall when that happens -- measured losing the door at
                        # 1.0 m every time.
                        corridor_travelling = False
                        # Once turned in, stay in: re-entering corridor travel from here swings
                        # the body back off the door and the pair of them oscillate, arriving
                        # further out each time -- measured three turn-ins and a final 0.77 m
                        # against 0.65 m going straight in, which is the difference between
                        # reaching the leaf and pushing at air.
                        committed = True
                        log.info("level with the %s, turning in", target)
                        self._turn_body_to(want)
                        self.robot.look_at(tracked_at)
                        continue

                turn = float(np.clip(steer_to * self.turn_gain, -1.2, 1.2))
                turn_last = turn
                scale, turn = self._avoid(
                    space, turn, target_bearing=bearing, target_distance=distance
                )

                if scale == 0.0 and turn == 0.0:
                    return NavResult(
                        NavState.BLOCKED, target, distance=distance, bearing=bearing, steps=steps
                    )

                # Slow down on approach, and while turning hard, so it does not overshoot.
                closing = min(1.0, max(0.25, (distance - self.arrive_distance) / 1.5))
                straightness = max(0.3, 1.0 - abs(turn))
                speed = self.cruise_speed * scale * closing * straightness
                # The gait has a dead zone: commanded 0.14 m/s it stands still against its
                # own stance friction (watched it parked for two thousand steps at exactly
                # that command). Three well-meant throttles multiplied together land right in
                # it. Walking slowly is a speed, not a fraction -- clamp to the slowest walk
                # that actually walks.
                if speed > 0.02:
                    speed = max(speed, MIN_WALK_SPEED)

                # Glance sideways as well as forward. The forward camera cannot see a wall the
                # robot is walking alongside -- that wall sits at 90 degrees, outside its
                # 75-degree view -- so an approach happily grinds along one: measured 675 steps
                # against the north wall with 0.17 m of room on the left for a body 0.455 m
                # wide. Reading the side cameras on the approach costs two renders and changes
                # what the loop sees, which is enough on its own to stop the robot ending up
                # in gaps it does not fit through.
                if distance > SIDE_LOOK_DISABLE_M:
                    self.robot.side_clearance()

                # Hold a gap off the walls on either side while closing in.
                #
                # The forward camera cannot see the flank the body is travelling on, and the
                # width check above stops at CLEARANCE_DISABLE_M -- so the last two metres of
                # an approach have nothing watching the sides at all. That is exactly where
                # the scraping is: measured 63 of 86 contact steps on one approach against
                # the north wall, between 2.0 m and 3.5 m along it, all of them inside the
                # width check's blind spot.
                #
                # Only when the head is roughly forward. The side cameras hang off it, so a
                # neck turned toward a door aims "left" down the corridor and "right" at the
                # floor; steering on that was measured making things worse. Straightening the
                # head is not an option either -- it is what keeps the door in sight.
                # Not on the last stretch. A doorway is 1.1 m against a 0.67 m body, so from
                # close up the target itself reads as walls on both sides -- steering off them
                # there is steering away from the door. Left running to the end, this
                # zeroed the approach's contact and then failed every errand in the suite,
                # 50% contact and nothing completed, by refusing to enter any opening.
                drift = 0.0
                if abs(self.robot.head_yaw) < SIDE_STEER_HEAD_RAD:
                    left, right = self.robot.side_clearance()
                    want = self.robot.half_width + SIDE_STEER_MARGIN_M
                    encroach = max(0.0, want - right) - max(0.0, want - left)
                    drift = float(np.clip(encroach * SIDE_STEER_GAIN, -0.25, 0.25))

                # Touch overrides the cameras, on the same command rather than in place of it.
                #
                # This is where the scraping survived every camera-based fix: measured five
                # contact episodes on one approach, all between x=2.4 and x=3.0 with the body
                # 0.33 m off a wall it is 0.336 m wide -- short by centimetres, with the head
                # turned toward the door and nothing looking at the flank. The reflex elsewhere
                # in this loop handles a body already stuck; this handles a body still walking,
                # by folding "something is touching my left" into the sideways term the
                # approach was going to use anyway. One command, one place, no second
                # controller to fight the first.
                touching = self.robot.wall_contact_side()
                if touching is not None:
                    drift = float(np.clip(drift - touching * SIDE_TOUCH_PUSH, -0.3, 0.3))
                self.robot.step(speed, drift, turn)
                # Re-measure the distance we are closing on next refresh.
                # The decrement assumes the commanded speed closed the gap, which is fiction
                # when the gait is stalled -- watched the "remaining" tick from 1.43 m to
                # 0.16 m while the robot stood pinned in one spot. Re-measuring against the
                # tracked position instead was tried, and it shifted the plain approach's
                # trajectory enough to land at the wrong door; the stuck watchdog above now
                # owns the stall case, so the optimistic estimate is tolerable again.
                last_seen = (located[0], bearing, max(0.0, distance - speed * self.robot.control_dt))
                continue

            if state is NavState.DETOUR:
                detour_steps += 1

                # Give up detouring rather than orbiting the building forever.
                if detour_steps > DETOUR_MAX_STEPS:
                    log.info("detour exhausted, searching again")
                    state = NavState.SEARCH
                    searched = 0
                    last_seen = None
                    tracked_at = surveyed.copy() if surveyed is not None else tracked_at
                        # Keep the last anchor when there is no survey to reset to.
                        # The door has not moved; forgetting where it was flips the
                        # loop onto the assumed-progress estimate, which once counted
                        # its way down from 6.9 m to an "arrival" 8.4 m from the door.
                    continue

                # Rejoin the direct approach once the way ahead genuinely opens up -- but only
                # after committing for a while, or it flips between the two every other frame.
                ahead = space.clearance_ahead(half_angle=0.30)
                if detour_steps > DETOUR_MIN_STEPS and ahead > DETOUR_CLEAR_M:
                    if located is not None:
                        log.info("route ahead is clear, resuming approach")
                        state = NavState.APPROACH
                        last_seen = located
                        continue
                    # Target not in frame but the way is open. Keep the tracked world position:
                    # it names the door we chose, and a position does not stop being right just
                    # because the robot walked round a wall. (The qualifier alone would --
                    # "the rightmost door" means something different from the far side of the
                    # corridor -- which is why this used to discard it and end up at whichever
                    # door was nearest.)
                    state = NavState.SEARCH
                    searched = 0
                    last_seen = None
                    continue

                # Track the wall: steer toward the chosen side while keeping some clearance in
                # front, so the robot slides along the obstruction rather than into it.
                self.robot.step(*self._wall_follow(space, detour_side))

        self.robot.face_forward()
        return NavResult(NavState.LOST, target, steps=steps)

    def face(self, target: str, max_steps: int = 200, where: str | None = None) -> NavResult:
        """Turn to put the target dead ahead, without walking anywhere."""
        tracked: np.ndarray | None = None
        for step in range(max_steps):
            detections, _, fovy, size, depth = self._observe(target)
            if tracked is None:
                located = self._locate(detections, depth, fovy, size, where=where)
            else:
                located = self._locate(detections, depth, fovy, size, tracking=tracked)
            if located is None:
                self.robot.step(0.0, 0.0, 0.6)
                continue
            _, bearing, distance = located
            tracked = self._world_position(bearing, distance)
            if abs(bearing) < 0.06:
                self.robot.stand(0.2)
                return NavResult(
                    NavState.ARRIVED, target, distance=distance, bearing=bearing, steps=step
                )
            self.robot.step(0.0, 0.0, float(np.clip(bearing * self.turn_gain, -0.9, 0.9)))
        return NavResult(NavState.LOST, target, steps=max_steps)
