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
# Ranges became accurate once the depth buffer's axial distance was converted to true range,
# so the robot now stops where it actually intended to -- which turned out to be a couple of
# centimetres too far back to get through. 0.55 puts the hand on the leaf again.

# Half the clear width of a doorway, in metres. Used to aim for the side of an opening the
# door does not swing across.
DOORWAY_HALF_WIDTH_M = 0.55

# How fast to lean on a door, m/s. Well below walking pace: a door opening at speed is a
# hazard to anyone standing behind it, and nothing about the task needs it done quickly.
DOOR_PUSH_SPEED = 0.15

# How far open to aim for before walking through, radians. 1.15 is about 66 degrees; the
# comment here used to claim 75, which it never was.
#
# Treat this as the target for the push, not as proof of fit. The geometric requirement looks
# tighter than it is -- a 0.92 m leaf (assets/office.xml, half-size 0.46) leaves
# 0.92 * (1 - cos(swing)) of clear width, which for a 0.67 m body wants 74 degrees. But the
# leaf keeps swinging as the robot moves into it, so the gap at the moment of transit is wider
# than the angle at the moment of measurement: measured getting through on a 61-degree peak.
# Raising this to 1.31 to "fix" that only made the push run longer for no gain.
#
# Nor is the peak angle a usable pass/fail test. Rejecting anything under 40 degrees as "too
# narrow to fit through" looked reasonable and broke the errand: doors that the robot does get
# through peak below that, and the run failed later, in leave(). Whether the robot fits is a
# question about the gap at the moment it steps through, which this number does not answer.
DOOR_OPEN_ENOUGH_RAD = 1.15

# Two door sightings this far apart in world heading are different doors. The office doors are
# 4 m apart, so from anywhere in the corridor they are tens of degrees apart.
COUNT_SEPARATION_RAD = 0.35

# Two bearings closer than this name the same door when checking which one is being faced.
SAME_DOOR_RAD = 0.4

# Sightings closer together than this are edges of one door, not separate doors. The office
# doors are 4 m apart and a leaf is about 1 m wide, so this sits comfortably between the two.
DISTINCT_DOORS_M = 2.5

# How many times to walk back and try again after finding the wrong door.
WRONG_DOOR_RETRIES = 2
PUSH_STANDOFF_M = 0.55


@dataclass
class SkillResult:
    """Outcome of one skill, in a form that can be messaged to a human."""

    ok: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    # Whether a failure should stop the rest of the plan.
    #
    # Most failures should: later steps assume the earlier ones worked, and a robot that could
    # not pick the towel up has no business trying to fold it. But some are simply "I cannot do
    # that particular thing" -- asking a robot to carry a basket that is bolted to the floor --
    # and abandoning a five-step errand over one impossible aside is the wrong response.
    # Skills set this False to say "note it and carry on".
    fatal: bool = True

    def __str__(self) -> str:
        return self.message


class Skills:
    """The robot's action repertoire."""

    def __init__(
        self,
        robot: Robot,
        grounder: Grounder,
        navigator: MaplessNavigator | None = None,
        planner: object | None = None,
    ):
        self.robot = robot
        self.grounder = grounder
        self.nav = navigator or MaplessNavigator(robot, grounder)
        # Optional. When it can think about a choice -- LLMPlanner can -- the robot asks it
        # before doing something irreversible. Anything without check_choice is ignored.
        self.planner = planner
        # Where the robot stood just before pushing through a doorway, so it can get back out.
        # Navigation is otherwise memoryless, and a room is a dead end without this: from
        # inside, the open door shows only its edge and no other door is visible at all, so
        # there is nothing for a purely reactive search to home in on. One remembered pose is
        # a much smaller concession than building a map.
        self._doorway_return: np.ndarray | None = None
        # Heading the robot had when it went through, so leaving can line up on the reverse.
        self._doorway_heading: float | None = None
        # Corridor-side spots of every doorway this robot has personally been through. A door
        # it opened itself may later be invisible -- the leaf settles ajar, angled into its
        # own doorway, and an oblique view sees only the recess -- but the robot's own history
        # is first-hand: there IS a door there, it walked through it. Used to reconcile "you
        # said three" with a frame that shows two.
        self._doors_opened: list[np.ndarray] = []
        # Which way to turn next time a shoulder catches on a door frame; flips each attempt so
        # repeated snags do not walk the robot along the frame into the opposite jamb.
        self._loose_nudge = 1.0
        # Where the robot was when it was given its instructions. 「最初にいる位置からみて」 --
        # "from where you are standing now" -- makes that the frame of reference for left and
        # right, and it is also the one spot with a clear view of every door.
        self._home = robot.position[:2].copy()
        self._home_heading = robot.yaw

    # -- navigation ---------------------------------------------------------------

    def goto(
        self, target: str, where: str | None = None, expect: int | None = None
    ) -> SkillResult:
        """Walk to something the camera can find.

        `where` picks between identical candidates: "the door on the right".
        """
        if where in ("left", "right", "middle") and expect:
            seen = self._ensure_expected_in_view(target, expect)
            if seen < expect:
                return SkillResult(
                    False,
                    f"You said there were {expect} {target}s, but I can only see {seen} "
                    f"from here, so I am not sure which one you mean.",
                    {"expected": expect, "seen": seen},
                )

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

    def return_home(self, max_steps: int = 1600) -> SkillResult:
        """Go back to where the robot was standing when it got its instructions.

        This is the vantage point the user described things from -- 「最初にいる位置からみて、
        三つ見えるドアのうち」 -- so it is where "left" and "right" mean what they were meant to.
        It is also the only place all three doors are in frame at once: from beside a doorway
        only one is, which is why picking a second room used to open whichever was nearest.

        The step budget covers a walk across the whole office with detours -- getting back from
        the pantry takes about 1100 steps, and at the old 900 it stopped 0.9 m short.
        """
        for _ in range(max_steps):
            delta = self._home - self.robot.position[:2]
            distance = float(np.linalg.norm(delta))
            if distance < 0.4:
                break
            desired = math.atan2(delta[1], delta[0])
            error = (desired - self.robot.yaw + math.pi) % (2 * math.pi) - math.pi
            turn = float(np.clip(error * 1.4, -1.0, 1.0))

            if self._clearance_ahead() < 0.55 or self._touching_door():
                # Something in the way, most likely the door just opened. Turn to face home
                # first, then push through: creeping around it drove the robot further onto the
                # leaf (clearance 0.41 to 0.17 over 300 steps), and reversing blindly pushed it
                # back into the room it had just left.
                self._turn_to(desired, max_steps=80)
                for _ in range(20):
                    self.robot.step(0.3, 0.0, 0.0)
                    if not self._touching_door() and self._clearance_ahead() > 0.7:
                        break
                continue

            self.robot.step(0.45 * max(0.35, 1.0 - abs(turn)), 0.0, turn)

        self._turn_to(self._home_heading)
        # Straighten the waist. The head camera hangs off the torso, so a waist left at -23
        # degrees points the view 23 degrees away from wherever the body is facing -- which is
        # why standing on the exact starting spot, facing the exact starting heading, showed
        # one door where it had shown three.
        self._straighten_waist()
        self.robot.stand(0.3)

        distance = float(np.linalg.norm(self._home - self.robot.position[:2]))
        seen = self._count_doors()
        if distance < 0.8:
            return SkillResult(True, "I went back to where I started.", {"doors_visible": seen})
        return SkillResult(
            False,
            f"I could not get back to where I started ({distance:.1f} m short).",
            {"doors_visible": seen},
        )

    def _straighten_waist(self, settle: float = 0.4) -> None:
        """Point the head camera where the body is facing.

        The waist drifts during a manoeuvre and does not come back on its own -- measured
        sitting at -30 degrees, which aims the camera 30 degrees off the body's heading and
        quietly changes what "left" means. Its servo is deliberately weak (kp=1) so the torso
        stays compliant while walking; stiffening it to correct this broke the gait badly
        enough to fail four tests, so the joint is reset directly instead.
        """
        import mujoco  # noqa: PLC0415 - only needed for this introspection

        joint = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_JOINT, "waist_yaw")
        if joint < 0:
            return
        adr = self.robot.model.jnt_qposadr[joint]
        self.robot.data.qpos[adr] = 0.0
        self.robot.data.qvel[self.robot.model.jnt_dofadr[joint]] = 0.0
        idx = self.robot._act.get("waist_yaw")  # noqa: SLF001 - Skills owns this robot
        if idx is not None:
            self.robot.data.ctrl[idx] = 0.0
        mujoco.mj_forward(self.robot.model, self.robot.data)
        self.robot.stand(settle)

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

    def _count_by_looking_around(self) -> int:
        """How many distinct doors the robot can find by turning its head, without moving.

        Counts by direction in the world, not by position: a bearing plus the robot's own
        heading is enough to say "that is a different door from the last one", and it avoids
        the range errors that made an earlier position-based count inflate to four doors in a
        room with one. Two sightings more than 15 degrees apart are different doors.
        """
        return len(self.headings_of_doors())

    def _walk_to_surveyed(self, where: str) -> bool:
        """Walk to the door the qualifier names, using the positions the survey just found.

        Returns False if there is nothing usable to walk to.
        """
        seen = self.nav.survey("door")
        if len(seen) < 2:
            return False
        chosen = self.nav._pick_bearing(seen, where)
        if chosen is None:
            return False
        self.nav._walk_to(self.nav._world_position(*chosen))
        return True

    def _at_the_right_door(self, where: str) -> bool | None:
        """Whether the door in front is the one the qualifier names.

        Looks around rather than trusting the forward frame: from right up against a door the
        others are outside a 75-degree view, so "is this the leftmost" cannot be answered
        without turning the head. Returns None when the question cannot be settled -- one door
        visible means there is nothing to compare against, and refusing to act on that would
        block every approach that ends up correctly alone in front of its target.
        """
        # Settle first. The check runs straight after an approach, which leaves the head
        # turned toward whatever it was tracking and the body still rocking; a survey started
        # from there sees a different set of doors than the same spot does at rest, and the
        # verdict flipped between runs because of it.
        self.robot.face_forward()
        self.robot.stand(0.5)

        seen = self.nav.survey("door")
        if len(seen) < 2:
            return None

        # Up against a door, its two edges read as two separate sightings -- measured (4.80,
        # 0.57) and (3.54, 0.91) from 0.9 m away, which is one door 1.3 m wide, not two doors.
        # Comparing "leftmost" against that says the robot is at the wrong door when it is
        # exactly where it should be. Only judge when the sightings are far enough apart in the
        # world to be different doors.
        positions = [self.nav._world_position(b, d) for b, d in seen]
        spread = max(
            float(np.linalg.norm(a - b_)) for a in positions for b_ in positions
        )
        if spread < DISTINCT_DOORS_M:
            return None

        bearings = sorted(b for b, _ in seen)
        wanted = {"left": bearings[-1], "right": bearings[0],
                  "middle": bearings[len(bearings) // 2]}[where]

        # The door about to be opened is the nearest one, not the one closest to straight
        # ahead. Standing 0.38 m to the side of the middle door, with the left one 4.65 m off,
        # the left one is nearer the centre of the frame -- so judging by bearing alone said
        # "yes, this is the left door" about a door the robot was pressed against.
        nearest = min(seen, key=lambda s: s[1])[0]
        verdict = abs(nearest - wanted) < SAME_DOOR_RAD

        # Ask the planner too, if there is one that can think about it. This is a standstill --
        # the robot has stopped, and is about to do something it cannot undo -- so a second
        # opinion is worth the second it costs. The model is given the same bearings in the
        # same terms the instruction used, and only gets to veto a "yes": a disagreement means
        # going back for another look, which is cheap, while overriding a geometric "no" would
        # let a hallucinated answer open the wrong door.
        thinking = getattr(self.planner, "check_choice", None)
        if verdict and thinking is not None:
            second = thinking(where, [math.degrees(b) for b in bearings])
            if second is False:
                log.info("on reflection, this may not be the %s door", where)
                return False
        return verdict

    def headings_of_doors(self) -> list[float]:
        """World headings of every door found by turning the head, left to right.

        Counting by direction rather than by position: a bearing plus the robot's own heading
        says "that is a different door from the last one" without needing a range, which is
        what an earlier position-based count got wrong -- range errors inflated it to four
        doors in a room with one.

        Measured from the corridor this finds all three at +21.0, +91.4 and +158.9 degrees
        against true values of +21.8, +90.0 and +158.2, from a spot where a single forward
        frame sees only one of them.
        """
        found: list[tuple[float, float]] = []  # heading, confidence
        for angle in (-1.0, -0.5, 0.0, 0.5, 1.0):
            self.robot.turn_head_to(angle)
            for det in self.grounder.find(self.robot.look().rgb, "door"):
                if det.confidence < 0.25:
                    continue
                bearing = self.robot.bearing_to_pixel(det.x * self.robot.camera_width)
                heading = self.robot.yaw + self.robot.head_yaw + bearing
                match = next(
                    (i for i, (h, _) in enumerate(found)
                     if abs(heading - h) < COUNT_SEPARATION_RAD),
                    None,
                )
                if match is None:
                    found.append((heading, det.confidence))
                elif det.confidence > found[match][1]:
                    found[match] = (heading, det.confidence)
        self.robot.face_forward()
        return sorted(h for h, _ in found)

    def _ensure_expected_in_view(self, target: str, expect: int, tries: int = 3) -> int:
        """Get to somewhere the stated number of objects is actually visible.

        The user said how many there are -- 「三つ見えるドアのうち」 -- and that is checkable.
        If only one door is in frame, "the leftmost" resolves against that one and the robot
        confidently opens the wrong thing. Better to go and look properly first.

        Returns how many ended up visible, which may still be short.
        """
        # Counts what is in the forward frame, deliberately not what turning the head can
        # find. The head does see more -- from the corridor it picks out all three doors at
        # +21.0, +91.4 and +158.9 degrees, against truth of +21.8, +90.0 and +158.2, from a
        # spot where the forward camera sees one. But the navigator still chooses from the
        # forward frame, and counting three while choosing from one is worse than not counting:
        # told "the left door" it confidently opened the middle one and reported success,
        # finishing at x=+0.18 for a door at x=-4.0. headings_of_doors is the piece that would
        # close this, once choosing can use it.
        seen = self._count_doors()
        if seen >= expect:
            log.info("counted %d %ss, as stated; choosing between them", seen, target)
            return seen

        log.info("expected %d %ss in view, can see %d; repositioning", expect, target, seen)
        self.return_home()
        self._straighten_waist()
        # Settle before counting. The count is taken from one frame, and a frame grabbed
        # while the body is still rocking from the walk clips a door at the edge of view,
        # so the robot concludes it can only see two and gives up on a scene where all
        # three are plainly there.
        self.robot.stand(0.8)
        seen = self._count_doors()
        log.info("look 1: counted %d %ss", seen, target)
        if seen >= expect:
            return seen

        # Still short: step BACK for a wider view, rather than recounting from the same spot.
        # return_home stops anywhere within its arrival tolerance, and a home 0.27 m nearer
        # the doors than usual (measured (0.26, -2.73) against the (0, -3.0) the count was
        # tuned at) pushes the outer doors past the edge of the frame -- from there the count
        # reads 2 however many times it is retaken. Backing up is what a person does when a
        # row of doors will not fit in view. Half a metre per look, twice at most, keeps well
        # clear of the wall behind (measured 1.27 m of corridor behind the worst home pose).
        for attempt in range(1, tries):
            for _ in range(int(0.5 / (0.25 * self.robot.control_dt))):
                self.robot.step(vx=-0.25)
            self.robot.stand(0.6)
            seen = self._count_doors()
            log.info("look %d: counted %d %ss (from further back)", attempt + 1, seen, target)
            if seen >= expect:
                return seen

        return seen

    def open_door(
        self, target: str = "door", side: str = "r", where: str | None = None,
        expect: int | None = None,
    ) -> SkillResult:
        """Walk up to a door and push it open.

        A push, not a handle turn. Pushing needs the hand somewhere on the leaf rather than
        precisely on a 3.6 cm handle, so it survives the pose error that navigation leaves
        behind - and it is what a person does to an unlatched office door anyway.

        The sequence: get close, square up, reach out at handle height, walk into the door so
        the arm loads it, then check the hinge actually moved.
        """
        # Which room this whole skill started from, for the crossing check at the end. Taken
        # at the very top: both the approach and the push can carry the robot over the
        # threshold on their own -- measured the approach drifting into the pantry through
        # the part-open door, so a check anchored any later called a successful entry a
        # failure to get through, from inside the room it had supposedly not entered.
        began_in = self.report_position().data.get("room")

        # A spatial qualifier is anchored to where the user was describing from, which is where
        # the robot was standing when it was told. Go back there before choosing: from beside a
        # doorway only one door is in frame, so "the left one" would mean whichever it is next
        # to. When the user also said how many there are, check that too and refuse rather than
        # guess -- opening the wrong door confidently is worse than saying you cannot tell.
        approach = None
        if where in ("left", "right", "middle"):
            wanted = expect or 3
            seen = self._ensure_expected_in_view(target, wanted)
            if seen < wanted:
                # The forward frame will not show the stated number -- but the frame is not
                # the only way to confirm the scene. A door this robot has already been
                # through stands ajar, and an ajar leaf seen obliquely through its own
                # doorway is a sliver the colour match misses: measured counting 2 of 3 from
                # every spot along home after opening the right-hand door. Turning the head
                # finds all three (the ajar one gets looked at square-on), and once the
                # count is confirmed, choose from the survey too -- then walk to the chosen
                # one and let the wrong-door check below judge the result as usual.
                # The stated number may still be accounted for. Doors found by turning the
                # head are one source; doorways this robot has personally walked through are
                # another, and that history is first-hand. (The landmark map was tried here
                # and reverted: it carries phantoms -- one logged 90 sightings in the middle
                # of the corridor -- so "the leftmost remembered door" could name a spot on
                # open floor.) A door the robot opened settles ajar, angled into its own
                # doorway, and vanishes from an oblique view; its doorway has not moved.
                sightings = self.nav.survey(target)
                spots = [(b, self.nav._world_position(b, d)) for b, d in sightings]
                for p in self._doors_opened:
                    if all(
                        float(np.linalg.norm(p - q)) >= DISTINCT_DOORS_M / 2
                        for _, q in spots
                    ):
                        spots.append((self.nav._relative_to(p)[0], p))
                chosen = (
                    self.nav._pick_bearing(spots, where) if len(spots) >= wanted else None
                )
                if chosen is None:
                    return SkillResult(
                        False,
                        f"You said there were {wanted} {target}s, but I can only see {seen} "
                        f"from here, so I am not sure which one you mean.",
                        {"expected": wanted, "seen": seen},
                    )
                log.info(
                    "%d of %d %ss in the frame, but counting the ones I have been through "
                    "accounts for all of them; choosing among those", seen, wanted, target,
                )
                # The default step budget covers about 4.5 m and the far door is 5.5 from
                # home: walked with the default, the robot stalled midway, and the goto that
                # followed locked onto a door 5.5 m in the other direction. Walk with room to
                # spare and face the chosen spot before letting goto pick a door, so the one
                # it picks is the one just walked to.
                self.nav._walk_to(chosen[1], max_steps=1500)
                self.nav._turn_body_to(self.nav._relative_to(chosen[1])[0])
                approach = self.nav.goto(target)

        # Stop within arm's length. The arm reaches ~0.43 m in front of the base at handle
        # height (measured by sweeping the shoulder/elbow range), so the default 0.85 m
        # stand-off leaves the hand half a metre short of the door.
        if approach is None:
            approach = self.nav.goto(target, where=where)
        if not approach.success:
            return SkillResult(False, approach.describe())

        # Check before committing. Arriving somewhere is not the same as arriving at the door
        # that was asked for: told the leftmost of three, the robot has finished at the middle
        # one and opened it, reporting success. Standing still and looking around is cheap
        # compared with opening the wrong door, and from here the answer is unambiguous --
        # the target should be the leftmost/rightmost/middle thing in view.
        if where in ("left", "right", "middle"):
            for attempt in range(WRONG_DOOR_RETRIES):
                verdict = self._at_the_right_door(where)
                if verdict is None or verdict:
                    break
                log.info("this is not the %s %s; going back for another look", where, target)
                # Walk to the door itself rather than restarting from home. The survey has just
                # located every door from here, so the one that was asked for has a position --
                # going back to the start and re-approaching throws that away and repeats the
                # same mistake. Measured ending 0.64 m past the left door this way, against
                # opening the middle one and calling it the left.
                if self._walk_to_surveyed(where):
                    # Standing at the right door now, so approach the nearest one rather than
                    # re-applying the qualifier. "The left door" means something different from
                    # here -- the robot is past it, and asking for the leftmost thing in view
                    # sends it away again: measured stalling 0.64 m from the door for the rest
                    # of the run, where a plain approach reaches it.
                    approach = self.nav.goto(target)
                else:
                    self.return_home()
                    self._straighten_waist()
                    approach = self.nav.goto(target, where=where)
                if not approach.success:
                    return SkillResult(False, approach.describe())
            else:
                # Every retry used up and the check still says this is the wrong door. Opening
                # it anyway is the failure the check exists to prevent: the robot ended up at
                # the middle door, was told twice that it was the middle door, and opened it.
                if self._at_the_right_door(where) is False:
                    return SkillResult(
                        False,
                        f"I could not get to the {where} {target} - I kept ending up at a "
                        f"different one, so I have not opened anything.",
                        {"where": where},
                    )

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

        # Square up to the doorway before closing in. Approaching a door at an angle -- the
        # left one is reached at 167 degrees for an opening that faces 90 -- means pushing
        # through sideways, and a body turned 80 degrees across a 0.98 m gap does not fit.
        # All the doorways are on the north wall, which is the one piece of layout this uses.
        self._turn_to(math.pi / 2 if self.robot.position[1] < 1.0 else -math.pi / 2)

        # Close the remaining gap until the door is within reach.
        for _ in range(140):
            space = self._clearance_ahead()
            if space <= PUSH_STANDOFF_M:
                break
            self.robot.step(vx=0.3)
        self.robot.stand(0.3)

        # Line up on the middle of the OPENING before anything else. Arrival promises only
        # "within arm's reach of the tracked position", which can just as well be in front of
        # the frame or the wall beside the opening: measured squaring up there and spending
        # the whole stroke pushing a blameless wall, reported as "it may be locked". Not on
        # the handle, though that was the first try -- the handle hangs at the leaf's free
        # edge, 0.19 m from the frame post, and a body centred there puts a shoulder into the
        # post on the way through (opened 77 degrees and could not follow). The opening's
        # middle clears both posts and the arm reaches the leaf from anywhere across it.
        hinge = self._nearest_hinge_x()
        if hinge is not None:
            # Every doorway here spans about a metre from its hinge toward +x.
            centre = np.array([hinge + DOORWAY_HALF_WIDTH_M, self.robot.position[1]])
            left_axis = np.array([-math.sin(self.robot.yaw), math.cos(self.robot.yaw)])
            for _ in range(60):
                lateral = float((centre - self.robot.position[:2]) @ left_axis)
                if abs(lateral) < 0.05:
                    break
                self.robot.step(0.0, float(np.clip(lateral * 1.5, -0.3, 0.3)), 0.0)
            self.robot.stand(0.3)

        # Remember where we are STANDING, which is the corridor side of the threshold -- that
        # is where leaving has to get back to. Recording the doorway itself (one arm's length
        # ahead) put the target inside the room: measured (4.54, 1.27) for a doorway at y=1.0,
        # so "returning" to it never left the pantry.
        self._doorway_return = self.robot.position[:2].copy()
        self._doorway_heading = self.robot.yaw

        angle_before = self._door_angle()

        # Push with the arm on the hinge side, so the other one stays clear of the jamb as the
        # robot walks through. The handle is on the far edge from the hinge, so reaching across
        # with the near arm keeps the body out of the opening.
        handle_offset = self._handle_offset(side)
        if handle_offset is not None and handle_offset > 0.15:
            side = "l"  # handle is to the left; use the left arm
        elif handle_offset is not None and handle_offset < -0.15:
            side = "r"

        # Best forward reach at handle height, found by sweeping the joint ranges:
        # shoulder pitch -1.10, elbow -0.20 puts the gripper 0.43 m ahead at z=1.03.
        self.robot.set_arm(side, shoulder_pitch=-1.10, shoulder_roll=0.0,
                           shoulder_yaw=0.0, elbow=-0.20)
        self.robot.grip(side, 0.35)
        self.robot.stand(0.6)

        # Push: keep walking forward so the extended arm loads the door. Deliberately slow.
        # There may be somebody on the other side, and a door that swings open at walking pace
        # is how you hit them; a person opening a door they cannot see through leans on it
        # gently and gives whoever is behind it time to notice. Same total travel, taken over
        # more than twice as long.
        # Twice the travel, because a slow push loses ground to the door's own spring: the
        # same distance walked at 0.15 m/s instead of 0.35 swings the door 67 degrees rather
        # than 109. Walking further at the slow speed reaches the same 109.
        opened_to = angle_before
        # The widest the leaf got at any point, not just where it ended up. The hinge has a
        # spring (assets/office.xml, class "door": stiffness 0.8), so the moment the body
        # stops bearing on the leaf it starts closing again -- measured swinging to 61 degrees,
        # carrying the robot 1.26 m into the room, and reading 10 degrees by the time the walk
        # finished. Judging on the final angle called that a door that never opened.
        widest = angle_before
        for _ in range(int(2 * 160 * 0.35 / DOOR_PUSH_SPEED)):
            self.robot.step(vx=DOOR_PUSH_SPEED)
            # Stop as soon as it is open enough to walk through. Pushing on past that just
            # grinds the robot into the frame for the rest of the stroke, and taking the push
            # slowly made that stretch more than twice as long.
            opened_to = self._door_angle()
            if abs(opened_to - angle_before) > abs(widest - angle_before):
                widest = opened_to
            if abs(opened_to - angle_before) > DOOR_OPEN_ENOUGH_RAD:
                break

        # No pushing past the threshold to park the leaf wide, though it is tempting -- a leaf
        # left near the threshold settles ajar and half-hides in its own doorway. Tried, and
        # it backfired: a leaf parked at 109 degrees stands square across the hinge half of
        # the opening, and the robot, entering on that half, walked into its edge and could
        # not get through at all. The leaf part-open is what funnels the body through.

        # Judge on how far the door was got open, not on how much THIS push added. Squeezing
        # past a door on the way to it can already have swung it (a detour nudged the pantry
        # door 35 degrees open before the arm ever touched it), and measuring only the delta
        # then reports a door standing wide open as "it did not open".
        swing = max(
            abs(math.degrees(widest)),
            abs(math.degrees(widest - angle_before)),
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
        # Walking blind, though, means a shoulder that catches the jamb just grinds there for
        # the rest of the push, which is what wedged the robot half in the opening. So drive
        # forward but back off and re-angle whenever it actually snags.
        self._push_through(90)
        # Then continue only while there is room, so it does not end up wedged in a corner.
        for _ in range(75):
            if self._clearance_ahead() < 0.9:
                break
            self.robot.step(vx=0.45)

        # The walk above is blind about rooms, and "went through" is checkable -- pushed from
        # a stance one step further back, it finished at y=0.89, eleven centimetres short of
        # the doorway line, and still reported success; leave() then found itself already in
        # the corridor and failed on a room it had never entered. Creep forward until the room
        # actually changes, and if it will not, say so instead of claiming it did.
        #
        # No clearance test here, deliberately: what reads as an obstacle a hand-width ahead
        # is the part-open leaf, which yields when leaned on -- breaking on it stranded the
        # robot at 66 degrees with the doorway centimetres away, which is exactly the trap
        # the comment above the fixed-distance walk describes. The arm is still out and the
        # speed is a lean, so a real wall just ends the creep at the same spot when the
        # budget runs out. The budget is sized generously: 80 steps commands 0.48 m, the
        # gait delivers less, and starting half a metre from the threshold that finished at
        # y=0.98 -- two centimetres short, every time.
        for _ in range(200):
            if self.report_position().data.get("room") != began_in:
                break
            self.robot.step(vx=0.3)
        self.robot.arm_home(side)
        self.robot.stand(0.4)

        if self.report_position().data.get("room") == began_in:
            return SkillResult(
                False,
                f"I opened the {target} (it swung {swing:.0f} degrees) but could not get "
                f"through the doorway.",
                {
                    "swing_degrees": swing,
                    # Where it gave up, so a failed run says which part of the opening the
                    # body was pressed against rather than leaving that to guesswork.
                    "stuck_at": [round(float(self.robot.position[0]), 2),
                                 round(float(self.robot.position[1]), 2)],
                    "stuck_heading_deg": round(math.degrees(self.robot.yaw), 1),
                },
            )

        # Crossing verified: remember this doorway first-hand, once per door.
        if self._doorway_return is not None and not any(
            float(np.linalg.norm(self._doorway_return - p)) < DISTINCT_DOORS_M / 2
            for p in self._doors_opened
        ):
            self._doors_opened.append(self._doorway_return.copy())

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

        # Tuck the arms in before going anywhere near the opening. An outstretched arm is the
        # widest thing on the robot: measured half-width 0.496 m against 0.334 m with the arms
        # down, so a 0.67 m body becomes a 0.99 m one trying to fit a 1.10 m doorway. That is
        # why it snags on the frame and cannot get clear -- the gap it is aiming for is barely
        # wider than it has made itself.
        self._tuck_arms()

        target = self._doorway_return.copy()

        # Aim for the far side of the opening from the hinge. A door pushed to 109 degrees does
        # not vanish -- the leaf stands square across the jamb it is hinged on, so the half of
        # the opening nearest the hinge is blocked by its own door. Retracing the way in leads
        # straight into that half: measured the robot grinding at x=-0.40, ten centimetres from
        # a hinge at x=-0.50, making a centimetre of progress per attempt for 398 attempts. The
        # other half of the same doorway is clear.
        hinge = self._nearest_hinge_x()
        if hinge is not None:
            away = 1.0 if target[0] >= hinge else -1.0
            target[0] = hinge + away * DOORWAY_HALF_WIDTH_M * 0.75

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
        squared_up = 0
        for _ in range(600):
            delta = target - self.robot.position[:2]
            distance = float(np.linalg.norm(delta))
            # Only stop early once we are actually out. Reaching the remembered pose is not the
            # same as having left: the robot got within 0.09 m of the threshold and stopped
            # there, still inside, with the spring closing the door on it.
            if distance < 0.25 and self.report_position().data.get("room") != started_in:
                break
            desired = math.atan2(delta[1], delta[0])
            error = (desired - self.robot.yaw + math.pi) % (2 * math.pi) - math.pi

            # Square up before moving when badly misaligned. Steering while walking is fine for
            # small errors, but in a doorway a large one never converges: the forward component
            # carries the robot sideways across the opening faster than the turn corrects it,
            # so it creeps along the threshold instead of through it. Measured stalled at
            # 0.18 m short of the corridor, facing 65 degrees off, for the full step budget --
            # and simply turning to face the way out cleared it in 40 steps.
            if abs(error) > 0.5:
                # Square up on the way *out*, not on the remembered spot. The two differ: the
                # return pose sits a little to one side, and turning to face it can point the
                # robot straight at the leaf. The reverse of the heading it came in on always
                # points through the opening, which is the one direction guaranteed to be a way
                # out of the room.
                exit_heading = desired
                if self._doorway_heading is not None:
                    exit_heading = self._doorway_heading + math.pi
                # Cap the corrections. Each one is expensive, and needing many of them means
                # the heading is not the problem -- carrying on walking is better than turning
                # forever a few centimetres from the corridor.
                if squared_up < 4:
                    squared_up += 1
                    self._turn_to(exit_heading, max_steps=60)
                    continue

            turn = float(np.clip(error * 1.4, -0.9, 0.9))
            self.robot.step(0.35 * max(0.4, 1.0 - abs(turn)), 0.0, turn)

            # Fouling the leaf. The spring is closing the door onto the robot -- measured
            # going from 21 to 13 degrees while it stood in the gap -- so the answer is to
            # hold the door open, not to squeeze past a shrinking opening. Put an arm out
            # against it and keep walking; the leaf gives way and the robot goes through.
            if self._touching_door():
                self._hold_door_open()
            elif self.robot.arm_is_blocked():
                # An arm has fouled something -- a desk edge, the frame. Left alone the robot
                # keeps pushing against it and stops making ground: measured stalled at y=1.04
                # with the doorway 4 cm away, arm jammed, for the rest of the step budget.
                # Pull the arms in, and if the contact still holds them out, turn slightly to
                # slide the arm off whatever it is caught on. Turning rather than reversing:
                # backing off mid-exit gives up the ground the loop is spending its budget to
                # win, and every variant that retreated here cost two other rooms, while a turn
                # keeps the robot where it is.
                self._narrow_arms()
                if self.robot.arm_is_blocked():
                    for _ in range(10):
                        self.robot.step(wz=self._loose_nudge * 0.4)
                    self._loose_nudge = -self._loose_nudge
                    self._narrow_arms()
            elif self.robot.is_touching("wall"):
                # Caught on the frame rather than the leaf. Holding the door open does nothing
                # for this -- the obstruction is the jamb against a shoulder -- so back off and
                # come at the gap from a slightly different angle, as when pushing in.
                self._work_loose()

        # A last straight push if the loop ran out while still inside. The steering loop stops
        # making ground once the heading error sits just under its squaring-up threshold --
        # measured 4 cm short of the opening, 23 degrees wide, walking sideways along the
        # threshold. Facing the way out and walking is all that is needed from there.
        if self.report_position().data.get("room") == started_in:
            exit_heading = (
                self._doorway_heading + math.pi
                if self._doorway_heading is not None
                else self.robot.yaw
            )
            self._turn_to(exit_heading)
            for _ in range(120):
                if self.report_position().data.get("room") != started_in:
                    break
                if self._touching_door():
                    self._hold_door_open()
                self.robot.step(0.35)

        # Get clear of the leaf before finishing. Ending the manoeuvre still in contact leaves
        # whatever runs next -- return_home, another door -- starting from 0.17 m of clearance
        # with nowhere to go.
        for _ in range(40):
            if not self._touching_door() and self._clearance_ahead() > 0.7:
                break
            # Never reverse back into the room. This step exists to get off the leaf, but it
            # reverses along the way the robot came, which is through the doorway: measured
            # reaching the corridor at y=0.92 and being pushed back to y=1.04, inside, so the
            # errand reported failure after actually succeeding. Sidestep instead once out.
            if self.report_position().data.get("room") != started_in:
                self.robot.step(0.0, 0.3, 0.0)
                continue
            self.robot.step(-0.35, 0.25, 0.0)
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

    def _nearest_hinge_x(self) -> float | None:
        """World x of the hinge of whichever door is nearest, or None if there is no door.

        Read from the simulator. On a real robot this is what a glance at the door tells you --
        which edge it is attached to, and therefore which way it swings clear.
        """
        import mujoco  # noqa: PLC0415 - only needed for this introspection

        best_x: float | None = None
        best_distance = math.inf
        here = self.robot.position[:2]
        for name in ("door_workspace", "door_meeting", "door_pantry"):
            bid = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                continue
            hinge = self.robot.data.xpos[bid][:2]
            distance = float(np.linalg.norm(hinge - here))
            if distance < best_distance:
                best_distance = distance
                best_x = float(self.robot.model.body_pos[bid][0])
        return best_x

    def _tuck_arms(self) -> None:
        """Bring both arms in close, so the body is as narrow as it can be.

        Narrower than arm_home, which rests them slightly out from the body. Used before
        anything that has to fit through a gap.
        """
        # Back off first if an arm is pressed against anything. A pinned arm cannot be pulled
        # in -- the servo is commanded home but the contact holds it out, and the robot stays as
        # wide as it was: measured tucking from 0.420 m down to only 0.391 m instead of the
        # 0.299 m the same command reaches in free space. A short reverse unloads it.
        #
        # Not just doors. In the workspace it was a fingertip resting on a desk that held the
        # arm out, which a door-only test missed entirely.
        for _ in range(30):
            if not self.robot.arm_is_blocked():
                break
            self.robot.step(vx=-0.3)

        self._narrow_arms()

    def _narrow_arms(self) -> None:
        """Command both arms in against the body. Does not move the feet."""
        for side in ("l", "r"):
            # Roll toward the body, not away from it. arm_home rolls outward by 0.12, which is
            # what leaves the forearms as the widest part of the robot; the opposite sign pulls
            # them in against the ribs. Measured 0.334 m half-width at rest against 0.299 m
            # here, so tucking is worth 7 cm on each side of a doorway.
            sign = 1.0 if side == "r" else -1.0
            self.robot.set_arm(side, shoulder_pitch=-0.25, shoulder_roll=sign * 0.05,
                               shoulder_yaw=0.0, elbow=-0.35)
            self.robot.grip(side, 0.0)
        self.robot.stand(0.3)


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

    def _push_through(self, steps: int, speed: float = 0.45) -> None:
        """Drive forward through a doorway, working loose if the body catches on the frame.

        A doorway is barely wider than the shoulders, so arriving a few degrees off square puts
        one shoulder into the jamb. Pushing harder does not help -- the contact is sideways, and
        the robot simply grinds against the frame until the step budget runs out, stuck half in
        the opening with the door resting on its back.

        Backing off a little and turning slightly is what a person does here, and it works for
        the same reason: reversing breaks the contact, and the small turn means the next attempt
        presents a different angle. Alternating the turn direction matters -- always turning the
        same way just walks the robot along the wall into the other side of the frame.
        """
        taken = 0
        while taken < steps:
            if self.robot.is_touching("wall") or self.robot.is_touching("door"):
                self._work_loose()
                taken += 20
                continue
            self.robot.step(vx=speed)
            taken += 1

    def _work_loose(self) -> None:
        """Back off a caught shoulder and re-aim before trying the gap again.

        Alternates which way it turns. Always turning the same way just walks the robot along
        the frame into the other side of it, so the direction flips on each attempt.
        """
        for _ in range(12):
            self.robot.step(vx=-0.3)
        for _ in range(8):
            self.robot.step(wz=self._loose_nudge * 0.5)
        self._loose_nudge = -self._loose_nudge

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
        self,
        action: str,
        argument: str | None = None,
        where: str | None = None,
        expect: int | None = None,
    ) -> SkillResult:
        """Execute one planner-issued action."""
        if action in ("goto", "face", "open", "open_door", "point_at"):
            argument = self._normalise_target(argument)
        handlers = {
            "goto": lambda: self.goto(argument or "door", where, expect),
            "face": lambda: self.face(argument or "door", where),
            "open": lambda: self.open_door(argument or "door", where=where, expect=expect),
            "pull": lambda: self.pull_door(argument or "door"),
            "open_door": lambda: self.open_door(argument or "door", where=where, expect=expect),
            "point_at": lambda: self.point_at(argument or "door", where=where),
            "leave": lambda: self.leave_room(),
            "close": lambda: self.close_door(),
            "home": lambda: self.return_home(),
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
