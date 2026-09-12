"""What Momo can do with laundry.

The same shape as brain/skills.py: one method per verb the planner may emit, each returning a
SkillResult carrying a sentence fit to send back over Pyunto. Failing is normal and reported,
not raised -- "I could not reach the towel" is a good answer to send someone.

The task this exists for, in the order a person would describe it:

    open the washer -> take the towel out -> put it in the basket
                    -> put it on the counter -> fold it

Two things about this domain are different from the office and shape everything below.

**Reach is short.** The hand can be PUT within 0.25 m of the base, not the 0.40 m the joint
chain spans: past that the extended arm loads the torso, the body yields, and the hand settles
a fifth of a metre short however many correction passes are run. So every manipulation skill
here begins by walking to a standing spot, and the standing spot is computed from the target
rather than remembered. See sim/reach.WORKING_REACH_M for the measurements.

**The object is cloth.** A towel has no pose -- it has 63 vertex positions, and which one the
hand is near is the whole question. Grasping is a weld to one vertex (sim/cloth.py), so a skill
has to choose a vertex, and the choice is not arbitrary: lifting a towel by its middle gathers
it into a bundle, and lifting two diagonally opposite corners does the same. Folding needs two
ADJACENT corners carried to the opposite edge, which is why ClothSheet names its corners by
which edge they sit on.

Known limitation, stated plainly because it is the one thing here that does not work:

    Folding a towel that has just come out of the drum does not produce a properly folded towel.

Every other step does. `open_washer`, `take_out`, `close_washer`, `put_in_basket` and
`put_on_counter` all pass in sequence, and `fold` is deterministic and repeatable on a towel
that is lying flat -- 0.50 m across to 0.28 m, from the `counter` keyframe.

What remains is the cloth's shape rather than the robot's aim. The towel is carried by a single
vertex, because a grasp is a weld to one point, so in the air it hangs as a curtain (measured
dz=0.363 against dx=0.253) and lands gathered however carefully it is set down. `put_on_counter`
now lays it out -- touching the far edge down and drawing the hand back, which trails the cloth
along the surface instead of dropping it in a heap -- and that roughly doubles the span it
lands at. It is still short of flat, and `_spread` cannot finish the job one-handed: pulling
one corner of a gathered sheet moves the bundle as much as it opens it.

A proper fold is TWO-HANDED and the two corners must move TOGETHER: take the two ends of the
near edge, carry them across in parallel, and lay them on the two ends of the far edge, so the
sheet hinges along the middle. That is what `fold` aims at, and it is why it walks between the
two corners rather than dragging one of them: measured a one-corner carry taking the near-left
corner from x=0.85 to x=1.22, right across the towel, while y barely moved -- the span fell
0.50 -> 0.21 m, which passes a naive "smaller than before" check and is a sheet pulled into a
diagonal, not a fold.

Doing both corners at the same instant was tried and does not work on this robot: the corners
are 0.40 m apart, the shoulders only 0.30 m, and the best standing spot found by a sweep that
scores candidates with BOTH arms out at once still leaves the worse hand 0.19 m away against a
0.075 m grasp tolerance. So the corners are carried one at a time from a spot chosen for each,
with the landing positions snapshotted BEFORE anything moves -- read them live and the second
corner aims at wherever the first one has already dragged the far edge.

So the honest summary is that the fold at the end of the errand is a carry-across rather than a
crease. `fold` measures the span before and after and says which it achieved; it will not claim
a fold it did not perform. The real fix is a two-handed spread -- pin one corner and drag the
opposite one -- which needs the arms to work together in a way nothing else here requires.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
import mujoco
import numpy as np

from ..perception.grounding import Grounder
from ..sim.cloth import ClothGrasp, ClothSheet
from ..sim.reach import WORKING_REACH_M
from ..nav.explore import MaplessNavigator
from ..sim.robot import Robot
from .skills import SkillResult

log = logging.getLogger(__name__)

# Where to stand relative to something being manipulated. Comfortably inside the working reach
# (0.45 m on AN-01), so a little drift while turning does not push the target out of the
# envelope -- but not so close that the body ends up pressed against the fixture, which is its
# own failure mode: a robot leaning on a counter cannot walk away from it.
STAND_OFF_M = 0.34

# How close the hand has to get before a grasp is attempted. The gripper spans 6 cm, so a miss
# larger than this would weld the palm to a vertex it is not actually touching -- the towel
# would snap to the hand, which looks like success and is not.
GRASP_TOLERANCE_M = 0.075

# Height the hand rises to while carrying. Enough to lift a hand towel clear of the counter and
# the basket rim (0.32 m) without hitting the arm's upper limit.
CARRY_HEIGHT_M = 1.15

# A flat 0.40 x 0.30 sheet spans about 0.50 m corner to corner. Folded once it should span
# appreciably less; this is the threshold the fold skill reports against, and it is measured
# rather than asserted -- see ClothGrasp.sheet_extent.
FOLDED_EXTENT_M = 0.40


@dataclass
class LaundrySkills:
    """Momo's laundry repertoire.

    Holds the small amount of state the task needs between steps: which towel is in hand, and
    where the robot was standing when it was given the instruction.
    """

    robot: Robot
    grounder: Grounder
    # Vision-driven navigation for crossing the room. Everything that walks any distance goes
    # through this rather than through _walk_to: it looks with the head camera, steers by what
    # it sees, and searches again when the target goes out of frame. _walk_to remains for the
    # last few decimetres, where the target is a computed point rather than a thing to see.
    nav: MaplessNavigator | None = None
    cloth: ClothGrasp = field(init=False)
    _home: np.ndarray = field(init=False)
    _home_heading: float = field(init=False)
    _holding_basket: bool = field(default=False, init=False)
    # Which hand(s) hold the basket. One on this robot; see basket_grip in home.xml.
    _basket_hands: tuple = field(default=(), init=False)
    # Which sheet the current errand is about. Set by whichever skill first names a towel, so
    # "take it out and fold it" does not need the towel named twice.
    _subject: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.cloth = ClothGrasp(self.robot.model, self.robot.data)
        if self.nav is None:
            self.nav = MaplessNavigator(self.robot, self.grounder)
        self._home = self.robot.position[:2].copy()
        self._home_heading = self.robot.yaw

    # -- helpers ------------------------------------------------------------------

    def _sheet(self, name: str | None) -> ClothSheet | None:
        """Resolve a towel by name, falling back to whatever the errand is already about.

        Accepts the words a person actually uses -- "the blue towel", 「青いタオル」 -- and maps
        them onto the sheet names in the scene.
        """
        if name:
            key = _canonical_towel(name)
            if key and key in self.cloth.sheets:
                self._subject = key
                return self.cloth.sheets[key]
        if self._subject:
            return self.cloth.sheets.get(self._subject)
        # No name and no subject: if there is exactly one towel, it is unambiguous.
        if len(self.cloth.sheets) == 1:
            only = next(iter(self.cloth.sheets))
            self._subject = only
            return self.cloth.sheets[only]
        # Otherwise prefer whichever is nearest, which is what "that one" almost always means.
        best, distance = None, float("inf")
        for key, sheet in self.cloth.sheets.items():
            offset = float(
                np.linalg.norm(self.cloth.sheet_centre(sheet)[:2] - self.robot.position[:2])
            )
            if offset < distance:
                best, distance = key, offset
        if best:
            self._subject = best
            return self.cloth.sheets[best]
        return None

    # -- getting there ------------------------------------------------------------
    #
    # Two ways to move, and the difference matters. `_travel_to` CROSSES THE ROOM: it steers by
    # what the head camera sees, so it copes with the target having moved, with the robot
    # having drifted, and with something being in the way. `_walk_to` below is dead reckoning
    # toward a computed point, which is right only for the last few decimetres where the
    # target is a coordinate rather than a thing that can be looked at.

    # Where the named places are, for a robot that has lost track of one. These are fallbacks:
    # the camera is asked first, and this is what "walk toward roughly where it should be"
    # means when nothing is visible.
    _PLACES = {
        "washer": "drum_ring_b",
        "counter": "counter_g",
        "basket": None,          # a free body now; found via basket_site
    }

    def _place_position(self, name: str) -> np.ndarray | None:
        """Where a named place is, by site or geom lookup."""
        name = name.strip().lower()
        if name in ("basket", "hamper", "kago"):
            return self._site_position("basket_site")
        geom = self._PLACES.get(name)
        if geom:
            return self._geom_position(geom)
        # Unknown name: let the grounder try, since it knows more words than this table.
        return None

    def _beside(self, target: np.ndarray, clearance: float) -> np.ndarray:
        """A spot `clearance` metres to the SIDE of `target`, along the room's long axis.

        Used for putting things down near an appliance without putting them in the way of it.
        The side is chosen as whichever leaves more room, so this works at the washer (against
        the north wall) and at the counter without a table of special cases.
        """
        target = np.asarray(target, dtype=float)[:2]
        # The appliances stand against the north wall, so "beside" means offset in x. Pick the
        # side with more floor: the washer sits at x=-1.45 in a room spanning -2.5..2.5, so
        # there is far more space to its +x side.
        room_centre = 0.0
        direction = 1.0 if target[0] < room_centre else -1.0
        # Stand off in y as well, or the drop point is inside the appliance's own footprint.
        return np.array([target[0] + direction * clearance, target[1] - 0.42])

    def _approach_point(self, target: np.ndarray, standoff: float) -> np.ndarray:
        """A spot `standoff` metres short of `target`, on the line from where we stand."""
        here = self.robot.position[:2]
        offset = np.asarray(target)[:2] - here
        distance = float(np.linalg.norm(offset))
        if distance < 1e-6:
            return here.copy()
        return np.asarray(target)[:2] - offset / distance * standoff

    def _travel_to(self, point: np.ndarray, target: str, tolerance: float = 0.45) -> float:
        """Cross the room to `point`, using the camera to find `target` on the way.

        Returns the final distance to `point`.

        The camera leads and dead reckoning follows. `MaplessNavigator.goto` walks toward what
        it can SEE, which is what makes this robust to the things that actually go wrong -- the
        basket is not where it was last time, the robot drifted while turning, a door is in the
        way -- and then the last stretch is closed by walking to the computed point, because a
        centroid in an image is not accurate to the centimetre.

        When the target cannot be found, escalate rather than give up, which is what a person
        does in an unfamiliar room:

            1. survey        turn on the spot and look again
            2. wander        take a few steps to change the viewpoint, then survey again
            3. dead reckon   walk to where the thing is supposed to be

        Only after all three does it return short, and the caller decides what that means.
        """
        goal = np.asarray(point, dtype=float)[:2]

        # DO NOT let the navigator drive the last stretch to something light.
        #
        # `goto` walks up to what it sees and stops against it, which is correct for a door or
        # a counter and destructive for a laundry basket: measured it driving the robot into
        # the basket and knocking it from level to a 0.85 quaternion before any hand was
        # raised. Everything downstream was then trying to pick up a box that had already been
        # shoved over.
        #
        # So for movable things the camera is used to FIND them and the approach is finished by
        # walking to the computed standoff, which is a point in free floor rather than the
        # object itself.
        movable = target in ("basket", "hamper")

        # FACE THE DESTINATION BEFORE LOOKING FOR IT.
        #
        # After fetching the basket the robot stands where the basket was, which can be
        # anywhere: measured it at (-0.36, 0.03) with the washing machine 173 degrees behind
        # it -- almost exactly at its back. The neck reaches +-100 degrees, so no amount of
        # head-turning can see something there, and the navigator then spends the whole
        # approach reporting "lost sight of washer" while walking on dead reckoning.
        #
        # Turning the body first is cheap and it is what a person does before setting off.
        # After it, the head sweep in `survey` has a real chance of finding the target.
        offset = goal - self.robot.position[:2]
        if float(np.linalg.norm(offset)) > 0.15:
            self._turn_to(math.atan2(offset[1], offset[0]))

        if self.nav is not None and target:
            if movable:
                if not self.nav.survey(target):
                    log.info("could not see the %s; looking around", target)
                    self._look_around_for(target)
                result = None
            else:
                result = self.nav.goto(target, max_steps=1600, search_steps=220)
            if result is not None and not getattr(result, "ok", False):
                log.info("could not see the %s; looking around", target)
                if not self._look_around_for(target):
                    log.info("still cannot see the %s; walking to where it should be", target)

        # Close the remainder by dead reckoning. Even a successful `goto` stops at a distance
        # judged from an image, which is not the centimetre-accurate spot a grasp needs.
        gap = float(np.linalg.norm(goal - self.robot.position[:2]))
        if gap > tolerance:
            gap = self._walk_to(goal, max_steps=1200, stop_at=min(0.12, tolerance))
        return gap

    def _look_around_for(self, target: str) -> bool:
        """Look for `target`, escalating only as far as necessary. True if found.

        The order is the point, and it is the order a person uses:

            1. TURN THE HEAD.   `survey` sweeps the neck across +-80 degrees, which is most of
                                what is in front of the robot, and costs nothing but a few
                                camera frames. Almost everything is found here.
            2. TURN THE BODY.   Only if the head sweep found nothing. Three 120-degree turns
                                cover the rest of the room from where the robot already
                                stands, each followed by another head sweep.
            3. WALK.            Only if turning found nothing, because the target may be behind
                                something -- the washer hides the basket from half this room --
                                and no amount of turning on the spot sees round an obstacle.

        Doing this in the wrong order is what the errand used to look like: the robot span its
        whole body through a full circle to find a washing machine that was already within a
        head-turn of straight ahead. Turning the body is slow, it drags whatever is being
        carried through the air, and it loses the standing position that the next step wants.
        """
        if self.nav is None:
            return False

        # 1. Head only.
        if self.nav.survey(target):
            log.info("found the %s by looking around", target)
            return True

        # 2. Body turns, with a head sweep at each.
        for _ in range(2):
            log.info("cannot see the %s; turning to look further round", target)
            self._turn_to(self.robot.yaw + 2.09)  # 120 degrees
            if self.nav.survey(target):
                log.info("found the %s after turning", target)
                return True

        # 3. Walk, to see round whatever is in the way.
        for attempt in range(2):
            log.info("still cannot see the %s; moving to get a different view", target)
            for _ in range(45):
                self.robot.step(vx=0.24, wz=0.35 if attempt == 0 else -0.35)
            self.robot.stand(0.4)
            if self.nav.survey(target):
                log.info("found the %s after moving", target)
                return True
        return False

    def _turn_to(self, heading: float, max_steps: int = 200) -> None:
        """Turn the body to a world heading."""
        for _ in range(max_steps):
            error = (heading - self.robot.yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(error) < 0.05:
                break
            self.robot.step(0.0, 0.0, float(np.clip(error * 1.6, -1.2, 1.2)))
        self.robot.stand(0.2)

    def _walk_to(self, point: np.ndarray, max_steps: int = 600, stop_at: float = 0.10) -> float:
        """Walk to a spot on the floor. Returns how close it got.

        Deliberately simple: this room is 5 x 4 m with everything against the walls, so there
        is nothing to navigate around between the washer, the basket and the counter. The
        office's obstacle-avoiding navigator would be the wrong tool -- it steers away from
        exactly the fixtures Momo is trying to walk up to.
        """
        target = np.asarray(point, dtype=float)[:2]
        for _ in range(max_steps):
            delta = target - self.robot.position[:2]
            distance = float(np.linalg.norm(delta))
            if distance < stop_at:
                break
            desired = math.atan2(delta[1], delta[0])
            error = (desired - self.robot.yaw + math.pi) % (2 * math.pi) - math.pi
            turn = float(np.clip(error * 1.5, -1.2, 1.2))
            # Slow down while turning hard, so the robot does not arc past the target.
            self.robot.step(0.40 * max(0.25, 1.0 - abs(turn)), 0.0, turn)
        self.robot.stand(0.2)
        return float(np.linalg.norm(target - self.robot.position[:2]))

    def _stand_near(self, target: np.ndarray, offset: float = STAND_OFF_M) -> float:
        """Walk to within arm's length of a point and face it.

        This is the step that makes manipulation work at all. The hand reaches 0.25 m, so a
        skill that reaches for something 0.45 m away misses by 0.2 m and reports a puzzling
        failure; positioning first turns that into an ordinary success.
        """
        target = np.asarray(target, dtype=float)
        here = self.robot.position[:2]
        direction = target[:2] - here
        distance = float(np.linalg.norm(direction))
        if distance < 1e-6:
            direction = np.array([math.cos(self.robot.yaw), math.sin(self.robot.yaw)])
            distance = 1.0
        unit = direction / distance
        # Stand back along the line from the target, then face it.
        self._walk_to(target[:2] - unit * offset)
        self._turn_to(math.atan2(unit[1], unit[0]))

        # Then close the rest of the way, because the ideal standing spot is often inside a
        # fixture. A towel in the drum sits 0.32 m beyond where a polite standoff leaves the
        # robot, and _walk_to gives up as soon as its own target is reached -- so the grasp
        # missed by 0.22 m against a 0.25 m reach, on a corner that was actually within range.
        # Walking forward until the body meets the appliance recovers that: she can get to
        # y=1.132 against a drum mouth at y=1.273, which puts the near corner in reach.
        # Stop at 0.9 of the working reach, NOT closer.
        #
        # Pressing in to two thirds of it was tried on AN-01 and is worse: the fold corner
        # went from 0.118 m to 0.240 m. A longer arm wants to work at arm's length, and
        # walking the body further in leaves the shoulder folded up with nowhere to put the
        # elbow -- the same reason a person steps BACK from a counter to work on it rather
        # than pressing against it.
        gap = float(np.linalg.norm(target[:2] - self.robot.position[:2]))
        stalled = 0
        for _ in range(160):
            if gap <= WORKING_REACH_M * 0.9:
                break
            before = self.robot.position[:2].copy()
            self.robot.step(vx=0.25)
            # Stop pressing once the body is against something and no longer making progress,
            # rather than grinding into it for the rest of the budget.
            if float(np.linalg.norm(self.robot.position[:2] - before)) < 1e-3:
                stalled += 1
                if stalled > 12:
                    break
            else:
                stalled = 0
            gap = float(np.linalg.norm(target[:2] - self.robot.position[:2]))
        self.robot.stand(0.2)
        return float(np.linalg.norm(target[:2] - self.robot.position[:2]))

    def _hand_for(self, point: np.ndarray) -> str:
        """Which hand can actually get to a point: the one on that side of the body.

        Not a nicety. The shoulders sit 0.15 m either side of the centre line and neither arm
        crosses the chest, so reaching for something off to the left with the right hand fails
        in a way that looks like a range problem and is not: measured a corner 0.22 m from the
        base -- comfortably inside the 0.25 m reach -- missed by 0.26 m, with the hand moving
        AWAY from it, because it lay to the left and the right arm has nowhere to go.
        """
        offset = np.asarray(point)[:2] - self.robot.position[:2]
        # Rotate into the body frame; +y is the robot's left.
        cos_yaw, sin_yaw = math.cos(-self.robot.yaw), math.sin(-self.robot.yaw)
        lateral = cos_yaw * offset[1] - sin_yaw * offset[0]
        return "l" if lateral > 0.0 else "r"

    def find_fold_spot(self, sheet: ClothSheet, near: list[int], span: float = 0.40) -> float:
        """Find a spot BOTH ends of an edge can be reached from, one per hand.

        find_standing_spot optimises for a single vertex, which is right for picking a towel up
        and wrong for folding: it happily walks to a place where one corner is perfect and the
        other is a metre away. Measured exactly that -- the robot ending at x=0.24 for a towel
        at x=0.87, with the right hand 0.17 m from its corner and the left hand 1.10 m from
        the other.

        Folding is two-handed, so the spot has to be scored on the WORSE of the two hands.
        """
        centre = self.cloth.sheet_centre(sheet)[:2]
        saved_qpos = self.robot.data.qpos.copy()
        saved_qvel = self.robot.data.qvel.copy()
        saved_ctrl = self.robot.data.ctrl.copy()

        def restore() -> None:
            self.robot.data.qpos[:] = saved_qpos
            self.robot.data.qvel[:] = saved_qvel
            self.robot.data.ctrl[:] = saved_ctrl
            mujoco.mj_forward(self.robot.model, self.robot.data)

        best_score = float("inf")
        best_at = self.robot.position[:2].copy()
        for dx in np.linspace(-span, span, 7):
            for dy in np.linspace(0.10, span, 4):
                restore()
                self.robot.data.qpos[0:2] = np.array([centre[0] + dx, centre[1] - dy])
                mujoco.mj_forward(self.robot.model, self.robot.data)
                self.robot.arm_home("r")
                self.robot.arm_home("l")
                self.robot.stand(0.3)
                # Face the towel, so "left" and "right" mean something.
                self._turn_to(
                    math.atan2(centre[1] - self.robot.position[1],
                               centre[0] - self.robot.position[0])
                )
                ends = sorted(near, key=lambda v: self._lateral_of(
                    self.cloth.vertex_position(sheet, v)))
                worst = 0.0
                for side, vertex in (("r", ends[0]), ("l", ends[-1])):
                    worst = max(
                        worst,
                        self.robot.reach_to(
                            self.cloth.vertex_position(sheet, vertex), side,
                            passes=2, settle_steps=70,
                        ),
                    )
                if worst < best_score:
                    best_score, best_at = worst, self.robot.position[:2].copy()

        restore()
        self.robot.data.qpos[0:2] = best_at
        mujoco.mj_forward(self.robot.model, self.robot.data)
        self.robot.arm_home("r")
        self.robot.arm_home("l")
        self.robot.stand(0.4)
        self._turn_to(
            math.atan2(centre[1] - self.robot.position[1], centre[0] - self.robot.position[0])
        )
        return best_score

    def find_standing_spot(
        self, sheet: ClothSheet, graspable: list[int], span: float = 0.45
    ) -> tuple[float, int, str]:
        """Search nearby floor for a spot the cloth is actually reachable from.

        Returns (best reach error, vertex, hand) and leaves the robot standing there.

        This exists because positioning by rule kept failing on a task that was perfectly
        solvable. An open door hangs across part of its own opening, the arm cannot cross the
        chest, and the hand bottoms out around z=0.70 -- three constraints whose intersection
        is a small patch of floor that no simple "stand 0.22 m back and face it" rule finds.
        Sweeping candidate spots and asking the IK what it can actually do found a position
        with 0.037 m of error where the heuristic was reporting 0.19 m and giving up.

        The sweep is cheap: solving IK is milliseconds, and stepping the simulation to each
        candidate is the only real cost, so a few dozen candidates is nothing next to the walk
        that follows.
        """
        origin = self.robot.position[:2].copy()
        centre = self.cloth.sheet_centre(sheet)[:2]

        # Snapshot the WHOLE world, not just the robot's position.
        #
        # Trying a candidate teleports the base, and a base that lands inside the washer drags
        # the towel out with it: measured the search returning a target vertex at z=0.004, the
        # sheet having been swept onto the floor by the trials themselves. The search is meant
        # to be a question about the world, not an edit to it, so every trial is rolled back.
        saved_qpos = self.robot.data.qpos.copy()
        saved_qvel = self.robot.data.qvel.copy()
        saved_ctrl = self.robot.data.ctrl.copy()

        def restore() -> None:
            self.robot.data.qpos[:] = saved_qpos
            self.robot.data.qvel[:] = saved_qvel
            self.robot.data.ctrl[:] = saved_ctrl
            mujoco.mj_forward(self.robot.model, self.robot.data)

        best = (float("inf"), graspable[0], "r", origin)
        # The dy range runs from right up against the target out to the arm's working reach.
        # It was 0.18..0.60 when the arm reached 0.25 m; AN-01 reaches 0.45 m, so standing that
        # close is not merely unnecessary, it is counterproductive -- the body ends up against
        # the fixture and the shoulder is left with nowhere to put the elbow. Sweeping out to
        # the real envelope is what lets the search find the spot it needs.
        for dx in np.linspace(-span, span, 9):
            for dy in np.linspace(0.15, WORKING_REACH_M + 0.10, 6):
                # Candidates are laid out around the cloth, not around the robot, so the sweep
                # stays useful however badly the approach ended up placed.
                # Candidates run from right up against the target out to arm's length. The
                # closest ones matter most: a towel lying inside a drum is only reachable from
                # a spot the robot can barely fit into, and a sweep that starts 0.20 m back
                # never offers it -- the search returned 0.21 m for a grasp that works at 0.03.
                spot = np.array([centre[0] + dx, centre[1] - abs(dy)])
                restore()
                self.robot.data.qpos[0:2] = spot
                # Start every candidate from the same arm pose. Leaving the arm wherever the
                # previous candidate put it makes the trial depend on the order they were
                # tried in, and an arm already stretched the wrong way cannot be recovered
                # within the reach loop's budget.
                self.robot.arm_home("r")
                self.robot.arm_home("l")
                self.robot.stand(0.3)
                # Try BOTH hands and keep whichever actually gets there, rather than deciding
                # in advance from geometry. Which arm can reach a point depends on the shoulder
                # position, the torso yaw and whatever the target is sitting in, and picking by
                # side alone discarded spots that worked.
                for hand in ("r", "l"):
                    # Reset the arms before each hand's trial. Testing the left arm from
                    # whatever pose the right arm's trial left the body in makes the result
                    # depend on the order the hands were tried in.
                    self.robot.arm_home("r")
                    self.robot.arm_home("l")
                    self.robot.stand(0.25)
                    vertex, _ = self.cloth.nearest_vertex(
                        sheet, self.robot.hand_position(hand), among=graspable
                    )
                    error = self.robot.reach_to(
                        self.cloth.vertex_position(sheet, vertex), hand,
                        passes=2, settle_steps=70,
                    )
                    if error < best[0]:
                        best = (error, vertex, hand, self.robot.position[:2].copy())
                # Deliberately NO early exit. Stopping at the first spot that clears the
                # tolerance sounds like a saving and is not: the candidates are swept in
                # geometric order, not in order of quality, so the first acceptable one is
                # rarely the best and the sweep would settle for 0.11 m where carrying on
                # finds 0.028 m. The whole sweep is a couple of seconds of simulation.

        # Return to the winning spot and re-run the reach there.
        #
        # Restoring the base position does NOT restore the arm: the servos still hold whatever
        # they were commanded at the last candidate, and the hand springs back toward rest.
        # Measured a search reporting 0.009 m and the very next reach missing by 0.46 m purely
        # because the arm had been left pointing somewhere else. The reach has to be redone
        # from the pose the robot is actually in.
        restore()
        self.robot.data.qpos[0:2] = best[3]
        mujoco.mj_forward(self.robot.model, self.robot.data)
        self.robot.arm_home("r")
        self.robot.arm_home("l")
        self.robot.stand(0.4)
        vertex, side = best[1], best[2]
        error = self.robot.reach_to(
            self.cloth.vertex_position(sheet, vertex), side, passes=3
        )
        return error, vertex, side

    def _spread(self, sheet: ClothSheet, pulls: int = 2) -> float:
        """Flatten a bunched towel by dragging opposite corners apart. Returns the new span.

        Not a nicety: a fold needs a flat sheet. A towel dropped on a surface after being
        carried is gathered around wherever it was held, and pulling one corner of a gathered
        towel moves the whole bundle rather than folding it.

        Each pull takes the corner furthest from the sheet's centre and drags it outward along
        that line, which is what a person does with two hands at once and this robot has to do
        one hand at a time.
        """
        for _ in range(max(1, pulls)):
            centre = self.cloth.sheet_centre(sheet)
            perimeter = sheet.perimeter()
            # The corner that has travelled least far from the middle is the one holding the
            # bundle together, so it is the one worth pulling out.
            offsets = {
                v: float(np.linalg.norm(self.cloth.vertex_position(sheet, v)[:2] - centre[:2]))
                for v in perimeter
            }
            vertex = min(offsets, key=lambda v: offsets[v])

            # Reach from where the robot already stands rather than searching for a new spot.
            # find_standing_spot teleports the base between candidates, and a base that lands
            # against the counter drags the towel with it -- which is the very thing this is
            # trying to undo. If the corner is out of reach from here, skip the pull.
            side = self._hand_for(self.cloth.vertex_position(sheet, vertex))
            grabbed, _ = self._grasp_vertex(sheet, vertex, side)
            if not grabbed:
                continue

            # Drag outward along the line from the centre, staying just above the surface.
            #
            # The distance is the SHORTFALL, not a fixed step. A flat sheet spans 0.50 m, and
            # a fixed 0.16 m pull is sized for a towel that is nearly there already: on one
            # gathered into a 0.21 m bundle it moves the corner a fraction of the way out and
            # the span barely changes, which is why repeated pulls used to plateau around
            # 0.14 m. Pulling out by roughly what is missing gets the corner clear of the pile
            # in one go, which is what actually opens the sheet.
            #
            # Capped at 0.22 m: the arm reaches 0.25 m from the shoulder, and asking for more
            # than that just drags the whole towel along with the hand.
            here = self.cloth.vertex_position(sheet, vertex)
            direction = here[:2] - centre[:2]
            norm = float(np.linalg.norm(direction))
            direction = direction / norm if norm > 1e-6 else np.array([1.0, 0.0])
            shortfall = max(0.0, 0.50 - self.cloth.sheet_extent(sheet))
            pull = float(np.clip(shortfall * 0.5, 0.16, 0.22))
            target = np.array([
                here[0] + direction[0] * pull,
                here[1] + direction[1] * pull,
                here[2] + 0.03,
            ])
            self.robot.reach_to(target, side, passes=3)
            self.robot.grip(side, 0.0)
            self.cloth.release(side)
            self.robot.arm_home(side)
            self.robot.stand(0.8)

        return self.cloth.sheet_extent(sheet)

    def _lateral_of(self, point: np.ndarray) -> float:
        """How far to the robot's left a world point is, in metres. Negative is right."""
        offset = np.asarray(point)[:2] - self.robot.position[:2]
        cos_yaw, sin_yaw = math.cos(-self.robot.yaw), math.sin(-self.robot.yaw)
        return float(cos_yaw * offset[1] - sin_yaw * offset[0])

    def _near_and_far_edges(self, sheet: ClothSheet) -> tuple[list[int], list[int]]:
        """The sheet's two opposite edges, ordered nearest-to-the-robot first.

        Which edge is "near" is a fact about where the robot is standing, not about the vertex
        grid. ClothSheet.edge names them by grid position, fixed when the model compiles; a
        fold that trusted those names reached across the towel for its far side and came away
        with nothing.

        Both axes are considered, so this works whichever way the sheet is lying and whichever
        way the robot approached it.
        """
        here = self.robot.position[:2]

        def distance(vertices: list[int]) -> float:
            points = np.array(
                [self.cloth.vertex_position(sheet, v)[:2] for v in vertices]
            )
            return float(np.linalg.norm(points.mean(axis=0) - here))

        pairs = [
            (sheet.edge("near"), sheet.edge("far")),
            (sheet.edge("left"), sheet.edge("right")),
        ]
        # Fold across whichever axis the robot is squarest to: that is the one where the near
        # edge is genuinely nearer, and folding across it moves cloth away rather than sideways.
        best = max(pairs, key=lambda pair: abs(distance(pair[0]) - distance(pair[1])))
        first, second = best
        return (first, second) if distance(first) <= distance(second) else (second, first)

    def _best_hold(
        self, sheet: ClothSheet, among: list[int] | None = None
    ) -> tuple[int, str, float]:
        """The (vertex, hand) pair that is genuinely easiest to grasp, and how far it is.

        Choosing the vertex and the hand separately does not work. The nearest point might be
        on the robot's left while the nearest hand is its right, and since neither arm crosses
        the chest, the pairing that looked best by distance is the one combination that cannot
        be done. Scoring both hands against every candidate picks the pair that is actually
        reachable.
        """
        candidates = among if among is not None else sheet.perimeter()
        best: tuple[int, str, float] = (candidates[0], "r", float("inf"))
        for side in ("r", "l"):
            hand = self.robot.hand_position(side)
            vertex, distance = self.cloth.nearest_vertex(sheet, hand, among=candidates)
            # Only consider points on this hand's own side of the body.
            if self._hand_for(self.cloth.vertex_position(sheet, vertex)) != side:
                distance += 0.25
            if distance < best[2]:
                best = (vertex, side, distance)
        return best

    def _grasp_vertex(self, sheet: ClothSheet, vertex: int, side: str) -> tuple[bool, float]:
        """Reach for one cloth vertex and close on it. Returns (grasped, miss distance).

        The miss is checked before welding: welding at a distance snaps the towel to the hand,
        which looks exactly like a successful grasp in a video and is not one.
        """
        self.robot.grip(side, 0.0)
        miss = float("inf")
        # Several attempts, re-reading the vertex each time. Cloth sags as the hand comes down
        # on it, so the point aimed at is never quite the point arrived at, and a single pass
        # lands just outside tolerance more often than not -- measured 0.08 m against a 0.075 m
        # limit, which is a failure report for a grasp that was one nudge from working.
        #
        # Between attempts the robot also SHUFFLES toward the target. Most of the residual is
        # the body being a few centimetres too far away rather than the arm being wrong, and
        # stepping in closes it where re-solving the same IK cannot.
        for attempt in range(4):
            point = self.cloth.vertex_position(sheet, vertex)
            self.robot.reach_to(point, side, passes=2)
            point = self.cloth.vertex_position(sheet, vertex)
            miss = float(np.linalg.norm(self.robot.hand_position(side) - point))
            if miss <= GRASP_TOLERANCE_M:
                break
            if attempt < 3:
                offset = point[:2] - self.robot.position[:2]
                if float(np.linalg.norm(offset)) > WORKING_REACH_M:
                    self._turn_to(math.atan2(offset[1], offset[0]))
                    for _ in range(22):
                        self.robot.step(vx=0.20)
                    self.robot.stand(0.2)
        if miss > GRASP_TOLERANCE_M:
            return False, miss
        self.robot.grip(side, 1.0)
        self.robot.stand(0.25)
        return self.cloth.grasp(sheet.name, vertex, side), miss

    def _lift_to(self, height: float, side: str, steps: int = 260) -> None:
        """Raise whatever is in the hand to a carrying height, straight up.

        Straight up rather than along an arc: an arc drags the towel across whatever it was
        lying on, and in the drum that means catching it on the lip.
        """
        hand = self.robot.hand_position(side)
        self.robot.reach_to(np.array([hand[0], hand[1], height]), side, passes=2,
                            settle_steps=steps // 2)

    # -- skills -------------------------------------------------------------------

    def open_washer(self) -> SkillResult:
        """Walk to the washing machine and pull its door open.

        The drum door is light and barely sprung, unlike the office doors, so it stays where it
        is put -- a door that swung shut while the robot was reaching into the drum would make
        the rest of the task impossible for reasons that have nothing to do with laundry.
        """
        handle = self._site_position("drum_handle_site")
        if handle is None:
            return SkillResult(False, "I cannot find the washing machine.")

        joint = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_JOINT, "washer_door")
        if joint < 0:
            return SkillResult(False, "There is no washing machine door in this room.")
        address = self.robot.model.jnt_qposadr[joint]
        before = float(self.robot.data.qpos[address])

        self._stand_near(handle)
        miss = self.robot.reach_to(handle, "r")
        if miss > 0.12:
            return SkillResult(
                False,
                f"I could not reach the washing machine door ({miss:.2f} m short).",
                {"miss": miss},
            )

        # Grip the handle and swing the door open by walking backwards while holding on. The
        # door opens toward the robot, so pulling is the only way; pushing shuts it.
        #
        # The grip has to be a WELD, not friction. Measured: the hand arriving 0.068 m from the
        # handle, closing the fingers, and walking back 160 steps moved the door 0.00 degrees.
        # A 4 cm handle slips out of a pinch long before the arm can swing even a light door,
        # which is the same conclusion the office scene reached about its own door handles.
        self.robot.grip("r", 1.0)
        self.robot.stand(0.3)
        welded = self.robot.grasp("drum_door")

        # Pull until the door is properly out of the way, not merely ajar.
        #
        # A half-open door is worse than a shut one for what comes next: at 54 degrees the leaf
        # sits at y=1.07, squarely across a drum mouth at y=1.27, and every subsequent reach
        # into the drum stopped dead against it. So this keeps hauling while the angle is still
        # improving, and swings wide rather than settling for "it opened".
        best = 0.0
        stalled = 0
        for _ in range(420):
            self.robot.step(vx=-0.30)
            angle = abs(math.degrees(float(self.robot.data.qpos[address])))
            if angle > best + 0.5:
                best, stalled = angle, 0
            else:
                stalled += 1
                # 130 degrees, not 90: the hinge is on the far side of the opening, so the
                # leaf sweeps ACROSS the mouth on its way round. At 70 degrees the panel still
                # sat at y=1.04 against a mouth at y=1.27 and blocked every reach into the
                # drum; by 130 it has swung out to x=-1.87, clear of a mouth spanning
                # x=-1.66..-1.24.
                if stalled > 60 or angle > 130.0:
                    break

        self.robot.release()
        self.robot.grip("r", 0.0)
        self.robot.arm_home("r")
        self.robot.stand(0.4)
        if not welded:
            log.info("no handle weld in this scene; the door was only pushed")

        after = float(self.robot.data.qpos[address])
        swing = abs(math.degrees(after))
        if swing < 15.0:
            return SkillResult(
                False,
                "I tried to open the washing machine but the door barely moved.",
                {"swing_degrees": swing, "before": math.degrees(before)},
            )
        return SkillResult(
            True,
            f"I opened the washing machine ({swing:.0f} degrees).",
            {"swing_degrees": swing},
        )

    def close_washer(self) -> SkillResult:
        """Push the washing machine door shut again.

        The mirror of open_washer, and it needs no weld: a door is closed by pushing, and the
        arm can push a light leaf without holding on to it. Only pulling needs a grip, because
        a hand cannot pull on something it is not attached to.
        """
        joint = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_JOINT, "washer_door")
        if joint < 0:
            return SkillResult(False, "There is no washing machine door in this room.")
        address = self.robot.model.jnt_qposadr[joint]
        before = abs(math.degrees(float(self.robot.data.qpos[address])))
        if before < 10.0:
            return SkillResult(True, "The washing machine is already closed.")

        # Both hands have to be free.
        #
        # Closing means taking hold of the handle and walking it round an arc, and an arm that
        # is holding a towel cannot do it: measured 105 degrees closing to 12 empty-handed, and
        # sticking at 74-94 with a towel in hand. This is a real constraint of having one pair
        # of hands, not a control problem to tune away, so say so and let the caller reorder.
        if self._loaded_hand() is not None:
            return SkillResult(
                False,
                "I need both hands to close the washing machine door -- let me put the "
                "laundry down first.",
                {"holding": True, "before": before},
                # Not fatal: the rest of the errand still makes sense, and the door being left
                # open does not stop the laundry being folded.
                fatal=False,
            )

        # Closing is opening in reverse, and it needs the same weld for the same reason.
        #
        # A first version tried to sweep the door shut with an outstretched arm and got
        # nowhere: the handle of a door standing 105 degrees open is 0.337 m from where the
        # robot can stand, against a 0.25 m reach, so the hand never touched it and the door
        # finished at 136 degrees -- further open than it started. Take hold of the handle and
        # walk it round, exactly as open_washer does.
        handle = self._site_position("drum_handle_site")
        if handle is None:
            return SkillResult(False, "I cannot find the washing machine door.")

        # Approach ALONG THE HANDLE'S OWN RADIUS, from outside the arc -- never across it.
        #
        # Standing in front of the drum mouth is where a person stands, and it is exactly wrong
        # for this robot: the open leaf sweeps that space, so the body drives the door further
        # open on the way in and the handle runs away from the hand. Measured, coming back from
        # the basket with the door at 73 degrees: the approach walk alone took it to 94, and
        # each "shuffle in and retry" added more -- 106, 120, 132 -- so the skill reported the
        # door still open after having pushed it wide itself.
        #
        # The handle can only move along the TANGENT to its hinge circle. So walking in along
        # the RADIUS -- the line from the hinge out through the handle, extended -- applies no
        # torque to the door: the body closes on the handle without disturbing it. Same spot a
        # person would pick after the first try, for the same reason.
        hinge_body = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_BODY, "drum_door")
        hinge = (
            self.robot.data.xpos[hinge_body][:2].copy()
            if hinge_body >= 0
            else np.array([-1.71, 1.27])
        )
        radial = handle[:2] - hinge
        norm = float(np.linalg.norm(radial))
        radial = radial / norm if norm > 1e-6 else np.array([0.0, -1.0])

        # Two legs. The first swings wide to get ONTO the radial line from wherever the errand
        # left the robot -- crossing the arc at 0.75 m costs nothing, because the leaf is only
        # 0.48 m long. The second comes straight down the radius to arm's length.
        #
        # Generous step budget: this is often called after a trip to the basket or the counter
        # at the far end of the room, so it is a walk across the room and not a shuffle. At the
        # default budget the robot arrived 0.87 m short and reported that it could not reach.
        self._walk_to(handle[:2] + radial * 0.75, max_steps=1400, stop_at=0.20)
        self._walk_to(handle[:2] + radial * 0.26, max_steps=600, stop_at=0.12)
        self._turn_to(math.atan2(handle[1] - self.robot.position[1],
                                 handle[0] - self.robot.position[0]))

        side = self._hand_for(handle)
        self.robot.grip(side, 0.0)
        miss = self.robot.reach_to(self._site_position("drum_handle_site"), side, passes=4)
        if miss > 0.14:
            # Try the other hand before giving up: which one can reach depends on which way
            # the leaf swung, and neither arm crosses the chest.
            other = "l" if side == "r" else "r"
            self.robot.arm_home(side)
            self.robot.stand(0.2)
            alternative = self.robot.reach_to(
                self._site_position("drum_handle_site"), other, passes=4
            )
            if alternative < miss:
                miss, side = alternative, other

        # Still short? Re-seat on the radius and try again -- do NOT just walk at the handle.
        #
        # Stepping straight toward the handle is what the earlier version did, and it is the
        # bug: the handle is not a fixed point, so driving at it pushes the leaf round and the
        # gap never closes. Re-reading the radius each pass keeps the approach torque-free even
        # though the door has moved, so a genuine few-centimetre shortfall converges instead of
        # running away.
        for _ in range(3):
            if miss <= 0.14:
                break
            target = self._site_position("drum_handle_site")
            radial = target[:2] - hinge
            norm = float(np.linalg.norm(radial))
            radial = radial / norm if norm > 1e-6 else np.array([0.0, -1.0])
            self._walk_to(target[:2] + radial * 0.24, max_steps=400, stop_at=0.10)
            self._turn_to(math.atan2(target[1] - self.robot.position[1],
                                     target[0] - self.robot.position[0]))
            # Try both hands again: re-seating moves the body, and which arm can get there
            # depends on where the leaf ended up, not on which one was chosen first.
            for candidate in (side, "l" if side == "r" else "r"):
                self.robot.arm_home("r")
                self.robot.arm_home("l")
                self.robot.stand(0.2)
                attempt = self.robot.reach_to(
                    self._site_position("drum_handle_site"), candidate, passes=3
                )
                if attempt < miss:
                    miss, side = attempt, candidate
                if miss <= 0.14:
                    break

        if miss > 0.14:
            return SkillResult(
                False,
                f"I could not reach the washing machine door to close it ({miss:.2f} m short).",
                {"miss": miss, "before": before},
            )

        self.robot.grip(side, 1.0)
        self.robot.stand(0.3)
        welded = self.robot.grasp("drum_door")

        # Walk the handle around the arc it actually travels on.
        #
        # The direction is computed, not searched. The handle rotates about the hinge, so the
        # only way it can move is along the TANGENT to that circle, and which of the two
        # tangent directions closes the door is decided by the sign of the joint angle. Two
        # earlier versions guessed a direction and corrected on feedback; both sometimes drove
        # the door the wrong way and left it further open than they found it -- 105 degrees
        # becoming 133, and 136. `hinge` is the door body's own origin, read above.

        best = before
        stalled = 0
        for _ in range(900):
            angle = float(self.robot.data.qpos[address])
            if abs(math.degrees(angle)) < 8.0:
                break

            grip = self.robot.hand_position(side)[:2]
            radius = grip - hinge
            # Tangent to the hinge circle. Rotating the radius by +90 degrees gives the
            # direction of increasing joint angle; closing means going the other way, so the
            # sign of the current angle picks which.
            tangent = np.array([-radius[1], radius[0]])
            norm = float(np.linalg.norm(tangent))
            if norm < 1e-6:
                break
            tangent = tangent / norm * (-1.0 if angle > 0 else 1.0)

            heading = np.array([math.cos(self.robot.yaw), math.sin(self.robot.yaw)])
            lateral = np.array([-heading[1], heading[0]])
            self.robot.step(
                vx=float(np.dot(tangent, heading)) * 0.24,
                vy=float(np.dot(tangent, lateral)) * 0.24,
            )

            magnitude = abs(math.degrees(angle))
            if magnitude < best - 0.5:
                best, stalled = magnitude, 0
            else:
                stalled += 1
                if stalled > 90:
                    break

        self.robot.release()
        self.robot.grip(side, 0.0)

        # NOTE: do not "finish the job" by letting go and shoving the leaf.
        #
        # That was tried and it is much worse than stopping short. Once the door is nearly
        # shut the robot is standing inside the arc, so pressing on the panel drives it back
        # OUT: measured a door left at 22 degrees going to 100, 110, and finally 137 over
        # successive pushes -- wider open than before the skill ran. Holding the handle and
        # walking the arc is the only motion here that reliably closes anything, so when that
        # stalls the honest move is to report the angle, which the caller can act on.

        # BACK OFF THE DOOR BEFORE SETTLING.
        #
        # Walking the handle round ends with the robot standing where the leaf now wants to
        # be, and letting go does not move the body out of the way. Standing still there
        # re-opens the door it just shut: measured 7.98 degrees at the moment of release
        # becoming 22.39 during the settle that followed -- the whole difference between a
        # door reported shut and one reported ajar, with no motion commanded at all.
        #
        # Retreating along the radial line clears the arc without touching the leaf, for the
        # same reason the approach used it: motion along the radius applies no torque.
        retreat = self._site_position("drum_handle_site")
        if retreat is not None:
            outward = retreat[:2] - hinge
            norm = float(np.linalg.norm(outward))
            outward = outward / norm if norm > 1e-6 else np.array([0.0, -1.0])
            heading = np.array([math.cos(self.robot.yaw), math.sin(self.robot.yaw)])
            lateral = np.array([-heading[1], heading[0]])
            for _ in range(60):
                self.robot.step(
                    vx=float(np.dot(outward, heading)) * 0.26,
                    vy=float(np.dot(outward, lateral)) * 0.26,
                )

        self.robot.arm_home(side)
        if not welded:
            log.info("no handle weld in this scene; the door was only pushed")
        self.robot.stand(0.4)
        after = abs(math.degrees(float(self.robot.data.qpos[address])))
        if after < 20.0:
            return SkillResult(
                True,
                f"I closed the washing machine ({before:.0f} degrees open, now {after:.0f}).",
                {"before": before, "after": after},
            )
        return SkillResult(
            False,
            f"I pushed the washing machine door but it is still {after:.0f} degrees open.",
            {"before": before, "after": after},
        )

    def bring_basket(self, where: str | None = None) -> SkillResult:
        """Pick the laundry basket up and carry it to where it is wanted.

        The basket used to be scenery bolted to its plinth and this method could only apologise
        for that. It is a free body now (see assets/home.xml), so "bring the basket over" is an
        errand the robot can actually run: walk to it, take the rim in both hands, carry it,
        and set it down.

        `where` names the destination -- "washer" by default, because that is what the basket
        is fetched FOR. Anything the grounder knows works: washer, counter, basket.
        """
        target = (where or "washer").strip().lower()
        destination = self._place_position(target)
        if destination is None:
            return SkillResult(False, f"I do not know where '{target}' is.")

        picked = self._pick_up_basket()
        if not picked.ok:
            return picked

        # Carry it, and set it down BESIDE the destination rather than in front of it.
        #
        # This is the whole difficulty of the errand and it is not obvious until you watch it
        # fail. Put the basket squarely in front of the washer -- which is what "bring it to
        # the washer" literally asks for -- and it lands exactly where the robot has to STAND
        # to reach into the drum: the approach wants y=0.94 and a basket at y=0.66 spans
        # 0.46..0.86 with its 0.52 x 0.40 footprint. The next step then fails 0.24 m short,
        # having been blocked by the thing it just carried there.
        #
        # A person puts the basket to one SIDE of the machine for the same reason, so that is
        # what this does: offset along the wall the appliance stands against, far enough that
        # the basket is clear of the working space and near enough to drop washing into.
        stand = self._site_position("washer_stand_site") if target == "washer" else None
        drop_at = stand[:2] if stand is not None else self._beside(destination[:2], 0.62)

        # Walk so that THE BASKET ends over the drop point, not the robot.
        #
        # The basket is carried in one hand, out to one side and in front: measured it hanging
        # 0.48 m from where the robot was standing. Aiming the robot at the target therefore
        # parks the basket half a metre away from it, and the put-down then has nowhere to go.
        # The offset is whatever the basket currently is relative to the body, so this works
        # whichever hand is carrying and however the load has swung.
        carried = self._site_position("basket_site")
        offset = (carried[:2] - self.robot.position[:2]) if carried is not None else np.zeros(2)
        self._travel_to(self._approach_point(drop_at - offset, standoff=0.10), target,
                        tolerance=0.20)

        placed = self._put_basket_down(
            floor_z=0.90 if stand is not None else 0.29,
            place=stand[:2] if stand is not None else None,
        )
        if not placed.ok:
            return placed

        # "At the washer" means WITHIN REACH OF IT, not on top of it.
        #
        # The destination is the machine's own centre, which is inside the cabinet -- a basket
        # can never be there. What the errand asks for is a basket a person could use while
        # standing at the machine, so the bar is the arm's working distance plus the basket's
        # own half-width, and anything inside that is where it was meant to go.
        basket = self._site_position("basket_site")
        gap = float(np.linalg.norm(basket[:2] - destination[:2])) if basket is not None else 9.9
        reachable = WORKING_REACH_M + 0.60
        if gap > reachable:
            return SkillResult(
                False,
                f"I carried the basket but it ended up {gap:.1f} m from the {target}.",
                {"gap": gap, "where": target},
            )
        return SkillResult(
            True,
            f"I brought the laundry basket to the {target}.",
            {"gap": gap, "where": target},
        )

    def _pick_up_basket(self) -> SkillResult:
        """Walk to the laundry basket and take its near rim in one hand.

        ONE hand, not two. AN-01's hands rest 0.379 m apart and neither arm crosses the chest,
        so holding two rims means holding two points 0.38 m (long sides) or 0.50 m (short ends)
        apart. An exhaustive sweep of standing positions found the best simultaneous two-hand
        grip anywhere to be 0.395 m from target, five times the grasp tolerance. That is the
        same limit that stops this robot folding a towel two-handed; it is a property of the
        arms. The basket is light (1.6 kg) so that one arm can carry it.
        """
        if self.nav is not None and not self.nav.survey("basket"):
            self._look_around_for("basket")

        grip = self._site_position("basket_grip")
        if grip is None:
            return SkillResult(False, "I cannot find the laundry basket.")

        # WHERE TO STAND: 0.40 m from the RIM. The window is narrow and was measured, not
        # chosen. Below about 0.38 the body walks into the basket and knocks it off its
        # plinth -- measured it ending tipped at z=0.235. Above about 0.44 the rim is past
        # what the arm can reach (0.35 m from a shoulder ~0.10 m forward of the base) and the
        # grasp stalls 0.18 m short however many passes it makes. At 0.40 the basket is still
        # sitting level and untouched at z=0.622 when the hand arrives.
        # Approach from the side the RIM FACES, not from wherever the robot happens to be.
        #
        # `_approach_point` draws its line from the robot's current position, so the standoff
        # lands on whichever side it arrived from. Coming from the north that put the body
        # between the basket's long wall and the hand, and the grasp missed by 0.06 m -- while
        # the identical code from the other keyframe succeeded. The grip site is on one
        # specific wall, so the robot has to stand off THAT wall.
        centre = self._site_position("basket_site")
        outward = grip[:2] - centre[:2]
        norm = float(np.linalg.norm(outward))
        outward = outward / norm if norm > 1e-6 else np.array([0.0, -1.0])
        self._walk_to(grip[:2] + outward * 0.40, max_steps=1200, stop_at=0.10)
        self._turn_to(math.atan2(grip[1] - self.robot.position[1],
                                 grip[0] - self.robot.position[0]))

        side = self._hand_for(grip)
        self.robot.grip(side, 0.0)

        # Come down ONTO the rim, and get the hand ABOVE RIM HEIGHT BEFORE moving it in.
        #
        # Two separate mistakes were made here, in order. Reaching straight at the rim drives
        # the palm into the near wall, because the grip target sat at exactly the wall's top
        # edge -- that is why the site is now 0.05 m proud of it. But simply aiming at the
        # raised target is not enough either: the arm hangs at z=0.58, below the rim, so the
        # straight-line path to a point above the rim goes THROUGH the basket. Measured that
        # sweeping it off its plinth onto the floor, 1.03 m away.
        #
        # So the hand is raised first, in place, and only then moved over the rim and down.
        # RAISE AND TRAVERSE IN SMALL STEPS. This is the part that matters.
        #
        # A single reach_to for a large move solves for the destination and drives straight
        # there; the arm swings through whatever lies between, and from a hand hanging at
        # z=0.58 the path to a point above the rim goes clean through the basket -- measured
        # it launched 1.0 m across the room. Asking for the same motion in eighths never
        # touches it at all: every intermediate pose is itself reachable, so the arm goes
        # round rather than through.
        hand = self.robot.hand_position(side)
        top = grip[2] + 0.12
        for k in range(1, 9):
            self.robot.reach_to(
                np.array([hand[0], hand[1], hand[2] + (top - hand[2]) * k / 8.0]),
                side, passes=1, settle_steps=40,
            )
        # Now traverse to over the rim, in steps for the same reason.
        start = self.robot.hand_position(side).copy()
        above = grip + np.array([0.0, 0.0, 0.10])
        for k in range(1, 7):
            self.robot.reach_to(
                start + (above - start) * k / 6.0, side, passes=1, settle_steps=40
            )

        # Reach, keeping the best attempt and stopping the moment the basket starts to move.
        #
        # Later passes can be far WORSE than the first -- measured 0.047 m then 0.999 m --
        # because the hand has brushed the rim, the basket has shifted, and the arm is now
        # chasing a target that runs away from it. Retrying in that state bats the basket
        # around the room, so the loop keeps the closest pose it ever reached and bails out
        # rather than continuing.
        miss = float("inf")
        best = float("inf")
        best_pose = self.robot.data.qpos.copy()
        anchor = self._site_position("basket_site").copy()
        for _ in range(4):
            point = self._site_position("basket_grip")
            if point is None:
                break
            miss = self.robot.reach_to(point, side, passes=2)
            if miss < best:
                best, best_pose = miss, self.robot.data.qpos.copy()
            if miss <= 0.045:
                break
            moved = float(
                np.linalg.norm(self._site_position("basket_site")[:2] - anchor[:2])
            )
            if moved > 0.02:
                log.info("the basket has shifted; not chasing it")
                break

        if best < miss:
            self.robot.data.qpos[:] = best_pose
            self.robot.data.qvel[:] = 0.0
            mujoco.mj_forward(self.robot.model, self.robot.data)
            point = self._site_position("basket_grip")
            if point is not None:
                miss = float(np.linalg.norm(self.robot.hand_position(side) - point))

        # A RIGID object needs a tighter grasp than cloth does. GRASP_TOLERANCE_M is 0.075 m,
        # sized for a towel: the gripper spans 6 cm and cloth deforms, so a weld made 7 cm
        # away looks fine. A basket does not deform -- welding it from 0.073 m froze it
        # hanging 7 cm off the palm, and since a weld is a hard constraint the contact solver
        # cannot push apart, the box then passed through whatever it met. 0.030 m is inside
        # the gripper's own span, so a weld made there is a hand really closed on the rim.
        # 0.045 m, from the gripper's geometry rather than a guess. The fingers sit 0.041 m
        # apart and the rim they close on is 0.020 m thick, so a palm centred within about
        # half the finger separation of the rim has the wall between its fingers. An earlier
        # 0.030 m was arbitrary and rejected grasps that would have held: the log read
        # "0.04 m off" for a hand that was in fact close enough.
        if miss > 0.045:
            self.robot.grip(side, 0.0)
            self.robot.arm_home(side)
            return SkillResult(
                False,
                f"I could not get hold of the laundry basket ({miss:.2f} m off).",
                {"miss": miss},
            )

        self.robot.grip(side, 1.0)
        self.robot.stand(0.3)
        self.robot.grasp("basket", side)
        self._holding_basket = True
        self._basket_hands = (side,)

        # Lift clear of the plinth, then CARRY IT LOW AND OUT TO THE SIDE.
        #
        # Where the basket rides is a perception problem, not an ergonomic one. Held up in
        # front it sits 0.66 m from the head camera and fills the frame: measured the washing
        # machine visible before the pickup and invisible after it, which is what produced 255
        # consecutive "lost sight of washer" lines in one errand. The robot was not confused --
        # it was carrying its own blindfold.
        #
        # So the basket goes down to hip height and out past the shoulder line, which is how a
        # person carries a laundry basket precisely because it keeps the way ahead visible.
        # Lift enough to clear the plinth, in steps so the arm does not swing through it.
        hand = self.robot.hand_position(side)
        for k in range(1, 5):
            self.robot.reach_to(
                np.array([hand[0], hand[1], hand[2] + 0.10 * k / 4.0]),
                side, passes=1, settle_steps=45,
            )
        self.robot.stand(0.2)

        # NO SIDEWAYS CARRY POSE. It was tried and it costs more than it buys.
        #
        # The basket does block the head camera while carried -- measured the washing machine
        # visible before the pickup and not after -- and swinging the arm out and down does
        # clear the lens. But it also moves the load away from where the put-down expects it,
        # and the put-down is the harder problem: with the pose in, the basket ended on the
        # floor 2.9 m from the washer; with it out, it lands on the stand.
        #
        # The blindness is handled where it actually belongs, in `_travel_to`, which turns the
        # body to FACE the destination before looking for it. That fixes the real complaint --
        # the robot walking a whole approach reporting "lost sight of washer" -- because the
        # problem was never the basket in the frame, it was the machine being 173 degrees
        # behind the robot where no camera could see it.
        self.robot.stand(0.3)
        return SkillResult(True, "I picked up the basket.", {"hand": side, "miss": miss})

    def _put_basket_down(self, floor_z: float = 0.29,
                         place: np.ndarray | None = None) -> SkillResult:
        """Set the basket down and let go.

        `floor_z` is the height basket_site reads once the basket is resting: 0.29 on the
        ground, 0.90 on a stand. The robot cannot actually reach the ground (see the note on
        washer_stand in home.xml), so in practice this is always a stand.
        """
        if not self._holding_basket:
            return SkillResult(False, "I am not holding the basket.")
        # Lower until THE BASKET is near the floor, watching the basket rather than the hand.
        #
        # Aiming the hand at a fixed height does not work: the hand and the basket are 0.25 m
        # apart, and "hand at 0.74" left the basket hanging at z=0.28 -- a quarter of a metre
        # up, from which it drops, tips, and spills whatever is inside. The measurement that
        # matters is the underside of the basket's own floor, so that is what this drives to.
        # Keep the basket LEVEL while lowering, and judge "down" by its floor, not by a site.
        #
        # Two failures made this necessary. Lowering by hand height alone let the basket swing
        # 0.66 m sideways on the way down; pinning x and y fixed that. Then judging "is it
        # down" from basket_site's height let a basket that had TIPPED pass the test while
        # still in the air -- it came to rest on basket_w3, a side wall, because a tipped
        # basket's centre is lower than an upright one's. So the test below is on the body
        # origin, which is where the basket's own floor is, and the hands stay level with each
        # other so it does not tip in the first place.
        #
        # Lower straight DOWN, holding x and y fixed at where the hands started.
        #
        # Commanding only the height and letting the IK pick x and y looks equivalent and is
        # not: the arm swings as it descends, and the basket travelled 0.66 m sideways on the
        # way to the floor -- from x=-1.09 to -0.43 -- landing it well away from the machine it
        # had just been carried to. Pinning the horizontal target is what makes "put it down"
        # mean "put it down HERE".
        # First bring the basket OVER the target, then lower. Lowering from wherever the walk
        # happened to stop puts it down next to the stand rather than on it: measured the
        # basket arriving 0.29 m to one side, which is most of a basket's width.
        # Iterate: each pass closes most of the remaining offset but not all of it, because
        # the arm is carrying a 1.6 kg box at full extension and settles short of its command.
        # Measured the offset shrinking 0.45 -> 0.18 -> 0.12 m over three passes, which is
        # convergence -- it simply needs more of them than the shape of the loop first allowed.
        if place is not None:
            for _ in range(8):
                basket = self._site_position("basket_site")
                if basket is None:
                    break
                offset = np.asarray(place)[:2] - basket[:2]
                if float(np.linalg.norm(offset)) < 0.10:
                    break
                for side in (self._basket_hands or ("r", "l")):
                    hand = self.robot.hand_position(side)
                    self.robot.reach_to(
                        np.array([hand[0] + offset[0], hand[1] + offset[1], hand[2]]),
                        side, passes=2,
                    )
                self.robot.stand(0.25)

        # Only the hand that is actually holding the basket. Driving both when one is empty
        # swings the free arm into the load.
        carrying = [s for s in ("r", "l") if self.cloth.holding(s) is None] if False else None
        anchors = {
            side: self.robot.hand_position(side).copy()
            for side in (self._basket_hands or ("r", "l"))
        }
        for _ in range(8):
            basket = self._site_position("basket_site")
            if basket is None:
                break
            # basket_site rides 0.275 above the body origin, which is itself 0.015 above the
            # ground once the basket rests on the floor -- so the site reads 0.29 when down.
            # Stop at whatever surface is under it. On the floor the site reads 0.29; on a
            # stand it reads 0.90. `floor_z` is set by the caller to whichever applies.
            if basket[2] <= floor_z + 0.05:
                break
            drop = min(0.08, basket[2] - floor_z)
            # One shared height for both hands. Letting each descend from its own current
            # height lets a lag on one side become a tilt, and a tilted basket lands on a
            # wall instead of its floor.
            level = min(self.robot.hand_position(s)[2] for s in anchors) - drop
            moved = False
            for side, anchor in anchors.items():
                before = self.robot.hand_position(side)[2]
                self.robot.reach_to(
                    np.array([anchor[0], anchor[1], level]), side, passes=2
                )
                moved = moved or (before - self.robot.hand_position(side)[2]) > 0.005
            self.robot.stand(0.3)
            # The arm bottoms out before the basket reaches the floor -- it cannot crouch --
            # so stop when lowering stops working rather than grinding against the limit.
            if not moved:
                log.info("arm cannot lower any further; releasing from here")
                break
        self.robot.stand(0.4)
        self.robot.release()
        for side in (self._basket_hands or ("r", "l")):
            self.robot.grip(side, 0.0)
        self._holding_basket = False
        self.robot.stand(1.0)
        # Arms away BEFORE stepping back, or the retreat drags the basket along with them:
        # letting go of a weld is not the same as the hands being clear of the rim.
        for side in (self._basket_hands or ("r", "l")):
            self.robot.arm_home(side)
        self.robot.stand(0.5)
        for _ in range(45):
            self.robot.step(vx=-0.28)
        self.robot.stand(0.3)
        return SkillResult(True, "I put the basket down.")

    def take_out(self, towel: str | None = None, side: str = "r") -> SkillResult:
        """Take a towel out of the drum and hold it.

        Grasps a corner rather than the middle. A towel lifted by its centre comes out as a
        bundle, which cannot then be laid flat or folded without putting it down and starting
        again.
        """
        sheet = self._sheet(towel)
        if sheet is None:
            return SkillResult(False, "I cannot see a towel to take out.")

        # Choose where to take hold BEFORE positioning, and position for that point rather
        # than for the middle of the towel. Standing off the centre of a sheet lying in the
        # drum left the grasp 0.19 m short: the centre is 0.13 m further into the cavity than
        # the near edge, and the whole reach budget is 0.25 m.
        #
        # Anywhere on the perimeter will do -- the sheet hangs from whatever edge point holds
        # it -- so this takes the nearest one rather than insisting on a corner. Corners are
        # for folding; a corner is the furthest part of the sheet from a robot standing square
        # on, and reaching for one missed by 0.18 m where an edge vertex was 0.16 m away.
        # Only the HALF OF THE PERIMETER NEAREST THE ROOM is worth offering.
        #
        # A towel lying in the drum spans y=1.31..1.57 with the mouth ring at y=1.29, so its
        # far edge is a quarter of a metre deeper into the cavity than its near edge -- past
        # the arm's whole 0.25 m budget. Offering the whole perimeter lets the search pick a
        # far-edge vertex and score the spot on a reach it can never make: measured it
        # returning vertex 55 at y=1.574, and the grasp missing by 0.226 m, while vertex 0 at
        # y=1.313 sat right at the mouth. Nothing downstream can recover from that choice, so
        # the deep half is filtered out before the search ever sees it.
        # Ranked by DEPTH INTO THE DRUM, not by distance from the robot. The robot has not
        # taken up its standing position yet, so "nearest to me" is measured from wherever the
        # last skill left it and ranks the towel wrongly -- tried that, and it picked worse
        # vertices still (0.46 m, and the sweep dragged the sheet to a 0.285 m span). How deep
        # a vertex sits is a fact about the machine, so it is read off the drum instead.
        perimeter = sheet.perimeter()
        mouth = self._geom_position("drum_ring_b")
        mouth_y = float(mouth[1]) if mouth is not None else 1.288
        depths = sorted(
            perimeter,
            key=lambda v: abs(float(self.cloth.vertex_position(sheet, v)[1]) - mouth_y),
        )
        graspable = depths[: max(3, len(depths) // 3)]
        vertex, _ = self.cloth.nearest_vertex(sheet, self.robot.position, among=graspable)
        self._stand_near(self.cloth.vertex_position(sheet, vertex))

        # Then find a spot the towel is genuinely reachable from. Walking up and facing it is
        # not enough at a washer: the open door hangs across its own opening, so the clear
        # line in is off to one side, and which side depends on how far the door swung.
        _, vertex, side = self.find_standing_spot(sheet, graspable)
        grasped, miss = self._grasp_vertex(sheet, vertex, side)
        if not grasped:
            return SkillResult(
                False,
                f"I reached for the towel but could not get hold of it ({miss:.2f} m off).",
                {"miss": miss},
            )

        # Draw it out of the drum before lifting: raising it inside the cavity jams it on the
        # roof, which is only 0.19 m above where it lies.
        for _ in range(90):
            self.robot.step(vx=-0.22)
        self._lift_to(CARRY_HEIGHT_M, side)

        height = float(self.cloth.sheet_centre(sheet)[2])
        if height < 0.55:
            return SkillResult(
                False, "I had hold of the towel but it slipped out.", {"height": height}
            )
        return SkillResult(
            True, "I took the towel out of the washing machine.", {"height": height}
        )

    def _loaded_hand(self, prefer: str | None = None) -> str | None:
        """Whichever hand is actually holding cloth.

        Callers must not assume the right hand. Which hand picks a towel up is decided at
        grasp time by where the cloth is -- neither arm crosses the chest -- so a skill that
        hard-codes "r" reports "I am not holding anything" while the left hand is holding it.
        """
        if prefer and self.cloth.holding(prefer):
            return prefer
        for side in ("r", "l"):
            if self.cloth.holding(side):
                return side
        return None

    def put_in_basket(self, side: str | None = None) -> SkillResult:
        """Carry whatever is in hand to the laundry basket and drop it in."""
        side = self._loaded_hand(side)
        if side is None:
            return SkillResult(False, "I am not holding anything to put in the basket.")
        held = self.cloth.holding(side)
        if held is None:
            return SkillResult(False, "I am not holding anything to put in the basket.")

        basket = self._site_position("basket_site")
        if basket is None:
            return SkillResult(False, "I cannot find the laundry basket.")

        # Stand well back from the basket and reach OVER it, rather than pressing up against
        # it the way _stand_near does for things that have to be grasped.
        #
        # The basket is 0.52 x 0.40 on a plinth, and the robot is 0.52 wide. Closing to arm's
        # length puts its feet inside the basket's own footprint -- measured ending at
        # (-0.42, -0.07) with the body spanning y -0.33..0.19 against a basket edge at -0.35,
        # so it was standing on the rim. It then could not walk away: 0.029 m of travel in 300
        # steps, which stranded every skill that ran after it.
        #
        # Dropping a towel in does not need the base close, only the hand high -- but it does
        # need the hand OVER THE MIDDLE, and 0.62 m back was too far for that. The arm reaches
        # 0.25 m, so from there the release happened 0.24 m short of the target: the towel came
        # out at x=-0.47 against a basket centre of x=-0.30, draping over the near rim and
        # slumping down the outside to z=0.44. It counted as "in the basket" on the footprint
        # test while actually hanging off it, and the next skill could not pick it up again.
        #
        # 0.40 m is the measured sweet spot: the reach lands at 0.059 m and the towel drops at
        # x=-0.324, genuinely inside. It is still well clear of the 0.52 x 0.40 basket's own
        # footprint, so the feet stay off the rim and the back-off below still frees the body.
        approach = basket[:2] + np.array([0.0, -0.40])
        self._walk_to(approach, max_steps=900, stop_at=0.14)
        self._turn_to(math.atan2(basket[1] - self.robot.position[1],
                                 basket[0] - self.robot.position[0]))
        # Hold the towel over the middle of the basket, then open the hand. Releasing at
        # carrying height rather than lowering into the basket keeps the arm clear of the rim.
        above = np.array([basket[0], basket[1], max(basket[2] + 0.12, 0.75)])
        self.robot.reach_to(above, side, passes=3)
        self.robot.grip(side, 0.0)
        self.cloth.release(side)
        # Move the arm away BEFORE waiting, then give the towel time to actually fall.
        # Releasing the weld is not the same as the towel being free: measured it hanging at
        # z=0.62 directly over a basket whose rim is at 0.32, draped across the forearm, which
        # reads as "did not land in the basket" for a drop that was perfectly aimed.
        self.robot.arm_home(side)
        self.robot.stand(2.0)

        # Then STEP BACK OFF the basket.
        #
        # Reaching over the rim leaves the robot pressed against it, and a body wedged on
        # scenery cannot walk away: measured 0.029 m of travel in 300 forward steps, with
        # basket_w1 and basket_floor in contact. Every skill that ran afterwards was stranded
        # -- close_washer reported the door 0.87 m out of reach when the real problem was that
        # the robot could not leave the basket. Backing off is cheap and frees it.
        for _ in range(70):
            self.robot.step(vx=-0.30)
        self.robot.stand(0.3)

        sheet = self.cloth.sheets.get(held[0])
        landed = self.cloth.sheet_centre(sheet) if sheet else None
        if landed is None:
            return SkillResult(True, "I put the towel in the basket.")
        # "In the basket" is judged against the basket's own floor, which is on a plinth, not
        # against the ground. A fixed z<0.45 test was left over from a floor-standing basket
        # and failed a towel sitting correctly inside the raised one.
        floor = self._geom_position("basket_floor")
        lip = float(floor[2]) if floor is not None else 0.0
        inside = (
            abs(landed[0] - basket[0]) < 0.34
            and abs(landed[1] - basket[1]) < 0.28
            and landed[2] < lip + 0.30
        )
        if not inside:
            return SkillResult(
                False,
                "I let go of the towel but it did not land in the basket.",
                {"where": landed.tolist()},
            )
        return SkillResult(True, "I put the towel in the basket.", {"where": landed.tolist()})

    def put_on_counter(self, towel: str | None = None, side: str | None = None) -> SkillResult:
        """Move a towel to the washstand counter and lay it out flat.

        The counter is the surface the task description calls "the space by the washbasin where
        laundry can be put down". Folding happens here because it is the only surface at a
        height the arm can work on: the floor is below the arm's range entirely.
        """
        counter = self._geom_position("counter_g")
        if counter is None:
            return SkillResult(False, "I cannot find the counter.")
        # Aim for the middle of the counter's depth, not its near lip. The arm can reach the
        # near half standing in front of it, but a towel released over the edge slides off:
        # the sheet is 0.30 m deep and the counter only 0.60 m, so letting go 0.18 m in from
        # the front leaves half of it hanging in mid-air. Aiming at the centre line puts the
        # whole sheet on the surface. Still clear of the basin, which is at the far end in x.
        surface = np.array([counter[0] - 0.20, counter[1], counter[2] + 0.03])

        side = self._loaded_hand(side)
        if side is None:
            # Nothing in hand yet, so pick the towel up first -- from the drum if that is where
            # it is, and off whatever surface it is lying on otherwise.
            #
            # Only try the drum if the towel is actually IN the drum. take_out approaches with
            # _stand_near, which walks the body up against the nearest fixture; run on a towel
            # sitting in the basket it wedges the robot on the rim, fails, and leaves the
            # general pick-up below to search from a pose it cannot recover from -- 0.21 m
            # where going straight to the search gets 0.059 m and succeeds.
            in_drum = False
            sheet_now = self._sheet(towel)
            drum = self._geom_position("drum_ring_b")
            if sheet_now is not None and drum is not None:
                where = self.cloth.sheet_centre(sheet_now)
                in_drum = (
                    abs(float(where[0]) - float(drum[0])) < 0.30
                    and float(where[1]) > float(drum[1]) - 0.10
                )
            picked = self.take_out(towel) if in_drum else SkillResult(False, "")
            side = self._loaded_hand()
            if not picked.ok or side is None:
                sheet = self._sheet(towel)
                if sheet is None:
                    return SkillResult(False, "I cannot find a towel to move.")
                # Take hold of the TOP of the pile, not just any vertex.
                #
                # Not the perimeter: a towel dropped into a basket is crumpled, so its "edge"
                # is wherever the folds put it. But offering all 63 vertices is no better --
                # most of them are buried under the rest of the sheet or below the basket's
                # 0.87 m rim, and the sweep spends its trials on points the hand can only get
                # to by going through a wall. Measured 0.239 m that way, on a pick-up where an
                # exhaustive check of the topmost vertices found 0.026 m.
                #
                # Height is what makes a vertex graspable here, so rank by it and offer the
                # sweep the highest dozen. That is the part of the towel standing proudest of
                # the pile, which is exactly what a person reaches for.
                graspable = sorted(
                    range(sheet.count),
                    key=lambda v: float(self.cloth.vertex_position(sheet, v)[2]),
                    reverse=True,
                )[:12]
                # Straight to the search -- NO _stand_near first.
                #
                # _stand_near walks the body forward until it meets the fixture, which is the
                # right move at a drum mouth and the wrong one at a basket: it ends up pressed
                # against the rim at 0.20 m from the cloth, and the sweep cannot undo that.
                # Every candidate it then tries starts from a wedged pose, and the search
                # returned 0.188 m where going straight in returns 0.045 m and the grasp
                # actually succeeds. find_standing_spot walks itself to the spot it picks, so
                # the approach was never needed here.
                _, vertex, side = self.find_standing_spot(sheet, graspable)
                grasped, miss = self._grasp_vertex(sheet, vertex, side)
                if not grasped:
                    return SkillResult(
                        False, f"I could not pick up the towel ({miss:.2f} m off).",
                        {"miss": miss},
                    )
                self._lift_to(CARRY_HEIGHT_M, side)

        held = self.cloth.holding(side)

        # Approach the counter from IN FRONT of it (south, -y), not from whichever side the
        # robot happens to be on. Walking straight at the surface from the west put the robot
        # level with the counter's end, and the towel went down across the corner at x=0.54
        # against an edge at x=0.52 -- half on, half off, and it slid.
        #
        # Standing back in y and reaching north puts the whole sheet over the surface.
        approach = np.array([surface[0], surface[1] - 0.75])
        self._walk_to(approach, stop_at=0.25)
        self._turn_to(math.pi / 2)
        self._stand_near(surface)

        # LAY the towel down, do not drop it.
        #
        # A carried towel hangs as a vertical curtain from the one vertex the hand holds --
        # measured dz=0.363 against dx=0.253 -- so releasing it over the surface lets it
        # collapse into a heap under itself: 0.367 m across in the air, 0.19 m on the counter.
        # Everything downstream then fails on a sheet that is not flat, and `_spread` cannot
        # open a bundle that tight once it has formed.
        #
        # Touching the far edge down first and drawing the hand back toward the robot trails
        # the cloth out along the surface instead, the way a person lays out a sheet. Measured
        # 0.249 m against 0.134 m for the same errand. It does not fully flatten the towel --
        # one hand cannot -- but it lands it open rather than balled up.
        far = np.array([surface[0], surface[1] + 0.13, surface[2] + 0.02])
        near = np.array([surface[0], surface[1] - 0.13, surface[2] + 0.02])
        self.robot.reach_to(far, side, passes=3)
        for fraction in np.linspace(0.0, 1.0, 6)[1:]:
            self.robot.reach_to(far + (near - far) * fraction, side, passes=2)
        self.robot.grip(side, 0.0)
        self.cloth.release(side)
        # Arm away first, THEN wait. Releasing the weld does not free the towel if it is still
        # draped over the forearm -- the same thing that made a perfectly aimed drop into the
        # basket read as a miss.
        self.robot.arm_home(side)
        self.robot.stand(2.0)
        # Step back off the counter, for the same reason as the basket: a robot leaning on a
        # fixture cannot walk away from it, and whatever skill runs next inherits the problem.
        for _ in range(70):
            self.robot.step(vx=-0.30)
        self.robot.stand(0.3)

        sheet = self.cloth.sheets.get(held[0]) if held else self._sheet(towel)
        if sheet is None:
            return SkillResult(True, "I put the towel on the counter.")
        landed = self.cloth.sheet_centre(sheet)
        # Judge against the counter's TOP SURFACE, and require the towel to be within its
        # footprint as well as at its height. The first version tested only
        # `z > counter_centre - 0.05`, which a towel lying on the floor at z=0.014 passes
        # nowhere near -- but a towel draped over the edge and hanging down passes easily,
        # and so does one that slid off onto a cabinet. Both were reported as success.
        top = float(counter[2]) + 0.02
        on_top = (
            landed[2] > top - 0.12
            and abs(landed[0] - counter[0]) < 0.70
            and abs(landed[1] - counter[1]) < 0.34
        )
        if not on_top:
            return SkillResult(
                False,
                "I let go of the towel but it slid off the counter.",
                {"where": landed.tolist()},
            )
        return SkillResult(
            True, "I put the towel on the counter.", {"where": landed.tolist()}
        )

    def fold(self, towel: str | None = None) -> SkillResult:
        """Fold a towel in half on the counter.

        Folding is two-handed by nature: take the two corners of the near edge, carry them over
        to the far edge, and put them down. One hand can only ever bring one corner across,
        which drags the sheet into a diagonal rather than folding it.

        Success is measured, not assumed. A flat 0.40 x 0.30 sheet spans about 0.50 m corner to
        corner; folded in half it should span appreciably less, and the skill reports the actual
        before-and-after span rather than claiming victory because the arms moved.
        """
        sheet = self._sheet(towel)
        if sheet is None:
            return SkillResult(False, "I cannot see a towel to fold.")

        before = self.cloth.sheet_extent(sheet)
        centre = self.cloth.sheet_centre(sheet)

        # A towel that arrived from the drum lands crumpled, and folding a crumpled sheet drags
        # it off the counter rather than folding it: the corners are bunched, so carrying one
        # across pulls the whole gathered mass with it. Measured a fold from the washer ending
        # with the towel on the floor at z=0.015 where the same fold on a flat sheet works.
        #
        # So spread it first if it is not already flat. A flat 0.40 x 0.30 sheet spans 0.50 m;
        # anything much under that is bunched up.
        if before < 0.42 and centre[2] > 0.6:
            log.info("towel is bunched (span %.2f m); spreading it before folding", before)
            self._spread(sheet)
            before = self.cloth.sheet_extent(sheet)
            centre = self.cloth.sheet_centre(sheet)

        if centre[2] < 0.6:
            return SkillResult(
                False,
                "The towel is not on a surface I can work on - put it on the counter first.",
                {"height": float(centre[2])},
            )

        self._stand_near(centre)

        # Work out which edge is actually nearest, rather than trusting the grid's own idea of
        # "near". ClothSheet names edges by their position in the vertex grid, which is fixed
        # at compile time and says nothing about where the robot ended up: on the counter the
        # grid's "near" edge sat at x=0.85 with the robot at x=1.43, so folding reached for the
        # far side of the towel and got nothing.
        near, far = self._near_and_far_edges(sheet)

        # Order both edges the same way across the body, so each near corner is carried to the
        # far corner on its own side and the sheet folds over itself instead of being dragged
        # into a diagonal. Sorting by world y would not do -- the robot may be facing any way --
        # so they are compared in the body frame.
        positions = {v: self.cloth.vertex_position(sheet, v) for v in near + far}
        near.sort(key=lambda v: self._lateral_of(positions[v]))
        far.sort(key=lambda v: self._lateral_of(positions[v]))

        # Fold ONE CORNER AT A TIME, repositioning between them.
        #
        # The intent was to take both corners of the near edge at once, as a person does with a
        # small towel. This arm cannot: the corners are 0.40 m apart, and the best standing
        # spot found by an exhaustive sweep still left the worse hand 0.22 m from its corner,
        # against a 0.075 m grasp tolerance. There is no position where both are in reach,
        # because the sheet is wider than the span the two hands share.
        #
        # So each corner is carried across on its own, and the robot walks between them. That
        # is also what a person does with a bath towel, and it converges for the same reason:
        # the constraint is the width of the cloth, not the number of hands.
        landings = {near[0]: far[0], near[-1]: far[-1]}
        folded = 0
        misses = []
        for vertex in (near[0], near[-1]):
            # Take the hand the search actually succeeded with. Re-deciding it afterwards from
            # geometry throws the result away: the sweep found a spot where the RIGHT hand was
            # 0.078 m from the corner, then _hand_for looked at the same corner, called it a
            # left-hand target, and the left hand could not get near it.
            _, vertex, side = self.find_standing_spot(sheet, [vertex], span=0.35)
            ok, miss = self._grasp_vertex(sheet, vertex, side)
            if not ok:
                misses.append(miss)
                log.info("could not grasp a corner for folding (%.3f m off)", miss)
                self.robot.arm_home(side)
                continue

            # Lift just clear of the surface, carry across, and set down.
            #
            # The lift height is narrow and was found by measurement, not chosen:
            #
            #   0.16 m  the corner peels the whole sheet off the counter and the towel ends up
            #           on the floor -- and it still LOOKS folded, because a towel gathered on
            #           the floor has a small span too (measured 0.28 -> 0.24 m at z=0.014)
            #   0.07 m  too low: the corner drags across the surface and nothing folds
            #   0.10 m  still nothing (span unchanged at 0.50 m)
            #   0.13 m  folds and stays put: 0.50 -> 0.28 m with the sheet at z=0.811
            #
            # The window is that tight because the sheet is only 0.30 m deep; lift much more
            # than a third of that and you are picking the towel up rather than folding it.
            hand = self.robot.hand_position(side)
            self.robot.reach_to(np.array([hand[0], hand[1], hand[2] + 0.13]), side, passes=2)
            landing = self.cloth.vertex_position(sheet, landings[vertex])
            self.robot.reach_to(
                np.array([landing[0], landing[1], landing[2] + 0.05]), side, passes=3
            )
            self.robot.grip(side, 0.0)
            self.cloth.release(side)
            self.robot.stand(0.6)
            self.robot.arm_home(side)
            folded += 1

        self.robot.stand(0.5)
        if folded == 0:
            closest = min(misses) if misses else float("inf")
            return SkillResult(
                False,
                f"I could not get hold of the towel to fold it ({closest:.2f} m off).",
                {"extent": before, "closest_miss": closest},
            )
        grabbed = folded

        after = self.cloth.sheet_extent(sheet)
        landed = self.cloth.sheet_centre(sheet)

        # A towel that ended up on the floor is not a folded towel, however small its span.
        # This check exists because the span test alone passed one: dragging a sheet off the
        # counter gathers it up, so the extent drops exactly as a real fold would make it drop
        # -- 0.28 m to 0.24 m -- and the skill reported success for a towel lying at z=0.014.
        if landed[2] < centre[2] - 0.25:
            return SkillResult(
                False,
                f"I pulled the towel off the surface while folding it "
                f"(it is {landed[2]:.2f} m up now).",
                {"extent_before": before, "extent_after": after, "height": float(landed[2])},
            )

        if after < FOLDED_EXTENT_M:
            return SkillResult(
                True,
                f"I folded the towel ({before:.2f} m across, now {after:.2f} m).",
                {"extent_before": before, "extent_after": after, "corners_folded": grabbed},
            )
        return SkillResult(
            False,
            f"I tried to fold the towel but it is still {after:.2f} m across.",
            {"extent_before": before, "extent_after": after, "corners_folded": grabbed},
        )

    # -- reporting ----------------------------------------------------------------

    def describe_view(self) -> SkillResult:
        """Say what is in front of Momo right now."""
        obs = self.robot.look()
        found = []
        for name in ("towel", "basket", "washer", "counter"):
            if self.grounder.find(obs.rgb, name):
                found.append(name)

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
        """Where Momo is, in terms someone can picture."""
        x, y = float(self.robot.position[0]), float(self.robot.position[1])
        washer = self._geom_position("drum_ring_b")
        counter = self._geom_position("counter_g")
        basket = self._site_position("basket_site")

        places = {"the washing machine": washer, "the counter": counter, "the basket": basket}
        nearest, distance = None, float("inf")
        for label, point in places.items():
            if point is None:
                continue
            offset = float(np.linalg.norm(point[:2] - np.array([x, y])))
            if offset < distance:
                nearest, distance = label, offset

        if nearest and distance < 1.2:
            where = f"next to {nearest}"
        elif nearest:
            where = f"in the middle of the room, {distance:.1f} m from {nearest}"
        else:
            where = "in the laundry room"
        return SkillResult(
            True,
            f"I am {where}.",
            {"x": x, "y": y, "nearest": nearest, "distance": distance},
        )

    def go_home(self) -> SkillResult:
        """Walk back to where Momo was standing when she was given the task."""
        distance = self._walk_to(self._home, stop_at=0.25)
        self._turn_to(self._home_heading)
        if distance < 0.5:
            return SkillResult(True, "I went back to where I started.")
        return SkillResult(
            False, f"I could not get back to where I started ({distance:.1f} m short)."
        )

    # -- scene lookups ------------------------------------------------------------

    def _site_position(self, name: str) -> np.ndarray | None:
        site = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site < 0:
            return None
        return self.robot.data.site_xpos[site].copy()

    def _geom_position(self, name: str) -> np.ndarray | None:
        geom = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom < 0:
            return None
        return self.robot.data.geom_xpos[geom].copy()

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
            "open_washer": lambda: self.open_washer(),
            "close_washer": lambda: self.close_washer(),
            "bring_basket": lambda: self.bring_basket(argument or where),
            "take_out": lambda: self.take_out(argument),
            "to_basket": lambda: self.put_in_basket(),
            "to_counter": lambda: self.put_on_counter(argument),
            "fold": lambda: self.fold(argument),
            "describe": lambda: self.describe_view(),
            "where": lambda: self.report_position(),
            "home": lambda: self.go_home(),
            "report": lambda: SkillResult(True, argument or "Done."),
        }
        handler = handlers.get(action)
        if handler is None:
            return SkillResult(False, f"I do not know how to '{action}'.")
        log.info("skill: %s(%s)", action, argument or "")
        return handler()


# What a person might call each towel, mapped onto the name the scene uses.
_TOWEL_WORDS: dict[str, str] = {
    "towel_a": "towel_a", "a": "towel_a",
    "blue": "towel_a", "青": "towel_a", "青い": "towel_a", "ブルー": "towel_a",
    "towel_b": "towel_b", "b": "towel_b",
    "pink": "towel_b", "ピンク": "towel_b", "桃": "towel_b",
}


def _canonical_towel(description: str) -> str | None:
    """Map free text onto a sheet name, longest match first."""
    text = description.lower().strip()
    matches = [(len(k), v) for k, v in _TOWEL_WORDS.items() if k in text]
    return max(matches)[1] if matches else None
