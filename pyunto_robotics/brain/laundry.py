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

    Folding a towel that has just come out of the drum does not reliably succeed.

Every individual step does. `open_washer`, `take_out` and `put_on_counter` all pass, and `fold`
is deterministic and repeatable on a towel that is lying flat -- 0.50 m across to 0.28 m, twice
out of two from the `counter` keyframe. But a towel carried out of the washer lands bunched, and
folding a bunched sheet drags the whole gathered mass off the counter instead of folding it.
`_spread` was written to flatten it first and does not do enough: the pull moves the bundle
rather than opening it out.

What that costs, concretely: the chain "take the towel out and fold it" gets three steps in and
then reports honestly that it pulled the towel off the surface. The fix is a proper two-handed
spread -- pin one corner and drag the opposite one -- which needs the arms to work together in
a way nothing else here requires.
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
from ..sim.robot import Robot
from .skills import SkillResult

log = logging.getLogger(__name__)

# Where to stand relative to something being manipulated. Slightly inside the working reach, so
# a little drift while turning does not push the target out of the envelope.
STAND_OFF_M = 0.22

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
    cloth: ClothGrasp = field(init=False)
    _home: np.ndarray = field(init=False)
    _home_heading: float = field(init=False)
    # Which sheet the current errand is about. Set by whichever skill first names a towel, so
    # "take it out and fold it" does not need the towel named twice.
    _subject: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.cloth = ClothGrasp(self.robot.model, self.robot.data)
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
        for dx in np.linspace(-span, span, 9):
            for dy in np.linspace(0.18, span + 0.15, 5):
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
            here = self.cloth.vertex_position(sheet, vertex)
            direction = here[:2] - centre[:2]
            norm = float(np.linalg.norm(direction))
            direction = direction / norm if norm > 1e-6 else np.array([1.0, 0.0])
            target = np.array([
                here[0] + direction[0] * 0.16,
                here[1] + direction[1] * 0.16,
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

        # Aim at the outer edge of the open leaf, which is the part that has to travel.
        handle = self._site_position("drum_handle_site")
        if handle is None:
            return SkillResult(False, "I cannot find the washing machine door.")

        self._stand_near(handle, offset=0.30)
        side = self._hand_for(handle)
        self.robot.grip(side, 0.0)
        self.robot.reach_to(handle, side, passes=3)

        # Sweep the arm across the door's arc. Walking into it would drive the robot into the
        # machine; the door has to be pushed sideways, which is what the arm is for.
        best = before
        stalled = 0
        for _ in range(300):
            self.robot.step(vx=0.10, wz=-0.35 if side == "r" else 0.35)
            angle = abs(math.degrees(float(self.robot.data.qpos[address])))
            if angle < best - 0.5:
                best, stalled = angle, 0
            else:
                stalled += 1
                if stalled > 70 or angle < 8.0:
                    break

        self.robot.arm_home(side)
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

    def bring_basket(self) -> SkillResult:
        """Report that the basket cannot be moved.

        The laundry basket is fixed scenery in this room: it stands on a plinth so its contents
        are inside the arm's reach band, and it has no free joint, so there is nothing for the
        robot to pick up and carry.

        This exists so that asking for it gets a straight answer rather than an unhelpful
        "I do not know how to that". Reporting a limit clearly is a better outcome than a
        silent no-op, and the robot can still do the useful half: go and stand by it.
        """
        basket = self._site_position("basket_site")
        if basket is None:
            return SkillResult(False, "I cannot find the laundry basket.")
        return SkillResult(
            False,
            "I cannot carry the basket -- it is fixed to the floor in this room. "
            "I can take the laundry to it instead.",
            {"basket": basket.tolist()},
        )

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
        graspable = sheet.perimeter()
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

        self._stand_near(basket)
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
            picked = self.take_out(towel)
            side = self._loaded_hand()
            if not picked.ok or side is None:
                sheet = self._sheet(towel)
                if sheet is None:
                    return SkillResult(False, "I cannot find a towel to move.")
                # Any vertex will do here, not just the perimeter. A towel that has been
                # dropped into a basket is crumpled, so its "edge" is wherever the folds put
                # it and the topmost reachable point is a better handle than a nominal corner
                # buried in the pile.
                graspable = list(range(sheet.count))
                vertex, _ = self.cloth.nearest_vertex(
                    sheet, self.robot.position, among=graspable
                )
                self._stand_near(self.cloth.vertex_position(sheet, vertex))
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
        self.robot.reach_to(np.array([surface[0], surface[1], surface[2] + 0.10]), side,
                            passes=3)
        self.robot.grip(side, 0.0)
        self.cloth.release(side)
        # Arm away first, THEN wait. Releasing the weld does not free the towel if it is still
        # draped over the forearm -- the same thing that made a perfectly aimed drop into the
        # basket read as a miss.
        self.robot.arm_home(side)
        self.robot.stand(2.0)

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
            "bring_basket": lambda: self.bring_basket(),
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
