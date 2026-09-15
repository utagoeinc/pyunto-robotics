"""Looking after a cat while the owner is out, with a small camera robot.

The other watching demonstration in this package deliberately has no indoor camera: the older
person living there did not ask to be filmed, and the design answer was to watch the home
instead of the person. Here the situation is the opposite in every respect. The only human is
the one holding the phone, it is their own flat, and the subject is a cat, who will not be
reassured or troubled either way. So a camera is the right instrument, and the interesting
question becomes how to aim it.

That question is the whole demo. A fixed forward-facing lens on a wheeled robot finds a cat
on the floor and nothing else, so the cat's usual places are at four different heights --
under the sofa, on the windowsill, in the cat tree, on top of the bookshelf. Two of them need
the camera tilted down, two need it tilted up, and none of them can be reached by driving
alone. The pan/tilt head is load-bearing.

`find` is a search, not a lookup. It could trivially read the cat's position out of the
simulator and drive there, and would look identical on screen while proving nothing. Instead
it visits the places a cat is likely to be, aims the camera at each, and checks whether she is
actually in frame -- so it fails honestly when she is somewhere unexpected, which is the
behaviour that matters when this is pointed at a real flat.
"""

from __future__ import annotations

import logging
import math

import mujoco
import numpy as np

from ..sim.robot import Robot
from .result import SkillResult

log = logging.getLogger(__name__)

# Where a cat in this flat is usually found, in the order worth trying.
#
# Ordered by likelihood rather than by distance: most of what a pet camera sees is a cat
# asleep, and a sleeping cat picks somewhere soft and out of the way long before it picks
# somewhere convenient for the robot. Each entry says where to park, where to aim, and what
# to call the place in a message.
#
# Measure through the REAL path -- drive there, then sweep -- never by teleporting and
# aiming. Teleporting sets yaw to zero and skips the body turn, and every viewpoint measured
# that way scored well and then failed in use: `_aim_at` turns the body, which moves the robot
# off the very spot being measured. That mismatch cost most of the time spent on this scene.
#
#   key: (stand_x, stand_y, aim_x, aim_y, aim_z, name_ja)
# The standing positions are MEASURED, not chosen by eye: the robot was teleported over a
# grid, aimed through its real pan/tilt head at each place, and the spot that saw the most cat
# was kept. Measuring through the actual head matters -- an earlier pass used a free camera
# placed at the cat's head, which flattered every viewpoint and produced numbers the robot
# could not reproduce.
#
# The first set was picked off the plan (stand a metre away, facing it) and every one failed:
# from close and square on, the sill slab, the cradle rim and the shelf edge each hide exactly
# the surface the cat is lying on. Standing back and to one side is what works.
#
#   key: (stand_x, stand_y, aim_x, aim_y, aim_z, name_ja)
HAUNTS: dict[str, tuple[float, float, float, float, float, str]] = {
    "sill":   ( 2.20, -2.20,  1.60, -2.45, 0.76, "窓辺の日なた"),
    "tree":   ( 2.60, -1.40,  3.07, -1.90, 1.12, "キャットタワー"),
    "shelf":  (-0.20,  1.80,  0.20,  2.37, 1.72, "本棚の上"),
    "sofa":   ( 0.60, -0.60, -1.03,  0.40, 0.14, "ソファの下"),
    "bowls":  ( 4.40,  0.60,  4.40, -0.60, 0.10, "ごはんのところ"),
}


# The kitchen is behind a half-height divider that spans y from 0.2 to 3.0, so the way
# through is round its south end. This robot steers straight at its target, which walked it
# into the divider every time; one waypoint is enough and is honest about what it is -- the
# alternative is a path planner, which this scene does not need and the SDK has elsewhere.
KITCHEN_GATE = (3.80, -1.20)
BEHIND_THE_DIVIDER_X = 3.6

# The dock, where the robot waits between errands.
DOCK = (-1.90, -2.40)

# How close counts as arrived, and how near the heading has to be before driving on.
ARRIVED_M = 0.12
HEADING_TOLERANCE_RAD = 0.15

# How long the robot is willing to spend getting anywhere, in control steps. A pet camera
# that wanders for a minute has lost the owner's attention; better to report honestly that it
# could not get there.
#
# Sized against the flat rather than guessed: the longest leg is about 6 m, and this robot's
# 45 mm wheels cover roughly 1 m per 200 steps. 900 was the first figure and it stopped the
# robot a third of the way across the room, which reads as a fault rather than as a limit.
TRAVEL_LIMIT = 2200

# The cat has to be this much of the frame to count as found. Small, because she is often far
# away and partly behind furniture -- but not zero, so a glimpse of a ginger cushion does not
# count.
# Sized against what the robot actually sees from its own viewpoints, not picked round: at
# these ranges a cat fills well under a percent of the frame, and 0.004 (the first guess)
# rejected true sightings of 0.0014 while still admitting stray pixels. 0.0006 is above the
# noise floor -- an empty room scores 0.0000 exactly, because segmentation is by object id
# rather than by colour -- and below every real sighting measured.
SEEN_FRACTION = 0.0006

# How far the lens sits above the base, for working out the tilt. Measured from the model
# rather than guessed: it decides whether the robot aims at a cat or at the underside of the
# shelf she is on.
LENS_ABOVE_BASE_M = 0.62

# How close to a haunt's aim point the cat has to be to count as being AT it. Generous,
# because "on the windowsill" means anywhere along the sill, not one spot on it.
AT_PLACE_M = 0.85


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class PetWatchSkills:
    """Drive around a flat, aim a camera, and say how the cat is."""

    actions = (
        "find", "look", "pan", "tilt", "patrol", "photo",
        "go", "home", "check", "report",
    )

    def __init__(self, robot: Robot, grounder=None):  # noqa: ANN001
        self.robot = robot
        self.grounder = grounder
        self.model = robot.model
        self.data = robot.data
        self._pan = 0.0
        self._tilt = 0.0
        self._last_seen: str | None = None
        self._cat_geoms = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in ("cat_body", "cat_head", "cat_haunch", "cat_chest", "cat_muzzle")
        ]
        self._cat_geoms = [g for g in self._cat_geoms if g >= 0]
        # Our own segmentation renderer. `Robot.look` returns rgb and depth only, and adding
        # a third buffer there would cost every other demo a render per frame for something
        # only this one needs.
        # 320x240, not 160x120. A cat across a room is a few dozen pixels, and at the lower
        # resolution the difference between "asleep on the shelf" and "not there" came down to
        # single pixels -- which made the threshold a coin toss rather than a measurement.
        self._seg = mujoco.Renderer(self.model, height=240, width=320)
        self._seg.enable_segmentation_rendering()
        self._hold_cat()

    def _hold_cat(self) -> None:
        """Tell the cat's actuators to hold her where the keyframe put her.

        A keyframe sets qpos but not ctrl, and her joints are position servos -- so with ctrl
        left at zero she slid steadily to the origin while the robot drove, arriving in the
        middle of the floor. The robot then correctly reported not finding her anywhere, which
        was a true statement about a scene that had quietly fallen apart.

        She is moved later by setting these same targets, so this is the resting state rather
        than a constraint.
        """
        for index, name in enumerate(("cat_x", "cat_y", "cat_z", "cat_yaw")):
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            actuator = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if joint >= 0 and actuator >= 0:
                self.data.ctrl[actuator] = float(self.data.qpos[self.model.jnt_qposadr[joint]])

    # -- the camera head --------------------------------------------------------------

    def _apply_head(self) -> None:
        """Push the pan and tilt targets to the actuators and let them settle."""
        pan = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cam_pan")
        tilt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cam_tilt")
        if pan >= 0:
            self.data.ctrl[pan] = float(np.clip(self._pan, -2.6, 2.6))
        if tilt >= 0:
            self.data.ctrl[tilt] = float(np.clip(self._tilt, -0.35, 1.15))
        for _ in range(60):
            self.robot.step(0.0, 0.0, 0.0)

    def pan(self, argument: str | None = None) -> SkillResult:
        """Turn the camera left or right, in degrees or by word."""
        delta = self._degrees(argument, default=30.0)
        if argument and any(w in str(argument) for w in ("左", "left")):
            delta = abs(delta)
        elif argument and any(w in str(argument) for w in ("右", "right")):
            delta = -abs(delta)
        self._pan += math.radians(delta)
        self._apply_head()
        side = "左" if delta > 0 else "右"
        return SkillResult(
            True,
            f"カメラを{side}に{abs(delta):.0f}度向けました。",
            {"pan_deg": round(math.degrees(self._pan), 1)},
        )

    def tilt(self, argument: str | None = None) -> SkillResult:
        """Aim the camera up or down.

        Note the sign: positive `cam_tilt` pitches the lens DOWN, because the hinge is about
        +y and the camera looks along +x. Converted here so no caller has to know that.
        """
        degrees = self._degrees(argument, default=20.0)
        up = bool(argument) and any(w in str(argument) for w in ("上", "up"))
        self._tilt += -math.radians(degrees) if up else math.radians(degrees)
        self._apply_head()
        where = "上" if up else "下"
        return SkillResult(
            True,
            f"カメラを{where}に{degrees:.0f}度向けました。",
            {"tilt_deg": round(math.degrees(self._tilt), 1)},
        )

    @staticmethod
    def _degrees(argument: str | None, default: float) -> float:
        digits = ""
        for char in str(argument or ""):
            if char.isdigit() or (char == "." and digits):
                digits += char
            elif digits:
                break
        try:
            return float(digits) if digits else default
        except ValueError:
            return default

    # -- driving ----------------------------------------------------------------------

    def _pose(self) -> tuple[float, float, float]:
        """Where the robot is and which way it faces."""
        base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        x, y = float(self.data.xpos[base][0]), float(self.data.xpos[base][1])
        # Forward is the body's +x axis in world coordinates.
        forward = self.data.xmat[base].reshape(3, 3)[:, 0]
        return x, y, math.atan2(float(forward[1]), float(forward[0]))

    def _drive_to(self, tx: float, ty: float) -> bool:
        """Drive to a point, going round the kitchen divider when it is in the way."""
        x, _y, _yaw = self._pose()
        crossing = (x < BEHIND_THE_DIVIDER_X) != (tx < BEHIND_THE_DIVIDER_X)
        if crossing and not self._drive_straight_to(*KITCHEN_GATE):
            return False
        return self._drive_straight_to(tx, ty)

    def _drive_straight_to(self, tx: float, ty: float) -> bool:
        """Turn toward a point and drive to it. True if it arrived."""
        for _ in range(TRAVEL_LIMIT):
            x, y, yaw = self._pose()
            dx, dy = tx - x, ty - y
            if math.hypot(dx, dy) <= ARRIVED_M:
                self.robot.step(0.0, 0.0, 0.0)
                return True
            error = _wrap(math.atan2(dy, dx) - yaw)
            if abs(error) > HEADING_TOLERANCE_RAD:
                self.robot.step(0.0, 0.0, float(np.clip(error * 2.2, -1.4, 1.4)))
            else:
                self.robot.step(float(np.clip(math.hypot(dx, dy) * 1.2, 0.20, 0.75)),
                                0.0, float(np.clip(error * 1.6, -0.6, 0.6)))
        return False

    def _face(self, wx: float, wy: float) -> None:
        """Turn the body toward a point, so the head does not have to do all the work.

        The pan joint reaches +-149 degrees, so aiming sideways is nominally possible -- and
        it does not work: at 116 degrees off the nose the shell around the lens is in its own
        way, and the robot stared at the cat tree seeing nothing. Turning the body first keeps
        the pan small, which is also how a person holding a camera would do it.
        """
        for _ in range(400):
            x, y, yaw = self._pose()
            error = _wrap(math.atan2(wy - y, wx - x) - yaw)
            if abs(error) < 0.08:
                break
            self.robot.step(0.0, 0.0, float(np.clip(error * 2.2, -1.4, 1.4)))
        self.robot.step(0.0, 0.0, 0.0)

    def _aim_at(self, wx: float, wy: float, wz: float) -> None:
        """Point the head at a world position, by pan and tilt rather than by driving."""
        self._face(wx, wy)
        base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        x, y, yaw = self._pose()
        z = float(self.data.xpos[base][2])
        self._pan = _wrap(math.atan2(wy - y, wx - x) - yaw)
        flat = math.hypot(wx - x, wy - y)
        # Positive tilt looks down, so the sign is inverted against the usual elevation.
        self._tilt = -math.atan2(wz - (z + LENS_ABOVE_BASE_M), max(flat, 0.05))
        self._apply_head()

    # -- seeing -----------------------------------------------------------------------

    def _sweep_for_cat(self, ax: float, ay: float, az: float) -> float:
        """Aim at a place, then sweep across it. Returns the best fraction seen.

        A single fixed aim point is too brittle: it points at the middle of a windowsill while
        the cat is at one end, and a 72-degree lens from across the room does not quite reach
        her. A real pet camera sweeps for exactly this reason, so this does too -- and it
        keeps `find` a search rather than a lookup, because the sweep does not know where she
        is either.

        The sweep is deliberately narrow. At +-0.45 rad it swung far enough past the sofa to
        catch the cat asleep on the bookshelf behind it, and then reported her as being under
        the sofa -- a confident, specific, wrong answer, which is worse than no answer for
        somebody who is out and cannot check. A sighting must belong to the place being
        looked at, so the sweep stays inside it.
        """
        self._aim_at(ax, ay, az)
        if not self._cat_near(ax, ay, az):
            return 0.0
        centre_pan, centre_tilt = self._pan, self._tilt
        best = self._cat_in_frame()
        for pan_offset in (-0.24, -0.12, 0.12, 0.24):
            if best >= SEEN_FRACTION:
                break
            self._pan = centre_pan + pan_offset
            self._tilt = centre_tilt
            self._apply_head()
            best = max(best, self._cat_in_frame())
        if best < SEEN_FRACTION:
            self._pan, self._tilt = centre_pan, centre_tilt
            self._apply_head()
        return best

    def _cat_near(self, ax: float, ay: float, az: float) -> bool:
        """Is the cat actually AT the place being looked at?

        Seeing her is not the same as her being there. This flat is one open room, so from the
        sofa the camera can also see the bookshelf, the sill and the cat tree -- and the first
        version of `find` cheerfully reported a cat asleep on the shelf as being under the
        sofa. Confident, specific and wrong is the worst answer for somebody who is out and
        cannot check for themselves.

        So a sighting has to pass two tests: she is in frame (the camera really can see her,
        which is what makes this a search) and she is near the place that was aimed at. The
        radius is generous, because "on the windowsill" covers the whole sill.
        """
        head = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cat_head")
        if head < 0:
            return True
        cx, cy, cz = (float(v) for v in self.data.geom_xpos[head])
        return math.dist((cx, cy, cz), (ax, ay, az)) <= AT_PLACE_M

    def _cat_in_frame(self) -> float:
        """How much of the camera frame the cat occupies, 0 if she is not visible.

        Uses the segmentation buffer rather than colour, because "is that ginger thing the cat
        or the wooden shelf" is exactly the ambiguity a demo should not paper over.
        """
        try:
            self._seg.update_scene(self.data, camera="head_cam")
            seg = self._seg.render()
        except Exception:  # noqa: BLE001 - a camera that cannot read is "did not see her"
            log.debug("segmentation render failed", exc_info=True)
            return 0.0
        # Channel 0 is the object id, channel 1 the object type. Geom ids are only meaningful
        # where the type is a geom, so mask on that before matching -- otherwise a body id
        # that happens to equal one of the cat's geom ids counts as a sighting.
        ids, kinds = seg[:, :, 0], seg[:, :, 1]
        hits = np.isin(ids, self._cat_geoms) & (kinds == mujoco.mjtObj.mjOBJ_GEOM)
        return float(hits.sum()) / float(hits.size)

    def close(self) -> None:
        """Release the segmentation renderer."""
        try:
            self._seg.close()
        except Exception:  # noqa: BLE001 - closing must never be fatal
            pass

    # -- the skills the owner actually asks for ---------------------------------------

    def find(self, argument: str | None = None) -> SkillResult:
        """Go looking for the cat, and say where she is.

        A search, not a lookup. The places are visited in order, the camera is aimed at each,
        and the frame is checked -- so if she is somewhere that is not on the list, this says
        so rather than inventing an answer.
        """
        tried: list[str] = []
        for key, (sx, sy, ax, ay, az, name) in HAUNTS.items():
            tried.append(name)
            if not self._drive_to(sx, sy):
                log.info("could not reach %s", key)
                continue
            fraction = self._sweep_for_cat(ax, ay, az)
            log.info("looked at %s: %.4f of frame", key, fraction)
            if fraction >= SEEN_FRACTION:
                self._last_seen = key
                return SkillResult(
                    True,
                    f"🐱 {name}にいました。カメラを向けています。",
                    {"where": key, "where_ja": name, "frame_fraction": round(fraction, 4),
                     "tried": tried},
                )
        return SkillResult(
            True,
            "見つけられませんでした。"
            + "、".join(tried)
            + "を見ましたが、どこにもいません。別の場所を指定してください。",
            {"where": None, "tried": tried},
            fatal=False,
        )

    def go(self, argument: str | None = None, where: str | None = None) -> SkillResult:
        """Drive to a named place in the flat."""
        key = self._place_in(where or argument)
        if key is None:
            return SkillResult(
                False,
                "どこへ行けばいいか分かりませんでした。"
                + "、".join(h[5] for h in HAUNTS.values())
                + "、ドックのいずれかを指定してください。",
            )
        if key == "home":
            return self.home()
        sx, sy, ax, ay, az, name = HAUNTS[key]
        if not self._drive_to(sx, sy):
            return SkillResult(False, f"{name}まで行けませんでした。何かに阻まれています。")
        self._aim_at(ax, ay, az)
        return SkillResult(True, f"{name}に来て、カメラを向けました。", {"where": key})

    def home(self, _argument: str | None = None) -> SkillResult:
        """Go back to the dock."""
        arrived = self._drive_to(*DOCK)
        self._pan = self._tilt = 0.0
        self._apply_head()
        return SkillResult(
            arrived,
            "ドックに戻りました。" if arrived else "ドックに戻れませんでした。",
            {"docked": arrived},
        )

    def patrol(self, _argument: str | None = None) -> SkillResult:
        """Visit every haunt in turn and report what was at each.

        What an owner wants after a day out is not one photograph but a round: the cat was on
        the sill, the water bowl is still full. So this reports every place, not only the one
        with the cat in it.
        """
        seen: list[str] = []
        empty: list[str] = []
        for key, (sx, sy, ax, ay, az, name) in HAUNTS.items():
            if not self._drive_to(sx, sy):
                continue
            if self._sweep_for_cat(ax, ay, az) >= SEEN_FRACTION:
                seen.append(name)
                self._last_seen = key
            else:
                empty.append(name)
        lines = []
        if seen:
            lines.append("🐱 " + "、".join(seen) + "にいました。")
        if empty:
            lines.append("（" + "、".join(empty) + "は空でした）")
        if not seen:
            lines.append("どこにも見当たりませんでした。")
        return SkillResult(True, "\n".join(lines),
                           {"seen": seen, "empty": empty})

    def check(self, _argument: str | None = None) -> SkillResult:
        """Is she in view right now, without moving?"""
        fraction = self._cat_in_frame()
        if fraction >= SEEN_FRACTION:
            return SkillResult(True, "🐱 いまカメラに写っています。",
                               {"visible": True, "frame_fraction": round(fraction, 4)})
        return SkillResult(
            True,
            "いまは写っていません。「探して」と言っていただければ見に行きます。",
            {"visible": False},
            fatal=False,
        )

    def photo(self, _argument: str | None = None) -> SkillResult:
        """Send what the camera sees right now.

        The image itself is attached by the backend; this only says what is in it, because a
        photograph with no caption is the thing owners complain about in pet cameras.
        """
        fraction = self._cat_in_frame()
        caption = ("🐱 写っています。" if fraction >= SEEN_FRACTION
                   else "いまの視界です（猫は写っていません）。")
        return SkillResult(True, caption,
                           {"image": "head_cam", "visible": fraction >= SEEN_FRACTION})

    def look(self, argument: str | None = None, where: str | None = None) -> SkillResult:
        """Aim at a named place without driving to it."""
        key = self._place_in(where or argument)
        if key is None or key == "home":
            return SkillResult(False, "どこを見ればいいか分かりませんでした。")
        _sx, _sy, ax, ay, az, name = HAUNTS[key]
        fraction = self._sweep_for_cat(ax, ay, az)
        if fraction >= SEEN_FRACTION:
            self._last_seen = key
            return SkillResult(True, f"🐱 {name}に向けました。写っています。",
                               {"where": key, "visible": True})
        return SkillResult(True, f"{name}に向けましたが、いません。",
                           {"where": key, "visible": False}, fatal=False)

    @staticmethod
    def _place_in(text: str | None) -> str | None:
        """Which place an instruction meant."""
        if not text:
            return None
        lowered = str(text).lower()
        for key, (_sx, _sy, _ax, _ay, _az, _name) in HAUNTS.items():
            if key in lowered:
                return key
        for key, words in (
            ("sofa", ("ソファ", "そふぁ", "sofa", "couch")),
            ("sill", ("窓", "まど", "日なた", "ひなた", "window", "sill", "sun")),
            ("tree", ("タワー", "キャットタワー", "tower", "cat tree", "tree")),
            ("shelf", ("本棚", "棚", "ほんだな", "shelf", "bookshelf")),
            ("bowls", ("ごはん", "ご飯", "えさ", "餌", "水", "food", "bowl", "water")),
            ("home", ("ドック", "どっく", "dock", "home", "戻", "帰")),
        ):
            if any(w in lowered or w in str(text) for w in words):
                return key
        return None

    # -- dispatch ---------------------------------------------------------------------

    def run(
        self,
        action: str,
        argument: str | None = None,
        where: str | None = None,
        expect: int | None = None,
    ) -> SkillResult:
        handlers = {
            "find": lambda: self.find(argument),
            "look": lambda: self.look(argument, where),
            "pan": lambda: self.pan(argument),
            "tilt": lambda: self.tilt(argument),
            "patrol": lambda: self.patrol(argument),
            "photo": lambda: self.photo(argument),
            "go": lambda: self.go(argument, where),
            "home": lambda: self.home(argument),
            "check": lambda: self.check(argument),
            "report": lambda: SkillResult(True, argument or "はい。"),
        }
        handler = handlers.get(action)
        if handler is None:
            return SkillResult(False, f"'{action}' はできません。")
        log.info("skill: %s(%s)", action, argument or "")
        return handler()
