"""Holding cloth in MuJoCo.

Cloth is not a rigid body and cannot be picked up like one. A flexcomp sheet is a grid of
vertex bodies joined by edge constraints, so "the towel" has no single pose to grasp -- there
are 63 of them, and which one the hand is near is the entire question.

Two facts drive everything here:

  * A friction grasp does not hold. The fingers slide off a 4 mm-thick sheet long before the
    arm can lift it, exactly as they slide off a door handle in the office scene. So a closed
    hand is modelled as a WELD to one vertex, which is the same compromise made there.

  * Which vertex is welded has to be decided at runtime. The scene declares a weld against
    vertex 0 as a placeholder; this module rewrites `eq_obj2id` to whichever vertex is actually
    nearest the palm and writes the relative pose into `eq_data`. Skipping the pose rewrite
    makes the solver enforce the offset compiled into the XML, and the towel snaps across the
    room to meet the hand.

Vertex indexing is deterministic and worth relying on: a `count="9 7 1"` grid numbers vertices
row-major, so index = row * 7 + col for a 9x7 sheet -- 0 is one corner, 62 the opposite one.
That is what makes "grasp two adjacent corners", which is how folding starts, expressible at
all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClothSheet:
    """One flexcomp sheet, addressed by name.

    `rows` and `cols` mirror the flexcomp's `count`, and are needed to turn a corner name into
    a vertex index. They are read from the model rather than passed in, so a sheet that changes
    size in the XML does not silently break the corner lookup.
    """

    name: str
    rows: int
    cols: int
    first_body: int  # model body id of vertex 0

    @property
    def count(self) -> int:
        return self.rows * self.cols

    def index(self, row: int, col: int) -> int:
        """Vertex index for a grid position. Row-major, matching MuJoCo's own ordering."""
        return row * self.cols + col

    def body_id(self, vertex: int) -> int:
        """Model body id for a vertex index."""
        return self.first_body + vertex

    def edge(self, which: str) -> list[int]:
        """Every vertex along one edge of the sheet.

        Corners are the right handle for FOLDING, where the two ends of an edge have to be
        carried across together. They are the wrong handle for picking a towel out of a drum:
        a corner sits at the extreme of the sheet, so it is the furthest part of it from a
        robot standing square on -- measured 0.18 m from the hand where the nearest edge
        vertex was 0.16 m, with a reach budget of 0.25 m and the corner off to one side where
        the arm cannot follow.

        Grabbing the middle of the near edge is also what a person does reaching into a washer.
        """
        if which == "near":
            return [self.index(0, c) for c in range(self.cols)]
        if which == "far":
            return [self.index(self.rows - 1, c) for c in range(self.cols)]
        if which == "left":
            return [self.index(r, 0) for r in range(self.rows)]
        if which == "right":
            return [self.index(r, self.cols - 1) for r in range(self.rows)]
        return []

    def perimeter(self) -> list[int]:
        """Every vertex on the outside of the sheet.

        The graspable set: anywhere on the edge can be picked up and the sheet will hang from
        it, whereas lifting from the middle gathers the towel into a bundle around the hand.
        """
        return sorted(
            set(self.edge("near") + self.edge("far") + self.edge("left") + self.edge("right"))
        )

    def corners(self) -> dict[str, int]:
        """The four corner vertices, named by their position in the grid.

        Folding needs two ADJACENT corners -- lifting diagonally opposite ones gathers the
        sheet into a bundle instead of folding it -- so the names have to be unambiguous about
        which edge they sit on.
        """
        return {
            "near_left": self.index(0, 0),
            "near_right": self.index(0, self.cols - 1),
            "far_left": self.index(self.rows - 1, 0),
            "far_right": self.index(self.rows - 1, self.cols - 1),
        }


def find_sheets(model: mujoco.MjModel) -> dict[str, ClothSheet]:
    """Discover every flexcomp sheet in a model.

    flexcomp names its vertex bodies `<name>_<i>`, which is the only handle the compiled model
    gives back -- `nflex` counts the sheets but the vertex-to-body mapping is not exposed. So
    the bodies are scanned once and grouped by prefix.

    Only flex sheets are wanted, so a group is kept only if the model has a flex of that name.
    Without that check any body family ending in a number -- `frame_1a`-style scenery, or a
    second robot's numbered links -- would be mistaken for cloth.
    """
    groups: dict[str, list[int]] = {}
    for body in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        prefix, sep, suffix = name.rpartition("_")
        if not sep or not suffix.isdigit():
            continue
        groups.setdefault(prefix, []).append(body)

    sheets: dict[str, ClothSheet] = {}
    for prefix, bodies in groups.items():
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, prefix) < 0:
            continue
        bodies.sort()
        total = len(bodies)
        rows, cols = _grid_shape(model, bodies, total)
        if rows * cols != total:
            continue
        sheets[prefix] = ClothSheet(prefix, rows, cols, bodies[0])
    return sheets


def _grid_shape(model: mujoco.MjModel, bodies: list[int], total: int) -> tuple[int, int]:
    """Rows and columns of a sheet, measured from where its vertex bodies sit.

    MuJoCo does not keep the flexcomp's `count` around after compilation, so the grid is
    recovered rather than declared -- which keeps this correct when the XML changes.

    It is measured from `body_pos`, NOT from `flex_vert`. flex_vert is the deformation buffer
    and reads as all zeros in the rest pose, which made every sheet look like a 63x1 strip and
    collapsed all four "corners" onto two vertices.

    The grid is laid out row-major along the second axis, so the number of distinct y offsets
    is the column count.
    """
    coords = np.array([model.body_pos[b] for b in bodies])
    cols = len(np.unique(np.round(coords[:, 1], 4)))
    if cols and total % cols == 0:
        return total // cols, cols
    # Fall back to as square a grid as divides evenly.
    for side in range(int(total**0.5), 0, -1):
        if total % side == 0:
            return total // side, side
    return total, 1


class ClothGrasp:
    """Welds a hand to a cloth vertex, and lets go again.

    One instance manages every cloth weld in a scene. It is kept separate from `Robot` because
    nothing about it is specific to a body plan -- a quadruped with a mouth gripper would use
    it the same way.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.sheets = find_sheets(model)
        # Which weld is holding what, so release() knows what to switch off and callers can ask
        # whether a hand is full without inspecting the solver.
        self._held: dict[str, tuple[str, int]] = {}

    # -- queries ------------------------------------------------------------------

    def sheet(self, name: str) -> ClothSheet | None:
        return self.sheets.get(name)

    def holding(self, side: str = "r") -> tuple[str, int] | None:
        """(sheet name, vertex) held by a hand, or None."""
        return self._held.get(side)

    def vertex_position(self, sheet: ClothSheet, vertex: int) -> np.ndarray:
        """Where a vertex is right now, in world coordinates."""
        return self.data.xpos[sheet.body_id(vertex)].copy()

    def sheet_centre(self, sheet: ClothSheet) -> np.ndarray:
        """Mean position of every vertex -- the middle of the towel, wherever it has got to."""
        ids = [sheet.body_id(v) for v in range(sheet.count)]
        return self.data.xpos[ids].mean(axis=0)

    def sheet_extent(self, sheet: ClothSheet) -> float:
        """Largest distance between any two vertices, in metres.

        This is the measurement that says whether a fold worked: a flat 0.40 x 0.30 sheet spans
        about 0.50 m corner to corner, and folding it in half should bring that down markedly.
        Comparing before and after is a far more honest test than asserting a hand reached a
        waypoint.
        """
        ids = [sheet.body_id(v) for v in range(sheet.count)]
        points = self.data.xpos[ids]
        spread = points[:, None, :] - points[None, :, :]
        return float(np.sqrt((spread**2).sum(axis=2)).max())

    def nearest_vertex(
        self, sheet: ClothSheet, point: np.ndarray, among: list[int] | None = None
    ) -> tuple[int, float]:
        """Closest vertex to a world point, and how far away it is.

        `among` restricts the search, which is what makes "the nearest CORNER" expressible --
        grasping the middle of a towel and lifting gathers it into a bag rather than picking
        it up by an edge.
        """
        candidates = among if among is not None else list(range(sheet.count))
        ids = [sheet.body_id(v) for v in candidates]
        offsets = self.data.xpos[ids] - np.asarray(point)
        distances = np.linalg.norm(offsets, axis=1)
        best = int(np.argmin(distances))
        return candidates[best], float(distances[best])

    # -- grasping -----------------------------------------------------------------

    def grasp(self, sheet_name: str, vertex: int, side: str = "r") -> bool:
        """Weld a palm to one cloth vertex, freezing the current relative pose.

        Returns False when the scene declares no weld for that hand and sheet, which is a
        setup error rather than a runtime failure -- a scene with cloth needs one weld per
        hand per sheet, because folding is two-handed and one weld holds one corner.
        """
        sheet = self.sheets.get(sheet_name)
        if sheet is None:
            log.warning("no cloth sheet named %r", sheet_name)
            return False
        eq = self._weld_for(sheet_name, side)
        if eq is None:
            log.warning("no weld defined for %s on hand %r", sheet_name, side)
            return False

        palm = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"palm_{side}")
        target = sheet.body_id(vertex)

        # Point the weld at the vertex actually being held. The XML names vertex 0 as a
        # placeholder; without this every grasp would drag the same corner.
        self.model.eq_obj2id[eq] = target

        # eq_data for a weld is [anchor(3), relpose(7: pos + quat), torquescale(1)], with the
        # relative pose expressed in body1's frame. Writing the CURRENT relative pose is what
        # keeps the towel where it was grabbed; leaving the compiled offset teleports it.
        palm_pos, palm_quat = self.data.xpos[palm], self.data.xquat[palm]
        target_pos = self.data.xpos[target]

        inv_palm = np.zeros(4)
        mujoco.mju_negQuat(inv_palm, palm_quat)
        rel_pos = np.zeros(3)
        mujoco.mju_rotVecQuat(rel_pos, target_pos - palm_pos, inv_palm)

        self.model.eq_data[eq, 0:3] = 0.0
        self.model.eq_data[eq, 3:6] = rel_pos
        # A cloth vertex is a point mass with no meaningful orientation, so the rotational part
        # of the weld is left as identity and the torque scale at zero. Constraining a vertex's
        # orientation to the palm's fights the edge constraints and makes the sheet twitch.
        self.model.eq_data[eq, 6:10] = np.array([1.0, 0.0, 0.0, 0.0])
        self.model.eq_data[eq, 10] = 0.0
        self.data.eq_active[eq] = 1
        self._held[side] = (sheet_name, vertex)
        mujoco.mj_forward(self.model, self.data)
        return True

    def release(self, side: str | None = None) -> None:
        """Let go with one hand, or with both when `side` is None."""
        sides = [side] if side else list(self._held)
        for hand in sides:
            held = self._held.pop(hand, None)
            if held is None:
                continue
            eq = self._weld_for(held[0], hand)
            if eq is not None:
                self.data.eq_active[eq] = 0
        mujoco.mj_forward(self.model, self.data)

    def _weld_for(self, sheet_name: str, side: str) -> int | None:
        """The equality index for a (sheet, hand) pair.

        Matched by NAME rather than by body, because eq_obj2id is rewritten on every grasp --
        looking the weld up by what it currently points at would stop finding it after the
        first grasp moved it to a different vertex.
        """
        wanted = f"grasp_{sheet_name.replace('towel_', '')}_{side}"
        eq = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, wanted)
        if eq >= 0:
            return eq
        # Fall back to a plain <sheet>_<side> spelling, so a scene is not forced to adopt the
        # "towel_" prefix convention.
        eq = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_{sheet_name}_{side}"
        )
        return eq if eq >= 0 else None
