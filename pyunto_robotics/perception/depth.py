"""Reading geometry out of the depth image.

No map is ever built. Every question is answered from the current frame:
"how far is it that way", "where can I walk", "how far is the thing at this pixel".

Working in bearing columns rather than a metric grid is deliberate. A grid would be a map by
another name, and it would drift as soon as the robot's odometry did. Columns are recomputed
from scratch each frame, so nothing accumulates and nothing goes stale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Depth samples closer than this are the robot's own body or contact noise.
MIN_VALID_M = 0.12


# How far ahead an obstacle has to be before it stops constraining how wide a path needs to be.
# Near enough that the robot is committed to passing it, far enough that a wall at the end of a
# corridor does not read as an immediate squeeze. Measured across a sweep: 2.4 m makes distant
# walls constrain the width and wall contact goes back up nearly fivefold, because the robot
# swerves for things it would have turned past anyway.
LOOKAHEAD_M = 1.2


@dataclass(frozen=True)
class FreeSpace:
    """How far the robot could travel in each of several directions."""

    bearings: np.ndarray  # (N,) radians, + is left
    ranges: np.ndarray  # (N,) metres of clear space in that direction

    def clearance_ahead(self, half_angle: float = 0.30) -> float:
        """Nearest obstacle within a cone straight ahead."""
        mask = np.abs(self.bearings) <= half_angle
        if not mask.any():
            return float(self.ranges.min())
        return float(self.ranges[mask].min())

    def widest_gap(self, heading: float, half_width: float) -> float:
        """Sideways room to spare when travelling along `heading`, in metres.

        Negative means the body does not fit. Positive is how much margin there is.

        Range alone cannot answer "will I fit". A wall a metre ahead and slightly to the left
        reads as a comfortable range in every direction, and the robot walks into it shoulder
        first. What matters is how far each obstacle sits from the line of travel, sideways,
        compared with how wide the robot is -- so that is what this measures: for every bearing
        with something in it, the perpendicular distance from that obstacle to the intended
        path, minus the half-width.
        """
        offsets = self.bearings - heading
        # Only obstacles roughly ahead can be walked into; something at 80 degrees is passed,
        # not hit, and including it makes every corridor look impassable.
        relevant = np.abs(offsets) < math.pi / 3.0
        if not relevant.any():
            return float("inf")

        # Perpendicular distance from the obstacle to the line of travel, and how far along
        # that line it sits.
        lateral = np.abs(self.ranges[relevant] * np.sin(offsets[relevant]))
        forward = self.ranges[relevant] * np.cos(offsets[relevant])

        # Only obstacles within the next stretch of travel constrain the width. A wall straight
        # ahead has a lateral offset of zero at any distance, so without this every direction in
        # an open room reports "will not fit" -- measured -0.32 m in all directions in an empty
        # lobby. What is far away gets steered around long before it is reached.
        soon = (forward > 0.0) & (forward < LOOKAHEAD_M)
        if not soon.any():
            return float("inf")
        return float(lateral[soon].min() - half_width)

    def clearest_heading(
        self, prefer: float, half_width: float, spread: float = 0.35
    ) -> float:
        """The heading near `prefer` that leaves the body the most room.

        Deliberately a small search around the direction the robot already wants to go, not a
        free choice of any direction. Steering purely for clearance walks away from the target;
        nudging the heading by up to `spread` keeps the errand while taking the roomier line
        through a gap.
        """
        options = prefer + np.linspace(-spread, spread, 9)
        gaps = [self.widest_gap(float(h), half_width) for h in options]

        # Prefer room, but break ties toward the heading actually wanted.
        scores = [g - 0.25 * abs(float(h) - prefer) for g, h in zip(gaps, options, strict=True)]
        return float(options[int(np.argmax(scores))])

    def best_bearing(self, prefer: float = 0.0, min_range: float = 1.0) -> float | None:
        """The most open direction, breaking ties toward `prefer`.

        Used when exploring: head into open space rather than at a wall.
        """
        open_enough = self.ranges >= min_range
        if not open_enough.any():
            return None
        candidates = self.bearings[open_enough]
        scores = self.ranges[open_enough] - 0.8 * np.abs(candidates - prefer)
        return float(candidates[int(np.argmax(scores))])


def _horizon_band(depth: np.ndarray) -> np.ndarray:
    """The rows worth trusting for obstacle checks.

    The top of the frame is mostly ceiling and the bottom is the floor right at the robot's
    feet; both would make a clear corridor look blocked. The middle band is what a person
    scanning a room for a way through would actually look at.
    """
    h = depth.shape[0]
    return depth[int(h * 0.35) : int(h * 0.75), :]


def free_space(
    depth: np.ndarray, fovy_deg: float, columns: int = 33, max_range: float = 12.0
) -> FreeSpace:
    """Collapse a depth image into per-bearing clearances.

    Each column reports its *nearest* return, because one obstacle in a column blocks it
    regardless of how much open space surrounds it.
    """
    band = _horizon_band(depth)
    h, w = depth.shape

    # Horizontal FOV follows from the vertical FOV and the aspect ratio.
    fovy = math.radians(fovy_deg)
    focal = (h / 2.0) / math.tan(fovy / 2.0)
    fovx = 2.0 * math.atan((w / 2.0) / focal)

    edges = np.linspace(0, w, columns + 1).astype(int)
    bearings = np.empty(columns, dtype=np.float64)
    ranges = np.empty(columns, dtype=np.float64)

    for i in range(columns):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        strip = band[:, lo:hi]
        valid = strip[(strip > MIN_VALID_M) & (strip < max_range)]
        ranges[i] = float(np.percentile(valid, 5)) if valid.size else max_range

        centre_px = (lo + hi) / 2.0
        # + bearing is left, matching Robot.bearing_to_pixel.
        bearings[i] = -math.atan2(centre_px - w / 2.0, focal)

    # Rays hitting a surface at a glancing angle read long; the true forward clearance is
    # the perpendicular distance, so project each column onto the view axis.
    ranges = ranges * np.cos(bearings)
    _ = fovx  # kept for readability; bearings already encode it
    return FreeSpace(bearings=bearings, ranges=ranges)


def nearest_off_axis(
    space: "FreeSpace", exclude_bearing: float | None = None, exclude_width: float = 0.30
) -> tuple[float, float] | None:
    """The closest thing that is not the target, as (bearing, range).

    Obstacle checks that only look straight ahead miss the case that actually hurts: brushing
    along a wall while walking past it. The clearance dead ahead stays comfortable the whole
    time, so nothing objects, and the robot arrives having scraped the length of the corridor.

    `exclude_bearing` blanks out the direction the robot is deliberately heading for, so the
    door it is about to touch does not read as something to avoid.
    """
    mask = np.ones(space.bearings.shape, dtype=bool)
    if exclude_bearing is not None:
        mask &= np.abs(space.bearings - exclude_bearing) > exclude_width
    if not mask.any():
        return None
    idx = int(np.argmin(np.where(mask, space.ranges, np.inf)))
    return float(space.bearings[idx]), float(space.ranges[idx])


def depth_at(
    depth: np.ndarray, u: float, v: float, patch: int = 5, percentile: float = 50.0
) -> float | None:
    """Depth in a small patch around a pixel.

    A single pixel lands on an edge often enough to matter, so this samples a patch. The
    percentile chooses what to take from it: 50 (the median) is right for a surface seen
    face-on, but for something seen at an angle the patch straddles both the target and
    whatever is in front of it, and the median then reports the nearer thing.
    """
    h, w = depth.shape
    x, y = int(round(u)), int(round(v))
    if not (0 <= x < w and 0 <= y < h):
        return None
    r = patch // 2
    window = depth[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1]
    valid = window[(window > MIN_VALID_M) & np.isfinite(window)]
    if valid.size == 0:
        return None
    return float(np.percentile(valid, percentile))


def target_offset(
    depth: np.ndarray,
    u: float,
    v: float,
    fovy_deg: float,
    image_size: tuple[int, int],
    camera_yaw: float = 0.0,
) -> tuple[float, float] | None:
    """Convert a pixel into (bearing, distance) in the robot's frame.

    This is the bridge from vision to motion: a vision model says "the door is here in the
    image", and this turns that into something `step(vx, vy, wz)` can chase.

    MuJoCo's depth buffer holds distance along the view axis, not distance to the point. For
    anything off-centre those differ by 1/cos(bearing), and at 45 degrees that is a factor of
    1.4: three doors 4.0, 5.66 and 5.66 m away all read as 3.90 m. Dividing by cos recovers
    the true range, which is what makes a world-coordinate track land on the right door.

    free_space already did this for its clearance columns; target_offset did not, and that was
    the whole reason tracking a door by position kept locking onto its neighbour.

    `camera_yaw` is where the head is pointing relative to the body. Everything downstream
    steers the body, so the bearing has to be reported in the body's frame; a head turned 45
    degrees toward a door puts that door in the middle of its own frame, which would otherwise
    read as "straight ahead". Adding it here rather than at each call site keeps the correction
    in one place -- applying it in two of them cancelled the head entirely.
    """
    w, h = image_size
    axial = depth_at(depth, u, v, patch=9, percentile=70.0)
    if axial is None:
        return None
    focal = (h / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    # Bearing within the image, before accounting for where the camera itself points.
    in_frame = -math.atan2(u - w / 2.0, focal)
    # The range correction uses the in-frame angle: it is about where the pixel sits on the
    # sensor, not about which way the head is turned.
    distance = axial / max(math.cos(in_frame), 1e-3)
    return in_frame + camera_yaw, distance
