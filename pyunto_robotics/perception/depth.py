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


def depth_at(depth: np.ndarray, u: float, v: float, patch: int = 5) -> float | None:
    """Median depth in a small patch around a pixel.

    A single pixel lands on an edge often enough to matter; the median of a patch is what
    makes "how far is that door" answerable in practice.
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
    return float(np.median(valid))


def target_offset(
    depth: np.ndarray,
    u: float,
    v: float,
    fovy_deg: float,
    image_size: tuple[int, int],
) -> tuple[float, float] | None:
    """Convert a pixel into (bearing, distance) in the robot's frame.

    This is the bridge from vision to motion: a vision model says "the door is here in the
    image", and this turns that into something `step(vx, vy, wz)` can chase.
    """
    w, h = image_size
    distance = depth_at(depth, u, v)
    if distance is None:
        return None
    focal = (h / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    bearing = -math.atan2(u - w / 2.0, focal)
    return bearing, distance
