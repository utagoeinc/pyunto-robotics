"""Keeping track of which door is which.

The grounder answers "where are the doors in this image". That is enough to walk toward one,
and not enough to keep walking toward the *same* one: three identical doors carry nothing that
tells them apart, so an approach that re-decides every frame drifts from the door it chose to
whichever is nearest now.

A LandmarkMap fixes that by giving each door an identity. Every sighting is matched against
what has been seen before -- by where it is in the world, which is the one thing about a door
that does not change -- and either updates an existing landmark or creates a new one. Chasing
landmark #2 then means the same thing from anywhere in the building.

This is not SLAM and not a prior map. Nothing is loaded from disk, nothing is built ahead of
time, and the robot still cannot navigate to a door it has not seen. It is the smallest amount
of memory that makes "that one, not the other one" expressible: a handful of points, each
averaged over its own sightings.

Averaging is what makes it hold up. A single frame puts a door 0.1-0.34 m from its true
position, and the error swings with viewpoint; a landmark that has been seen from several
angles settles close to the truth, which is what lets the robot walk down the middle of a
corridor with all three doors in frame and still know which one it is going to.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

# Two sightings this far apart in the world are different objects. The doors are 4 m apart, so
# this leaves generous room for the per-frame error without ever merging neighbours.
MATCH_RADIUS_M = 1.5

# How much a new sighting moves the stored estimate. Low, because single frames are noisy and
# there is no hurry: a landmark seen ten times should sit near the average of those ten, not
# wherever it was last seen from.
BLEND = 0.25


@dataclass
class Landmark:
    """One physical object the robot has seen, tracked across sightings."""

    id: int
    label: str
    position: np.ndarray  # (2,) world x, y
    sightings: int = 1
    last_seen_step: int = 0
    _samples: list[np.ndarray] = field(default_factory=list, repr=False)

    @property
    def spread(self) -> float:
        """How far its sightings scatter, in metres.

        A real object is seen in roughly the same place from every angle. A phantom -- one bad
        range estimate that started its own landmark -- has nothing holding it together, so its
        samples wander.
        """
        if len(self._samples) < 2:
            return 0.0
        samples = np.array(self._samples[-8:])
        return float(np.mean(np.linalg.norm(samples - samples.mean(axis=0), axis=1)))

    @property
    def confident(self) -> bool:
        """Whether this has been seen enough times, consistently enough, to act on.

        Both tests earn their place. Three sightings, because a bad range estimate lands far
        enough from a real door to start a landmark of its own and those are usually seen once
        or twice -- measured two phantoms alongside three real doors on one walk. And a tight
        spread, because a phantom that does keep being re-matched drifts, while a real door
        stays put.
        """
        return self.sightings >= 3 and self.spread < 0.8

    def observe(self, position: np.ndarray, step: int) -> None:
        """Fold in a new sighting."""
        self.sightings += 1
        self.last_seen_step = step
        self._samples.append(position.copy())
        # Average over recent sightings rather than tracking the latest. Individual frames are
        # noisy and biased by viewpoint; the mean of several angles is much closer to truth.
        recent = self._samples[-8:]
        self.position = (1 - BLEND) * self.position + BLEND * np.mean(recent, axis=0)


class LandmarkMap:
    """The objects the robot has noticed, and where they are.

    Deliberately tiny: a list of points with running averages, no graph, no optimisation, no
    loop closure. It exists to answer one question -- "is this the thing I was already looking
    at?" -- which bearing alone cannot.

    Landmarks of the same label are expected to be co-linear, which is how phantoms get
    rejected. Doors in a building sit along walls, and a bad range estimate -- typically from
    seeing a leaf edge-on, which reads several metres too far -- lands well off that line.
    Nothing about the office is assumed: the line is fitted to whatever has actually been seen.
    """

    def __init__(self, match_radius: float = MATCH_RADIUS_M):
        self.match_radius = match_radius
        self._landmarks: dict[int, Landmark] = {}
        self._next_id = 1
        self._step = 0

    def __len__(self) -> int:
        return len(self._landmarks)

    def __iter__(self):
        return iter(self._landmarks.values())

    def get(self, landmark_id: int) -> Landmark | None:
        return self._landmarks.get(landmark_id)

    def observe(self, label: str, position: np.ndarray) -> Landmark:
        """Record a sighting, matching it to an existing landmark or creating a new one."""
        self._step += 1
        match = self._nearest(label, position)
        if match is not None:
            match.observe(position, self._step)
            return match

        landmark = Landmark(
            id=self._next_id,
            label=label,
            position=position.copy(),
            last_seen_step=self._step,
            _samples=[position.copy()],
        )
        self._landmarks[landmark.id] = landmark
        self._next_id += 1
        log.debug("new landmark #%d %s at %s", landmark.id, label, np.round(position, 2))
        return landmark

    def observe_all(self, label: str, positions: list[np.ndarray]) -> list[Landmark]:
        """Record the sightings from one frame.

        Near-duplicates are merged first. Colour matching splits a door into two blobs when it
        is close enough to fill the frame -- its two edges catch the light differently -- and
        without merging, one blob matches the existing landmark and the other is pushed out by
        the exclusion below into a landmark of its own. Measured one door becoming two, 0.89 m
        apart, both with hundreds of sightings.

        What remains is matched greedily so two genuinely distinct detections cannot both claim
        the same landmark.
        """
        merged = self._merge_duplicates(positions)

        claimed: set[int] = set()
        results: list[Landmark] = []
        for position in merged:
            match = self._nearest(label, position, exclude=claimed)
            if match is not None:
                self._step += 1
                match.observe(position, self._step)
                claimed.add(match.id)
                results.append(match)
            else:
                landmark = self.observe(label, position)
                claimed.add(landmark.id)
                results.append(landmark)
        return results

    @staticmethod
    def _merge_duplicates(
        positions: list[np.ndarray], radius: float = 0.6
    ) -> list[np.ndarray]:
        """Collapse detections in one frame that are too close to be separate objects.

        0.6 m: wide enough to join the two edges of one door seen close up, narrow enough that
        two real doors 4 m apart can never merge. At 1.2 m the target door was being absorbed
        into its neighbour on the final approach and disappeared from the map entirely.
        """
        clusters: list[list[np.ndarray]] = []
        for position in positions:
            for cluster in clusters:
                if float(np.linalg.norm(cluster[0] - position)) < radius:
                    cluster.append(position)
                    break
            else:
                clusters.append([position])
        return [np.mean(cluster, axis=0) for cluster in clusters]

    def of_label(self, label: str, confident_only: bool = True) -> list[Landmark]:
        """Every landmark with this label.

        Confident-only by default: a caller asking "which doors are there" wants the real ones,
        not the phantoms a noisy range throws off along the way.
        """
        if confident_only:
            self._consolidate()
        candidates = [
            lm
            for lm in self._landmarks.values()
            if lm.label == label and (lm.confident or not confident_only)
        ]
        return self._drop_outliers(candidates) if confident_only else candidates

    @staticmethod
    def _drop_outliers(landmarks: list[Landmark], tolerance: float = 0.7) -> list[Landmark]:
        """Remove landmarks that do not lie on the line the others form.

        Doors along a wall are co-linear. A phantom from an edge-on range reading is not, and
        it survives the sighting-count and spread tests because it keeps being re-detected in
        the same wrong place -- measured one sitting 1.4 m off the wall with a tighter spread
        than a real door. Fitting a line to the group and dropping what misses it catches
        exactly that, without hard-coding where any wall is.
        """
        if len(landmarks) < 3:
            return landmarks  # too few to say which is the odd one out

        points = np.array([lm.position for lm in landmarks])

        def offsets_from_line(subset: np.ndarray) -> np.ndarray:
            centre = subset.mean(axis=0)
            # Principal direction of the group; the smaller singular vector is the normal.
            _, _, vt = np.linalg.svd(subset - centre)
            return np.abs((points - centre) @ vt[1])

        # Fit once, drop the worst offender, then refit. A single phantom drags the line toward
        # itself enough to push a real door outside the tolerance -- measured a true door at
        # 0.83 against a 0.7 threshold while the phantom sat at 1.56. Refitting without it puts
        # the real ones back on the line.
        first = offsets_from_line(points)
        if float(first.max()) > tolerance and len(landmarks) > 3:
            keep = np.ones(len(points), dtype=bool)
            keep[int(np.argmax(first))] = False
            offsets = offsets_from_line(points[keep])
        else:
            offsets = first

        return [lm for lm, off in zip(landmarks, offsets, strict=True) if off <= tolerance]

    def _nearest(
        self, label: str, position: np.ndarray, exclude: set[int] | None = None
    ) -> Landmark | None:
        best: Landmark | None = None
        best_distance = self.match_radius
        for landmark in self._landmarks.values():
            if landmark.label != label:
                continue
            if exclude and landmark.id in exclude:
                continue
            distance = float(np.linalg.norm(landmark.position - position))
            if distance < best_distance:
                best_distance = distance
                best = landmark
        return best

    def _consolidate(self, radius: float = 1.3) -> None:
        """Merge landmarks that have converged onto the same object.

        Frame-level de-duplication is not enough on its own: the two edges of a door can be
        detected in *different* frames, each starting its own landmark before either has moved
        near the other. Measured one door held as two entries 0.89 m apart, both with 160-odd
        sightings. Sightings are pooled so the merged landmark keeps the evidence of both.

        1.3 m sits between the two scales that matter: the split halves of one door end up
        about 1.0 m apart, and real doors are 4 m apart, so this cannot join two of them.
        """
        by_label: dict[str, list[Landmark]] = {}
        for landmark in self._landmarks.values():
            by_label.setdefault(landmark.label, []).append(landmark)

        for group in by_label.values():
            group.sort(key=lambda lm: -lm.sightings)
            for i, keeper in enumerate(group):
                if keeper.id not in self._landmarks:
                    continue
                for other in group[i + 1 :]:
                    if other.id not in self._landmarks or other.id == keeper.id:
                        continue
                    if float(np.linalg.norm(keeper.position - other.position)) < radius:
                        total = keeper.sightings + other.sightings
                        keeper.position = (
                            keeper.position * keeper.sightings + other.position * other.sightings
                        ) / total
                        keeper.sightings = total
                        keeper._samples = (keeper._samples + other._samples)[-8:]
                        del self._landmarks[other.id]

    def forget(self) -> None:
        """Drop everything. Used when the robot is teleported or the scene resets."""
        self._landmarks.clear()
        self._next_id = 1
        self._step = 0
