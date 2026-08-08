"""Landmark map tests.

The map exists to answer one question the grounder cannot: "is this the same door I was
already looking at?" These lock in the properties that make that answer trustworthy.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyunto_robotics.perception.landmarks import LandmarkMap


def test_repeated_sightings_become_one_landmark():
    """The same door seen from several places is one thing, not several."""
    m = LandmarkMap()
    for offset in (0.0, 0.2, -0.15, 0.1):
        m.observe("door", np.array([4.0 + offset, 1.0]))
    assert len(m) == 1
    assert m.of_label("door", confident_only=False)[0].sightings == 4


def test_distant_sightings_are_different_landmarks():
    m = LandmarkMap()
    m.observe("door", np.array([-4.0, 1.0]))
    m.observe("door", np.array([0.0, 1.0]))
    m.observe("door", np.array([4.0, 1.0]))
    assert len(m) == 3


def test_position_averages_toward_the_truth():
    """Single frames are 0.1-0.34 m out; the average of several should beat any one of them."""
    m = LandmarkMap()
    truth = np.array([4.0, 1.0])
    rng = np.random.default_rng(0)
    for _ in range(12):
        m.observe("door", truth + rng.normal(0, 0.3, 2))

    estimate = m.of_label("door", confident_only=False)[0].position
    assert float(np.linalg.norm(estimate - truth)) < 0.3


def test_one_sighting_is_not_yet_confident():
    """A bad range reading starts a landmark of its own; it should not be trusted at once."""
    m = LandmarkMap()
    m.observe("door", np.array([1.0, 3.0]))
    assert m.of_label("door") == []


def test_three_sightings_earn_confidence():
    m = LandmarkMap()
    for _ in range(3):
        m.observe("door", np.array([4.0, 1.0]))
    assert len(m.of_label("door")) == 1


def test_split_detections_in_one_frame_merge():
    """Close up, colour matching splits a door into two blobs at its edges.

    Left alone they become two landmarks about a metre apart, both heavily observed.
    """
    m = LandmarkMap()
    for _ in range(4):
        m.observe_all("door", [np.array([3.7, 1.0]), np.array([4.3, 1.0])])
    assert len(m.of_label("door")) == 1


def test_two_real_doors_never_merge():
    """The merge must not be so eager that it joins neighbours 4 m apart."""
    m = LandmarkMap()
    for _ in range(4):
        m.observe_all("door", [np.array([0.0, 1.0]), np.array([4.0, 1.0])])
    assert len(m.of_label("door")) == 2


def test_off_wall_phantom_is_dropped():
    """A door seen edge-on reads several metres too far and lands off the line of the others.

    Doors sit along a wall; something well off that line is a range error, not a door.
    """
    m = LandmarkMap()
    for _ in range(5):
        m.observe("door", np.array([-4.0, 1.0]))
        m.observe("door", np.array([0.0, 1.0]))
        m.observe("door", np.array([4.0, 1.0]))
        m.observe("door", np.array([1.4, 3.2]))  # phantom, well off the wall

    kept = m.of_label("door")
    assert len(kept) == 3
    assert all(abs(lm.position[1] - 1.0) < 0.5 for lm in kept)


def test_labels_do_not_mix():
    m = LandmarkMap()
    m.observe("door", np.array([0.0, 1.0]))
    m.observe("whiteboard", np.array([0.1, 1.0]))
    assert len(m) == 2


def test_forget_clears_everything():
    m = LandmarkMap()
    m.observe("door", np.array([0.0, 1.0]))
    m.forget()
    assert len(m) == 0


@pytest.mark.slow
def test_map_matches_the_real_office():
    """End to end: walking the corridor should produce three doors near their true positions."""
    from pyunto_robotics.nav.explore import MaplessNavigator
    from pyunto_robotics.perception.grounding import ColorGrounder
    from pyunto_robotics.sim.robot import Robot

    robot = Robot("office.xml", keyframe="lobby")
    try:
        nav = MaplessNavigator(robot, ColorGrounder())
        nav.goto("door", where="middle")

        doors = sorted(nav.landmarks.of_label("door"), key=lambda lm: lm.position[0])
        assert len(doors) >= 2, f"expected to map several doors, got {len(doors)}"
        # Every mapped door should be near one of the real ones at x = -4, 0, +4.
        for door in doors:
            nearest = min(abs(door.position[0] - x) for x in (-4.0, 0.0, 4.0))
            assert nearest < 1.5, f"mapped a door at x={door.position[0]:.2f}, nowhere near a real one"
    finally:
        robot.close()
