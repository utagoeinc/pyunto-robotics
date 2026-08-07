"""Perception and navigation tests.

The headline test is `test_navigates_to_door_without_a_map`: from across the office, using
nothing but camera frames, the robot has to end up at a door. Everything else here supports
that claim.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyunto_robotics.nav.explore import MaplessNavigator, NavState
from pyunto_robotics.perception.depth import depth_at, free_space, target_offset
from pyunto_robotics.perception.grounding import ColorGrounder, _parse_detections
from pyunto_robotics.sim.robot import Robot


@pytest.fixture(scope="module")
def robot():
    r = Robot("office.xml", keyframe="lobby")
    yield r
    r.close()


# -- depth geometry ------------------------------------------------------------------


def test_free_space_matches_known_geometry(robot):
    """From the lobby the corridor wall is 4 m ahead; the depth reading must agree."""
    robot.reset("lobby")
    robot.stand(0.3)
    obs = robot.look()

    space = free_space(obs.depth, robot.camera_fovy())
    expected = 1.0 - robot.position[1]  # door plane at y=+1.0
    assert abs(space.clearance_ahead() - expected) < 0.6, (
        f"clearance {space.clearance_ahead():.2f} m vs geometry {expected:.2f} m"
    )


def test_free_space_is_symmetric_in_a_symmetric_view(robot):
    """Facing straight down a symmetric corridor, left and right must read alike."""
    robot.reset("lobby")
    robot.stand(0.3)
    space = free_space(robot.look().depth, robot.camera_fovy())

    left = space.ranges[space.bearings > 0.3]
    right = space.ranges[space.bearings < -0.3]
    assert abs(left.mean() - right.mean()) < 0.5


def test_free_space_bearings_span_the_field_of_view(robot):
    space = free_space(robot.look().depth, robot.camera_fovy())
    assert math.degrees(space.bearings.max()) > 30
    assert math.degrees(space.bearings.min()) < -30
    assert space.bearings[0] > space.bearings[-1], "bearings must run left to right"


def test_depth_at_rejects_out_of_bounds():
    depth = np.full((20, 30), 2.0, dtype=np.float32)
    assert depth_at(depth, 15, 10) == pytest.approx(2.0)
    assert depth_at(depth, -5, 10) is None
    assert depth_at(depth, 100, 10) is None


def test_target_offset_sign_convention():
    """Left of centre is a positive bearing, matching Robot.bearing_to_pixel."""
    depth = np.full((100, 200), 3.0, dtype=np.float32)
    left = target_offset(depth, 20, 50, 75.0, (200, 100))
    right = target_offset(depth, 180, 50, 75.0, (200, 100))
    centre = target_offset(depth, 100, 50, 75.0, (200, 100))

    assert left is not None and right is not None and centre is not None
    assert left[0] > 0 > right[0]
    assert abs(centre[0]) < 1e-6
    assert centre[1] == pytest.approx(3.0)


# -- grounding -----------------------------------------------------------------------


def test_finds_three_doors_from_the_lobby(robot):
    """All three doorways are visible from the lobby; the grounder should see them."""
    robot.reset("lobby")
    robot.stand(0.3)
    detections = ColorGrounder().find(robot.look().rgb, "door")

    assert len(detections) >= 3, f"expected 3 doors, found {len(detections)}"
    xs = sorted(d.x for d in detections[:3])
    assert xs[0] < 0.25 and 0.35 < xs[1] < 0.65 and xs[2] > 0.75, (
        f"doors should be spread left/centre/right, got {xs}"
    )


def test_japanese_and_english_descriptions_agree(robot):
    """The instruction may arrive in either language."""
    robot.reset("lobby")
    robot.stand(0.3)
    rgb = robot.look().rgb
    grounder = ColorGrounder()
    assert len(grounder.find(rgb, "オフィスのドア")) == len(grounder.find(rgb, "door"))


def test_unknown_object_returns_nothing(robot):
    assert ColorGrounder().find(robot.look().rgb, "a purple giraffe") == []


def test_detection_pixel_conversion():
    from pyunto_robotics.perception.grounding import Detection

    det = Detection(label="door", x=0.25, y=0.5, confidence=1.0)
    assert det.pixel(400, 200) == (100.0, 100.0)


@pytest.mark.parametrize(
    "reply,expected",
    [
        ('[{"label":"door","x":0.5,"y":0.4,"confidence":0.9}]', 1),
        ('Here it is:\n```json\n[{"label":"door","x":0.5,"y":0.4}]\n```', 1),
        ("[]", 0),
        ("I cannot see a door.", 0),
        ('[{"label":"door","x":500,"y":400}]', 1),  # 0-1000 coords get rescaled
        ('[{"label":"door","x":"bad"}]', 0),
    ],
)
def test_vlm_reply_parsing(reply, expected):
    """Models wrap JSON in prose, use code fences, and sometimes ignore the 0-1 request."""
    assert len(_parse_detections(reply)) == expected


def test_vlm_coordinates_are_normalised():
    dets = _parse_detections('[{"label":"door","x":500,"y":400,"confidence":0.8}]')
    assert dets[0].x == pytest.approx(0.5)
    assert dets[0].y == pytest.approx(0.4)


# -- navigation ----------------------------------------------------------------------


def test_navigates_to_door_without_a_map(robot):
    """The core claim: cross the office to a door using only camera frames."""
    robot.reset("lobby")
    start = robot.position.copy()

    result = MaplessNavigator(robot, ColorGrounder()).goto("door", max_steps=900)

    assert result.state is NavState.ARRIVED, f"navigation failed: {result.describe()}"
    assert result.distance is not None and result.distance < 1.0
    # It must actually have travelled toward the doors (which are at +y).
    assert robot.position[1] > start[1] + 1.5, "did not make progress toward the doors"


def test_navigation_reports_failure_for_absent_target(robot):
    robot.reset("lobby")
    result = MaplessNavigator(robot, ColorGrounder()).goto(
        "a purple giraffe", max_steps=120, search_steps=60
    )
    assert not result.success
    assert "could not find" in result.describe()


def test_face_turns_toward_the_target(robot):
    """After facing a door it should be near the centre of the image."""
    robot.reset("lobby")
    robot.stand(0.3)
    nav = MaplessNavigator(robot, ColorGrounder())

    result = nav.face("door", max_steps=150)
    assert result.success
    assert result.bearing is not None and abs(result.bearing) < 0.12
