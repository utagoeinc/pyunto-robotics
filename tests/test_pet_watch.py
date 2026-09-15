"""The pet camera robot: can it find a cat at four different heights, and admit when it cannot.

These are slow -- each one drives a robot around a flat -- but they guard the two properties
that make the demonstration worth anything. It must find her where she actually is, and it
must NOT claim to have found her somewhere she is not.
"""

from __future__ import annotations

import pytest

from pyunto_robotics.brain.pet_watch import HAUNTS, PetWatchSkills
from pyunto_robotics.sim.robot import Robot
from pyunto_robotics.sim.wheel_drive import LEFT_WHEELS_4, RIGHT_WHEELS_4, SkidDrive


def flat(keyframe: str = "dock") -> tuple[Robot, PetWatchSkills]:
    robot = Robot(
        "pet.xml",
        gait=SkidDrive(left_wheels=LEFT_WHEELS_4, right_wheels=RIGHT_WHEELS_4,
                       slip_factor=1.2),
        keyframe=keyframe,
    )
    return robot, PetWatchSkills(robot)


# Which keyframe puts the cat in which haunt.
WHERE = {"dock": "sofa", "sill": "sill", "shelf": "shelf", "tree": "tree"}


@pytest.mark.slow
@pytest.mark.parametrize("keyframe,expected", sorted(WHERE.items()))
def test_it_finds_her_at_every_height(keyframe: str, expected: str):
    """Under the sofa, on the sill, in the tree, on top of the bookshelf.

    Four places at four heights, and none reachable by driving alone -- which is the entire
    reason this robot has a pan/tilt head rather than a fixed lens.
    """
    robot, skills = flat(keyframe)
    try:
        result = skills.find()
        assert result.data.get("where") == expected, result.message
    finally:
        skills.close()
        robot.close()


@pytest.mark.slow
def test_it_does_not_put_her_somewhere_she_is_not():
    """Confident, specific and wrong is the worst possible answer here.

    The owner is out and cannot check. An early version swept the camera wide enough to catch
    the cat asleep on the bookshelf while looking at the sofa, and reported her as being under
    the sofa. A sighting has to belong to the place that was looked at.
    """
    robot, skills = flat("shelf")
    try:
        sx, sy, ax, ay, az, _name = HAUNTS["sofa"]
        skills._drive_to(sx, sy)
        assert skills._sweep_for_cat(ax, ay, az) == 0.0
    finally:
        skills.close()
        robot.close()


@pytest.mark.slow
def test_it_can_reach_every_place_including_the_kitchen():
    """The kitchen is behind a divider, and driving straight at it wedges the robot."""
    robot, skills = flat()
    try:
        for key, (sx, sy, *_rest) in HAUNTS.items():
            assert skills._drive_to(sx, sy), f"could not reach {key}"
        assert skills.home().data["docked"]
    finally:
        skills.close()
        robot.close()


@pytest.mark.slow
def test_the_camera_is_what_finds_her_not_the_wheels():
    """Aimed at the right place from the right spot, she is in frame; level, she is not.

    If this ever passes with the head held level, the scene has stopped testing what it was
    built to test and the hiding places need moving back up and down.
    """
    robot, skills = flat("shelf")
    try:
        sx, sy, ax, ay, az, _name = HAUNTS["shelf"]
        skills._drive_to(sx, sy)
        skills._aim_at(ax, ay, az)
        aimed = skills._cat_in_frame()
        skills._tilt = 0.0
        skills._apply_head()
        level = skills._cat_in_frame()
        assert aimed > level, f"tilt made no difference: aimed={aimed} level={level}"
    finally:
        skills.close()
        robot.close()


def test_the_renderer_is_not_built_until_it_is_used():
    """The demo builds skills AFTER opening the viewer.

    On macOS `launch_passive` hands the GL context to the UI thread, so a renderer constructed
    at init belongs to a context somebody else then owns -- the robot answered the first
    message and went quiet afterwards. Building it lazily puts it on the thread that renders.
    """
    robot, skills = flat()
    try:
        assert skills._seg is None, "the renderer must not exist before the first look"
        skills._cat_in_frame()
        assert skills._seg is not None, "the renderer should be built on first use"
    finally:
        skills.close()
        robot.close()


@pytest.mark.slow
def test_a_broken_camera_is_not_reported_as_an_empty_flat():
    """The owner is out. "She is not in any of her usual places" would send them home."""

    class Broken:
        def update_scene(self, *args, **kwargs):
            raise RuntimeError("GL context is not current")

        def close(self):
            pass

    robot, skills = flat("sill")
    try:
        assert skills.find().data.get("where") == "sill"
        skills._seg = Broken()
        result = skills.find()
        assert not result.ok
        assert result.data.get("camera_failed")
        assert "カメラ" in result.message
    finally:
        skills.close()
        robot.close()
