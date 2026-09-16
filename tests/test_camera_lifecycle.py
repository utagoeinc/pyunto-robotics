"""The camera has to survive more than one instruction.

The demo constructs a Robot and THEN opens the viewer. On macOS `launch_passive` hands the GL
context to the UI thread, so renderers built in `Robot.__init__` belong to a context somebody
else then owns: the robot answered the first instruction, photographed it, and stalled on the
second. Building them on first use puts them on the thread that actually renders.

This is the second time the same fault has been found -- the pet camera's segmentation
renderer had it too -- which is why it is pinned for every robot rather than for one.
"""

from __future__ import annotations

import pytest

from pyunto_robotics.sim.robot import Robot


def test_renderers_are_not_built_until_something_looks():
    robot = Robot("solar.xml", gait=None)
    try:
        assert robot._renderer is None, "a renderer was built before anything looked"
        robot.look("head_cam")
        assert robot._renderer is not None, "looking did not build a renderer"
    finally:
        robot.close()


def test_closing_without_ever_looking_is_safe():
    """A robot that is opened and shut without a camera frame must not raise."""
    robot = Robot("solar.xml", gait=None)
    robot.close()


@pytest.mark.slow
def test_the_camera_still_works_on_the_second_errand():
    """The reported symptom, as a test: one round worked, the next did not."""
    from pyunto_robotics import registry

    setup = registry.get("solar")
    robot = Robot(setup.scene, gait=setup.gait(), keyframe=setup.default_keyframe,
                  max_depth=setup.max_depth)
    skills = setup.skills(robot, None)
    try:
        for round_number in (1, 2):
            assert skills.run("find_sun").ok, f"errand {round_number} failed"
            frame = robot.look("head_cam")
            assert frame.rgb.size, f"no camera frame on round {round_number}"
            skills.run("home")
    finally:
        robot.close()


@pytest.mark.slow
@pytest.mark.parametrize("key", ["solar", "mars", "orchard", "hotel", "pet"])
def test_every_robot_with_a_camera_survives_a_second_round(key: str):
    """The fault was in Robot, so it was every robot's fault, not one robot's.

    `watch` is excluded deliberately: the watching flat has no camera by design, and the
    reporter is expected to degrade rather than raise -- covered separately below.
    """
    from pyunto_robotics import registry

    setup = registry.get(key)
    robot = Robot(setup.scene, gait=setup.gait() if setup.gait else None,
                  keyframe=setup.default_keyframe, max_depth=setup.max_depth)
    skills = setup.skills(robot, None)
    try:
        for round_number in (1, 2):
            frame = robot.look("head_cam")
            assert frame.rgb.size, f"{key}: no camera frame on round {round_number}"
    finally:
        if hasattr(skills, "close"):
            skills.close()
        robot.close()


def test_a_robot_with_no_camera_does_not_break_the_reporter():
    """The watching flat has no camera. Asking it for a photograph must degrade, not raise."""
    from pyunto_robotics import registry
    from pyunto_robotics.reporting import ThreadReporter

    class FakeClient:
        uuid = "x"

        def send(self, *args, **kwargs):
            return {"uuid": "1"}

        def send_image(self, *args, **kwargs):
            return {"uuid": "1"}

    setup = registry.get("watch")
    robot = Robot(setup.scene, gait=None, keyframe=setup.default_keyframe)
    reporter = ThreadReporter(FakeClient(), "space", "thread", robot=robot, camera="head_cam")
    try:
        for _ in range(3):
            reporter.show("no camera here")          # must not raise
    finally:
        robot.close()
