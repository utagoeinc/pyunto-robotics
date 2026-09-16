"""The Mars rover: told where to go in a diary entry, finds it by camera.

What is pinned here is mapless navigation actually working -- the rover holds no map, and the
targets are found by looking. The failures these tests would have caught were all silent ones:
the rover reported seeing a target, drove confidently, and arrived somewhere else.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyunto_robotics.brain.rover import RoverSkills
from pyunto_robotics.perception.grounding import ColorGrounder
from pyunto_robotics.sim.robot import Robot
from pyunto_robotics.sim.wheel_drive import SkidDrive

# The rover's own far-field limit. The indoor default of 12 m makes every target beyond it
# read as exactly 12 m, which is the bug these tests exist to keep out.
MARS_DEPTH_M = 60.0


def build(keyframe: str = "channel") -> tuple[Robot, RoverSkills]:
    robot = Robot("mars.xml", gait=SkidDrive(), keyframe=keyframe, max_depth=MARS_DEPTH_M)
    for _ in range(200):
        robot.step()
    return robot, RoverSkills(robot, ColorGrounder())


@pytest.mark.slow
@pytest.mark.parametrize("target", ["cache", "beacon", "lander"])
def test_the_rover_reaches_each_target(target):
    robot, skills = build()
    try:
        result = skills.run("goto", target)
        assert result.ok, result.message
    finally:
        robot.close()


@pytest.mark.slow
def test_depth_is_not_clipped_at_the_indoor_limit():
    """Every target past the limit collapses onto it, and the rover drives to a phantom.

    The beacon is 19 m away. At the indoor 12 m the rover read it as 12, drove to a point 7 m
    short, and circled there -- while reporting, truthfully, that it had seen the beacon.
    """
    robot, _ = build()
    try:
        depth = robot.look().depth
        assert depth.max() > 12.5, "depth is still clipped at the indoor limit"
    finally:
        robot.close()


@pytest.mark.slow
def test_landmarks_are_not_confused_with_the_ground():
    """Mars is one colour. A landmark that shares it is worse than no landmark.

    A red beacon measured three sightings scattered across open regolith, and a gold lander
    matched 58,000 pixels of pure terrain from 21 m.
    """
    robot, _ = build()
    try:
        grounder = ColorGrounder()
        # Facing away from every piece of hardware, nothing should be detected.
        import mujoco

        robot.data.qpos[3:7] = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]  # +y, open ground
        mujoco.mj_forward(robot.model, robot.data)
        for _ in range(60):
            robot.step()
        frame = robot.look().rgb
        for target in ("beacon", "lander", "cache"):
            assert grounder.find(frame, target) == [], f"false {target} on open ground"
    finally:
        robot.close()


@pytest.mark.slow
def test_going_home_does_not_depend_on_seeing_the_lander():
    """The one place the rover knows without looking is where it started.

    Searching for the lander by camera from the channel floor caught glimpses over a bank and
    lost them as the terrain rolled -- 9000 steps of going nowhere while reporting a distance.
    """
    robot, skills = build()
    try:
        for _ in range(600):
            robot.step(0.5, 0.0, 0.0)
        assert skills._from_base() > 3.0, "the rover did not leave"
        result = skills.home()
        assert result.ok, result.message
        assert skills._from_base() < 3.0
    finally:
        robot.close()


@pytest.mark.slow
def test_tilt_is_measured_and_reported():
    """On a planet nobody can see the machine; the numbers are all an operator has."""
    robot, skills = build()
    try:
        result = skills.attitude()
        assert result.ok
        assert 0.0 <= result.data["tilt_deg"] < 90.0
    finally:
        robot.close()


def test_instructions_reach_the_right_action():
    from pyunto_robotics.brain.domains import DOMAINS

    domain = DOMAINS["mars"]
    assert domain.verb("drive to the sample") == "goto"
    assert domain.object_in("drive to the sample") == "cache"
    assert domain.verb("drive to the beacon") == "goto"
    assert domain.object_in("drive to the beacon") == "beacon"
    assert domain.verb("go back to the lander") == "home"
    assert domain.verb("how steep is it") == "attitude"
