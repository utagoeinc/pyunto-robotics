"""The orchard quadruped: fetch fruit from the rows, carry it to the shed.

The round trip is the demonstration -- out to something it was not given the position of,
and back to somewhere it was. What these pin is that the robot does not claim to have done
either half when it has not.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyunto_robotics.brain.orchard import CRATES_PER_TRIP, OrchardSkills
from pyunto_robotics.perception.grounding import ColorGrounder
from pyunto_robotics.sim.quad_gait import TrotGait
from pyunto_robotics.sim.robot import Robot

ORCHARD_DEPTH_M = 40.0


def build(keyframe: str = "shed") -> tuple[Robot, OrchardSkills]:
    robot = Robot("orchard.xml", gait=TrotGait(), keyframe=keyframe, max_depth=ORCHARD_DEPTH_M)
    for _ in range(400):
        robot.step()
    return robot, OrchardSkills(robot, ColorGrounder())


@pytest.mark.slow
def test_a_trotting_robot_stays_on_the_ground():
    """Built as a bare heightfield, this scene threw the robot 500 m up in 300 steps.

    A trot plants and lifts feet many times a second, and hfield contact normals flip at cell
    boundaries in a way a plane's do not. The ground is a plane with the rough patch laid on
    top, and this is what says it stayed that way.
    """
    robot, _ = build()
    try:
        for _ in range(600):
            robot.step(0.4, 0.0, 0.0)
        assert robot.position[2] < 1.0, f"robot is at z={robot.position[2]:.1f}"
    finally:
        robot.close()


@pytest.mark.slow
def test_the_whole_errand_delivers_the_crates():
    robot, skills = build()
    try:
        result = skills.run("fetch")
        assert result.ok, result.message
        assert result.data["delivered"] == CRATES_PER_TRIP
        assert result.data["carrying"] == 0
        # And it ended up back at the shed, not merely claiming to have.
        assert float(np.linalg.norm(robot.position[:2] - skills.base)) < 2.5
    finally:
        robot.close()


@pytest.mark.slow
def test_loading_is_refused_when_the_crates_are_not_there():
    """Reporting fruit loaded tells a grower it is on the way when it is still in the row."""
    robot, skills = build()
    try:
        result = skills.collect()
        assert result.ok is False
        assert skills.carrying == 0
    finally:
        robot.close()


@pytest.mark.slow
def test_the_robot_walks_to_the_crates_by_looking():
    """The crates' position is never given to the robot; it finds them down the lane."""
    robot, skills = build()
    try:
        result = skills.goto("crates")
        assert result.ok, result.message
        # The crates are at y=+9; the robot started at y=-7.5.
        assert robot.position[1] > 5.0, f"ended at {robot.position[:2]}"
    finally:
        robot.close()


@pytest.mark.slow
def test_carrying_is_reported_honestly():
    robot, skills = build()
    try:
        assert skills.carrying_what().data["carrying"] == 0
        skills.carrying = 2
        assert "2 crates" in skills.carrying_what().message
    finally:
        robot.close()


def test_instructions_reach_the_right_action():
    from pyunto_robotics.brain.domains import DOMAINS

    domain = DOMAINS["orchard"]
    assert domain.verb("fetch the crate of apples") == "fetch"
    assert domain.verb("fetch the apples") == "fetch"
    assert domain.verb("take them to the shed") == "deliver"
    assert domain.verb("what are you carrying") == "carrying"
    assert domain.object_in("go to the crates") == "crates"
