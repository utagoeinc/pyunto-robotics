"""The hotel cleaner: two corridors, a lift between them.

The lift is what is being demonstrated, so what these pin is that riding it is real -- the
robot goes up because it is standing on a platform that moves -- and that the robot's own
report of which floor it is on comes from measuring its height rather than from counting the
instructions it was given.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyunto_robotics.brain.hotel import HotelSkills
from pyunto_robotics.perception.grounding import ColorGrounder
from pyunto_robotics.sim.robot import Robot

HOTEL_DEPTH_M = 30.0


def build(keyframe: str = "corridor") -> tuple[Robot, HotelSkills]:
    robot = Robot("hotel.xml", keyframe=keyframe, max_depth=HOTEL_DEPTH_M)
    for _ in range(400):
        robot.step()
    return robot, HotelSkills(robot, ColorGrounder())


@pytest.mark.slow
def test_the_robot_rides_the_lift_to_the_other_floor():
    """A full floor of travel, standing on the car while it moves."""
    robot, skills = build("in_lift")
    try:
        before = robot.position[2]
        result = skills.ride()
        assert result.ok, result.message
        assert robot.position[2] - before > 3.0, f"{before:.2f} -> {robot.position[2]:.2f}"
        assert skills.current_floor() == 2
    finally:
        robot.close()


@pytest.mark.slow
def test_riding_is_refused_when_not_in_the_car():
    """Otherwise the robot reports arriving on a floor it never left."""
    robot, skills = build("corridor")
    try:
        result = skills.ride()
        assert result.ok is False
        assert skills.current_floor() == 1
    finally:
        robot.close()


@pytest.mark.slow
def test_the_floor_the_robot_reports_is_the_floor_it_is_on():
    """Measured from height, not counted from instructions.

    This caught a real failure: the robot rode up, stepped out of the car into a gap where the
    upper floor had not been built, and fell back to the ground -- while its report said it
    had reached floor 2, which by its own bookkeeping it had.
    """
    robot, skills = build("corridor")
    try:
        assert skills.current_floor() == 1
        skills.board()
        skills.ride()
        assert skills.current_floor() == 2
        # And it is still there after working the corridor, rather than having fallen off.
        skills.clean_floor()
        assert skills.current_floor() == 2, f"fell to z={robot.position[2]:.2f}"
    finally:
        robot.close()


@pytest.mark.slow
def test_the_whole_job_cleans_both_corridors():
    robot, skills = build("corridor")
    try:
        result = skills.run("clean")
        assert result.ok, result.message
        assert result.data["cleaned"] == [1, 2]
        assert skills.current_floor() == 2
    finally:
        robot.close()


@pytest.mark.slow
def test_boarding_puts_the_robot_in_the_car():
    robot, skills = build("corridor")
    try:
        result = skills.board()
        assert result.ok, result.message
        assert skills._in_car(), f"ended at {robot.position[:2]}"
    finally:
        robot.close()


def test_instructions_reach_the_right_action():
    from pyunto_robotics.brain.domains import DOMAINS

    domain = DOMAINS["hotel"]
    assert domain.verb("clean both floors") == "clean"
    assert domain.verb("clean the corridor") == "clean_floor"
    assert domain.verb("get in the lift") == "board"
    assert domain.verb("go up a floor") == "ride"
    assert domain.verb("which floor are you on") == "floor"
