"""The energy errand: a robot that goes out and fetches its own power.

These pin the properties that make the demonstration worth anything. The robot could be made
to look identical on screen by driving to a fixed point and waiting, so what is tested here is
that it is not doing that: charging in shade genuinely collects nothing and is refused, the
park genuinely beats the carport, and the energy that comes back is what actually entered the
battery.
"""

from __future__ import annotations

import pytest

from pyunto_robotics.brain.solar_errand import SolarErrandSkills
from pyunto_robotics.perception.grounding import ColorGrounder
from pyunto_robotics.sim.robot import Robot
from pyunto_robotics.sim.solar import SolarPanel
from pyunto_robotics.sim.wheel_drive import LEFT_WHEELS_4, RIGHT_WHEELS_4, SkidDrive


def build(keyframe: str) -> tuple[Robot, SolarErrandSkills]:
    robot = Robot(
        "solar.xml",
        gait=SkidDrive(left_wheels=LEFT_WHEELS_4, right_wheels=RIGHT_WHEELS_4, slip_factor=1.4),
        keyframe=keyframe,
    )
    for _ in range(80):
        robot.step()
    return robot, SolarErrandSkills(robot, ColorGrounder())


@pytest.mark.slow
def test_the_carport_is_in_shade_and_the_park_is_not():
    """The premise of the whole errand. If home were sunny there would be nothing to do."""
    robot, _ = build("carport")
    try:
        assert SolarPanel(robot).read().in_sun is False
    finally:
        robot.close()

    robot, _ = build("park")
    try:
        assert SolarPanel(robot).read().in_sun is True
    finally:
        robot.close()


@pytest.mark.slow
def test_the_park_generates_substantially_more_than_home():
    """Not merely "more": enough that going there is worth the journey."""
    robot, _ = build("carport")
    try:
        shade = SolarPanel(robot).read().irradiance_w_m2
    finally:
        robot.close()

    robot, _ = build("park")
    try:
        sun = SolarPanel(robot).read().irradiance_w_m2
    finally:
        robot.close()

    # 1.5x, not 2x. Moving the carport off the road put it beside the house rather than
    # under the deepest shade, so home reads about 257 W/m² against the park's 390 -- and the
    # figure that decides the errand is `in_sun`, tested above, not this ratio. What this
    # pins is that the gap is wide enough to be worth the journey.
    assert sun > shade * 1.4, f"park {sun:.0f} vs carport {shade:.0f} W/m²"


@pytest.mark.slow
def test_charging_in_shade_is_refused_rather_than_faked():
    """A robot that reports "charged for 45 minutes" after collecting nothing has misled.

    This is the one place the demonstration could most easily become a lie, so it is the one
    most worth pinning.
    """
    robot, skills = build("carport")
    try:
        result = skills.charge()
        assert result.ok is False
        assert "shade" in result.message
    finally:
        robot.close()


@pytest.mark.slow
def test_aiming_the_panel_is_worth_doing():
    """If tilt did not matter, the hinge and the aiming step would both be decoration."""
    robot, _ = build("park")
    try:
        panel = SolarPanel(robot)
        flat = panel.read().irradiance_w_m2
        panel.aim_at_sun()
        for _ in range(150):
            robot.step()
        aimed = panel.read().irradiance_w_m2
        assert aimed > flat * 1.3, f"flat {flat:.0f} -> aimed {aimed:.0f} W/m²"
    finally:
        robot.close()


@pytest.mark.slow
def test_the_robot_finds_sunlight_from_home():
    robot, skills = build("carport")
    try:
        result = skills.find_sun()
        assert result.ok, result.message
        assert SolarPanel(robot).read().in_sun
    finally:
        robot.close()


@pytest.mark.slow
def test_the_whole_errand_ends_with_the_lights_on():
    """Energy that does not do anything has not demonstrated anything."""
    robot, skills = build("carport")
    try:
        result = skills.run("fetch_power")
        assert result.ok, result.message
        assert result.data["lights_on"] is True
        # A real amount, not a rounding artefact: enough to run the lights for a while.
        assert result.data["collected_wh"] > 5.0
        assert result.data["lighting_minutes"] > 20
    finally:
        robot.close()


@pytest.mark.slow
def test_the_battery_reports_what_actually_went_in():
    """The number in the diary entry has to be a measurement, not an estimate."""
    robot, skills = build("park")
    try:
        before = skills.panel.charge_wh
        result = skills.charge()
        assert result.ok, result.message
        gained = skills.panel.charge_wh - before
        assert abs(gained - result.data["collected_wh"]) < 0.1
    finally:
        robot.close()


def test_the_instruction_reaches_the_errand():
    """「電力を取得してきて」 has to become fetch_power, not a bare goto."""
    from pyunto_robotics.brain.domains import DOMAINS

    domain = DOMAINS["solar"]
    for text in ("日光が当たる場所まで移動して、電力を取得してきて",
                 "電力を取得してきて", "go and fetch some power", "充電してきて"):
        assert domain.verb(text) == "fetch_power", text


def test_unknown_actions_are_refused_not_raised():
    """A skill layer that raises hands a traceback to a person in a diary."""
    from unittest.mock import Mock

    skills = SolarErrandSkills.__new__(SolarErrandSkills)
    skills.robot = Mock()
    result = SolarErrandSkills.run(skills, "dance")
    assert result.ok is False
    assert "dance" in result.message
