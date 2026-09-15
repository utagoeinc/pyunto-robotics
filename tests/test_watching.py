"""The watching flat: sensors, an older person, and what reaches the diary.

What is pinned here is mostly about NOT speaking. A watching system that reports every room
change, or cries wolf every night, is one nobody reads by the end of the week -- and then it
is worse than nothing, because the family believe they are being told.
"""

from __future__ import annotations

import pytest

from pyunto_robotics.brain.watching import WatchingSkills
from pyunto_robotics.sim.household import Day
from pyunto_robotics.sim.robot import Robot


def flat(day: Day | None = None) -> tuple[Robot, WatchingSkills]:
    # gait=None: this scene has no robot in it, and Robot's humanoid default reads a free
    # joint that does not exist here.
    robot = Robot("watch.xml", gait=None, keyframe="asleep")
    return robot, WatchingSkills(robot, None, day=day or Day.ordinary())


def run_day(skills: WatchingSkills, hours: int = 16) -> list[str]:
    said = []
    for _ in range(hours // 4):
        result = skills.watch("4")
        if result.data.get("notes"):
            said.extend(result.message.split("\n"))
    return said


@pytest.mark.slow
def test_the_sensors_follow_a_body_through_the_rooms():
    """The readings have to come from where the person is, or none of this is testable."""
    robot, skills = flat()
    try:
        skills.advance(8 * 60)  # through breakfast, on the sofa by now
        reading = skills.sensors.read(skills.minute)
        # Out of the bedroom, which is what the sensors are being asked to show.
        assert reading.room in ("kitchen", "living", "hallway", "bathroom")
        # `lying` is True here and that is correct: she is sitting on the sofa, and the
        # sensor cannot tell sitting from lying -- nor should it pretend to. What it must
        # not do is confuse either with standing.
        assert reading.position[0] > 0, "she should have left the bedroom (x < 0)"
    finally:
        robot.close()


@pytest.mark.slow
def test_an_ordinary_day_raises_no_alarm():
    """The hardest requirement. Anything that fires on a normal day gets the system muted."""
    robot, skills = flat(Day.ordinary())
    try:
        said = run_day(skills)
        assert said, "a whole day and it said nothing at all"
        assert not [line for line in said if "⚠️" in line], said
    finally:
        robot.close()


@pytest.mark.slow
def test_sleeping_through_the_night_is_not_an_alarm():
    """Seven motionless hours is alarming at two in the afternoon and normal at three a.m."""
    robot, skills = flat(Day.ordinary())
    try:
        result = skills.watch("8")  # midnight to 08:00
        assert "⚠️" not in result.message, result.message
    finally:
        robot.close()


@pytest.mark.slow
def test_a_long_bathroom_visit_is_reported():
    """The event these systems exist for."""
    robot, skills = flat(Day.long_bathroom())
    try:
        said = run_day(skills)
        warnings = [line for line in said if "⚠️" in line]
        assert warnings, said
        assert "bathroom" in warnings[0]
        assert any(k == "long_bathroom" for _, k in skills.events)
    finally:
        robot.close()


@pytest.mark.slow
def test_still_in_bed_at_midday_is_reported():
    """The quietest emergency, and the one plain stillness cannot catch.

    At eleven in the morning the stillness timer has only just started, because sleep does
    not count toward it. Time of day is what makes this visible.
    """
    robot, skills = flat(Day.did_not_get_up())
    try:
        said = run_day(skills, hours=16)
        warnings = [line for line in said if "⚠️" in line]
        assert warnings, "nobody noticed she never got up"
        assert "still in bed" in warnings[0]
    finally:
        robot.close()


@pytest.mark.slow
def test_walking_through_a_room_is_not_settling_in_it():
    """Passing through the living room announced her settling there one minute before she
    reached the kitchen, which reads as confusion rather than as watching."""
    robot, skills = flat(Day.ordinary())
    try:
        said = run_day(skills, hours=12)
        rooms = [line for line in said if "kitchen" in line or "living" in line]
        if len(rooms) >= 2:
            # Whatever order they come in, each room is announced at most once.
            assert len([r for r in rooms if "kitchen" in r]) == 1
            assert len([r for r in rooms if "living" in r]) == 1
    finally:
        robot.close()


@pytest.mark.slow
def test_check_answers_where_she_is_now():
    robot, skills = flat()
    try:
        skills.advance(9 * 60)
        result = skills.check()
        assert result.ok
        assert result.data["room"]
        assert ":" in result.message  # it says the time
    finally:
        robot.close()


@pytest.mark.slow
def test_the_devices_still_work():
    """The flat has an air conditioner and a lock, and watching without acting is half a
    product: noticing a room is 29°C is worth something, turning the aircon on is worth more."""
    robot, skills = flat()
    try:
        assert skills.run("lock").ok
        assert skills.run("lock_status").data["locked"] is True
        assert skills.run("set_temperature", "24度").data["target_c"] == 24.0
    finally:
        robot.close()


def test_instructions_reach_the_right_action():
    from pyunto_robotics.brain.domains import DOMAINS

    domain = DOMAINS["watch"]
    assert domain.verb("母の様子はどう？") == "check"
    assert domain.verb("how is she") == "check"
    assert domain.verb("今日は何してた？") == "today"
    assert domain.verb("2時間見ていて") == "watch"
    assert domain.verb("エアコンをつけて") == "aircon_on"
    assert domain.verb("鍵はかかってる？") == "lock_status"
