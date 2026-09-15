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


@pytest.mark.slow
def test_the_wall_clock_shows_her_time_of_day():
    """A viewer cannot tell a quiet afternoon from a frozen simulation without one."""
    import mujoco

    robot, skills = flat()
    try:
        skills.advance(7 * 60 + 35)  # 07:35

        def lit(name: str) -> bool:
            geom = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            on = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_MATERIAL, "seg_on")
            return robot.model.geom_matid[geom] == on

        # "0" lights every segment but the middle bar; "1" lights only the right-hand pair.
        assert lit("clock_d0_a") and not lit("clock_d0_g"), "first digit should read 0"
        assert lit("clock_d1_b") and lit("clock_d1_c"), "second digit should read 7"
    finally:
        robot.close()


@pytest.mark.slow
def test_the_speed_shown_is_measured_not_declared():
    """A fixed number would be decoration: the rate depends on the machine."""
    robot, skills = flat()
    try:
        skills.advance(120)
        assert 1 <= skills.speed_pips <= 5
        # This runs far faster than real time, so it should be at the top of the scale.
        assert skills.speed_pips >= 4, skills.speed_pips
    finally:
        robot.close()


@pytest.mark.slow
def test_she_still_has_a_life_on_the_second_day():
    """The routine repeats. It did not, and she lay in bed from day two onward.

    `place_at` took an absolute minute against a schedule covering one day, so once the clock
    passed midnight every entry matched and it returned the last one -- "bed" -- forever. In a
    system whose whole purpose is noticing that someone has stopped moving, a bug that fakes
    it is the worst one available.
    """
    robot, skills = flat()
    try:
        skills.advance(24 * 60)          # through to day two
        rooms = set()
        for _ in range(24 * 60):
            skills.advance(1)
            rooms.add(skills.sensors.read(skills.minute).room)
        assert rooms - {"bedroom"}, f"she never left the bedroom on day two: {rooms}"
        assert "kitchen" in rooms, f"no meals on day two: {rooms}"
    finally:
        robot.close()


@pytest.mark.slow
def test_the_house_keeps_speaking_on_the_second_day():
    """Every "first of the day" was really a first of the run.

    `_seen_rooms`, the warning flags and the `got_up` event were set once and never cleared,
    so from day two the house said nothing at all -- and silence from a watching system reads
    as "nothing is wrong", which is the most dangerous thing it could get wrong.
    """
    robot, skills = flat()
    try:
        day_one = skills.advance(24 * 60)
        day_two = skills.advance(24 * 60)
        assert day_one, "nothing was said on day one"
        assert day_two, "the house went silent on day two"
        assert any("she is up" in line for line in day_two), day_two
    finally:
        robot.close()


@pytest.mark.slow
def test_an_unusual_day_does_not_quietly_resolve_itself():
    """A long bathroom visit must not end because midnight came round."""
    robot, skills = flat(day=Day.long_bathroom())
    try:
        skills.advance(26 * 60)          # well past midnight
        assert skills.sensors.read(skills.minute).room == "bathroom"
    finally:
        robot.close()


def test_there_is_no_camera_inside_the_flat():
    """A constraint, not an omission. See the module docstring in brain/watching.py.

    The person being watched did not ask for any of this; her family did. Indoor footage is
    the line between a product somebody would install in their mother's home and one they
    would not, so it is pinned here rather than left to whoever edits the scene next.
    """
    import mujoco

    robot, _ = flat()
    try:
        cameras = [
            mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(robot.model.ncam)
        ]
        assert not cameras, f"the watching flat must have no indoor camera: {cameras}"
    finally:
        robot.close()


@pytest.mark.slow
def test_unanswered_calls_corroborate_a_bad_day():
    """The doorphone earns its place by being a better sensor, not only a safer one.

    Three unanswered callers on a day she did not get up is evidence a motion sensor alone
    cannot give -- and it is obtained without a lens pointed at her.
    """
    robot, skills = flat()
    try:
        skills.advance(19 * 60)
        ordinary = skills.run("visitors").data["visitors"]
        assert ordinary, "nobody called on an ordinary day"
        assert all(c["answered"] for c in ordinary), ordinary
    finally:
        robot.close()

    robot, skills = flat(day=Day.did_not_get_up())
    try:
        skills.advance(19 * 60)
        bad = skills.run("visitors").data["visitors"]
        assert bad and not any(c["answered"] for c in bad), bad
    finally:
        robot.close()
