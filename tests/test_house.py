"""The house: air conditioning, lights, a front door lock, thermometers.

No simulator, which is the point. These run in milliseconds and they pin the two things that
make this a diary feature rather than a smart-home app: the replies answer the question that
was actually asked, and a refusal is honest.
"""

from __future__ import annotations

import pytest

from pyunto_robotics.brain.domains import DOMAINS
from pyunto_robotics.brain.home_devices import (
    MAX_TEMP_C,
    MIN_TEMP_C,
    HomeDevices,
    HomeSkills,
)


def house() -> HomeSkills:
    return HomeSkills(HomeDevices())


def test_locking_says_whether_it_was_already_locked():
    """Someone writing "did I lock up?" from a train is asking exactly this.

    "Locked." alone does not answer it -- it is true whether the robot just turned the key or
    found it already turned, and those are different facts to the person asking.
    """
    home = house()
    first = home.lock()
    assert first.data["changed"] is True
    second = home.lock()
    assert second.data["changed"] is False
    assert "already" in second.message


def test_asking_whether_the_door_is_locked_does_not_lock_it():
    """A question and an instruction share the word "lock". Getting this backwards on a front
    door is the worse way round."""
    home = house()
    result = home.run("lock_status")
    assert result.ok
    assert home.devices.locked is False


def test_a_temperature_is_reported_with_what_it_means():
    """"24.5" is data. "Cooler than outside" is what was being asked about."""
    home = house()
    home.devices.outside_c = 30.0
    result = home.temperature("living room")
    assert "24.5" in result.message
    assert "cooler than outside" in result.message


@pytest.mark.parametrize("wanted", ["2", "45", "100"])
def test_absurd_temperatures_are_refused_rather_than_clamped(wanted):
    """A person who typed 2 meant something; silently setting 16 hides the mistake."""
    home = house()
    result = home.set_temperature(wanted)
    assert result.ok is False
    assert str(int(MIN_TEMP_C)) in result.message
    assert str(int(MAX_TEMP_C)) in result.message


def test_a_temperature_is_read_out_of_japanese_phrasing():
    home = house()
    result = home.set_temperature("24度")
    assert result.ok, result.message
    assert result.data["target_c"] == 24.0


def test_the_room_named_in_the_instruction_is_the_room_acted_on():
    home = house()
    home.run("lights_on", where="bedroom")
    assert home.devices.rooms["bedroom"].lights_on is True
    assert home.devices.rooms["living room"].lights_on is False


def test_an_unnamed_room_defaults_and_the_reply_says_which():
    """A wrong guess has to be visible, or the person cannot correct it."""
    home = house()
    result = home.run("lights_on")
    assert "living room" in result.message


def test_nudging_stops_at_the_limit_and_says_so():
    home = house()
    for _ in range(20):
        result = home.cooler()
    assert home.devices.rooms["living room"].aircon_target_c == MIN_TEMP_C
    assert "as cooler as I go" in result.message or "already" in result.message


def test_unknown_actions_are_refused_not_raised():
    """A traceback is not an answer to send a person."""
    result = house().run("make_coffee")
    assert result.ok is False
    assert "make_coffee" in result.message


@pytest.mark.parametrize(
    ("text", "action"),
    [
        ("鍵はかかってる？", "lock_status"),
        ("is the door locked?", "lock_status"),
        ("戸締まりして", "lock"),
        ("鍵を開けて", "unlock"),
        ("リビングのエアコンを24度にして", "set_temperature"),
        ("寝室の温度は？", "temperature"),
        ("暑い", "cooler"),
        ("寒い", "warmer"),
        ("電気を消して", "lights_off"),
        ("エアコンをつけて", "aircon_on"),
        ("家の様子は？", "status"),
    ],
)
def test_instructions_reach_the_right_action(text, action):
    assert DOMAINS["house"].verb(text) == action, text


@pytest.mark.parametrize(
    ("text", "room"),
    [("リビングのエアコン", "living room"), ("寝室の電気", "bedroom"), ("キッチンの温度", "kitchen")],
)
def test_rooms_are_recognised(text, room):
    assert DOMAINS["house"].object_in(text) == room


def test_the_whole_house_fits_in_one_reply():
    home = house()
    home.lock()
    home.run("aircon_on", where="bedroom")
    result = home.status()
    assert "locked" in result.message
    for room in ("living room", "bedroom", "kitchen"):
        assert room in result.message
