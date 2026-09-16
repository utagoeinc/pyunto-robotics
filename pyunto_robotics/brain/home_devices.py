"""The devices in a house, answering in a diary.

The simplest robot in this package, and the one most people could actually use. No simulator,
no camera, no legs -- a house full of things that can be asked and told. An air conditioner, a
door lock, a thermometer, the lights.

Why this belongs in a robotics SDK: it is the clearest demonstration that `RobotSkills` is the
whole contract. A robot here is a Python class with a `run()` method; whether behind that
method there is a MuJoCo humanoid or a lock on a front door makes no difference to Pyunto. If
someone wants to attach their own house, this file is the forty lines they change.

What makes it a diary feature rather than a smart-home app: the house writes back. Asked to
lock up it says whether it was already locked; asked about the temperature it gives a number
and says whether that is warm for the time of year. Those replies land in the same diary as
everything else, so "I locked up at 11" sits in the record next to the rest of the day.

Everything here is a simulated house. `HomeDevices` holds the state and `run()` changes it,
which is exactly the shape a real integration takes -- swap the bodies of these methods for
calls into HomeKit, Matter, ECHONET Lite or a vendor API and nothing else has to change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from .result import SkillResult

log = logging.getLogger(__name__)

# What a house is allowed to be set to. Not because the hardware could not do more, but
# because a diary entry is a blunt instrument: a typo that reads as 32 degrees should not cook
# anyone, and one that reads as 5 should not freeze the pipes.
MIN_TEMP_C = 16.0
MAX_TEMP_C = 30.0

# Where the thermostat goes when told "warm it up" or "cool it down" without a number.
COMFORT_C = 22.0
STEP_C = 2.0


@dataclass
class DoorCall:
    """Somebody at the front door, and whether it was answered.

    The doorphone is the one camera in this house, and it points OUTWARD at the porch. That
    distinction is the whole privacy argument: a camera watching a person in their own living
    room is surveillance of them, while one watching who rings their bell is a record of
    visitors -- the same thing the door's peephole has always been, kept for later.
    """

    minute: float
    who: str
    answered: bool
    # Whether a still from the porch camera is available for this call.
    has_image: bool = True


@dataclass
class Room:
    """One room, and the things in it."""

    name: str
    temperature_c: float
    humidity_pct: float = 52.0
    lights_on: bool = False
    aircon_on: bool = False
    aircon_target_c: float = COMFORT_C


@dataclass
class HomeDevices:
    """A simulated house: rooms, an air conditioner, lights, and a front door lock.

    State lives here rather than in the skills so that a real integration can replace this
    class wholesale -- reading from a hub instead of from memory -- without touching the part
    that turns a sentence into an action.
    """

    rooms: dict[str, Room] = field(default_factory=lambda: {
        "living room": Room("living room", 24.5),
        "bedroom": Room("bedroom", 22.0),
        "kitchen": Room("kitchen", 25.5),
    })
    locked: bool = False
    outside_c: float = 27.0
    outside_humidity_pct: float = 62.0
    # Who has rung the bell, most recent last. A real integration replaces this with the
    # doorphone's own log.
    door_calls: list[DoorCall] = field(default_factory=list)

    def room(self, name: str | None) -> Room:
        """The named room, or the living room when nobody said which.

        Defaulting rather than asking: "turn the aircon on" in a house almost always means the
        room the person is in, and a robot that replies "which room?" to that is tiresome.
        The reply always names the room it acted on, so a wrong guess is visible and correctable.
        """
        if name:
            key = name.strip().lower()
            if key in self.rooms:
                return self.rooms[key]
        return self.rooms["living room"]


class HomeSkills:
    """Ask the house things, and tell it to do things."""

    actions = (
        "temperature", "humidity", "set_temperature", "warmer", "cooler",
        "aircon_on", "aircon_off", "lights_on", "lights_off",
        "lock", "unlock", "lock_status", "visitors", "doorphone", "status", "report",
    )

    def __init__(self, devices: HomeDevices | None = None):
        self.devices = devices or HomeDevices()

    # -- the thermometer ------------------------------------------------------------

    def temperature(self, room_name: str | None = None) -> SkillResult:
        room = self.devices.room(room_name)
        outside = self.devices.outside_c
        # Say what the number means, not just what it is. "24.5" is data; "warmer than
        # outside" is the thing a person was actually asking about.
        if room.temperature_c > outside + 1.0:
            note = f"That is warmer than outside ({outside:.0f}°C)."
        elif room.temperature_c < outside - 1.0:
            note = f"That is cooler than outside ({outside:.0f}°C)."
        else:
            note = f"About the same as outside ({outside:.0f}°C)."
        return SkillResult(
            True,
            f"The {room.name} is {room.temperature_c:.1f}°C. {note}",
            {"room": room.name, "temperature_c": round(room.temperature_c, 1),
             "outside_c": round(outside, 1)},
        )

    # -- the air conditioner --------------------------------------------------------

    def set_temperature(self, argument: str | None, room_name: str | None = None) -> SkillResult:
        """Set the target. Refuses values outside what a house should be asked for."""
        wanted = _number_in(argument)
        if wanted is None:
            return SkillResult(
                False,
                "I did not catch a temperature. Try \"set the aircon to 24 degrees\".",
                {},
            )
        if not MIN_TEMP_C <= wanted <= MAX_TEMP_C:
            # Refuse rather than clamp silently. A person who typed 2 meant something, and
            # setting 16 without saying so would hide a mistake worth noticing.
            return SkillResult(
                False,
                f"{wanted:.0f}°C is outside what I will set — I keep it between "
                f"{MIN_TEMP_C:.0f} and {MAX_TEMP_C:.0f}°C.",
                {"requested_c": wanted},
            )
        room = self.devices.room(room_name)
        room.aircon_target_c = wanted
        room.aircon_on = True
        return SkillResult(
            True,
            f"The {room.name} air conditioner is set to {wanted:.0f}°C.",
            {"room": room.name, "target_c": wanted, "aircon_on": True},
        )

    def warmer(self, room_name: str | None = None) -> SkillResult:
        return self._nudge(room_name, +STEP_C, "warmer")

    def cooler(self, room_name: str | None = None) -> SkillResult:
        return self._nudge(room_name, -STEP_C, "cooler")

    def _nudge(self, room_name: str | None, delta: float, word: str) -> SkillResult:
        room = self.devices.room(room_name)
        target = min(max(room.aircon_target_c + delta, MIN_TEMP_C), MAX_TEMP_C)
        if target == room.aircon_target_c:
            return SkillResult(
                True,
                f"The {room.name} is already at {target:.0f}°C, which is as {word} as I go.",
                {"room": room.name, "target_c": target},
            )
        room.aircon_target_c = target
        room.aircon_on = True
        return SkillResult(
            True,
            f"I made the {room.name} {word} — now set to {target:.0f}°C.",
            {"room": room.name, "target_c": target, "aircon_on": True},
        )

    def aircon_on(self, room_name: str | None = None) -> SkillResult:
        room = self.devices.room(room_name)
        if room.aircon_on:
            return SkillResult(
                True,
                f"The {room.name} air conditioner is already on, set to "
                f"{room.aircon_target_c:.0f}°C.",
                {"room": room.name, "aircon_on": True},
            )
        room.aircon_on = True
        return SkillResult(
            True,
            f"I turned the {room.name} air conditioner on, set to "
            f"{room.aircon_target_c:.0f}°C.",
            {"room": room.name, "aircon_on": True, "target_c": room.aircon_target_c},
        )

    def aircon_off(self, room_name: str | None = None) -> SkillResult:
        room = self.devices.room(room_name)
        if not room.aircon_on:
            return SkillResult(
                True, f"The {room.name} air conditioner is already off.",
                {"room": room.name, "aircon_on": False},
            )
        room.aircon_on = False
        return SkillResult(
            True, f"I turned the {room.name} air conditioner off.",
            {"room": room.name, "aircon_on": False},
        )

    # -- the lights -----------------------------------------------------------------

    def lights_on(self, room_name: str | None = None) -> SkillResult:
        room = self.devices.room(room_name)
        room.lights_on = True
        return SkillResult(
            True, f"The {room.name} lights are on.", {"room": room.name, "lights_on": True}
        )

    def lights_off(self, room_name: str | None = None) -> SkillResult:
        room = self.devices.room(room_name)
        room.lights_on = False
        return SkillResult(
            True, f"The {room.name} lights are off.", {"room": room.name, "lights_on": False}
        )

    # -- the lock -------------------------------------------------------------------

    def lock(self) -> SkillResult:
        """Lock the front door.

        Says whether it was already locked rather than just "done". Someone who writes "did I
        lock up?" from a train is asking exactly that, and "locked" alone does not answer it.
        """
        if self.devices.locked:
            return SkillResult(
                True, "The front door was already locked.",
                {"locked": True, "changed": False},
            )
        self.devices.locked = True
        return SkillResult(
            True, "I locked the front door.", {"locked": True, "changed": True}
        )

    def unlock(self) -> SkillResult:
        if not self.devices.locked:
            return SkillResult(
                True, "The front door was already unlocked.",
                {"locked": False, "changed": False},
            )
        self.devices.locked = False
        return SkillResult(
            True, "I unlocked the front door.", {"locked": False, "changed": True}
        )

    def lock_status(self) -> SkillResult:
        state = "locked" if self.devices.locked else "unlocked"
        return SkillResult(
            True, f"The front door is {state}.", {"locked": self.devices.locked}
        )

    # -- everything at once ---------------------------------------------------------

    # -- humidity ------------------------------------------------------------------

    def humidity(self, room_name: str | None = None) -> SkillResult:
        """How damp a room is, and whether that is worth doing anything about.

        Worth asking after in an older person's flat for two ordinary reasons rather than any
        dramatic one: dry air in winter and mould in summer. Like the thermometer, this says
        what the number means, because "52%" is data and "a little dry" is the answer.
        """
        room = self.devices.room(room_name)
        rh = room.humidity_pct
        if rh >= 70.0:
            note = "That is damp enough for mould to take hold."
        elif rh >= 60.0:
            note = "A little humid."
        elif rh >= 40.0:
            note = "Comfortable."
        elif rh >= 30.0:
            note = "A little dry — worth a humidifier if her throat is sore."
        else:
            note = "Very dry."
        return SkillResult(
            True,
            f"The {room.name} is at {rh:.0f}% humidity. {note}",
            {"room": room.name, "humidity_pct": round(rh, 1),
             "outside_humidity_pct": round(self.devices.outside_humidity_pct, 1)},
        )

    # -- the front door, from outside ------------------------------------------------

    def visitors(self, _argument: str | None = None) -> SkillResult:
        """Who has been to the door, and whether she answered.

        This is the part of watching that does not require watching HER. That somebody called
        at 14:20 and she did not answer is worth a family member knowing, and it is arrived at
        without a single camera pointed inside the flat.
        """
        calls = self.devices.door_calls
        if not calls:
            return SkillResult(True, "Nobody has been to the door today.",
                               {"visitors": []})
        lines = []
        for call in calls:
            when = f"{int(call.minute) // 60 % 24:02d}:{int(call.minute) % 60:02d}"
            lines.append(
                f"  {when}  {call.who} — "
                + ("she answered" if call.answered else "no answer")
            )
        unanswered = sum(1 for c in calls if not c.answered)
        head = f"{len(calls)} caller(s) at the door today"
        if unanswered:
            head += f", {unanswered} unanswered"
        return SkillResult(
            True,
            head + ":\n" + "\n".join(lines),
            {"visitors": [
                {"minute": int(c.minute), "who": c.who, "answered": c.answered,
                 "has_image": c.has_image}
                for c in calls
            ]},
        )

    def doorphone(self, _argument: str | None = None) -> SkillResult:
        """The porch camera's view of the most recent caller.

        The only camera in this house, and it faces the street. Asking for "the camera" in a
        flat where someone lives alone should get you the doorstep, not her sitting room --
        see `DoorCall`. A real integration returns the doorphone's own still here.
        """
        calls = [c for c in self.devices.door_calls if c.has_image]
        if not calls:
            return SkillResult(True, "No doorphone picture — nobody has rung the bell today.",
                               {"image": None})
        last = calls[-1]
        when = f"{int(last.minute) // 60 % 24:02d}:{int(last.minute) % 60:02d}"
        return SkillResult(
            True,
            f"📷 The doorphone at {when}: {last.who} — "
            + ("she answered." if last.answered else "no answer."),
            {"image": "doorphone", "minute": int(last.minute), "who": last.who,
             "answered": last.answered},
        )

    def status(self) -> SkillResult:
        """The whole house in one reply, for "is everything alright?"."""
        lines = [f"The front door is {'locked' if self.devices.locked else 'unlocked'}."]
        calls = self.devices.door_calls
        if calls:
            unanswered = sum(1 for c in calls if not c.answered)
            lines.append(
                f"{len(calls)} caller(s) at the door today"
                + (f", {unanswered} unanswered." if unanswered else ".")
            )
        for room in self.devices.rooms.values():
            bits = [f"{room.temperature_c:.1f}°C", f"{room.humidity_pct:.0f}%RH"]
            if room.aircon_on:
                bits.append(f"aircon on at {room.aircon_target_c:.0f}°C")
            if room.lights_on:
                bits.append("lights on")
            lines.append(f"  {room.name}: {', '.join(bits)}")
        return SkillResult(
            True,
            "\n".join(lines),
            {
                "locked": self.devices.locked,
                "rooms": {
                    r.name: {
                        "temperature_c": round(r.temperature_c, 1),
                        "humidity_pct": round(r.humidity_pct, 1),
                        "aircon_on": r.aircon_on,
                        "lights_on": r.lights_on,
                    }
                    for r in self.devices.rooms.values()
                },
            },
        )

    # -- dispatch ------------------------------------------------------------------

    def run(
        self,
        action: str,
        argument: str | None = None,
        where: str | None = None,
        expect: int | None = None,
    ) -> SkillResult:
        # `where` carries the room, because that is what it means everywhere else in this
        # package: which one of several identical things the instruction meant.
        room = where or _room_in(argument)
        handlers = {
            "temperature": lambda: self.temperature(room),
            "humidity": lambda: self.humidity(room),
            "set_temperature": lambda: self.set_temperature(argument, room),
            "warmer": lambda: self.warmer(room),
            "cooler": lambda: self.cooler(room),
            "aircon_on": lambda: self.aircon_on(room),
            "aircon_off": lambda: self.aircon_off(room),
            "lights_on": lambda: self.lights_on(room),
            "lights_off": lambda: self.lights_off(room),
            "lock": lambda: self.lock(),
            "unlock": lambda: self.unlock(),
            "lock_status": lambda: self.lock_status(),
            "visitors": lambda: self.visitors(argument),
            "doorphone": lambda: self.doorphone(argument),
            "status": lambda: self.status(),
            "report": lambda: SkillResult(True, argument or "Done."),
        }
        handler = handlers.get(action)
        if handler is None:
            return SkillResult(False, f"I do not know how to '{action}'.")
        log.info("skill: %s(%s)", action, argument or "")
        return handler()


def _number_in(text: str | None) -> float | None:
    """The first number in a phrase, or None. Handles "24 degrees" as well as "24"."""
    if not text:
        return None
    digits = ""
    for char in str(text):
        if char.isdigit() or (char == "." and digits):
            digits += char
        elif digits:
            break
    try:
        return float(digits) if digits else None
    except ValueError:
        return None


def _room_in(text: str | None) -> str | None:
    """A room named inside the instruction itself, for when the planner did not split it out."""
    if not text:
        return None
    lowered = str(text).lower()
    for english, japanese in (
        ("living room", "lounge"),
        ("bedroom", "bed room"),
        ("kitchen", "galley"),
    ):
        if english in lowered or japanese in str(text):
            return english
    return None
