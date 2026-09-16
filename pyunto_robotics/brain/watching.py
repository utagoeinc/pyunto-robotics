"""A flat that watches an older person living alone, and writes what it sees in a diary.

Nothing here is commanded. The other robots in this package do what they are told; this one
is asked questions and, more importantly, speaks up on its own. That is the shape the product
actually takes: a daughter in another city does not want to interrogate a sensor hub, she
wants to open a diary and see that her mother got up, ate, and went to bed.

So the reply is written for her, not for an operator. "She has been in the bathroom for
25 minutes" rather than "pir_bathroom active 1500s", and the quiet reassurance -- "up at 7:10,
breakfast, and she is watching television now" -- matters as much as the alarm, because a
watching system that only ever speaks when something is wrong is a system nobody trusts is
working.

The devices from the old `house` robot live here too: an older person's flat has an air
conditioner and lights and a front door, and the reason to connect them is that watching and
acting belong together. Noticing the room is 29°C is worth something; turning the air
conditioning on is worth more.

No cameras inside the flat. Not an omission -- a constraint, and the one that decides whether
this is a product somebody would put in their mother's home. The person being watched did not
ask for any of this; the daughter did. So the watching is built from things that describe a
home rather than record a person: floor-level motion sensors in each room, a bed sensor,
temperature and humidity, the door lock, and the doorphone.

The doorphone is the only camera, and it faces the street. That single fact carries the
argument: a camera in the living room is surveillance of her, while one on the porch is a
record of visitors -- the same thing a peephole has always been, kept for later. And it turns
out to be the better sensor anyway. Three unanswered calls on a day she did not get up is
corroboration a motion sensor alone cannot give, arrived at without watching her at all.

Anyone extending this should keep that line. If a request seems to need indoor footage, the
answer is another sensor, not a lens.
"""

from __future__ import annotations

import logging
import math
import time

import mujoco

from ..sim.household import (
    BATHROOM_CONCERN_MINUTES,
    MINUTES_PER_DAY,
    STILL_IN_BED_BY_MINUTE,
    PLACES,
    STILL_CONCERN_MINUTES,
    Day,
    HouseholdSensors,
)
from .home_devices import DoorCall, HomeDevices, HomeSkills
from .result import SkillResult

log = logging.getLogger(__name__)

# How many simulated minutes pass per second of watching. A day has to fit in a demonstration,
# and the sensors are read every simulated minute either way.
MINUTES_PER_TICK = 1.0

# Simulation steps per simulated minute. Enough for the person to actually walk between rooms
# under their position actuators rather than teleporting.
STEPS_PER_MINUTE = 60

# How the day is paced when nobody is asking anything. A demonstration has to show a whole
# day in the minutes somebody will actually watch for, and the interesting events -- a long
# bathroom visit, a morning she does not get up -- are hours apart.
MINUTES_PER_SECOND = 120.0

# Don't step on every redraw; the viewer runs far faster than the day needs to move.
TICKS_PER_SECOND = 10.0

# Who calls at the door, as (minute, who, answered-if-she-is-up).
#
# The doorphone faces the street, so this is the one camera in the flat and it watches
# visitors rather than the person -- which is the whole reason the watching can be thorough
# without being surveillance. Whether she answers is not scripted: it depends on where she
# actually is when the bell goes, which is what makes "14:20, no answer" worth reading.
DOOR_CALLS = (
    (615, "a parcel delivery"),   # 10:15
    (860, "the postman"),         # 14:20
    (1125, "the neighbour"),      # 18:45
)


def _clock(minute: float) -> str:
    return f"{int(minute) // 60 % 24:02d}:{int(minute) % 60:02d}"


class WatchingSkills:
    """Watch the flat, answer questions about it, and work its devices."""

    actions = (
        "check", "today", "watch", "time", "temperature", "set_temperature", "warmer", "cooler",
        "aircon_on", "aircon_off", "lights_on", "lights_off",
        "lock", "unlock", "lock_status", "status", "report",
    )

    def __init__(self, robot, grounder=None, day: Day | None = None):  # noqa: ANN001
        # `robot` here is a plain MuJoCo holder rather than a mobile machine: this scene has
        # no robot in it at all, which is rather the point.
        self.sim = robot
        self.sensors = HouseholdSensors(robot.model, robot.data)
        self.day = day or Day.ordinary()
        self.minute = 0.0
        self.events: list[tuple[float, str]] = []
        self._seen_rooms: set[str] = set()
        self._warned_bathroom = False
        self._warned_still = False
        # The flat's own devices, so "it is 29 degrees in there" can become "I turned the air
        # conditioning on".
        self.devices = HomeSkills(HomeDevices())
        # How fast the day is running, shown as pips and measured rather than declared.
        #
        # A fixed number would be a decoration. The rate depends on the machine and on
        # whether a viewer window is redrawing, so it is timed from the simulation itself and
        # the pips follow: 1 is near real time, 5 is a day in seconds.
        self.speed_pips = 3
        self._elapsed_wall = 0.0
        self._elapsed_sim = 0.0
        self._last_tick: float | None = None
        self._day_index = 0
        self.sensors.show_time(self.minute, speed_pips=self.speed_pips)

    # -- the day runs whether or not anyone is asking ------------------------------

    def tick(self) -> list[str]:
        """Advance the day a little. Safe to call as often as the viewer redraws.

        Without this the flat is frozen between messages: the clock holds one time and she
        never leaves the bed, so a viewer watching the window sees a photograph. The point of
        the scene is that it keeps living while nobody is asking it anything -- that is what
        makes "she has been in the bathroom 20 minutes" mean something when it arrives.

        Paced against the wall clock rather than the call count, because the idle hook fires
        at whatever rate the viewer happens to redraw at.
        """
        now = time.monotonic()
        if self._last_tick is None:
            self._last_tick = now
            return []
        elapsed = now - self._last_tick
        if elapsed < 1.0 / TICKS_PER_SECOND:
            return []
        self._last_tick = now
        return self.advance(elapsed * MINUTES_PER_SECOND)

    # -- watching -------------------------------------------------------------------

    def advance(self, minutes: float) -> list[str]:
        """Run the day forward, moving the person and reading the sensors.

        Returns anything worth saying. Called by `watch`, and by the demo's idle hook so the
        flat keeps living while nobody is asking it anything.
        """
        said: list[str] = []
        started = time.monotonic()
        for _ in range(int(minutes / MINUTES_PER_TICK)):
            self.minute += MINUTES_PER_TICK
            place = self.day.place_at(self.minute)
            target = PLACES.get(place, PLACES["bed"])
            self.sim.data.ctrl[:4] = target
            for _ in range(STEPS_PER_MINUTE):
                # Step the physics directly rather than through Robot.step, which drives a
                # gait: there is no robot in this scene, and the humanoid gait it defaults to
                # reads a free joint that does not exist here.
                mujoco.mj_step(self.sim.model, self.sim.data)

            self._roll_over_at_midnight()
            reading = self.sensors.read(self.minute)
            self._ring_the_doorbell(reading)
            for room in ("bedroom", "bathroom", "hallway", "living", "kitchen"):
                self.sensors.show_sensor(room, room == reading.room)
            # Her clock, on the wall, so a viewer can see the day moving.
            self.sensors.show_time(self.minute, speed_pips=self.speed_pips)

            note = self._notice(reading)
            if note:
                said.append(note)

        # Time how fast the day actually ran, and set the pips from it. Averaged over the
        # whole call rather than per minute, because a single minute is too short to time.
        self._elapsed_wall += time.monotonic() - started
        self._elapsed_sim += minutes * 60.0
        # 5 ms, not 50. Two simulated hours can pass in 24 ms here, and at the higher
        # threshold the pips never updated at all on a short call -- they sat at their
        # default, which is exactly the decoration this was meant to replace.
        if self._elapsed_wall > 0.005:
            ratio = self._elapsed_sim / self._elapsed_wall
            # 1 pip: real time. 5 pips: a day in a few seconds. A log scale, because the
            # range this spans is four orders of magnitude and a linear bar would sit at 5.
            self.speed_pips = max(1, min(5, int(math.log10(max(ratio, 1.0)) + 1)))
        return said

    def _ring_the_doorbell(self, reading) -> None:  # noqa: ANN001
        """Let the day's callers arrive, and record whether she got to the door.

        Answered is decided by where she is, not by a script: if she is in bed or the bathroom
        when the bell goes, nobody comes. That is the point of keeping the log -- a family
        member seeing "14:20 the postman — no answer" learns something real, and learns it
        without anyone watching her.
        """
        minute_of_day = self.minute % MINUTES_PER_DAY
        for when, who in DOOR_CALLS:
            if not (when <= minute_of_day < when + MINUTES_PER_TICK):
                continue
            if any(int(c.minute) == int(minute_of_day) for c in self.devices.devices.door_calls):
                continue
            # Reclining on the sofa is not "cannot reach the door" -- she gets up for the
            # bell like anyone else. What stops her answering is being asleep in bed or
            # occupied in the bathroom, which is also exactly when an unanswered call is
            # worth reading about.
            answered = reading.room not in ("bathroom", "bedroom")
            self.devices.devices.door_calls.append(
                DoorCall(minute=minute_of_day, who=who, answered=answered)
            )
            log.info("doorbell: %s at %s answered=%s", who, _clock(self.minute), answered)

    def _roll_over_at_midnight(self) -> None:
        """Start each day with a clean slate.

        Every "first of the day" here was really a first of the *run*: `_seen_rooms`,
        `_warned_bathroom`, `_warned_still` and the `got_up` event were set once and never
        cleared. So from day two the house went silent -- it had already said everything it
        knew how to say, and a watching system that stops watching after a day is worse than
        none, because the quiet reads as "nothing wrong".
        """
        day = int(self.minute // MINUTES_PER_DAY)
        if day == self._day_index:
            return
        self._day_index = day
        self._seen_rooms.clear()
        self._warned_bathroom = False
        self._warned_still = False
        # `events` is the diary `today` reads back, so it is emptied with the rest. Anything
        # worth keeping across days has already been said into the thread.
        self.events.clear()
        # The doorphone log is "today's callers", so it turns over with the day too.
        self.devices.devices.door_calls.clear()

    def _notice(self, reading) -> str | None:  # noqa: ANN001
        """Decide whether this reading is worth a word in the diary.

        Most are not. A watching system that narrates every room change is noise, and noise is
        how one gets muted. What earns a line is a first of the day -- she got up, she has had
        breakfast -- and anything that departs from the ordinary.
        """
        # Only count a room she has actually stopped in. Walking through the living room on
        # the way to the kitchen announced "she has settled in the living room" one minute
        # before "she is in the kitchen", which reads as confusion rather than as watching.
        if reading.room and reading.room not in self._seen_rooms:
            if reading.minutes_in_room < 3:
                return None
            self._seen_rooms.add(reading.room)
            first = {
                "bedroom": None,  # she starts there; not news
                "bathroom": None,  # nor is the first visit
                "hallway": None,
                "kitchen": f"🍵 {_clock(self.minute)} — she is in the kitchen.",
                "living": f"📺 {_clock(self.minute)} — she has settled in the living room.",
            }.get(reading.room)
            if first:
                self.events.append((self.minute, reading.room))
                return first

        # Up for the day: standing, out of the bedroom, for the first time.
        if not reading.lying and reading.room and reading.room != "bedroom":
            if not any(k == "got_up" for _, k in self.events):
                self.events.append((self.minute, "got_up"))
                return f"☀️ {_clock(self.minute)} — she is up."

        if (
            reading.room == "bathroom"
            and reading.minutes_in_room >= BATHROOM_CONCERN_MINUTES
            and not self._warned_bathroom
        ):
            self._warned_bathroom = True
            self.events.append((self.minute, "long_bathroom"))
            # Said plainly, with the number, and without diagnosing. The house does not know
            # whether she has fallen; it knows how long she has been in there, and that is
            # exactly what to pass on.
            return (
                f"⚠️ {_clock(self.minute)} — she has been in the bathroom for "
                f"{reading.minutes_in_room:.0f} minutes. That is longer than usual."
            )

        # Not while she is asleep in bed at night. Three hours without moving is alarming at
        # two in the afternoon and completely normal at three in the morning, and a system
        # that cries wolf every single night is one nobody reads by the end of the week.
        # 22:00 to 08:00. The first attempt ended the night at 06:00 and fired at exactly
        # 06:00 for someone who gets up at seven -- the alarm landed an hour before she was
        # due to move, which is the definition of a false one. The window has to cover when
        # people actually sleep, not when a clock says morning.
        night = (self.minute % 1440) < 480 or (self.minute % 1440) >= 1320
        asleep = reading.lying and reading.room == "bedroom"
        # Still in bed long after she should be up. The quietest emergency there is, and the
        # one plain stillness cannot catch: at eleven in the morning the timer has only just
        # started, because sleep does not count toward it.
        if (
            reading.lying
            and reading.room == "bedroom"
            and (self.minute % 1440) >= STILL_IN_BED_BY_MINUTE
            and not any(k == "got_up" for _, k in self.events)
            and not self._warned_still
        ):
            self._warned_still = True
            self.events.append((self.minute, "not_up"))
            return (
                f"⚠️ {_clock(self.minute)} — she is still in bed and has not been up today."
            )

        if (
            reading.minutes_still >= STILL_CONCERN_MINUTES
            and not self._warned_still
            and not (night and asleep)
        ):
            self._warned_still = True
            self.events.append((self.minute, "still"))
            where = reading.room or "the flat"
            return (
                f"⚠️ {_clock(self.minute)} — no movement in {where} for "
                f"{reading.minutes_still / 60:.0f} hours."
            )
        return None

    # -- answering ------------------------------------------------------------------

    def check(self) -> SkillResult:
        """"How is she?" -- the question this whole thing exists to answer."""
        reading = self.sensors.read(self.minute)
        if not reading.room:
            return SkillResult(
                True,
                f"{_clock(self.minute)} — I cannot see her in any room just now.",
                {"room": None, "minute": int(self.minute)},
            )

        where = {
            "bedroom": "in the bedroom", "bathroom": "in the bathroom",
            "hallway": "in the hallway", "living": "in the living room",
            "kitchen": "in the kitchen",
        }[reading.room]
        posture = "lying down" if reading.lying else "up and about"
        message = f"{_clock(self.minute)} — she is {where}, {posture}."

        if reading.minutes_in_room >= 60:
            message += f" She has been there {reading.minutes_in_room / 60:.0f} hours."
        elif reading.minutes_in_room >= 5:
            message += f" For the last {reading.minutes_in_room:.0f} minutes."

        return SkillResult(
            True, message,
            {"room": reading.room, "lying": reading.lying,
             "minutes_in_room": round(reading.minutes_in_room),
             "minute": int(self.minute)},
        )

    def today(self) -> SkillResult:
        """The day so far, as a family would want it: what happened and when."""
        if not self.events:
            return SkillResult(
                True, f"{_clock(self.minute)} — nothing to report yet today.",
                {"events": []},
            )
        lines = [f"{_clock(self.minute)} — today so far:"]
        described = {
            "got_up": "got up",
            "kitchen": "went to the kitchen",
            "living": "settled in the living room",
            "long_bathroom": "⚠️ a long bathroom visit",
            "still": "⚠️ a long spell without moving",
            "not_up": "⚠️ still in bed, not up",
        }
        for minute, kind in self.events:
            lines.append(f"  {_clock(minute)}  {described.get(kind, kind)}")
        return SkillResult(
            True, "\n".join(lines),
            {"events": [{"at": _clock(m), "kind": k} for m, k in self.events]},
        )

    def watch(self, argument: str | None = None) -> SkillResult:
        """Run the day forward and report anything worth saying.

        This is what the demonstration runs on: the flat lives, and speaks when there is
        something to say rather than when asked.
        """
        hours = 1.0
        if argument:
            digits = "".join(c for c in str(argument) if c.isdigit() or c == ".")
            if digits:
                hours = min(float(digits), 24.0)
        said = self.advance(hours * 60)
        if not said:
            reading = self.sensors.read(self.minute)
            return SkillResult(
                True,
                f"{_clock(self.minute)} — a quiet {hours:.0f} hours. "
                f"She is {'lying down' if reading.lying else 'up'} "
                f"{'in the ' + reading.room if reading.room else 'somewhere I cannot see'}.",
                {"minute": int(self.minute), "notes": 0},
            )
        return SkillResult(True, "\n".join(said), {"minute": int(self.minute),
                                                   "notes": len(said)})

    def time(self) -> SkillResult:
        """What time it is in the flat, and how fast the day is running.

        The wall clock shows this, but somebody reading the diary on a phone is not looking at
        the window -- and the first thing anybody asks of a scene that compresses a day is how
        far along it is. Saying the rate too, because "14:52" alone invites the reasonable
        assumption that it is 14:52 where the reader is.
        """
        # Only quote a rate once one has actually been timed. Before that `speed_pips` is
        # its default, and stating it would be inventing a measurement.
        if self._elapsed_wall > 0.005:
            rate = {1: "about real time", 2: "tens of times real time",
                    3: "about 100x real time", 4: "about 1000x real time",
                    5: "a day in seconds"}.get(self.speed_pips, "")
            text = f"It is {_clock(self.minute)} in the flat, running at {rate}."
        else:
            text = f"It is {_clock(self.minute)} in the flat. It has only just started."
        log.info("skill: time -> %s", _clock(self.minute))
        return SkillResult(True, text, {"minute": int(self.minute),
                                        "clock": _clock(self.minute),
                                        "speed_pips": self.speed_pips})

    # -- dispatch -------------------------------------------------------------------

    def run(
        self,
        action: str,
        argument: str | None = None,
        where: str | None = None,
        expect: int | None = None,
    ) -> SkillResult:
        if action in ("check", "today", "watch", "time"):
            handler = {"check": lambda: self.check(),
                       "today": lambda: self.today(),
                       "time": lambda: self.time(),
                       "watch": lambda: self.watch(argument)}[action]
            log.info("skill: %s(%s)", action, argument or "")
            return handler()
        # Everything else is the flat's own devices, unchanged from the house robot.
        result = self.devices.run(action, argument, where, expect)
        if result.ok and action in ("lights_on", "lights_off"):
            # Mirror it in the viewer, so switching a light is visible rather than asserted.
            room = (where or "living").split()[0]
            lamp = {"living": "living_lamp", "bedroom": "bedside_lamp",
                    "kitchen": "kitchen_lamp"}.get(room, "living_lamp")
            self.sensors.light(lamp, action == "lights_on")
        return result
