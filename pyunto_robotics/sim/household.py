"""The sensors in a flat, and the person they are watching.

This is the watching demonstration's physics half: a person who actually moves through actual
rooms, and sensors that read where they are rather than being told.

Why model the person at all, rather than a state machine that prints "in the kitchen": because
then the sensors would be reading the script instead of the world, and the detection logic
would be untestable. Here a motion sensor fires because a body is inside its volume, and "in
bed" is told from "standing beside the bed" by how low the body is. Change the routine and the
readings follow, which is the only way to find out whether the alerting works.

Swapping this for a real flat means replacing `HouseholdSensors.read()` with calls to whatever
hub the sensors report to. Nothing above it changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import mujoco
import numpy as np

log = logging.getLogger(__name__)

# The rooms, and the sensor volume that covers each.
ROOMS = ("bedroom", "bathroom", "hallway", "living", "kitchen")

# Below this pitch the person is lying down or sitting rather than standing. Read from the
# pitch joint rather than from height: a figure that tips flat is unmistakably in bed, where
# one that merely sinks is a person standing in a hole. -0.4 rad separates upright from
# reclining with room either side.
LYING_PITCH = -0.4

# Kept for the height test on the bed sensor, which still asks how low the body is.
LYING_Z = 0.90

# How long in the bathroom before it is worth remarking on. Twenty minutes is long for a visit
# and short enough that a fall would not go unnoticed for an hour. Real systems use something
# in this range for the same reason.
BATHROOM_CONCERN_MINUTES = 20.0

# How long with no movement at all before the house says so. An older person sits still for a
# long time quite normally -- a nap, a television programme -- so this is deliberately hours
# rather than minutes. Crying wolf is how a watching system gets switched off.
STILL_CONCERN_MINUTES = 300.0

# The other half of watching: not that she stopped, but that she never started.
#
# Someone sitting through a long afternoon of television is fine; someone still in bed at
# midday is not, and the second is the quieter emergency. Time of day is what separates them,
# which is why stillness alone cannot carry both.
STILL_IN_BED_BY_MINUTE = 660.0  # 11:00


# Which segments light for each digit, in the usual seven-segment lettering:
#      a
#    f   b
#      g
#    e   c
#      d
DIGIT_SEGMENTS = {
    0: "abcdef", 1: "bc", 2: "abdeg", 3: "abcdg", 4: "bcfg",
    5: "acdfg", 6: "acdefg", 7: "abc", 8: "abcdefg", 9: "abcdfg",
}


@dataclass
class Reading:
    """What the sensors say at one moment."""

    room: str
    """Which room the person is in, or "" if no sensor sees them."""

    lying: bool
    """Lying down or sitting, rather than standing."""

    minutes_in_room: float
    """How long they have been in this room."""

    minutes_still: float
    """How long since they last changed room."""

    position: tuple[float, float]


@dataclass
class DayEvent:
    """Something the house noticed, in the person's day."""

    minute: float
    kind: str
    """got_up, kitchen, bathroom, living, bed, still, long_bathroom."""

    detail: str = ""


class HouseholdSensors:
    """Reads a simulated flat: which room, standing or lying, and for how long.

    Attaches to a MuJoCo model containing a body named `person` and geoms named `pir_<room>`.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self._person = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "person")
        if self._person < 0:
            raise ValueError("this scene has no body named 'person'")
        self._volumes = {}
        for room in ROOMS:
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"pir_{room}")
            if geom >= 0:
                self._volumes[room] = geom
        # Where the person was when they last changed room, in simulated minutes.
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "person_pitch")
        self._pitch = int(model.jnt_qposadr[joint]) if joint >= 0 else None
        self._room = ""
        self._room_since = 0.0
        self._last_change = 0.0

    def read(self, minute: float) -> Reading:
        position = self.data.xpos[self._person].copy()
        room = ""
        for name, geom in self._volumes.items():
            centre = self.data.geom_xpos[geom]
            half = self.model.geom_size[geom]
            # Footprint only. The markers are flat pads on the floor so they do not obscure
            # the room from above, and testing their height would mean nobody is ever in
            # range -- a PIR sensor covers a room's floor area, not a 12 mm slab of air.
            if bool(np.all(np.abs(position[:2] - centre[:2]) <= half[:2])):
                room = name
                break

        pitch_now = float(self.data.qpos[self._pitch]) if self._pitch is not None else 0.0
        if room and room != self._room:
            self._room = room
            self._room_since = minute
            self._last_change = minute

        # Lying down in bed resets the stillness clock rather than accumulating on it.
        #
        # Sleep is not stillness worth reporting, and counting it means the timer is already
        # seven hours deep the moment someone wakes -- so the first reading after a normal
        # night fired the alarm at 07:00, an hour after she was safely asleep and a minute
        # before she got up. What the clock should measure is time motionless while awake.
        if room == "bedroom" and pitch_now < LYING_PITCH:
            self._last_change = minute

        pitch = float(self.data.qpos[self._pitch]) if self._pitch is not None else 0.0
        return Reading(
            room=room,
            lying=bool(pitch < LYING_PITCH),
            minutes_in_room=minute - self._room_since if self._room else 0.0,
            minutes_still=minute - self._last_change,
            position=(float(position[0]), float(position[1])),
        )

    def show_time(self, minute: float, speed_pips: int = 3) -> None:
        """Put the time of day on the wall clock, and how fast it is running.

        A demonstration that compresses a day into minutes has to say so. Without a clock a
        viewer cannot tell a quiet afternoon from a simulation that has stopped, and the first
        question anybody asks of this scene -- "how long is that in real time?" -- has no
        visible answer.

        Drawn with geometry rather than an overlay because this scene mostly lives in
        screenshots and recordings, and an overlay appears in neither.
        """
        hours, minutes = int(minute) // 60 % 24, int(minute) % 60
        for position, value in enumerate((hours // 10, hours % 10,
                                          minutes // 10, minutes % 10)):
            for segment in "abcdefg":
                self._set_material(
                    f"clock_d{position}_{segment}",
                    "seg_on" if segment in DIGIT_SEGMENTS[value] else "seg_off",
                )
        for pip in range(1, 6):
            self._set_material(f"speed_{pip}", "speed_on" if pip <= speed_pips else "speed_off")

    def _set_material(self, geom_name: str, material_name: str) -> None:
        geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        material = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, material_name)
        if geom >= 0 and material >= 0:
            self.model.geom_matid[geom] = material

    def light(self, name: str, on: bool) -> None:
        """Switch one of the flat's lamps, for the viewer."""
        material = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "lamp_on" if on else "lamp_off"
        )
        geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if material >= 0 and geom >= 0:
            self.model.geom_matid[geom] = material

    def show_sensor(self, room: str, firing: bool) -> None:
        """Light up a sensor volume in the viewer when it sees someone.

        A watching demonstration should show what is being watched. An invisible sensor is
        indistinguishable from no sensor, and the whole question a family has is "what does
        this thing actually see".
        """
        geom = self._volumes.get(room)
        if geom is None:
            return
        material = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "pir_fired" if firing else "pir_idle"
        )
        if material >= 0:
            self.model.geom_matid[geom] = material


# Where each part of the day happens, in world coordinates, and whether the person is down.
# x, y, height, pitch. Pitch is 0 standing and -1.5 flat, so the figure lies down on the bed
# and reclines on the sofa instead of sinking into them upright.
PLACES: dict[str, tuple[float, float, float, float]] = {
    # -0.09 puts her lowest part at 0.62, which is exactly the top of the mattress.
    #
    # It was -0.20, and that is the wrong DIRECTION: lying down was treated as lowering the
    # body, but a bed is above the floor. She sank into it -- torso at 0.58, legs at 0.52,
    # against a mattress surface of 0.62 -- and only her head showed, so she appeared to be a
    # person with no legs buried in the bedding. Measured, not guessed.
    "bed": (-1.6, 1.6, -0.09, -1.5),
    "bedside": (-1.6, 0.4, 0.0, 0.0),
    "bathroom": (2.4, 2.6, 0.0, 0.0),
    "kitchen": (7.8, 1.0, 0.0, 0.0),
    "sofa": (4.4, -1.5, -0.30, -1.0),
    "hallway": (1.2, -1.0, 0.0, 0.0),
}


@dataclass
class Day:
    """A day in the flat, as a list of (minute, place) the person moves between.

    A routine rather than random wandering, because the point of watching is noticing when a
    day departs from the usual one. `unusual` swaps in a day that a family would want to hear
    about -- a long bathroom visit, or not getting up at all.
    """

    schedule: list[tuple[float, str]] = field(default_factory=list)

    @classmethod
    def ordinary(cls) -> Day:
        """An unremarkable day: up at seven, meals, television, bed at ten."""
        return cls([
            (0, "bed"),        # midnight
            (420, "bedside"),  # 07:00 up
            (425, "bathroom"),
            (435, "kitchen"),  # breakfast
            (470, "sofa"),
            (720, "kitchen"),  # lunch
            (760, "sofa"),
            (900, "bathroom"),
            (910, "sofa"),
            (1080, "kitchen"),  # supper
            (1130, "sofa"),
            (1320, "bathroom"),  # 22:00
            (1330, "bed"),
        ])

    @classmethod
    def long_bathroom(cls) -> Day:
        """The same day, except a bathroom visit that does not end.

        This is the event these systems exist for, and it is why the bathroom timer is
        separate from the general stillness timer: someone on the bathroom floor is not
        "resting quietly".
        """
        return cls([
            (0, "bed"),
            (420, "bedside"),
            (425, "bathroom"),
            (435, "kitchen"),
            (470, "sofa"),
            (720, "kitchen"),
            (760, "sofa"),
            (900, "bathroom"),   # and stays there
        ])

    @classmethod
    def did_not_get_up(cls) -> Day:
        """Still in bed at midday. The quietest emergency there is."""
        return cls([(0, "bed")])

    def place_at(self, minute: float) -> str:
        """Where the person is meant to be at this minute."""
        place = self.schedule[0][1] if self.schedule else "bed"
        for when, where in self.schedule:
            if minute >= when:
                place = where
            else:
                break
        return place
