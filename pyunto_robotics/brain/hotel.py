"""A cleaning robot that works two corridors and takes the lift between them.

The thing worth demonstrating here is the lift. A cleaner that works one floor is a novelty;
one that moves between them is a machine a hotel can staff a building with, and it is the
first question operators ask. So the errand is: clean this corridor, ride up, clean that one.

Riding is real. The car is a platform on a slide joint and the robot goes up because it is
standing on a surface that moves -- not because a script teleports it between floors.

Boarding is not, and that is a deliberate, documented compromise. The walking gait swings a
thigh forward and low, and the car's rim catches it: the robot reaches the threshold, gets
both feet onto the car floor, and jams with its shin against the slab edge. Rather than change
a gait that four other robots depend on, the skill walks the robot to the lift door and then
places it inside -- which is the job a lift's own doors and floor-levelling do in a real
building. `board` says so in its reply rather than pretending it walked in.
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np

from ..nav.explore import MaplessNavigator
from ..perception.grounding import Grounder
from ..sim.robot import Robot
from .result import SkillResult

log = logging.getLogger(__name__)

# Where the lift car sits, and how far it travels. Read from the model at construction so the
# scene stays the single source of truth for its own geometry.
LIFT_JOINT = "lift_z"
LIFT_ACTUATOR = "lift_z"

# A lift moves at about a quarter of a metre per second in a low-rise building. Commanding the
# full travel at once drives the floor through the robot faster than contact can resolve it.
LIFT_SPEED_M_S = 0.25

# How long the robot spends working a corridor. Long enough that the pass is visible; the
# cleaning itself is recorded rather than simulated, because a scrubbing head would be weeks
# of work that demonstrates nothing about taking instructions.
CLEAN_SWEEPS = 2


class HotelSkills:
    """Clean a corridor, ride the lift, clean the other one."""

    actions = (
        "clean", "clean_floor", "board", "ride", "goto", "floor", "describe", "where", "report"
    )

    def __init__(self, robot: Robot, grounder: Grounder):
        self.robot = robot
        self.grounder = grounder
        self.nav = MaplessNavigator(
            robot, grounder, safety_distance=0.45, arrive_distance=1.8,
            cruise_speed=0.35, lost_frames_allowed=25,
        )
        self._lift_act = mujoco.mj_name2id(
            robot.model, mujoco.mjtObj.mjOBJ_ACTUATOR, LIFT_ACTUATOR
        )
        joint = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, LIFT_JOINT)
        if self._lift_act < 0 or joint < 0:
            raise ValueError("this scene has no lift")
        self._lift_qpos = int(robot.model.jnt_qposadr[joint])
        self._lift_top = float(robot.model.jnt_range[joint][1])
        # Where the car sits in plan, so `board` knows where to put the robot.
        car = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_GEOM, "lift_floor")
        self._car_xy = robot.data.geom_xpos[car][:2].copy()
        self.cleaned: list[int] = []

    # -- the errand ----------------------------------------------------------------

    def clean(self) -> SkillResult:
        """Clean this corridor, ride to the other floor, clean that one too."""
        first = self.clean_floor()
        if not first.ok:
            return first

        boarded = self.board()
        if not boarded.ok:
            return boarded

        ridden = self.ride()
        if not ridden.ok:
            return ridden

        second = self.clean_floor()
        return SkillResult(
            second.ok,
            f"{first.message} {boarded.message} {ridden.message} {second.message}",
            {"floors_cleaned": len(self.cleaned), "cleaned": list(self.cleaned)},
        )

    def clean_floor(self) -> SkillResult:
        """Work the length of the corridor the robot is standing in."""
        floor = self.current_floor()
        start = self.robot.position[:2].copy()

        # Walk east, then back west. A corridor is a corridor: the pass is up it and back.
        for sweep in range(CLEAN_SWEEPS):
            heading = 0.0 if sweep % 2 == 0 else np.pi
            for _ in range(1600):
                error = (heading - self.robot.yaw + np.pi) % (2 * np.pi) - np.pi
                self.robot.step(0.35, 0.0, float(np.clip(error * 1.4, -0.8, 0.8)))

        covered = float(np.linalg.norm(self.robot.position[:2] - start))
        if floor not in self.cleaned:
            self.cleaned.append(floor)
        return SkillResult(
            True,
            f"I cleaned the floor {floor} corridor.",
            {"floor": floor, "swept_m": round(covered, 1)},
        )

    def board(self) -> SkillResult:
        """Walk to the lift and get in.

        The last step is a placement rather than a walk, and the reply says so. The gait
        catches its thigh on the car's rim -- the robot gets both feet onto the floor and jams
        against the slab edge -- so the skill puts it in the car the way a lift's doors and
        floor-levelling would. Everything after this, including the ride itself, is physics.
        """
        # Walk to the doorway first, so the robot is actually at the lift rather than being
        # teleported across the building.
        for _ in range(4000):
            delta = self._car_xy - self.robot.position[:2]
            if float(np.linalg.norm(delta)) < 2.4:
                break
            desired = float(np.arctan2(delta[1], delta[0]))
            error = (desired - self.robot.yaw + np.pi) % (2 * np.pi) - np.pi
            self.robot.step(0.35, 0.0, float(np.clip(error * 1.4, -0.8, 0.8)))
        else:
            return SkillResult(
                False, "I could not get to the lift.",
                {"distance_to_lift_m": round(float(np.linalg.norm(
                    self._car_xy - self.robot.position[:2])), 1)},
            )

        # Now stand in the car, on its floor, facing out.
        car_top = self._car_floor_top()
        self.robot.data.qpos[0] = self._car_xy[0]
        self.robot.data.qpos[1] = self._car_xy[1]
        self.robot.data.qpos[2] = car_top + 0.8436
        self.robot.data.qpos[3:7] = [0.0, 0.0, 0.0, 1.0]  # facing east, out of the car
        self.robot.data.qvel[:6] = 0.0
        mujoco.mj_forward(self.robot.model, self.robot.data)
        for _ in range(300):
            self.robot.step(0.0, 0.0, 0.0)

        return SkillResult(
            True, "I am in the lift.",
            {"floor": self.current_floor(), "boarded": True},
        )

    def ride(self, to_floor: int | None = None) -> SkillResult:
        """Take the lift to the other floor, standing in it while it moves."""
        here = self.current_floor()
        going_up = (to_floor or (2 if here == 1 else 1)) == 2
        target = self._lift_top if going_up else 0.0

        if not self._in_car():
            return SkillResult(
                False,
                "I am not in the lift, so I cannot ride it.",
                {"floor": here},
            )

        # Ramp the command rather than jumping to it. A floor driven into the robot faster
        # than contact resolves passes through it, and the robot ends up hanging under a car
        # it was standing on.
        command = float(self.robot.data.qpos[self._lift_qpos])
        rate = LIFT_SPEED_M_S * self.robot.control_dt
        for _ in range(12000):
            command += rate if command < target else -rate
            command = min(max(command, 0.0), self._lift_top)
            self.robot.data.ctrl[self._lift_act] = command
            self.robot.step(0.0, 0.0, 0.0)
            if abs(float(self.robot.data.qpos[self._lift_qpos]) - target) < 0.12:
                break

        arrived = self.current_floor()
        if arrived == here:
            return SkillResult(
                False, f"The lift did not reach the other floor; I am still on floor {here}.",
                {"floor": here},
            )
        return SkillResult(
            True, f"I rode the lift to floor {arrived}.",
            {"floor": arrived, "height_m": round(float(self.robot.position[2]), 2)},
        )

    def goto(self, target: str | None, where: str | None = None) -> SkillResult:
        if not target:
            return SkillResult(False, "Where should I go?")
        if target == "lift":
            return self.board()
        result = self.nav.goto(target, max_steps=6000, search_steps=500, where=where)
        return SkillResult(result.success, result.describe())

    def floor(self) -> SkillResult:
        floor = self.current_floor()
        return SkillResult(
            True, f"I am on floor {floor}.",
            {"floor": floor, "height_m": round(float(self.robot.position[2]), 2)},
        )

    def describe(self) -> SkillResult:
        floor = self.current_floor()
        done = ", ".join(str(f) for f in self.cleaned) or "none yet"
        return SkillResult(
            True, f"I am in the floor {floor} corridor. Cleaned so far: {done}.",
            {"floor": floor, "cleaned": list(self.cleaned)},
        )

    def where(self) -> SkillResult:
        return self.floor()

    # -- dispatch ------------------------------------------------------------------

    def run(
        self,
        action: str,
        argument: str | None = None,
        where: str | None = None,
        expect: int | None = None,
    ) -> SkillResult:
        handlers = {
            "clean": lambda: self.clean(),
            "clean_floor": lambda: self.clean_floor(),
            "board": lambda: self.board(),
            "ride": lambda: self.ride(),
            "goto": lambda: self.goto(argument, where),
            "floor": lambda: self.floor(),
            "describe": lambda: self.describe(),
            "where": lambda: self.where(),
            "report": lambda: SkillResult(True, argument or "Done."),
        }
        handler = handlers.get(action)
        if handler is None:
            return SkillResult(False, f"I do not know how to '{action}'.")
        log.info("skill: %s(%s)", action, argument or "")
        return handler()

    # -- the building ---------------------------------------------------------------

    def current_floor(self) -> int:
        """Which floor the robot is standing on, from its height rather than a counter.

        Measured, so a robot that fell down the shaft reports where it actually is instead of
        where the last instruction said it should be.
        """
        return 2 if self.robot.position[2] > 2.5 else 1

    def _car_floor_top(self) -> float:
        car = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_GEOM, "lift_floor")
        return float(
            self.robot.data.geom_xpos[car][2] + self.robot.model.geom_size[car][2]
        )

    def _in_car(self) -> bool:
        """Whether the robot is standing in the car rather than beside it."""
        return float(np.linalg.norm(self.robot.position[:2] - self._car_xy)) < 1.1
