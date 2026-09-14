"""A legged robot that fetches fruit from the rows and carries it to the packing shed.

The errand is the point, not the walking. A grower writes "the crates by the far tree are
ready" in a diary and the robot goes, finds them, and brings them in -- which is a round trip
with two legs to it, and neither end is a coordinate the robot was given.

Why legs are worth the trouble here: orchard ground is soft and rutted between the rows, which
is where the fruit is. A wheeled machine either stays on a prepared track or sinks. That is
also why this scene's ground is uneven rather than flat -- the terrain is the justification
for the machine.

Carrying is modelled rather than simulated. The robot has no arms, and building a gripper that
can pick up a crate would be a month of work that teaches nothing about the thing being
demonstrated, which is instruction-to-errand. So `collect` records that the robot is carrying,
`deliver` records that it is not, and both report honestly what was actually done.
"""

from __future__ import annotations

import logging

import numpy as np

from ..nav.explore import MaplessNavigator
from ..perception.grounding import Grounder
from ..sim.robot import Robot
from .result import SkillResult

log = logging.getLogger(__name__)

# How many crates a trip brings back. Three is what is stacked in the scene, and a number the
# report can be specific about.
CRATES_PER_TRIP = 3


class OrchardSkills:
    """Walk the rows, fetch crates, carry them to the shed."""

    actions = ("fetch", "goto", "collect", "deliver", "carrying", "describe", "where", "report")

    def __init__(self, robot: Robot, grounder: Grounder):
        self.robot = robot
        self.grounder = grounder
        # Outdoor settings, as the Mars rover uses and for the same reasons: rough ground is
        # not a wall, and a target across an orchard is further off than anything indoors.
        self.nav = MaplessNavigator(
            robot, grounder, safety_distance=0.35, arrive_distance=2.0,
            cruise_speed=0.45, lost_frames_allowed=40,
        )
        # Where the errand started, which is the shed. Taken from the simulation so moving the
        # shed in the scene moves it here too.
        self.base = robot.position[:2].copy()
        self.carrying = 0

    # -- the errand ----------------------------------------------------------------

    def fetch(self) -> SkillResult:
        """The whole round trip: out to the crates, load, back to the shed, unload."""
        out = self.goto("crates")
        if not out.ok:
            return out

        loaded = self.collect()
        if not loaded.ok:
            return loaded

        back = self.deliver_to_shed()
        return SkillResult(
            back.ok,
            f"{out.message} {loaded.message} {back.message}",
            {**out.data, **loaded.data, **back.data},
        )

    def goto(self, target: str | None, where: str | None = None) -> SkillResult:
        if not target:
            return SkillResult(False, "Where should I go?")
        if target == "shed":
            return self.deliver_to_shed()
        before = self.robot.position[:2].copy()
        result = self.nav.goto(target, max_steps=9000, search_steps=700, where=where)
        walked = float(np.linalg.norm(self.robot.position[:2] - before))
        message = result.describe()
        if result.success:
            message += f" I walked {walked:.0f} m."
        return SkillResult(result.success, message, {"walked_m": round(walked, 1)})

    def collect(self) -> SkillResult:
        """Load the crates, if the robot is actually beside them."""
        if not self.nav.observe_landmarks("crates"):
            # Refuse rather than pretend. A robot that reports loading fruit it cannot see has
            # told a person their crates are on the way when they are still in the row.
            return SkillResult(
                False,
                "I cannot see any crates from here, so I have not loaded anything.",
                {"carrying": self.carrying},
            )
        self.carrying = CRATES_PER_TRIP
        return SkillResult(
            True, f"I have loaded {self.carrying} crates.", {"carrying": self.carrying}
        )

    def deliver_to_shed(self) -> SkillResult:
        """Carry what the robot has back to the shed.

        By dead reckoning, as the Mars rover returns to its lander, and for the same reason:
        the shed is where the errand started, so the robot knows the way without looking, and
        looking for it down a row of trees means catching the door between trunks and losing
        it again.
        """
        start = self.robot.position[:2].copy()
        for _ in range(12000):
            delta = self.base - self.robot.position[:2]
            distance = float(np.linalg.norm(delta))
            if distance < 2.0:
                walked = float(np.linalg.norm(self.robot.position[:2] - start))
                if self.carrying:
                    delivered = self.carrying
                    self.carrying = 0
                    return SkillResult(
                        True,
                        f"I carried {delivered} crates {walked:.0f} m to the shed.",
                        {"delivered": delivered, "walked_m": round(walked, 1), "carrying": 0},
                    )
                return SkillResult(
                    True, f"I am back at the shed, {walked:.0f} m.",
                    {"walked_m": round(walked, 1), "carrying": 0},
                )
            desired = float(np.arctan2(delta[1], delta[0]))
            error = (desired - self.robot.yaw + np.pi) % (2 * np.pi) - np.pi
            self.robot.step(0.45, 0.0, float(np.clip(error * 1.3, -0.8, 0.8)))

        return SkillResult(
            False,
            f"I could not get back to the shed — I am still "
            f"{np.linalg.norm(self.base - self.robot.position[:2]):.0f} m away, "
            f"carrying {self.carrying} crates.",
            {"carrying": self.carrying},
        )

    def carrying_what(self) -> SkillResult:
        if not self.carrying:
            return SkillResult(True, "I am not carrying anything.", {"carrying": 0})
        return SkillResult(
            True, f"I am carrying {self.carrying} crates.", {"carrying": self.carrying}
        )

    def describe(self) -> SkillResult:
        visible = [t for t in ("crates", "shed") if self.nav.observe_landmarks(t)]
        if not visible:
            return SkillResult(True, "Just trees and rows from here.", {"visible": 0})
        return SkillResult(
            True, f"I can see the {' and the '.join(visible)}.", {"visible": len(visible)}
        )

    def where(self) -> SkillResult:
        distance = float(np.linalg.norm(self.robot.position[:2] - self.base))
        return SkillResult(
            True, f"I am {distance:.0f} m from the shed.",
            {"distance_from_shed_m": round(distance, 1), "carrying": self.carrying},
        )

    # -- dispatch ------------------------------------------------------------------

    def run(
        self,
        action: str,
        argument: str | None = None,
        where: str | None = None,
        expect: int | None = None,
    ) -> SkillResult:
        handlers = {
            "fetch": lambda: self.fetch(),
            "goto": lambda: self.goto(argument, where),
            "collect": lambda: self.collect(),
            "deliver": lambda: self.deliver_to_shed(),
            "carrying": lambda: self.carrying_what(),
            "describe": lambda: self.describe(),
            "where": lambda: self.where(),
            "report": lambda: SkillResult(True, argument or "Done."),
        }
        handler = handlers.get(action)
        if handler is None:
            return SkillResult(False, f"I do not know how to '{action}'.")
        log.info("skill: %s(%s)", action, argument or "")
        return handler()
