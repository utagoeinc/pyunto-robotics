"""The public contract for attaching a robot to a Pyunto diary.

This is the only file you need to read to connect your own robot -- real hardware, another
simulator (Newton, Isaac, Gazebo), or a machine that exists only as an API. Everything else in
this package is our own implementation of these interfaces for the bundled MuJoCo robots.

There are two ways in, and you almost certainly want the first.

1. Implement `RobotSkills`. You get the Pyunto transport, end-to-end encryption, message
   handling, planning and replies for free; you write what your robot does when told to do
   something. This works for any robot at all, because we never touch your machine.

2. Implement `RobotBody` as well, if you want to reuse *our* skills (door opening, mapless
   navigation, patrol routes) on your own hardware. Narrower and more demanding, because our
   skills assume a mobile base with a camera.

A worked example of (1) is in `examples/my_robot.py` -- about forty lines.

Design notes worth knowing before you implement:

* **Failing is normal and is reported, not raised.** "I could not find the door" is a perfectly
  good answer to send a person; a traceback is not. Return `SkillResult(ok=False, message=...)`.
* **The message is read by a human**, in a diary, on a phone. Write sentences, not status codes.
* **You are called on the main thread**, one message at a time, and you may block for as long
  as the action takes. The network runs elsewhere.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# Re-exported so there is one import site for everything public.
from .brain.result import SkillResult
from .perception.grounding import Detection, Grounder
from .reporting import NullReporter, Reporter
from .sim.gait import Gait

__all__ = [
    "Detection",
    "Gait",
    "Grounder",
    "NullReporter",
    "Observation",
    "Reporter",
    "RobotBody",
    "RobotSkills",
    "SkillResult",
]


# `Reporter` is how the robot narrates while it works -- what it understood before it moves,
# each step as it finishes, and a camera frame at the end. You do not implement it: the
# package supplies one that posts into the thread the instruction came from. It is exported
# here because a `RobotSkills` implementation may want to accept one and report from inside a
# long step, and because a test wants `NullReporter`.


@runtime_checkable
class RobotSkills(Protocol):
    """What your robot can do. The main way to attach a robot.

    One method. It is called once per step of a plan; a message like "go to the door and open
    it" becomes two calls. Unknown actions should return a failure rather than raise.

    An optional `actions` attribute -- a sequence of the verbs you handle -- is used when the
    robot has to tell someone it did not understand them, so the reply can say what would
    have worked instead of just "no"::

        class MyRobot:
            def run(self, action, argument=None, where=None, expect=None):
                if action == "goto":
                    ok = my_arm.move_to(argument)
                    return SkillResult(ok, f"I moved to the {argument}." if ok
                                       else f"I could not reach the {argument}.")
                return SkillResult(False, f"I do not know how to '{action}'.")

    The verbs you receive are the ones in your `Domain` (see `brain/domains.py`); if you do not
    supply a domain, they come from the general humanoid vocabulary in brain/planner.py.
    """

    def run(
        self,
        action: str,
        argument: str | None = None,
        where: str | None = None,
        expect: int | None = None,
    ) -> SkillResult:
        """Carry out one step.

        Args:
            action: the verb, e.g. "goto", "open", "describe".
            argument: what it acts on, e.g. "door", "sample". May be None for verbs like "home".
            where: a spatial qualifier the speaker used, e.g. "right", "far". May be None.
            expect: how many of something the speaker said there were, when they said it.

        Returns:
            A `SkillResult` whose `.message` is a sentence to send back to the person.
        """
        ...


@runtime_checkable
class RobotBody(Protocol):
    """A machine our own skills can drive. Only needed if you reuse our skills.

    The bundled MuJoCo `Robot` satisfies this. Three members, deliberately:

    * `step` is the single movement primitive -- everything (walking, turning, strafing) is
      expressed as a body-frame velocity held for one control interval.
    * `look` returns what the robot sees, or None if it has no camera. Skills that need vision
      degrade to a reported failure rather than crashing.
    * `control_dt` is how long one `step` lasts, in seconds.
    """

    control_dt: float

    def step(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> None:
        """Move at this body-frame velocity for one control interval.

        vx forward (m/s), vy left (m/s), wz yaw rate (rad/s).
        """
        ...

    def look(self, camera: str = "head_cam") -> "Observation | None":
        """A camera frame with depth, or None if this robot cannot see."""
        ...


# Imported late: `Observation` lives with the MuJoCo body, but the Protocol above only needs
# its name. A customer implementing RobotBody without MuJoCo can return their own object with
# the same shape (`rgb`, `depth`).
from .sim.robot import Observation  # noqa: E402
