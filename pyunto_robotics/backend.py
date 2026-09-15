"""Making a robot the other half of a diary conversation.

`pyunto_agent.Bridge` already does the hard parts of being a diary member: receiving encrypted
entries, filtering out its own posts and undecryptable ones, keeping thread history, rate
limiting, replying in the right thread. It asks a `Backend` what to say.

A robot is simply a `Backend` whose "what to say" is "go and do it, then report back".

This is why the robot needs no message loop of its own -- an earlier version of this package
had one, duplicating Bridge's job with fewer safeguards.
"""

from __future__ import annotations

import logging

from pyunto_agent.backends import Context

from .agent import RobotAgent
from .reporting import ThreadReporter

log = logging.getLogger(__name__)


class RobotBackend:
    """Adapts a `RobotAgent` to the `pyunto_agent.backends.Backend` protocol."""

    name = "robot"
    #: Replying here means the machine moves, so the Bridge applies its stricter gate:
    #: never take instructions from another program, and act only when addressed by name.
    #: Set on the class so every way of building a robot Bridge gets it, including callers
    #: written before the flag existed.
    acts_physically = True

    def __init__(self, agent: RobotAgent, client=None, robot=None, camera: str = "head_cam",
                 send_images: bool = True):  # noqa: ANN001
        self.agent = agent
        # Given a client, the robot narrates into the thread as it works instead of going
        # quiet and posting one sentence at the end. Without one it behaves as before.
        self.client = client
        self.robot = robot if robot is not None else getattr(agent, "robot", None)
        self.camera = camera
        self.send_images = send_images

    def reply(self, ctx: Context) -> str | None:
        """Act on the newest entry from a PERSON, and return what happened.

        The last turn overall is not good enough. This robot narrates as it works, so its own
        progress notes land in the thread and come back as history -- and taking the last turn
        then hands the robot its own "Understood: ..." to carry out as a fresh instruction,
        which it dutifully re-plans and re-runs. That is what filled a thread with repeated
        plans and repeated "I raised my right hand".

        Diary entries are instructions, not a conversation to be summarised, so it is still
        only the newest one that is acted on -- just the newest one somebody else wrote.
        """
        instruction = ""
        for turn in reversed(ctx.turns):
            if turn.role == "user":
                instruction = turn.text.strip()
                break
        if not instruction:
            return None
        log.info("instruction: %s", instruction)
        reporter = None
        if self.client is not None and ctx.chat_space_id:
            reporter = ThreadReporter(
                self.client, ctx.chat_space_id, ctx.thread_id,
                robot=self.robot, camera=self.camera, send_images=self.send_images,
                # Notify whoever gave the instruction, so their phone tells them the robot
                # answered. The narration is posted by the reporter rather than returned to
                # Bridge, so it has to name the recipient itself.
                notify_users=[ctx.sender_uuid] if ctx.sender_uuid else None,
            )
        execution = self.agent.execute(instruction, report=reporter)
        if reporter is not None:
            # The narration already said everything the summary would repeat, and a final
            # duplicate of it reads as the robot saying the same thing twice.
            return None
        return execution.reply()
