"""The robot as a Pyunto correspondent.

Wires the pieces together: a message arrives over Socket.IO, the planner decides what it means,
the skills carry it out in simulation, and a reply goes back to the same thread.

One instruction is executed at a time. The simulator is single-threaded and a half-finished
walk is worse than a queued one, so a message arriving mid-task gets a short acknowledgement
rather than being interleaved.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass

from .brain.planner import Plan, RulePlanner
from .brain.skills import Skills
from .comms.client import IncomingMessage, PyuntoClient
from .perception.grounding import Grounder
from .sim.robot import Robot

log = logging.getLogger(__name__)


@dataclass
class Execution:
    """What happened when a plan was carried out."""

    plan: Plan
    messages: list[str]
    ok: bool

    def reply(self) -> str:
        """The text to send back."""
        if not self.messages:
            return "Done."
        return " ".join(self.messages)


class RobotAgent:
    """Listens on Pyunto, acts in simulation, replies with what happened."""

    def __init__(
        self,
        robot: Robot,
        grounder: Grounder,
        client: PyuntoClient | None = None,
        planner: RulePlanner | None = None,
        max_steps_per_message: int = 4,
        on_idle: Callable[[], None] | None = None,
    ):
        self.robot = robot
        self.skills = Skills(robot, grounder)
        self.client = client
        self.planner = planner or RulePlanner()
        self.max_steps_per_message = max_steps_per_message
        # Called repeatedly while waiting for work. Used to keep a viewer window responsive;
        # it runs on the same thread as the simulation, which is where MuJoCo needs it.
        self.on_idle = on_idle

        self._work: queue.Queue[IncomingMessage] = queue.Queue()
        self._busy = threading.Event()
        self._stop = threading.Event()

    # -- executing ----------------------------------------------------------------

    def execute(self, text: str) -> Execution:
        """Plan and carry out one instruction."""
        plan = self.planner.plan(text)
        log.info("plan for %r: %s", text, plan)

        if not plan.steps:
            return Execution(plan, [plan.reply or "I did not understand that."], ok=True)

        messages: list[str] = []
        ok = True
        # Cap the plan length: a model that emits twenty steps has misunderstood, and running
        # them would strand the robot somewhere unexpected.
        for step in plan.steps[: self.max_steps_per_message]:
            result = self.skills.run(step.action, step.argument, step.where, step.expect)
            messages.append(result.message)
            if not result.ok:
                ok = False
                break  # later steps assume the earlier ones worked

        return Execution(plan, messages, ok)

    # -- Pyunto loop --------------------------------------------------------------

    def _handle(self, message: IncomingMessage) -> None:
        """Socket.IO callback. Must return fast; the work happens on the worker thread."""
        if self._busy.is_set():
            self._reply(message, "I am in the middle of something - I will get to that next.")
        self._work.put(message)

    def _reply(self, message: IncomingMessage, text: str) -> None:
        if self.client is None:
            return
        try:
            self.client.send(message.chat_space_id, text, thread_id=message.thread_id)
            log.info("-> %s", text)
        except Exception:  # noqa: BLE001 - a send failure must not kill the robot
            log.exception("could not send reply")

    def run(self) -> None:
        """Listen for instructions until stopped. Blocks.

        The simulation runs on THIS thread, not a worker. MuJoCo's offscreen renderer binds its
        GL context to the thread that created it, and calling it from another one aborts the
        process with a Metal assertion on macOS. So the Socket.IO listener gets its own thread
        and the main thread owns the simulator, rather than the other way round.
        """
        if self.client is None:
            raise RuntimeError("RobotAgent needs a PyuntoClient to run()")

        listener = threading.Thread(
            target=self.client.listen, args=(self._handle,), daemon=True, name="pyunto-listener"
        )
        listener.start()
        log.info("robot online, listening for messages")

        try:
            self._pump()
        finally:
            self._stop.set()
            self.client.stop()
            listener.join(timeout=2.0)

    def _pump(self) -> None:
        """Drain the instruction queue on the calling thread until stopped."""
        while not self._stop.is_set():
            try:
                # Short timeout so `on_idle` still runs while nothing is queued -- that hook is
                # what keeps a simulator window redrawing between instructions.
                message = self._work.get(timeout=0.05)
            except queue.Empty:
                if self.on_idle is not None:
                    self.on_idle()
                continue

            self._busy.set()
            try:
                log.info("<- [%s] %s", message.sender_name, message.text)
                execution = self.execute(message.text)
                self._reply(message, execution.reply())
            except Exception:  # noqa: BLE001 - keep listening whatever happens
                log.exception("failed while executing an instruction")
                self._reply(message, "Something went wrong while I was doing that.")
            finally:
                self._busy.clear()
                self._work.task_done()

    def pump_once(self, timeout: float = 0.0) -> bool:
        """Handle at most one queued instruction. Returns True if one was handled.

        Lets a caller own the loop -- useful for tests and for driving the simulator from a
        viewer's frame loop.
        """
        try:
            message = self._work.get(timeout=timeout) if timeout else self._work.get_nowait()
        except queue.Empty:
            return False

        self._busy.set()
        try:
            execution = self.execute(message.text)
            self._reply(message, execution.reply())
        finally:
            self._busy.clear()
            self._work.task_done()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self.client is not None:
            self.client.stop()
