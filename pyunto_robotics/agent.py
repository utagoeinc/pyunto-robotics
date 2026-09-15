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
from dataclasses import dataclass, field

from .brain.planner import Plan, RulePlanner
from pyunto_agent.client import IncomingMessage, PyuntoClient
from .perception.grounding import Grounder
from .reporting import NullReporter, Reporter
from .sim.robot import Robot

log = logging.getLogger(__name__)


@dataclass
class Execution:
    """What happened when a plan was carried out."""

    plan: Plan
    messages: list[str]
    ok: bool
    # What each skill measured, in the order they ran. Kept so a caller can report on the
    # run rather than just pass/fail -- a door that opened 17 degrees and one that opened
    # 109 both "worked", and only the number tells them apart.
    data: list[dict] = field(default_factory=list)

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
        max_replans: int = 2,
        on_idle: Callable[[], None] | None = None,
        skills: object | None = None,
    ):
        self.robot = robot
        self.client = client
        self.planner = planner or RulePlanner()
        # Any object with `run(action, argument, where, expect) -> SkillResult` will do. There
        # is no default: what a robot can do is the one thing that genuinely differs between
        # machines, and inventing a fallback would mean a robot silently driving with another
        # machine's abilities. The registry supplies these per robot.
        if skills is None:
            raise ValueError(
                "RobotAgent needs skills: an object with "
                "run(action, argument, where, expect) -> SkillResult. "
                "See pyunto_robotics/api.py."
            )
        self.skills = skills
        self.max_steps_per_message = max_steps_per_message
        # How many times one message may be re-planned after a step fails. Two is enough to
        # get out of the situations that actually arise (something moved, something is in the
        # way) and few enough that a planner stuck on a bad idea cannot spin.
        self.max_replans = max_replans
        # Called repeatedly while waiting for work. Used to keep a viewer window responsive;
        # it runs on the same thread as the simulation, which is where MuJoCo needs it.
        self.on_idle = on_idle

        self._work: queue.Queue[IncomingMessage] = queue.Queue()
        self._busy = threading.Event()
        self._stop = threading.Event()

    # -- executing ----------------------------------------------------------------

    def execute(self, text: str, report: Reporter | None = None) -> Execution:
        """Plan and carry out one instruction.

        `report`, when given, is told what is happening as it happens: what the robot
        understood before it moves, each step as it finishes, and a camera frame at the end.
        Without it the behaviour is unchanged -- everything arrives in one reply at the end.
        """
        plan = self.planner.plan(text)
        log.info("plan for %r: %s", text, plan)

        if not plan.steps:
            # Not understood. Say what was heard and what the robot does know how to do, so the
            # person can rephrase rather than guess. This is a far more common outcome than a
            # failed action, and answering it with silence is the worst of the options.
            message = plan.reply or self._did_not_understand(text)
            if report is not None:
                report.say(message)
            return Execution(plan, [message], ok=True)

        if report is not None:
            # Phase 1: what the robot can see BEFORE it moves, then how it read the
            # instruction. The picture comes first so the plan is read against the scene it
            # was made in -- afterwards there is no way to know what the robot was looking at.
            report.show("🤖 Before: “" + text.strip() + "”")
            report.say(self._understood(text, plan))

        messages: list[str] = []
        measurements: list[dict] = []
        ok = True
        # Cap the plan length: a model that emits twenty steps has misunderstood, and running
        # them would strand the robot somewhere unexpected.
        remaining = list(plan.steps[: self.max_steps_per_message])
        replans = 0
        while remaining:
            step = remaining.pop(0)
            if report is not None:
                report.step_started(step)
            result = self.skills.run(step.action, step.argument, step.where, step.expect)
            messages.append(result.message)
            measurements.append({"step": str(step), "ok": result.ok, **result.data})
            if report is not None:
                report.step_finished(step, result)
            if result.ok:
                continue

            ok = False
            # Most failures stop the plan, because later steps assume the earlier ones
            # worked. A skill can mark its failure non-fatal to say "I could not do that
            # one, but the rest still makes sense" -- which is the right answer to an
            # impossible aside inside an otherwise perfectly good errand.
            if not getattr(result, "fatal", True):
                continue

            # STOP AND THINK AGAIN before abandoning the errand.
            #
            # A step failing does not always mean the errand is impossible; more often the
            # world has moved on from what the plan assumed. Carrying the basket to the washer
            # puts it exactly where the robot wanted to stand to reach into the drum, and the
            # original plan has no way to know that -- it was written before the basket moved.
            #
            # So the robot describes what is actually true now (what it can see, what it is
            # holding, what just failed) and asks the planner for the rest of the errand from
            # here. That is the third and last tier of recovery, after looking around and
            # wandering, both of which live in the skills where the target is known.
            #
            # Bounded to `max_replans`: a planner that keeps proposing the same failing step
            # would otherwise loop until the step cap, and repeating a failure is not thinking.
            if replans >= self.max_replans:
                break
            revised = self._replan(plan, step, result, remaining)
            if revised is None:
                break
            replans += 1
            log.info("replanned after %s failed: %s", step.action,
                     " -> ".join(str(s) for s in revised))
            messages.append("Let me try that a different way.")
            if report is not None:
                report.say("🤖 Let me try that a different way.")
            remaining = revised[: self.max_steps_per_message]
            ok = True  # the revised plan gets a fair chance to succeed

        # SAY SO when the cap bites. A six-part errand truncated to four used to finish with a
        # cheerful report of the four it did, and the user had no way to tell that the last two
        # were never attempted -- which is indistinguishable from the robot deciding it was
        # done. Silently doing less than asked is the one failure mode worth being loud about.

        dropped = len(plan.steps) - self.max_steps_per_message
        if dropped > 0 and ok:
            skipped = ", ".join(str(s) for s in plan.steps[self.max_steps_per_message:])
            messages.append(
                f"That was {len(plan.steps)} steps and I only do "
                f"{self.max_steps_per_message} at a time, so I have not done: {skipped}."
            )
            ok = False

        if report is not None:
            # Phase 4: one closing message that stands on its own -- did it work, what
            # happened at each step, and a final picture. Someone scrolling back a day later
            # reads this one entry instead of reassembling the running commentary.
            report.finished(text, ok, measurements)

        return Execution(plan, messages, ok, measurements)

    def _understood(self, text: str, plan) -> str:
        """What the robot is about to do, said before it moves.

        The person has just written a sentence to a machine and has no idea whether it landed.
        Saying "I heard X, so I will do Y" first means a misunderstanding is caught in the two
        seconds before the robot walks off, not after.
        """
        steps = " → ".join(self._describe_step(s) for s in plan.steps[: self.max_steps_per_message])
        return f"🤖 Understood: “{text.strip()}”\nI will: {steps}"

    def _did_not_understand(self, text: str) -> str:
        """Explain the failure to understand, and offer the vocabulary that would work."""
        known = self._known_actions()
        lines = [f"🤖 I did not understand “{text.strip()}”."]
        if known:
            lines.append("I know how to: " + ", ".join(known) + ".")
        return "\n".join(lines)

    def _known_actions(self) -> list[str]:
        """The verbs this robot actually has, asked of the skills rather than hard-coded."""
        for attr in ("actions", "verbs"):
            value = getattr(self.skills, attr, None)
            if callable(value):
                try:
                    value = value()
                except Exception:  # noqa: BLE001 - introspection must never break a reply
                    value = None
            if value:
                return sorted(str(v) for v in value)
        domain = getattr(self.planner, "domain", None)
        if domain is not None and getattr(domain, "verbs", None):
            return sorted({action for action, _ in domain.verbs})
        return []

    @staticmethod
    def _describe_step(step) -> str:
        parts = [str(step.action)]
        if getattr(step, "argument", None):
            parts.append(str(step.argument))
        if getattr(step, "where", None):
            parts.append(f"({step.where})")
        return " ".join(parts)

    def _replan(
        self,
        plan: Plan,
        failed: object,
        result: object,
        remaining: list,
    ) -> list | None:
        """Ask the planner for the rest of the errand, given what just went wrong.

        Returns the new steps, or None if the planner cannot help -- which includes the rule
        matcher, since it has no way to reason about a failure. Only the LLM planner is asked.
        """
        describe = getattr(self.planner, "replan", None)
        if not callable(describe):
            return None

        # What the robot can see right now. This is the part that makes re-planning worth
        # doing: without it the planner is guessing from the same information that produced
        # the plan which just failed.
        view = ""
        try:
            look = self.skills.run("describe", None, None, None)
            view = look.message
        except Exception:  # noqa: BLE001 - a failed look must not break recovery
            log.debug("could not look around before replanning", exc_info=True)

        try:
            return describe(
                original=str(plan),
                failed_step=str(failed),
                failure=getattr(result, "message", ""),
                remaining=[str(s) for s in remaining],
                view=view,
            )
        except Exception:  # noqa: BLE001 - the planner is remote/model code; never fatal
            log.exception("replanning failed")
            return None

    # -- Pyunto loop --------------------------------------------------------------

    def _handle(self, message: IncomingMessage) -> None:
        """Socket.IO callback. Must return fast; the work happens on the worker thread.

        This is the standalone path (``RobotAgent.run()``), used when the robot listens for
        itself rather than through ``pyunto_agent.Bridge``. It therefore has to repeat the
        Bridge's refusal: a machine takes orders from people, never from another program.
        Leaving it out here would make the protection depend on which entry point a
        deployment happened to use.
        """
        if message.sender_is_agent:
            log.info("ignoring entry from %s: a robot takes orders only from people",
                     message.sender_name)
            return
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
