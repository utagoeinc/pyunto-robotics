"""Telling the person what the robot is doing, while it does it.

A robot that accepts an instruction, goes silent for ninety seconds, and then posts one
sentence is indistinguishable from a robot that has crashed. The person cannot tell whether
their words were understood, whether anything is happening, or whether to write again -- and
the most common outcome of all, "I did not understand that", is exactly the one that most
needs saying out loud.

So the robot narrates. Before it moves it says what it understood and what it intends to do;
as each step finishes it says how that went; at the end it posts a picture of what it is
looking at. Every message lands in the same thread the instruction came from, so the diary
reads as a conversation rather than a log.

`Reporter` is the protocol the agent talks to. `ThreadReporter` is the one that posts to
Pyunto; `NullReporter` throws it away, which is what `--say` on the command line wants.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Protocol

log = logging.getLogger(__name__)

# How long a step must run before a "still working" note is worth sending. Below this the
# note arrives after the step has already finished and just adds noise; a person waiting on a
# walk across an office, which takes tens of seconds, wants to know it started.
SLOW_STEP_SECONDS = 6.0


class Reporter(Protocol):
    """What the agent tells about its progress. All methods are best-effort and must not raise."""

    def say(self, text: str) -> None:
        """Post one line of narration."""

    def step_started(self, step) -> None:  # noqa: ANN001 - Step is a planner type
        """A step is about to run."""

    def step_finished(self, step, result) -> None:  # noqa: ANN001
        """A step has finished, successfully or not."""

    def show(self, caption: str = "") -> None:
        """Post what the robot can currently see."""


class NullReporter:
    """Reports nowhere. The default, so `execute()` works with no Pyunto connection."""

    def say(self, text: str) -> None: ...
    def step_started(self, step) -> None: ...  # noqa: ANN001
    def step_finished(self, step, result) -> None: ...  # noqa: ANN001
    def show(self, caption: str = "") -> None: ...


class ThreadReporter:
    """Posts narration and camera frames into the thread the instruction arrived in.

    Failures here are logged and swallowed. A dropped progress note is a small
    disappointment; an exception raised out of a progress note would abandon a half-finished
    instruction with the robot standing in a doorway, which is much worse.
    """

    def __init__(
        self,
        client,  # noqa: ANN001 - pyunto_agent.client.PyuntoClient
        chat_space_id: str,
        thread_id: str | None,
        robot=None,  # noqa: ANN001 - anything with .look(camera) -> Observation
        camera: str = "head_cam",
        send_images: bool = True,
    ):
        self.client = client
        self.chat_space_id = chat_space_id
        self.thread_id = thread_id
        self.robot = robot
        self.camera = camera
        # Pictures need a thread to live in -- the upload endpoint is per-thread -- and they
        # are the one part of this a person may not want, so it is switchable.
        self.send_images = send_images and thread_id is not None and robot is not None
        self._started_at: float | None = None

    # -- narration ----------------------------------------------------------------

    def say(self, text: str) -> None:
        if not text:
            return
        try:
            self.client.send(self.chat_space_id, text, thread_id=self.thread_id)
        except Exception:  # noqa: BLE001 - narration must never break the errand
            log.warning("could not post progress: %s", text[:60], exc_info=True)

    def step_started(self, step) -> None:  # noqa: ANN001
        self._started_at = time.monotonic()

    def step_finished(self, step, result) -> None:  # noqa: ANN001
        """One line per step: what it was, and what came of it.

        Sent after the fact rather than before. "I am about to walk to the door" followed by
        "I walked to the door" is two messages for one event; the person already knows the
        plan, because it was posted before anything moved.
        """
        elapsed = time.monotonic() - (self._started_at or time.monotonic())
        mark = "✅" if result.ok else "⚠️"
        message = (result.message or "").strip()
        line = f"{mark} {_describe(step)} — {message}" if message else f"{mark} {_describe(step)}"
        if elapsed >= SLOW_STEP_SECONDS:
            line += f" ({elapsed:.0f}s)"
        self.say(line)

    # -- pictures -----------------------------------------------------------------

    def show(self, caption: str = "") -> None:
        """Post a frame from the robot's camera.

        This is the answer to "did it actually do what it said". A sentence claiming a door
        is open is a claim; the picture is the evidence, and for a buyer watching a demo it
        is the whole point.
        """
        if not self.send_images:
            return
        png = self._frame()
        if png is None:
            return
        try:
            self.client.send_image(
                self.chat_space_id, png, thread_id=self.thread_id, caption=caption or None
            )
        except Exception:  # noqa: BLE001
            log.warning("could not post camera frame", exc_info=True)
            # One failure is usually permanent (an old server without the endpoint), and
            # retrying it once per step would flood the log for the rest of the session.
            self.send_images = False

    def _frame(self) -> bytes | None:
        """The current camera image as PNG bytes, or None if it cannot be produced."""
        try:
            observation = self.robot.look(self.camera)
            from PIL import Image

            buf = io.BytesIO()
            Image.fromarray(observation.rgb).save(buf, format="PNG")
            return buf.getvalue()
        except Exception:  # noqa: BLE001
            log.warning("could not render a camera frame", exc_info=True)
            return None


def _describe(step) -> str:  # noqa: ANN001
    parts = [str(step.action)]
    if getattr(step, "argument", None):
        parts.append(str(step.argument))
    if getattr(step, "where", None):
        parts.append(f"({step.where})")
    return " ".join(parts)
