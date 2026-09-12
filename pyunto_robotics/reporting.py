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

    def finished(self, instruction: str, ok: bool, measurements: list[dict]) -> None:
        """The closing report: whether it worked, what each step did, and a last picture."""


class NullReporter:
    """Reports nowhere. The default, so `execute()` works with no Pyunto connection."""

    def say(self, text: str) -> None: ...
    def step_started(self, step) -> None: ...  # noqa: ANN001
    def step_finished(self, step, result) -> None: ...  # noqa: ANN001
    def show(self, caption: str = "") -> None: ...
    def finished(self, instruction: str, ok: bool, measurements: list[dict]) -> None: ...


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
        self._step_number = 0

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
        """One message per step: what it was, what came of it, and what the sensors read.

        Sent after the fact rather than before. "I am about to walk to the door" followed by
        "I walked to the door" is two messages for one event; the person already knows the
        plan, because it was posted before anything moved.
        """
        self._step_number += 1
        elapsed = time.monotonic() - (self._started_at or time.monotonic())
        mark = "✅" if result.ok else "⚠️"
        message = (result.message or "").strip()
        line = f"{mark} {self._step_number}. {_describe(step)}"
        if message:
            line += f" — {message}"
        if elapsed >= SLOW_STEP_SECONDS:
            line += f" ({elapsed:.0f}s)"
        readings = _readings(getattr(result, "data", None))
        if readings:
            # What the robot measured, not just what it claims. "the door is 0.42 m away" is
            # checkable; "I walked to the door" is a story.
            line += "\n📡 " + readings
        self.say(line)
        # A frame per step is the point of a step-by-step report: it is how someone sees the
        # arm actually go up rather than reading that it did.
        self.show(f"📷 {self._step_number}. {_describe(step)}")

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

    def finished(self, instruction: str, ok: bool, measurements: list[dict]) -> None:
        """One self-contained closing message, plus a final picture.

        The running commentary is for whoever is watching live. This is for everyone else:
        the person who looks at the diary tonight wants one entry that says whether the
        thing they asked for happened, not eight fragments to reassemble.
        """
        headline = "✅ Done" if ok else "⚠️ Not finished"
        lines = [f"{headline}: “{instruction.strip()}”"]
        for i, measured in enumerate(measurements, start=1):
            mark = "✅" if measured.get("ok") else "⚠️"
            readings = _readings({k: v for k, v in measured.items() if k not in ("step", "ok")})
            line = f"  {mark} {i}. {measured.get('step', '')}"
            if readings:
                line += f" — {readings}"
            lines.append(line)
        self.say("\n".join(lines))
        self.show("📷 " + ("Done" if ok else "Stopped here") + f": {instruction.strip()}")

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


def _readings(data) -> str:  # noqa: ANN001
    """Sensor values as a short phrase. Empty when there is nothing worth saying.

    A raw dict pasted into a diary is unreadable, and most of what skills record is for the
    logs. Distances, angles and counts are the parts a person can actually check.
    """
    if not data:
        return ""
    parts = []
    for key, value in data.items():
        if value is None or isinstance(value, (list, dict, bytes)):
            continue
        label = str(key).replace("_", " ")
        if isinstance(value, float):
            if key.endswith("_m"):
                parts.append(f"{label[:-2].strip()} {value:.2f} m")
            elif key.endswith("_deg"):
                parts.append(f"{label[:-4].strip()} {value:.0f}°")
            else:
                parts.append(f"{label} {value:.2f}")
        else:
            parts.append(f"{label} {value}")
    return ", ".join(parts[:4])


def _describe(step) -> str:  # noqa: ANN001
    parts = [str(step.action)]
    if getattr(step, "argument", None):
        parts.append(str(step.argument))
    if getattr(step, "where", None):
        parts.append(f"({step.where})")
    return " ".join(parts)
