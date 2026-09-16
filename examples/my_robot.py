#!/usr/bin/env python3
"""Attaching your own robot to a Pyunto diary.

This is the whole contract: one class with one method. Everything else -- the encrypted
transport, joining a space, understanding what the person wrote, replying in the right thread --
is handled for you.

Run it:

    pip install 'pyunto-robotics[llm] @ git+https://github.com/utagoeinc/pyunto-robotics'
    python examples/my_robot.py

A QR code appears. Scan it with the Pyunto app, choose a diary, and the robot starts
listening -- then write "go to the door" in that diary from your phone.

Swap the bodies of these methods for calls into your own control stack -- ROS, a vendor SDK, a
serial link, another simulator -- and the diary is talking to your machine instead of ours.
"""

from __future__ import annotations

import argparse
import sys

from pyunto_agent.bridge import Bridge
from pyunto_agent.pairing import encode_payload, pairing_payload, render_qr, wait_for_scan

from pyunto_robotics.api import SkillResult
from pyunto_robotics.backend import RobotBackend
from pyunto_robotics.connect import connect


class MyRobot:
    """A robot that only pretends to move. Satisfies `pyunto_robotics.api.RobotSkills`.

    Two rules worth keeping when you replace the pretending with real motion:

    * Never raise. "I could not reach the door" is a fine answer to send a person; a traceback
      is not. Return `SkillResult(ok=False, ...)` instead.
    * Write sentences. The message goes straight into someone's diary, on their phone.
    """

    def __init__(self) -> None:
        self.location = "the charging dock"

    def run(self, action, argument=None, where=None, expect=None) -> SkillResult:
        if action in ("goto", "go", "move"):
            target = argument or "somewhere"
            # ---- your robot moves here ----
            self.location = target
            return SkillResult(True, f"I went to {target}.", data={"location": target})

        if action in ("home", "return"):
            self.location = "the charging dock"
            return SkillResult(True, "I am back at the dock.")

        if action in ("where", "describe", "report"):
            return SkillResult(True, f"I am at {self.location}.")

        # Unknown verbs are reported, not raised: the person gets an answer either way.
        return SkillResult(False, f"I do not know how to '{action}' yet.")


def main() -> int:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()

    connection = connect(display_name="My Robot")
    print(f"robot: {connection.identity.display_name} ({connection.user_id})")

    # Pairing is a QR code somebody scans. Drawing the QR code and waiting is the whole handshake:
    # scanning IS the approval, so there is nothing else to type at either end.
    payload = pairing_payload(
        user_id=connection.user_id,
        display_name=connection.identity.display_name,
        public_key=connection.identity_store.public_key_b64,
        operator="",
        runtime="self_hosted",
    )
    print()
    print(render_qr(encode_payload(payload)) or encode_payload(payload))
    print()
    print(f"Scan this in the Pyunto app to let {connection.identity.display_name} into a diary.")
    print("waiting for the scan… (Ctrl-C to stop)")
    try:
        space_id = wait_for_scan(connection.client)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    if space_id is None:
        print("Nobody scanned it. Run this again when you are ready.")
        return 1
    print("paired ✓")
    print("Open that space in the Pyunto app once so the robot is given the key.")

    # A robot is a Bridge backend whose "reply" is "do it, then report".
    # RobotBackend expects anything with `.execute(text) -> something with .reply()`, which is
    # what RobotAgent provides. Here we skip the planner and map messages to skills directly.
    class DirectAgent:
        def __init__(self, skills):
            self.skills = skills

        def execute(self, text: str):
            verb, _, rest = text.strip().partition(" ")
            result = self.skills.run(verb.lower(), rest.strip() or None)
            return type("Execution", (), {"reply": lambda _self=None, r=result: r.message})()

    bridge = Bridge(connection.client, RobotBackend(DirectAgent(MyRobot())),
                    persona="", space_ids={space_id} if space_id else None, history=2)
    print("listening — message the robot from the Pyunto app. Ctrl-C to stop.")
    try:
        bridge.run()
    except KeyboardInterrupt:
        bridge.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
