#!/usr/bin/env python3
"""Run the robot: it logs into Pyunto and does what you message it.

    python scripts/run_robot.py                    # listen for messages
    python scripts/run_robot.py --llm              # use Gemma 4 instead of rules
    python scripts/run_robot.py --say "open the door"   # run one instruction, no Pyunto
    python scripts/run_robot.py --join CODE        # join a chat space first

Then message the robot's account from the Pyunto app:

    「オフィスのドアを開けて」  ->  it walks across the office and pushes the door open

Credentials come from .env (see .env.example).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyunto_robotics.agent import RobotAgent  # noqa: E402
from pyunto_robotics.brain.planner import LLMPlanner, RulePlanner  # noqa: E402
from pyunto_robotics.comms.auth import AuthError, Session  # noqa: E402
from pyunto_robotics.comms.client import PyuntoClient  # noqa: E402
from pyunto_robotics.comms.keys import RawSpaceKeyProvider  # noqa: E402
from pyunto_robotics.perception.grounding import ColorGrounder, VLMGrounder  # noqa: E402
from pyunto_robotics.sim.robot import Robot  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--say", metavar="TEXT",
                    help="execute one instruction locally and exit (no Pyunto)")
    ap.add_argument("--llm", action="store_true",
                    help="plan with Gemma 4 instead of the rule matcher")
    ap.add_argument("--vlm", action="store_true",
                    help="find objects with a vision model instead of colour matching")
    ap.add_argument("--join", metavar="CODE", help="join a chat space with an invite code")
    ap.add_argument("--scene", default="office.xml")
    ap.add_argument("--keyframe", default="lobby",
                    help="where the robot starts: lobby (far) or start (by the doors)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )
    # These are chatty and never interesting here.
    for noisy in ("httpx", "urllib3", "engineio", "socketio", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    load_dotenv()

    grounder = VLMGrounder() if args.vlm else ColorGrounder()
    planner = LLMPlanner() if args.llm else RulePlanner()
    print(f"planner : {type(planner).__name__}")
    print(f"grounder: {type(grounder).__name__}")

    robot = Robot(args.scene, keyframe=args.keyframe)
    print(f"scene   : {args.scene} (starting {args.keyframe})")

    # -- one-shot mode: no network, just do the thing ---------------------------------
    if args.say:
        agent = RobotAgent(robot, grounder, planner=planner)
        print(f"\ninstruction: {args.say!r}")
        execution = agent.execute(args.say)
        print(f"plan       : {execution.plan}")
        print(f"reply      : {execution.reply()}")
        robot.close()
        return 0 if execution.ok else 1

    # -- connected mode ----------------------------------------------------------------
    email = os.getenv("PYUNTO_EMAIL")
    password = os.getenv("PYUNTO_PASSWORD")
    base_url = os.getenv("PYUNTO_BASE_URL", "https://api.pyunto.com")
    if not email or not password:
        print("ERROR: set PYUNTO_EMAIL and PYUNTO_PASSWORD in .env (see .env.example)")
        robot.close()
        return 2

    session = Session(base_url, email, password)
    try:
        identity = session.login()
    except AuthError as e:
        print(f"ERROR: {e}")
        robot.close()
        return 1

    client = PyuntoClient(session, RawSpaceKeyProvider(session))
    print(f"account : {identity}")

    if args.join:
        client.join_with_code(args.join)

    spaces = client.list_spaces()
    shared = [s for s in spaces if not (s.get("is_self") or s.get("isSelf"))]
    print(f"spaces  : {len(spaces)} ({len(shared)} shared)")
    if not shared:
        print(
            "\nNote: the robot is only in its own self-space, so nobody else can message it.\n"
            "      Generate an invite code in the Pyunto app and rerun with --join CODE."
        )

    agent = RobotAgent(robot, grounder, client=client, planner=planner)
    print("\nlistening - message the robot from the Pyunto app (Ctrl-C to stop)")
    print('try: "open the door" / 「オフィスのドアを開けて」\n')

    try:
        agent.run()
    except KeyboardInterrupt:
        print("\nstopping...")
        agent.stop()
    finally:
        robot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
