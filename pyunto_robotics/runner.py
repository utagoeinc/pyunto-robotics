"""The shared body of every `run_*.py` script.

scripts/run_robot.py grew this logic for the office humanoid: open a viewer if asked, run one
instruction and exit if `--say` was given, otherwise log into Pyunto and act on messages. All
four robots need exactly that, and none of it depends on which robot it is -- so it lives here
once and each script supplies the parts that differ: the scene, the gait, the skills, and the
planning domain.

The office script is deliberately left alone. It works, it is the reference the others were
written against, and rewriting it to import this would be a change with no benefit to it.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from pyunto_agent.auth import AuthError

from .agent import RobotAgent
from .brain.domains import DOMAINS, Domain, DomainLLMPlanner, DomainRulePlanner
from .brain.planner import looks_multi_step
from . import registry
from .connect import connect
from .registry import RobotSetup
from .viewer import open_viewer
from .perception.grounding import ColorGrounder, VLMGrounder
from .sim.robot import Robot


def build_parser(setup: RobotSetup, description: str) -> argparse.ArgumentParser:
    """The command line every run_* script shares."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--say", metavar="TEXT",
                        help="execute one instruction locally and exit (no Pyunto)")
    parser.add_argument("--llm", action="store_true",
                        help="plan with Gemma 4 instead of the rule matcher")
    parser.add_argument("--vlm", action="store_true",
                        help="find objects with a vision model instead of colour matching")
    parser.add_argument("--join", metavar="CODE",
                        help="join a chat space with an invite code")
    parser.add_argument("--scene", default=setup.scene)
    parser.add_argument("--keyframe", default=setup.default_keyframe,
                        help=setup.keyframe_help or "where the robot starts")
    parser.add_argument("--view", action="store_true",
                        help="open the simulator window (macOS: run under mjpython)")
    parser.add_argument("--speed", type=float, default=3.0,
                        help="playback speed when --view is on (1 = real time)")
    parser.add_argument("--hold", type=float, default=0.0, metavar="SECONDS",
                        help="with --view and --say, close the window after this many seconds "
                             "instead of waiting for you to close it (0 = wait)")
    parser.add_argument("--pose", action="store_true",
                        help="just stand in the scene and hold the window open, for looking "
                             "at the model (needs --view)")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(setup: RobotSetup, description: str) -> int:
    """Run one robot. Returns a process exit code."""
    args = build_parser(setup, description).parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )
    for noisy in ("httpx", "urllib3", "engineio", "socketio", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    load_dotenv()

    grounder = VLMGrounder() if args.vlm else ColorGrounder()
    planner = (
        setup.planner(args.llm)
        if setup.planner is not None
        else (DomainLLMPlanner(setup.domain) if args.llm else DomainRulePlanner(setup.domain))
    )

    # A chained instruction under the rule matcher silently does the wrong single thing.
    if args.say and not args.llm and looks_multi_step(args.say):
        print(
            "\nNote: that instruction has several parts, and the rule planner only does one.\n"
            "      Add --llm to plan it with Gemma 4."
        )
    print(f"robot   : {setup.name}")
    print(f"planner : {type(planner).__name__} ({setup.domain.name})")
    print(f"grounder: {type(grounder).__name__}")

    robot = Robot(
        args.scene,
        gait=setup.gait() if setup.gait else None,
        keyframe=args.keyframe,
    )
    print(f"scene   : {args.scene} (starting {args.keyframe})")

    viewer_ctx = open_viewer(robot, args.speed, setup.camera) if args.view else None
    if args.view and viewer_ctx is None:
        robot.close()
        return 1

    try:
        skills = setup.skills(robot, grounder, planner=planner)
    except TypeError:
        skills = setup.skills(robot, grounder)

    # 8, not the office robot's 4. A household errand is genuinely long -- "open the washer,
    # take the laundry out, shut the door, put it in the basket, carry it to the folding
    # table" is five actions before anything unusual is asked for -- and at 4 the tail of a
    # perfectly reasonable request was dropped.
    max_steps = 8

    # -- look at the robot, do nothing else --------------------------------------------
    #
    # For inspecting the model itself. Without this the only way to get a window open was to
    # give it an errand and watch, which takes minutes and moves the robot away from wherever
    # you wanted to look at it.
    if args.pose:
        if viewer_ctx is None:
            print("--pose needs --view (macOS: run under mjpython)")
            robot.close()
            return 1
        print("\nholding the start pose - close the window to exit")
        try:
            while viewer_ctx.running:
                robot.step()
                # robot.step already syncs the viewer when --view is on, but sync again so the
                # window stays responsive even if the physics is paused from the UI.
                viewer_ctx.sync()
        except KeyboardInterrupt:
            print("\nclosing...")
        viewer_ctx.close()
        robot.close()
        return 0

    # -- one-shot mode: no network, just do the thing ---------------------------------
    if args.say:
        agent = RobotAgent(
            robot, grounder, planner=planner, skills=skills,
            max_steps_per_message=max_steps,
        )
        print(f"\ninstruction: {args.say!r}")
        execution = agent.execute(args.say)
        print(f"plan       : {execution.plan}")
        print(f"reply      : {execution.reply()}")
        if viewer_ctx is not None:
            # Leave the window OPEN when the errand finishes, and let the user close it.
            #
            # It used to hold the final pose for four seconds and then shut itself, which is
            # long enough to confirm a skill worked and far too short to actually look at the
            # robot -- and looking at the robot is most of what this window is for while the
            # model is being worked on. Waiting on `is_running()` costs nothing: the loop is
            # just redrawing, and it ends the moment the window is closed.
            #
            # `--hold 4` restores the old behaviour for a scripted run that should not block.
            if args.hold > 0:
                end = time.time() + args.hold
                while viewer_ctx.running and time.time() < end:
                    viewer_ctx.sync()
                    time.sleep(1 / 60)
            else:
                print("\nsimulator window is open - close it to exit")
                try:
                    while viewer_ctx.running:
                        viewer_ctx.sync()
                        time.sleep(1 / 60)
                except KeyboardInterrupt:
                    print("\nclosing...")
            viewer_ctx.close()
        robot.close()
        return 0 if execution.ok else 1

    # -- connected mode ----------------------------------------------------------------
    # No credentials needed: an anonymous robot account is created and remembered.
    try:
        connection = connect(display_name=setup.name)
    except AuthError as e:
        print(f"ERROR: {e}")
        robot.close()
        return 1

    client = connection.client
    identity = connection.identity
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

    def redraw() -> None:
        if viewer_ctx is not None and viewer_ctx.running:
            viewer_ctx.sync()

    agent = RobotAgent(
        robot, grounder, client=client, planner=planner, skills=skills,
        max_steps_per_message=max_steps,
        on_idle=redraw if viewer_ctx is not None else None,
    )
    print("\nlistening - message the robot from the Pyunto app (Ctrl-C to stop)")
    for example in setup.examples:
        print(f'try: "{example}"')
    print()

    try:
        agent.run()
    except KeyboardInterrupt:
        print("\nstopping...")
        agent.stop()
    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()
        robot.close()
    return 0


def setup_for(name: str) -> RobotSetup:
    """Look up a robot by name. Kept for the tests and older scripts; see registry.get."""
    return registry.get(name)
