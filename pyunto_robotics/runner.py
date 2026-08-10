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

from .agent import RobotAgent
from .brain.domains import DOMAINS, Domain, DomainLLMPlanner, DomainRulePlanner
from .comms.auth import AuthError, Session
from .comms.client import PyuntoClient
from .comms.keys import RawSpaceKeyProvider
from .brain.planner import looks_multi_step
from .perception.grounding import ColorGrounder, VLMGrounder
from .sim.robot import Robot


@dataclass
class RobotSetup:
    """What one robot needs that the others do not."""

    name: str
    scene: str
    domain: Domain
    # Builds the skills object. Takes (robot, grounder) and returns something with a
    # `run(action, argument, where, expect)` method -- the same contract brain/skills.Skills
    # has, which is what lets RobotAgent drive any of them.
    skills: Callable[[Robot, object], object]
    # Builds the gait, or None to use the humanoid default.
    gait: Callable[[], object] | None = None
    default_keyframe: str = "start"
    keyframe_help: str = ""
    examples: tuple[str, ...] = ()


def _open_viewer(robot: Robot, speed: float) -> object | None:
    """Open the simulator window and make every robot.step redraw it.

    All of this stays on the main thread on purpose: MuJoCo binds its renderer to the thread
    that created it, and calling Robot.look() from anywhere else aborts the process on macOS.
    The Pyunto listener is the part that gets a background thread.
    """
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        print("mujoco.viewer is unavailable; running without a window")
        return None

    if sys.platform == "darwin" and getattr(mujoco.viewer, "_MJPYTHON", None) is None:
        launcher = Path(sys.executable).with_name("mjpython")
        # shlex.quote each argument: instructions contain spaces and Japanese punctuation, so
        # an unquoted suggestion cannot be pasted back in.
        argv = " ".join(shlex.quote(a) for a in sys.argv[1:])
        print("\nThe simulator window needs mjpython on macOS. Run:")
        print(f"    {launcher} {sys.argv[0]} {argv}".rstrip())
        return None

    viewer = mujoco.viewer.launch_passive(robot.model, robot.data)
    real_step = robot.step

    def step_and_draw(*a, **kw):
        real_step(*a, **kw)
        viewer.sync()
        time.sleep(max(0.0, robot.control_dt / max(speed, 0.1) - 0.002))

    robot.step = step_and_draw  # type: ignore[method-assign]
    print("simulator window open - watch the robot carry out what you message it")
    return viewer


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
        DomainLLMPlanner(setup.domain) if args.llm else DomainRulePlanner(setup.domain)
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

    viewer_ctx = _open_viewer(robot, args.speed) if args.view else None
    if args.view and viewer_ctx is None:
        robot.close()
        return 1

    skills = setup.skills(robot, grounder)

    # -- one-shot mode: no network, just do the thing ---------------------------------
    if args.say:
        agent = RobotAgent(robot, grounder, planner=planner, skills=skills)
        print(f"\ninstruction: {args.say!r}")
        execution = agent.execute(args.say)
        print(f"plan       : {execution.plan}")
        print(f"reply      : {execution.reply()}")
        if viewer_ctx is not None:
            # Hold the finished pose long enough to see it.
            end = time.time() + 4.0
            while viewer_ctx.is_running() and time.time() < end:
                viewer_ctx.sync()
                time.sleep(1 / 60)
            viewer_ctx.close()
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

    def redraw() -> None:
        if viewer_ctx is not None and viewer_ctx.is_running():
            viewer_ctx.sync()

    agent = RobotAgent(
        robot, grounder, client=client, planner=planner, skills=skills,
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
    """Look up a robot by name. Used by the tests and by scripts/run_all.py."""
    from .brain.laundry import LaundrySkills  # noqa: PLC0415 - avoids import cycles
    from .brain.lunar import LunarSkills  # noqa: PLC0415
    from .brain.patrol import PatrolSkills  # noqa: PLC0415
    from .sim.quad_gait import TrotGait  # noqa: PLC0415
    from .sim.wheel_drive import SkidDrive  # noqa: PLC0415

    setups = {
        "home": RobotSetup(
            name="Momo (home assistant)",
            scene="home.xml",
            domain=DOMAINS["home"],
            skills=lambda robot, grounder: LaundrySkills(robot, grounder),
            default_keyframe="start",
            keyframe_help="start (at the washer), middle (centre of room), counter (at the counter)",
            examples=("タオルを洗濯機から出して畳んで", "open the washing machine"),
        ),
        "patrol": RobotSetup(
            name="Q1 (patrol quadruped)",
            scene="campus.xml",
            domain=DOMAINS["patrol"],
            skills=lambda robot, grounder: PatrolSkills(robot, grounder),
            gait=TrotGait,
            default_keyframe="start",
            keyframe_help="start (south of the building), corner (SE corner), steps (at the stairs)",
            examples=("ビルの周りを1周して", "patrol around the building"),
        ),
        "lunar": RobotSetup(
            name="R1 (lunar rover)",
            scene="lunar.xml",
            domain=DOMAINS["lunar"],
            skills=lambda robot, grounder: LunarSkills(robot, grounder),
            gait=SkidDrive,
            default_keyframe="plain",
            keyframe_help="plain (open surface), start (beside the lander)",
            examples=("クレーターの縁まで行って", "drive to the beacon"),
        ),
    }
    if name not in setups:
        raise KeyError(f"unknown robot {name!r}; known: {', '.join(sorted(setups))}")
    return setups[name]
