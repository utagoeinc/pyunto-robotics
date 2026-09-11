"""`pyunto-robotics` on the command line.

    pyunto-robotics demo --pair K3F9QZ     # the one-command demonstration
    pyunto-robotics robots                 # what machines are installed
    pyunto-robotics whoami                 # this robot's account and spaces
    pyunto-robotics pair K3F9QZ            # join a space without opening a window

On macOS a MuJoCo window must be owned by `mjpython`, not `python`. Rather than print an
instruction and stop -- which turns a one-command demo into a two-command one -- this
re-executes itself under mjpython automatically.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from pyunto_agent.auth import AuthError

from . import registry
from .connect import connect


def _reexec_under_mjpython_if_needed(wants_window: bool) -> None:
    """On macOS, hand the process over to mjpython so the window can open."""
    if not wants_window or sys.platform != "darwin" or os.environ.get("PYUNTO_NO_REEXEC"):
        return
    try:
        import mujoco.viewer
    except ImportError:
        return
    if getattr(mujoco.viewer, "_MJPYTHON", None) is not None:
        return  # already there
    launcher = Path(sys.executable).with_name("mjpython")
    if not launcher.exists():
        print("The simulator window needs mjpython, which ships with the mujoco package.")
        print(f"Expected it at {launcher}. Reinstall mujoco, or rerun with --no-window.")
        return
    os.environ["PYUNTO_NO_REEXEC"] = "1"
    os.execv(str(launcher), [str(launcher), "-m", "pyunto_robotics.cli", *sys.argv[1:]])


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="pyunto-robotics", description=__doc__.split("\n")[0])
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("demo", help="open a robot and answer messages from the app")
    p_demo.add_argument("--pair", help="pairing code shown by the Pyunto app")
    p_demo.add_argument("--robot", default="office", help="which machine (see `robots`)")
    p_demo.add_argument("--llm", action="store_true", help="use the local language model to plan")
    p_demo.add_argument("--no-window", dest="view", action="store_false", help="run headless")
    p_demo.add_argument("--speed", type=float, default=1.0, help="playback speed (1.0 = real time)")
    p_demo.add_argument("--no-photos", dest="send_images", action="store_false",
                        help="report in words only; do not post camera pictures to the diary")

    sub.add_parser("robots", help="list the installed machines")
    sub.add_parser("whoami", help="show this robot's account and the spaces it is in")
    p_pair = sub.add_parser("pair", help="join a space with a pairing code, then exit")
    p_pair.add_argument("code")

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.cmd == "robots":
        for key, setup in sorted(registry.setups().items()):
            print(f"{key:8} {setup.name}")
            for example in setup.examples[:2]:
                print(f"{'':8}   e.g. {example}")
        return 0

    if args.cmd == "whoami":
        try:
            connection = connect()
        except AuthError as e:
            print(f"ERROR: {e}")
            return 2
        print(f"robot   : {connection.identity.display_name}")
        print(f"user id : {connection.user_id}")
        print(f"identity: {connection.identity_store.public_key_b64}")
        for space in connection.client.list_spaces():
            sid = str(space.get("uuid"))
            has = "yes" if connection.keys.has_key(sid) else "no"
            print(f"  - {space.get('name')}  {sid}  key={has}")
        return 0

    if args.cmd == "pair":
        try:
            connection = connect()
            space_id = connection.client.join(args.code)
        except AuthError as e:
            print(f"ERROR: {e}")
            return 2
        except Exception as e:  # noqa: BLE001 - a stale code is a user error
            print(f"ERROR: could not join with that code ({e}).")
            print("       Codes expire after 10 minutes — make a new one in the app.")
            return 1
        print(f"joined space {space_id}.")
        print("Open that space in the Pyunto app once so the robot is given the key.")
        return 0

    if args.cmd == "demo":
        _reexec_under_mjpython_if_needed(args.view)
        from .demo import run_demo

        return run_demo(
            robot_name=args.robot,
            pair=args.pair,
            use_llm=args.llm,
            view=args.view,
            speed=args.speed,
            send_images=args.send_images,
        )

    return 1


if __name__ == "__main__":
    sys.exit(main())
