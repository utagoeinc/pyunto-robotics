"""`pyunto-robotics` on the command line.

    pyunto-robotics showqr                 # show a square to scan, then open the robot
    pyunto-robotics demo                   # open the robot (already paired)
    pyunto-robotics robots                 # what machines are installed
    pyunto-robotics whoami                 # this robot's account and spaces

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


def _print_what_to_try(robot_name: str) -> None:
    """After pairing, say what to write and which other robots exist.

    Pairing succeeds and then the window opens, which is the moment somebody has no idea what
    to type. The examples are read from the registry rather than written here, so they cannot
    drift from what the robots actually answer.
    """
    from . import registry

    setup = registry.get(robot_name)
    print()
    print(f"Opening {setup.name}. Write one of these in the diary on your phone:")
    for example in setup.examples or ("where are you?",):
        print(f'    "{example}"')

    others = [key for key in registry.names() if key != robot_name]
    if others:
        print()
        print("Other robots, once you have stopped this one with Ctrl-C:")
        for key in others:
            other = registry.get(key)
            example = (other.examples or ("",))[0]
            line = f"    pyunto-robotics demo --robot {key}"
            print(f'{line:<42}# "{example}"' if example else line)
    print()


def _understanding(command_mode: bool) -> bool:
    """Whether the robot reads sentences, or matches a fixed list of commands.

    Reading is the default, and not as a convenience. A keyword table only matches the
    phrasings somebody thought to write down -- told "move somewhere sunny" the matcher went
    looking for a landmark of that name -- and every miss needs another pattern, in every
    language the product ships in. That table has no end, which is the whole reason a robot
    you can simply write to is worth building.

    Command mode is the deliberate opposite, for a site that WANTS a closed vocabulary:
    equipment with a fixed command set, an operator who types the same six instructions all
    day, a safety case that will not accept a model deciding what was meant. It is a choice
    about the deployment, not a fallback, and `--command-mode` is how a customer asks for it.

    The model also runs on this machine, so nothing is sent anywhere to be understood.

    When it cannot run at all, this says so in one line and matches keywords anyway rather
    than refusing to start -- a robot that will not open because a 5.5 GB download is missing
    is worse than one that understands less.
    """
    if command_mode:
        print("planner : command mode — matching the command list, not reading sentences.")
        return False
    if sys.platform != "darwin":
        print("note    : the local model needs Apple silicon, so this robot is matching")
        print("          commands instead of reading sentences. See `--command-mode`.")
        return False
    try:
        import mlx_vlm  # noqa: F401, PLC0415
    except ImportError:
        print("note    : the local model is not installed, so this robot is matching commands")
        print("          instead of reading sentences.")
        if sys.version_info >= (3, 13):
            # Do not send somebody to a command that cannot help them. `download_model`
            # refuses on 3.13+, so telling a 3.14 user to run it is an instruction to go and
            # read an error message.
            print(f"          mlx-vlm has no build for Python {sys.version_info.major}."
                  f"{sys.version_info.minor}, so the model cannot be installed here --")
            print("          build the environment on 3.11 or 3.12 to use it.")
        else:
            print("          To let it read what you write:")
            print("              python -m pyunto_robotics.download_model")
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="pyunto-robotics", description=__doc__.split("\n")[0])
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("demo", help="open a robot and answer messages from the app")
    # The solar errand robot: the demonstration the SDK leads with, and the one that shows a
    # robot finding something by measurement rather than following a script. The old default
    # was "office", a robot that no longer exists -- so a bare `demo` raised KeyError.
    p_demo.add_argument("--robot", default="solar", help="which machine (see `robots`)")
    # Reading sentences is the default and needs no flag. See `_understanding`.
    p_demo.add_argument("--command-mode", action="store_true",
                        help="match a fixed command list instead of reading what you wrote")
    p_demo.add_argument("--commands", metavar="FILE",
                        help="a JSON command list for this site; implies --command-mode")
    p_demo.add_argument("--no-window", dest="view", action="store_false", help="run headless")
    p_demo.add_argument("--speed", type=float, default=1.0, help="playback speed (1.0 = real time)")
    p_demo.add_argument("--no-photos", dest="send_images", action="store_false",
                        help="report in words only; do not post camera pictures to the diary")

    sub.add_parser("robots", help="list the installed machines")
    sub.add_parser("whoami", help="show this robot's account and the spaces it is in")
    p_qr = sub.add_parser(
        "showqr", help="show a QR code for someone to scan in the Pyunto app"
    )
    p_qr.add_argument("--operator", default="",
                      help="who runs this robot; shown to the person before they approve")
    p_qr.add_argument("--big", action="store_true",
                      help="draw the square larger; use it when a phone will not scan")
    p_qr.add_argument("--robot", default="solar", help="which machine to open once paired")
    p_qr.add_argument("--no-run", action="store_true",
                      help="draw the square and exit, instead of opening the robot once paired")
    p_qr.add_argument("--command-mode", action="store_true",
                      help="match a fixed command list instead of reading what you wrote")
    p_qr.add_argument("--no-window", action="store_true")
    p_qr.add_argument("--speed", type=float, default=1.0)
    p_qr.add_argument("--no-photos", action="store_true")

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

    if args.cmd == "showqr":
        # The same payload and renderer the agent uses, so one scanner path in the app
        # handles both. A robot is not a different kind of guest -- it is another program
        # asking to be let into a diary, and the person answers the same question.
        from pyunto_agent.pairing import encode_payload, pairing_payload, render_qr

        try:
            connection = connect()
        except AuthError as e:
            print(f"ERROR: {e}")
            return 2

        payload = pairing_payload(
            user_id=connection.user_id,
            display_name=connection.identity.display_name,
            public_key=connection.identity_store.public_key_b64,
            operator=args.operator,
            runtime="self_hosted",
        )
        text = encode_payload(payload)
        qr = render_qr(text, big=args.big)
        print()
        if qr:
            print(qr)
        else:
            # An install from before qrcode became a hard dependency. Say how to repair it
            # in one line, and print the payload underneath so the session is not wasted --
            # the app cannot scan it, but it proves the robot got this far.
            print("The QR code needs `qrcode`, which this environment does not have:")
            print()
            print("    pip install qrcode")
            print()
            print("Then run `pyunto-robotics showqr` again. The raw pairing payload is")
            print("below; it is what the square would encode, and nothing in it is secret.")
            print()
            print(text)
        print()
        print(f"Scan this in the Pyunto app to let {connection.identity.display_name} into a diary.")
        print("The app asks which space, and shows who runs this robot before anything is shared.")
        print("Nothing here is secret: it names the account asking, and the decision stays with")
        print("whoever holds the phone.")
        if args.no_run:
            print()
            print("Afterwards, open that space in the app once so the robot is given the key.")
            return 0

        # Wait for the scan, then open the robot. Drawing a square and exiting made the
        # person run a second command, and gave them no way to tell whether the scan had
        # worked -- the square just sat there either way. Scanning IS the approval.
        from pyunto_agent.pairing import wait_for_scan

        print()
        print("waiting for the scan… (Ctrl-C to stop)")
        try:
            space_id = wait_for_scan(connection.client)
        except KeyboardInterrupt:
            # Stopping on purpose is not a crash. A traceback here reads as one, and it is
            # the last thing somebody sees after following the README.
            print("\nStopped. Run `pyunto-robotics showqr` again when you are ready.")
            return 0
        if space_id is None:
            print("Nobody scanned it. Run this again when you are ready.")
            return 1
        print("paired ✓")
        _print_what_to_try(args.robot)
        return run_demo(
            robot_name=args.robot,
            pair=None,
            use_llm=_understanding(args.command_mode),
            view=not args.no_window,
            speed=args.speed,
            send_images=not args.no_photos,
        )

    if args.cmd == "demo":
        _reexec_under_mjpython_if_needed(args.view)
        from .demo import run_demo

        return run_demo(
            robot_name=args.robot,
            pair=None,
            use_llm=_understanding(args.command_mode or bool(args.commands)),
            commands=args.commands,
            view=args.view,
            speed=args.speed,
            send_images=args.send_images,
        )

    return 1


if __name__ == "__main__":
    sys.exit(main())
