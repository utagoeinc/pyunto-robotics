"""`pyunto-robotics` on the command line.

    pyunto-robotics demo --pair K3F9QZ     # the one-command demonstration
    pyunto-robotics robots                 # what machines are installed
    pyunto-robotics whoami                 # this robot's account and spaces
    pyunto-robotics pair K3F9QZ            # join a space without opening a window
    pyunto-robotics showqr                 # show a QR for someone to scan in the app

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


def _llm_wanted(asked: bool) -> bool:
    """Whether to read the sentence with the local model, saying why when it cannot.

    Understanding is on by default. Keyword tables only match the phrasings somebody thought
    to write down -- told "move somewhere sunny" the matcher went looking for a landmark
    of that name -- and every miss needs another pattern, in every language the product ships
    in. That is not a table anybody can finish.

    The model reads the sentence instead, and it runs on this machine, so nothing is sent
    anywhere to be understood. When it is not available this says so in one line and carries
    on with keywords rather than refusing to start: a robot that will not open because a
    5 GB download is missing is worse than one that understands less.
    """
    if not asked:
        return False
    if sys.platform != "darwin":
        print("note    : the local model needs Apple silicon; matching keywords instead.")
        return False
    try:
        import mlx_vlm  # noqa: F401, PLC0415
    except ImportError:
        print("note    : the local model is not installed, so keywords are being matched.")
        print("          To let it read sentences instead of matching words:")
        print("              python scripts/download_model.py")
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="pyunto-robotics", description=__doc__.split("\n")[0])
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("demo", help="open a robot and answer messages from the app")
    p_demo.add_argument("--pair", help="pairing code shown by the Pyunto app")
    # The solar errand robot: the demonstration the SDK leads with, and the one that shows a
    # robot finding something by measurement rather than following a script. The old default
    # was "office", a humanoid that no longer exists -- so a bare `demo` raised KeyError.
    p_demo.add_argument("--robot", default="solar", help="which machine (see `robots`)")
    # On by default. See `_llm_wanted`: a keyword table cannot be finished, least of all in
    # every language, and the fallback when the model is missing is the table anyway.
    p_demo.add_argument("--no-llm", action="store_true",
                        help="match keywords instead of reading the sentence with the local model")
    p_demo.add_argument("--no-window", dest="view", action="store_false", help="run headless")
    p_demo.add_argument("--speed", type=float, default=1.0, help="playback speed (1.0 = real time)")
    p_demo.add_argument("--no-photos", dest="send_images", action="store_false",
                        help="report in words only; do not post camera pictures to the diary")

    sub.add_parser("robots", help="list the installed machines")
    sub.add_parser("whoami", help="show this robot's account and the spaces it is in")
    p_pair = sub.add_parser("pair", help="join a space with a pairing code, then exit")
    p_pair.add_argument("code")

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
    p_qr.add_argument("--no-llm", action="store_true",
                      help="match keywords instead of reading the sentence with the local model")
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
            print("(install the 'qr' extra to draw this as a scannable square:")
            print("     pip install 'pyunto-agent[qr]')")
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
        space_id = wait_for_scan(connection.client)
        if space_id is None:
            print("Nobody scanned it. Run this again when you are ready.")
            return 1
        print("paired — opening the robot.\n")
        return run_demo(
            robot_name=args.robot,
            pair=None,
            use_llm=_llm_wanted(not args.no_llm),
            view=not args.no_window,
            speed=args.speed,
            send_images=not args.no_photos,
        )

    if args.cmd == "demo":
        _reexec_under_mjpython_if_needed(args.view)
        from .demo import run_demo

        return run_demo(
            robot_name=args.robot,
            pair=args.pair,
            use_llm=_llm_wanted(not args.no_llm),
            view=args.view,
            speed=args.speed,
            send_images=args.send_images,
        )

    return 1


if __name__ == "__main__":
    sys.exit(main())
