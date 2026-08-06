#!/usr/bin/env python3
"""Phase 1 verification: prove the robot can log in, read and write Pyunto messages.

Usage:
    python scripts/test_comms.py                 # login, list spaces/threads, read history
    python scripts/test_comms.py --send "hello"  # also post a message
    python scripts/test_comms.py --listen        # stream incoming messages (Ctrl-C to stop)
    python scripts/test_comms.py --join CODE     # join a human's space with an invite code

Credentials come from .env (see .env.example). Nothing is hardcoded.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyunto_robotics.comms.auth import AuthError, Session  # noqa: E402
from pyunto_robotics.comms.client import PyuntoClient  # noqa: E402
from pyunto_robotics.comms.keys import RawSpaceKeyProvider  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--send", metavar="TEXT", help="post this message")
    ap.add_argument("--thread", metavar="UUID", help="thread to post into (default: newest)")
    ap.add_argument("--space", metavar="UUID", help="chat space to use (default: first non-self)")
    ap.add_argument("--listen", action="store_true", help="stream incoming messages")
    ap.add_argument("--join", metavar="CODE", help="join a chat space with an invite code")
    ap.add_argument("--history", type=int, default=5, help="messages to show per thread")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    load_dotenv()

    email = os.getenv("PYUNTO_EMAIL")
    password = os.getenv("PYUNTO_PASSWORD")
    base_url = os.getenv("PYUNTO_BASE_URL", "https://api.pyunto.com")
    if not email or not password:
        print("ERROR: set PYUNTO_EMAIL and PYUNTO_PASSWORD (copy .env.example to .env)")
        return 2

    session = Session(base_url, email, password)
    try:
        identity = session.login()
    except AuthError as e:
        print(f"ERROR: {e}")
        return 1

    client = PyuntoClient(session, RawSpaceKeyProvider(session))
    print(f"\nrobot: {identity}\n")

    if args.join:
        client.join_with_code(args.join)

    spaces = client.list_spaces()
    print(f"chat spaces ({len(spaces)}):")
    for sp in spaces:
        kind = "self" if (sp.get("is_self") or sp.get("isSelf")) else "shared"
        members = sp.get("users") or sp.get("Users") or []
        names = ", ".join(u.get("display_name", "?") for u in members)
        print(f"  [{kind:6}] {sp.get('uuid')}  {sp.get('name')!r}  members: {names or '-'}")

    # Prefer a shared space -- that's where a human can talk to the robot.
    target = None
    if args.space:
        target = next((s for s in spaces if s.get("uuid", "").lower() == args.space.lower()), None)
    if target is None:
        target = next((s for s in spaces if not (s.get("is_self") or s.get("isSelf"))), None)
    if target is None and spaces:
        target = spaces[0]
    if target is None:
        print("\nno chat spaces. Use --join CODE with an invite code from the Pyunto app.")
        return 1

    space_id = target["uuid"]
    is_self = target.get("is_self") or target.get("isSelf")
    print(f"\nusing space {space_id} ({target.get('name')!r}{', self-space' if is_self else ''})")

    threads = client.list_threads(space_id, limit=50)
    print(f"threads ({len(threads)}):")
    for t in threads[:10]:
        # last_message_at is null for threads that only ever held an image, so coerce first.
        last = (t.get("last_message_at") or "")[:19] or "-"
        print(
            f"  {t.get('uuid')}  msgs={str(t.get('message_count', '?')):>3}  "
            f"last={last:19}  {str(t.get('title'))[:40]!r}"
        )

    if threads and args.history:
        newest = threads[0]
        print(f"\nlast {args.history} message(s) in {newest.get('uuid')}:")
        for m in client.get_messages(newest["uuid"], space_id)[-args.history:]:
            print(f"  {m}")

    if args.send:
        thread_id = args.thread or (threads[0]["uuid"] if threads else None)
        where = f"thread {thread_id}" if thread_id else "a NEW thread"
        print(f"\nsending to {where}: {args.send!r}")
        client.send(space_id, args.send, thread_id=thread_id)
        print("sent. Check the Pyunto app -- it should be readable there.")

    if args.listen:
        print("\nlistening for messages (Ctrl-C to stop)...")
        print("Send a message from the Pyunto app to see it arrive here.\n")

        def on_message(msg) -> None:  # noqa: ANN001
            print(f"  RECEIVED  {msg}")

        try:
            client.listen(on_message)
        except KeyboardInterrupt:
            client.stop()
            print("\nstopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
