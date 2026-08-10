#!/usr/bin/env python3
"""Run Q1, the four-legged patrol robot: it logs into Pyunto and does what you message it.

    python scripts/run_patrol.py                              # listen for messages
    python scripts/run_patrol.py --llm                        # plan with Gemma 4
    python scripts/run_patrol.py --say "ビルの周りを1周して"     # one instruction, no Pyunto
    python scripts/run_patrol.py --join CODE                  # join a chat space first

Then message the robot's account from the Pyunto app:

    「ビルの周りを1周して」  -> it walks the four corners of the site and reports what it saw
    「階段を上って」        -> it climbs the three steps to the building entrance

To watch it, launch with mjpython rather than python:

    ./.venv/bin/mjpython scripts/run_patrol.py --view --say "ビルの周りを1周して"

Credentials come from .env (see .env.example).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyunto_robotics.runner import main, setup_for  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(setup_for("patrol"), __doc__ or ""))
