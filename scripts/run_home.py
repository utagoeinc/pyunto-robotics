#!/usr/bin/env python3
"""Run Momo, the home assistant: she logs into Pyunto and does what you message her.

    python scripts/run_home.py                                    # listen for messages
    python scripts/run_home.py --llm                              # plan with Gemma 4
    python scripts/run_home.py --say "タオルを洗濯機から出して畳んで"    # one instruction, no Pyunto
    python scripts/run_home.py --join CODE                        # join a chat space first

Then message the robot's account from the Pyunto app:

    「タオルを洗濯機から出して畳んで」 -> she opens the washer, takes the towel out,
                                       lays it on the counter and folds it

To watch her do it, launch with mjpython rather than python:

    ./.venv/bin/mjpython scripts/run_home.py --view --llm --say "タオルを出して畳んで"

Credentials come from .env (see .env.example).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyunto_robotics.runner import main, setup_for  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(setup_for("home"), __doc__ or ""))
