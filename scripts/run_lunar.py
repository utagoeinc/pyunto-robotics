#!/usr/bin/env python3
"""Run R1, the lunar rover: it logs into Pyunto and does what you message it.

    python scripts/run_lunar.py                                   # listen for messages
    python scripts/run_lunar.py --llm                             # plan with Gemma 4
    python scripts/run_lunar.py --say "クレーターの縁まで行って"      # one instruction, no Pyunto
    python scripts/run_lunar.py --join CODE                       # join a chat space first

Then message the robot's account from the Pyunto app:

    「ビーコンまで行って」        -> it drives across the regolith to the survey beacon
    「氷まで行って」             -> it routes around the craters to the ice deposit
    「周りを見て」              -> it turns a full circle and reports what is lit and what is not

To watch it, launch with mjpython rather than python:

    ./.venv/bin/mjpython scripts/run_lunar.py --view --say "ビーコンまで行って"

Driving is slow here on purpose: lunar gravity is 1.62 m/s^2, so the rover has a sixth of the
traction it would have on Earth. `--speed 6` makes a long drive watchable.

Credentials come from .env (see .env.example).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyunto_robotics.runner import main, setup_for  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(setup_for("lunar"), __doc__ or ""))
