#!/usr/bin/env python3
"""Kept so `python scripts/download_model.py` still works from a clone.

The real module moved into the package, because somebody who installed with pip has no
`scripts/` directory and the README was telling them to run a file they did not have.
"""

from pyunto_robotics.download_model import main

if __name__ == "__main__":
    raise SystemExit(main())
