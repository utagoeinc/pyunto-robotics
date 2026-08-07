#!/usr/bin/env python3
"""Look at the simulation.

    python scripts/view_sim.py                    # interactive viewer, drive with --walk
    python scripts/view_sim.py --walk             # walk a lap of the corridor
    python scripts/view_sim.py --shot out.png     # save an overview + eye view instead
    python scripts/view_sim.py --scene scene.xml  # robot on an empty floor

In the interactive viewer: drag to orbit, scroll to zoom, double-click a body then Ctrl-drag
to shove it around.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

try:  # the interactive viewer is optional; --shot works without a display
    import mujoco.viewer

    HAVE_VIEWER = True
except ImportError:  # pragma: no cover
    HAVE_VIEWER = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyunto_robotics.sim.robot import Robot  # noqa: E402


def _patrol(t: float) -> tuple[float, float, float]:
    """A simple scripted route so there is something to watch."""
    phase = t % 16.0
    if phase < 5.0:
        return 0.7, 0.0, 0.0  # walk up the corridor
    if phase < 7.0:
        return 0.0, 0.0, 1.0  # turn
    if phase < 12.0:
        return 0.7, 0.0, 0.0  # walk back
    return 0.0, 0.0, 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="office.xml", help="scene file in assets/")
    ap.add_argument("--keyframe", default="start", help="starting keyframe")
    ap.add_argument("--walk", action="store_true", help="follow a scripted patrol route")
    ap.add_argument("--shot", metavar="PATH", help="save stills instead of opening a viewer")
    ap.add_argument("--seconds", type=float, default=60.0, help="how long to run")
    args = ap.parse_args()

    robot = Robot(args.scene, keyframe=args.keyframe)
    print(f"scene    : {args.scene}")
    print(f"actuators: {robot.model.nu}   bodies: {robot.model.nbody}")
    print(f"start    : pos={np.round(robot.position, 2)} yaw={math.degrees(robot.yaw):.0f} deg")

    if args.shot:
        from PIL import Image

        robot.stand(0.5)
        if args.walk:
            for _ in range(150):
                robot.step(vx=0.7)

        overview = mujoco.Renderer(robot.model, height=720, width=960)
        cam = "overview" if args.scene == "office.xml" else "track"
        overview.update_scene(robot.data, camera=cam)
        wide = overview.render()
        overview.close()

        eye = np.array(Image.fromarray(robot.look().rgb).resize((960, 720)))
        Image.fromarray(np.vstack([wide, eye])).save(args.shot)
        print(f"saved {args.shot} (overview above, robot's eye below)")
        robot.close()
        return 0

    if not HAVE_VIEWER:
        print("mujoco.viewer unavailable; use --shot to save stills instead")
        robot.close()
        return 1

    # On macOS the viewer needs the Cocoa event loop on the main thread, which only the
    # mjpython launcher (shipped with the mujoco wheel) sets up. Detect it the same way
    # mujoco.viewer does -- sys.executable still reads "python3" under mjpython, so checking
    # the interpreter name would wrongly reject a correct invocation.
    launched_by_mjpython = getattr(mujoco.viewer, "_MJPYTHON", None) is not None
    if sys.platform == "darwin" and not launched_by_mjpython:
        mjpython = Path(sys.executable).with_name("mjpython")
        launcher = mjpython if mjpython.exists() else Path("mjpython")
        # Keep the paths as the user typed them so the suggestion is copy-pasteable.
        script = sys.argv[0]
        args = " ".join(sys.argv[1:])
        print("\nThe interactive viewer needs mjpython on macOS. Run:")
        print(f"    {launcher} {script} {args}".rstrip())
        print("\nOr save stills instead, which works under plain python:")
        print(f"    {sys.executable} {script} --shot out.png {args}".rstrip())
        robot.close()
        return 1

    print("\nopening viewer - close the window to stop")
    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        start = time.time()
        while viewer.is_running() and (time.time() - start) < args.seconds:
            cmd = _patrol(robot.time) if args.walk else (0.0, 0.0, 0.0)
            robot.step(*cmd)
            viewer.sync()
            # Run at wall-clock speed so it is watchable.
            time.sleep(max(0.0, robot.control_dt - 0.001))
    robot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
