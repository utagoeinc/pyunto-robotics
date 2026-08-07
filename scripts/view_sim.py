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
import shlex
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
    """A canned route: forward, turn, forward, turn.

    This is NOT the robot thinking -- it ignores the camera entirely and just replays a
    timed sequence of velocity commands, so it is only useful for checking that the gait
    and the collision handling look right. Use --say to watch the robot actually perceive
    and decide.
    """
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
    ap.add_argument("--walk", action="store_true",
                    help="replay a canned route (ignores the camera; just checks the gait)")
    ap.add_argument("--say", metavar="TEXT",
                    help="carry out an instruction while you watch, e.g. \"open the right door\"")
    ap.add_argument("--llm", action="store_true", help="plan with Gemma 4 instead of rules")
    ap.add_argument("--shot", metavar="PATH", help="save stills instead of opening a viewer")
    ap.add_argument("--seconds", type=float, default=180.0, help="how long to run")
    ap.add_argument("--speed", type=float, default=2.0,
                    help="playback speed for --say (1 = real time; higher finishes sooner)")
    args = ap.parse_args()

    robot = Robot(args.scene, keyframe=args.keyframe)
    print(f"scene    : {args.scene}")
    if args.say:
        print(f"instruct : {args.say!r}")
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
        # shlex.quote each argument: instructions contain spaces and Japanese punctuation,
        # so an unquoted suggestion cannot be pasted back in.
        quoted = " ".join(shlex.quote(a) for a in sys.argv[1:])
        print("\nThe interactive viewer needs mjpython on macOS. Run:")
        print(f"    {launcher} {script} {quoted}".rstrip())
        print("\nOr save stills instead, which works under plain python:")
        print(f"    {sys.executable} {script} --shot out.png {quoted}".rstrip())
        robot.close()
        return 1

    print("\nopening viewer - close the window to stop")

    if args.say:
        return _run_instruction_in_viewer(robot, args)

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


def _run_instruction_in_viewer(robot: Robot, args: argparse.Namespace) -> int:
    """Carry out one instruction while the viewer redraws.

    Everything runs on THIS thread. MuJoCo's offscreen renderer is bound to the thread that
    created it, so calling Robot.look() from a worker kills the process outright on macOS --
    no exception, just a Metal assertion. Instead the viewer is refreshed from inside the
    robot's own step loop, by wrapping Robot.step.

    Stepping is also throttled to wall-clock speed; without that the whole errand finishes in
    about a second and there is nothing to watch.
    """
    from pyunto_robotics.brain.planner import LLMPlanner, RulePlanner  # noqa: PLC0415
    from pyunto_robotics.brain.skills import Skills  # noqa: PLC0415
    from pyunto_robotics.perception.grounding import ColorGrounder  # noqa: PLC0415

    planner = LLMPlanner() if args.llm else RulePlanner()
    skills = Skills(robot, ColorGrounder())

    plan = planner.plan(args.say)
    print(f"plan     : {plan}")
    if not plan.steps:
        print(f"reply    : {plan.reply}")
        robot.close()
        return 0

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        deadline = time.time() + args.seconds
        real_step = robot.step

        def step_and_draw(*a, **kw):
            real_step(*a, **kw)
            viewer.sync()
            # Pace the playback. Real time is the most natural to watch, but a full errand to
            # a far door takes over a minute, so the default runs a little fast.
            time.sleep(max(0.0, robot.control_dt / max(args.speed, 0.1) - 0.002))
            if not viewer.is_running() or time.time() > deadline:
                raise _ViewerClosed

        robot.step = step_and_draw  # type: ignore[method-assign]

        messages = []
        try:
            for step in plan.steps[:4]:
                result = skills.run(step.action, step.argument, step.where)
                messages.append(result.message)
                if not result.ok:
                    break
        except _ViewerClosed:
            messages.append("(stopped early)")
        finally:
            robot.step = real_step  # type: ignore[method-assign]

        print(f"reply    : {' '.join(messages)}")
        # Hold the final pose briefly so the result is visible.
        end = time.time() + 3.0
        while viewer.is_running() and time.time() < end:
            viewer.sync()
            time.sleep(1 / 60)

    robot.close()
    return 0


class _ViewerClosed(Exception):
    """Raised inside the step loop when the window is closed, to unwind the skill cleanly."""


if __name__ == "__main__":
    raise SystemExit(main())
