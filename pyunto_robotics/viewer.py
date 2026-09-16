"""The simulator window.

Deliberately plain: the 3D scene and nothing else. MuJoCo's own panels (joint sliders, solver
statistics, contact visualisation) are for people debugging a physics model, and they make the
window look like an instrument rather than a robot. `show_left_ui=False, show_right_ui=False`
turns them off.

macOS needs `mjpython` rather than `python` to own a MuJoCo window. Rather than fail, we print
the exact command to run. `pyunto_robotics.cli` goes further and re-executes itself under
mjpython automatically, so the buyer's one command works as typed.
"""

from __future__ import annotations

import logging
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Camera:
    """Where the window looks from. Chosen per robot so the machine fills the frame."""

    distance: float = 4.0
    elevation: float = -20.0
    azimuth: float = 135.0
    height: float = 0.9  # metres above the robot's base to aim at


class ViewerHandle:
    """A live window. `sync()` redraws it; `pace()` keeps playback near real time."""

    def __init__(self, viewer, control_dt: float, speed: float = 1.0):
        self._viewer = viewer
        self._control_dt = control_dt
        self._speed = max(speed, 0.1)

    @property
    def running(self) -> bool:
        return bool(self._viewer.is_running())

    def sync(self) -> None:
        if self._viewer.is_running():
            self._viewer.sync()

    def pace(self) -> None:
        """Call once per simulation step: redraw, then wait so it does not run away."""
        self.sync()
        time.sleep(max(0.0, self._control_dt / self._speed - 0.002))

    def close(self) -> None:
        try:
            self._viewer.close()
        except Exception:  # noqa: BLE001 - closing a window must never be fatal
            pass


def mjpython_hint() -> str | None:
    """The command to rerun under, or None if the window can be opened as-is."""
    if sys.platform != "darwin":
        return None
    try:
        import mujoco.viewer
    except ImportError:
        return None
    if getattr(mujoco.viewer, "_MJPYTHON", None) is not None:
        return None  # already running under mjpython
    launcher = Path(sys.executable).with_name("mjpython")
    argv = " ".join(shlex.quote(a) for a in sys.argv[1:])
    return f"{launcher} {sys.argv[0]} {argv}".rstrip()


def open_viewer(robot, speed: float = 1.0, camera: Camera | None = None) -> ViewerHandle | None:
    """Open the plain simulator window, or return None with an explanation printed."""
    try:
        import mujoco.viewer
    except ImportError:
        print("The simulator window needs the mujoco package; running without a window.")
        return None

    hint = mjpython_hint()
    if hint:
        print("\nThe simulator window needs mjpython on macOS. Run:")
        print(f"    {hint}\n")
        return None

    viewer = mujoco.viewer.launch_passive(
        robot.model,
        robot.data,
        show_left_ui=False,
        show_right_ui=False,
    )

    cam = camera or Camera()
    viewer.cam.distance = cam.distance
    viewer.cam.elevation = cam.elevation
    viewer.cam.azimuth = cam.azimuth
    try:
        base = robot.position()
        viewer.cam.lookat[:] = [base[0], base[1], cam.height]
    except Exception:  # noqa: BLE001 - framing is cosmetic; never fail the run over it
        log.debug("could not read the robot position for framing", exc_info=True)
    viewer.sync()

    handle = ViewerHandle(viewer, robot.control_dt, speed)
    # Redraw during long actions too, not just while idle: a walk across a room is one
    # blocking call, and a window that only updates between messages looks frozen.
    robot.on_step = handle.pace
    return handle
