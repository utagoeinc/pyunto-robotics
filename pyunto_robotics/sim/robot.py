"""The simulated robot: physics, sensing, and the arm.

This is the boundary the rest of the system talks to. Navigation says `step(vx, vy, wz)` and
`look()`; manipulation says `reach()` and `grip()`. Nothing above here knows about MuJoCo
joint names, and nothing knows whether the gait is kinematic or a trained policy.

Camera performance on this machine, measured rather than assumed: 224x224 RGB renders in
~1.4-2.0 ms and depth in ~0.6-0.9 ms, so perception can run as fast as the planner wants.
The bottleneck is the vision model, not the renderer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .gait import Gait, KinematicGait

ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"

# Depth beyond this is treated as "no return". mac OpenGL lacks ARB_clip_control, so far-field
# depth precision is poor; the navigation logic only ever needs the near field anyway.
MAX_DEPTH_M = 12.0


@dataclass
class Observation:
    """One perception frame from the robot's head camera."""

    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32, metres; MAX_DEPTH_M where nothing was hit
    position: np.ndarray  # (3,) world position of the base
    yaw: float  # world heading, radians

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.depth.shape
        return w, h


class Robot:
    """A humanoid in a MuJoCo scene."""

    def __init__(
        self,
        scene: str | Path = "office.xml",
        gait: Gait | None = None,
        keyframe: str | None = "start",
        cam_width: int = 424,
        cam_height: int = 320,
        control_hz: float = 50.0,
    ):
        path = Path(scene)
        if not path.is_absolute():
            path = ASSETS / path
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)

        self.gait: Gait = gait if gait is not None else KinematicGait()
        self.control_dt = 1.0 / control_hz
        self._steps_per_control = max(1, round(self.control_dt / self.model.opt.timestep))

        self._renderer = mujoco.Renderer(self.model, height=cam_height, width=cam_width)
        self._depth_renderer = mujoco.Renderer(self.model, height=cam_height, width=cam_width)
        self._depth_renderer.enable_depth_rendering()

        self._act = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(self.model.nu)
        }
        self.reset(keyframe)

    # -- lifecycle ----------------------------------------------------------------

    def reset(self, keyframe: str | None = "start") -> None:
        """Return to a named keyframe (or the default pose)."""
        if keyframe:
            kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
            if kid >= 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
            else:
                mujoco.mj_resetData(self.model, self.data)
        else:
            mujoco.mj_resetData(self.model, self.data)

        # Servos hold whatever pose we just loaded, otherwise the robot collapses on step 1.
        for name, idx in self._act.items():
            joint = self.model.actuator_trnid[idx, 0]
            self.data.ctrl[idx] = self.data.qpos[self.model.jnt_qposadr[joint]]

        mujoco.mj_forward(self.model, self.data)
        self.gait.reset(self.model, self.data)

    def close(self) -> None:
        self._renderer.close()
        self._depth_renderer.close()

    def __enter__(self) -> Robot:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- state --------------------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        """World position of the base."""
        return self.data.qpos[0:3].copy()

    @property
    def yaw(self) -> float:
        """World heading in radians. 0 = facing +x."""
        w, x, y, z = self.data.qpos[3:7]
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @property
    def time(self) -> float:
        return float(self.data.time)

    # -- locomotion ---------------------------------------------------------------

    def step(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> None:
        """Advance one control interval at the requested body-frame velocity.

        vx forward (m/s), vy left (m/s), wz yaw rate (rad/s). This is the only movement
        primitive the rest of the system uses.
        """
        self.gait.apply(self.model, self.data, vx, vy, wz, self.control_dt)
        for _ in range(self._steps_per_control):
            mujoco.mj_step(self.model, self.data)

    def stand(self, seconds: float = 0.5) -> None:
        """Hold still, letting the physics settle."""
        for _ in range(max(1, round(seconds / self.control_dt))):
            self.step(0.0, 0.0, 0.0)

    # -- sensing ------------------------------------------------------------------

    def look(self, camera: str = "head_cam") -> Observation:
        """Capture one RGB-D frame from the robot's point of view."""
        self._renderer.update_scene(self.data, camera=camera)
        rgb = self._renderer.render().copy()

        self._depth_renderer.update_scene(self.data, camera=camera)
        depth = self._depth_renderer.render().copy()
        # MuJoCo returns the far-plane value for rays that hit nothing; normalise that to a
        # single sentinel so downstream code has one thing to test for.
        depth[~np.isfinite(depth)] = MAX_DEPTH_M
        depth = np.clip(depth, 0.0, MAX_DEPTH_M)

        return Observation(rgb=rgb, depth=depth, position=self.position, yaw=self.yaw)

    def camera_fovy(self, camera: str = "head_cam") -> float:
        """Vertical field of view in degrees.

        Always ask for it by name -- cam_fovy[0] is whichever camera happens to be declared
        first in the scene, which is not the robot's.
        """
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        return float(self.model.cam_fovy[cid])

    def camera_intrinsics(self, camera: str = "head_cam") -> tuple[float, float, float]:
        """(fx, cx, cy) in pixels, derived from the camera's vertical FOV."""
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        fovy_deg = float(self.model.cam_fovy[cid])
        h = self._renderer.height
        w = self._renderer.width
        fy = (h / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
        return fy, w / 2.0, h / 2.0

    def bearing_to_pixel(self, u: float, camera: str = "head_cam") -> float:
        """Horizontal angle (radians, +left) to an image column.

        This is what turns "the door is at x=0.62 in the image" into "turn 8 degrees right".
        """
        f, cx, _ = self.camera_intrinsics(camera)
        return -math.atan2(u - cx, f)

    # -- manipulation -------------------------------------------------------------

    def set_arm(
        self,
        side: str = "r",
        shoulder_pitch: float | None = None,
        shoulder_roll: float | None = None,
        shoulder_yaw: float | None = None,
        elbow: float | None = None,
    ) -> None:
        """Command arm joint angles directly (radians). Unset joints keep their target."""
        targets = {
            f"sh_pitch_{side}": shoulder_pitch,
            f"sh_roll_{side}": shoulder_roll,
            f"sh_yaw_{side}": shoulder_yaw,
            f"elbow_{side}": elbow,
        }
        for joint, value in targets.items():
            if value is None:
                continue
            idx = self._act.get(joint)
            if idx is None:
                continue
            lo, hi = self.model.actuator_ctrlrange[idx]
            self.data.ctrl[idx] = float(np.clip(value, lo, hi))

    def grip(self, side: str = "r", closed: float = 1.0) -> None:
        """Close (1.0) or open (0.0) a gripper."""
        idx = self._act.get(f"grip_{side}")
        if idx is None:
            return
        lo, hi = self.model.actuator_ctrlrange[idx]
        self.data.ctrl[idx] = float(lo + (hi - lo) * np.clip(closed, 0.0, 1.0))

    def hand_position(self, side: str = "r") -> np.ndarray:
        """World position of a gripper tip."""
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"grip_{side}")
        return self.data.site_xpos[sid].copy()

    def arm_home(self, side: str = "r") -> None:
        """Return the arm to its resting pose."""
        sign = -1.0 if side == "r" else 1.0
        self.set_arm(side, shoulder_pitch=-0.25, shoulder_roll=sign * 0.12,
                     shoulder_yaw=0.0, elbow=-0.35)
        self.grip(side, 0.0)
