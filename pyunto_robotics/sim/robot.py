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
        # Fill any procedural heightfields before the first step. A hfield declared in XML has
        # a size but no elevation data, so an outdoor scene loaded without this is dead flat.
        # Harmless for scenes with no heightfield, which is why it lives here rather than
        # being something every caller has to remember.
        from .terrain import apply as apply_terrain  # noqa: PLC0415 - avoids a circular import

        apply_terrain(self.model)
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
        # Built on first use by reach_to. Not every robot in this project has arms -- the
        # quadruped and the rover do not -- so constructing an IK solver eagerly would allocate
        # a scratch MjData for a chain that does not exist.
        self._solver: object | None = None
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

    def side_clearance(self, above_horizon: bool = False) -> tuple[float, float]:
        """How much room there is to the left and right, in metres.

        The forward camera cannot answer this. A wall the robot is walking alongside sits at
        90 degrees, outside a 75-degree field of view, so the clearance ahead stays comfortable
        the whole way down a corridor while the robot scrapes along the side of it. Two more
        head cameras turned 60 degrees out, 90 degrees each, cover the rest.

        Reports the 5th percentile rather than the minimum: the very closest pixel is often the
        robot's own shoulder at the edge of frame, and a percentile ignores that without
        needing to know the geometry.

        `above_horizon` reads only the top half of each frame -- walls, not floor. The full
        frame always contains floor within a stride or two, which caps the reading around
        1.6 m on BOTH sides however far the walls are. That is the right answer to "do I fit
        beside this" and the wrong one to "which wall is closer": centring on the full frame
        declared the middle of a 5 m corridor wherever the robot happened to stand.
        """
        left = self.look("look_left").depth
        right = self.look("look_right").depth
        if above_horizon:
            half = left.shape[0] // 2
            left, right = left[:half], right[:half]
        return float(np.percentile(left, 5)), float(np.percentile(right, 5))

    def wall_contact_side(self) -> float | None:
        """Which side of the body is pressed against a wall or door frame: +1 left, -1 right.

        A person feels a shoulder brush; this robot walked whole corridors pressed against a
        wall without any part of the control loop knowing -- the test harness counted 663
        control steps of contact in one errand, every one of them invisible to behaviour.
        Reads the simulator's contact list the way the door-touch check does; on a real robot
        this is what joint-current and IMU disturbance sensing are for.

        Door leaves are deliberately excluded: leaning on a leaf is how doors get opened, and
        a reflex that recoils from it would undo the push. Everything else counts. The first
        cut listed walls and frames by name, and the robot then spent two thousand steps
        pressed against the reception counter -- which is neither -- without noticing.
        """
        robot_parts = (
            "torso", "pelvis", "uarm", "farm", "palm", "fing", "thigh", "shin", "foot",
            "head", "neck", "visor", "chest",
        )
        left_axis = np.array([-math.sin(self.yaw), math.cos(self.yaw)])
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            n1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
            n2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
            if contact.dist >= 0 or "floor" in n1 or "floor" in n2:
                continue
            first_is_body = any(p in n1 for p in robot_parts)
            second_is_body = any(p in n2 for p in robot_parts)
            if first_is_body == second_is_body:  # self-contact, or two world geoms
                continue
            other = n2 if first_is_body else n1
            if "door" in other:  # the leaf; pushing it is deliberate
                continue
            lateral = float((contact.pos[:2] - self.position[:2]) @ left_axis)
            return 1.0 if lateral > 0 else -1.0
        return None

    @property
    def half_width(self) -> float:
        """How far the body sticks out sideways from its centre line, in metres.

        Measured from the model rather than declared, so it stays true if the robot changes.
        Currently 0.334 m, set by the forearms -- wider than the torso, which is why an
        approach that clears the chest can still catch an arm on a door frame.

        This is what turns "there is a wall 0.3 m to my left" into "I do not fit", and a real
        robot needs the same number for the same reason. It is a property, not a constant,
        because the arms move: reaching out makes the robot wider.
        """
        base = self.position[:2]
        cos_yaw, sin_yaw = math.cos(-self.yaw), math.sin(-self.yaw)
        root = self.model.body_rootid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        ]

        widest = 0.0
        for geom in range(self.model.ngeom):
            if self.model.body_rootid[self.model.geom_bodyid[geom]] != root:
                continue
            offset = self.data.geom_xpos[geom][:2] - base
            # Rotate into the body frame; y is the sideways axis.
            lateral = abs(cos_yaw * offset[1] - sin_yaw * offset[0])
            widest = max(widest, lateral + float(self.model.geom_size[geom].max()))
        return widest

    def arm_is_blocked(self) -> bool:
        """Whether either arm is pressed against something it cannot pull away from.

        An arm resting on a desk or held out by a door leaf will not come in when the servo is
        commanded home -- the contact wins -- and the robot stays at its widest. Knowing this
        is what lets it back off first and then tuck.
        """
        arm_parts = ("uarm", "farm", "palm", "fing")
        root = self.model.body_rootid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        ]
        for i in range(self.data.ncon):
            geoms = (self.data.contact.geom1[i], self.data.contact.geom2[i])
            names = [
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
                for geom in geoms
            ]
            if not any(any(p in n for p in arm_parts) for n in names):
                continue
            # Only scenery counts. Tucked-in arms rest against the robot's own chest, and
            # treating that as a jam meant the tuck reported failure the moment it succeeded --
            # measured "blocked" with a perfectly normal 0.330 m half-width, the contacts being
            # farm_r and farm_l against torso_g.
            if any(self.model.body_rootid[self.model.geom_bodyid[g]] != root for g in geoms):
                return True
        return False

    def is_touching(self, keyword: str) -> bool:
        """Whether the robot is in contact with scenery whose geom name contains `keyword`.

        The cameras answer "how much room is there", which is a different question: a doorway
        the robot is squarely inside reads as tight whether or not it is actually caught on the
        frame. Contact is a physical fact, and the simulator already knows it. A real robot
        would read this from bumper or joint-torque sensing, which is why it belongs here with
        the hardware rather than in the navigator.
        """
        for i in range(self.data.ncon):
            for geom in (self.data.contact.geom1[i], self.data.contact.geom2[i]):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
                if keyword in name:
                    return True
        return False

    # How far the neck can turn either way, in radians. Matches the joint range in the model.
    NECK_LIMIT_RAD = 1.75

    # Most the head may turn in one control step, radians -- about 4 rad/s at 50 Hz, which is
    # fast for a head but well short of a saccade. There is a real optimum here: measured wall
    # and frame contact over three approaches at 0.015 / 0.05 / 0.08 / 0.12 as 438 / 180 / 6 /
    # 206 steps. Too slow and the head lags behind the target the body is turning away from;
    # too fast and every frame is taken mid-swing, pairing a depth reading with a bearing the
    # camera has already left.
    NECK_RATE_RAD = 0.08

    @property
    def head_yaw(self) -> float:
        """Where the head is pointing relative to the body, in radians. + is left."""
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "neck_yaw")
        return float(self.data.qpos[self.model.jnt_qposadr[joint]])

    def look_at(self, world_point: np.ndarray) -> float:
        """Hold the head on a fixed point in the world, whatever the body is doing.

        This is the gaze-stabilising half of looking. Commanding a body-relative angle is not
        enough on its own: walking swings the torso through 66 degrees of yaw, and a head held
        at a constant angle to it swings with it, so the target crosses the frame twice a
        stride. Aiming at a world point instead makes each step's command absorb whatever the
        body just did -- which is what eyes do for a person, and why they can walk toward a
        door while looking straight at it.

        Returns the body-relative angle actually commanded.
        """
        delta = np.asarray(world_point)[:2] - self.position[:2]
        bearing = math.atan2(delta[1], delta[0]) - self.yaw
        return self.look_toward((bearing + math.pi) % (2 * math.pi) - math.pi)

    def look_toward(self, bearing: float) -> float:
        """Turn the head toward a body-relative bearing. Returns what it can actually reach.

        This is what lets the body walk one line while the eyes stay on another. Steering
        around a wall used to swing the cameras off the door being approached, and with three
        identical doors in the office the robot would come back to whichever was nearest --
        so the body could not avoid anything without losing track of where it was going.
        """
        target = float(np.clip(bearing, -self.NECK_LIMIT_RAD, self.NECK_LIMIT_RAD))
        index = self._act.get("neck_yaw")
        if index is not None:
            # Move the head at a bounded rate. Snapping it to a new angle each control step
            # means every frame is taken mid-swing, and a frame taken while the camera is
            # rotating pairs a depth reading with the wrong bearing: measured a tracked door
            # landing 1.71 m from where it actually is at a head angle of -28 degrees, against
            # 0.05 m with the head still. A person does not flick their eyes to a new target
            # every twentieth of a second either.
            current = float(self.data.ctrl[index])
            step = float(np.clip(target - current, -self.NECK_RATE_RAD, self.NECK_RATE_RAD))
            self.data.ctrl[index] = current + step
            return current + step
        return target

    def face_forward(self) -> None:
        """Bring the head back to straight ahead, and wait for it to get there."""
        self.turn_head_to(0.0)

    def turn_head_to(self, bearing: float, max_steps: int = 80) -> float:
        """Turn the head to a bearing and hold still until it arrives.

        look_toward only moves the head one step's worth, because a head that snaps to a new
        angle takes every frame mid-swing. So a caller that wants to *look* somewhere -- rather
        than to track something that is moving -- has to keep asking. Measured: a single call
        followed by stand(0.5) reached 4.4 degrees of a commanded 57.
        """
        target = float(np.clip(bearing, -self.NECK_LIMIT_RAD, self.NECK_LIMIT_RAD))
        for _ in range(max_steps):
            self.look_toward(target)
            self.step()
            if abs(self.head_yaw - target) < 0.02:
                break
        return self.head_yaw

    @property
    def camera_width(self) -> int:
        """Width of a rendered frame in pixels."""
        return self._renderer.width

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

    def grasp(self, body: str) -> bool:
        """Weld the right palm to `body`, freezing the current relative pose.

        A friction grasp does not hold in MuJoCo -- the fingers slip off a 3.6 cm handle long
        before the arm can move a 20 kg door leaf -- so a firm grip is modelled as a weld. The
        relative pose has to be written into eq_data at the moment of contact; without it the
        solver enforces whatever offset was compiled in and the door teleports into the hand.

        Returns False if there is no weld defined for that body.
        """
        eq = self._weld_for(body)
        if eq is None:
            return False

        palm = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "palm_r")
        target = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)

        # eq_data layout for a weld: [anchor(3), relpose(7: pos + quat), torquescale(1)].
        # Express body2's frame in body1's frame, which is what the solver holds constant.
        palm_pos, palm_quat = self.data.xpos[palm], self.data.xquat[palm]
        target_pos, target_quat = self.data.xpos[target], self.data.xquat[target]

        inv_palm = np.zeros(4)
        mujoco.mju_negQuat(inv_palm, palm_quat)
        rel_pos = np.zeros(3)
        mujoco.mju_rotVecQuat(rel_pos, target_pos - palm_pos, inv_palm)
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, inv_palm, target_quat)

        self.model.eq_data[eq, 0:3] = 0.0        # anchor at body1's origin
        self.model.eq_data[eq, 3:6] = rel_pos
        self.model.eq_data[eq, 6:10] = rel_quat
        self.model.eq_data[eq, 10] = 1.0         # torque scale
        self.data.eq_active[eq] = 1
        mujoco.mj_forward(self.model, self.data)
        return True

    def release(self) -> None:
        """Drop whatever a hand is welded to.

        Deactivates every weld naming a palm, rather than a hard-coded list of door welds. The
        office scene has three (`grasp_1..3`, one per door); the home scene has one per towel
        per hand plus one for the washer door, and a fixed list would silently leave those
        latched -- a robot that cannot let go of a towel is stuck for the rest of the errand.
        """
        palms = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"palm_{side}")
            for side in ("r", "l")
        }
        palms.discard(-1)
        for eq in range(self.model.neq):
            if self.model.eq_type[eq] != mujoco.mjtEq.mjEQ_WELD:
                continue
            if self.model.eq_obj1id[eq] in palms or self.model.eq_obj2id[eq] in palms:
                self.data.eq_active[eq] = 0
        mujoco.mj_forward(self.model, self.data)

    def _weld_for(self, body: str) -> int | None:
        """The equality index whose weld targets `body`, if any."""
        target = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if target < 0:
            return None
        for i in range(self.model.neq):
            if self.model.eq_type[i] != mujoco.mjtEq.mjEQ_WELD:
                continue
            if target in (self.model.eq_obj1id[i], self.model.eq_obj2id[i]):
                return i
        return None

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

    # -- reaching -----------------------------------------------------------------

    def reach_to(
        self,
        point: np.ndarray,
        side: str = "r",
        passes: int = 4,
        settle_steps: int = 150,
    ) -> float:
        """Put a gripper on a world point. Returns how close it got, in metres.

        A reach is a LOOP, not a single solve. Solving once and driving to the answer leaves
        the hand short, because the extended arm loads the torso and the body yields: measured
        an IK residual of 0.024 m arriving as a 0.20 m miss. Each pass re-measures where the
        hand actually is and asks for the remaining correction.

        Beyond about 0.25 m in front of the base the loop does not converge at all -- the error
        grew pass over pass -- so a caller that wants something further away has to walk closer
        rather than reach harder. See sim/reach.WORKING_REACH_M.

        The returned distance is the honest outcome and callers are expected to check it: a
        grasp aimed at a corner the hand never got to would weld thin air to the palm.
        """
        from .reach import ArmSolver  # noqa: PLC0415 - avoids a circular import at module load

        if self._solver is None:
            self._solver = ArmSolver(self.model)

        target = np.asarray(point, dtype=float)
        error = float("inf")
        for _ in range(max(1, passes)):
            angles, _ = self._solver.solve(self.data, target, side)
            if not angles:
                return float("inf")
            for joint, value in angles.items():
                index = self._act.get(joint)
                if index is None:
                    continue
                lo, hi = self.model.actuator_ctrlrange[index]
                self.data.ctrl[index] = float(np.clip(value, lo, hi))
            for _ in range(settle_steps):
                self.step()
            error = float(np.linalg.norm(self.hand_position(side) - target))
        return error
