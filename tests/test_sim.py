"""Simulation tests: the robot model, the office, and the velocity interface.

These lock in the properties the demo depends on. If the humanoid stops standing, or its hands
stop reaching a door handle, everything downstream breaks in confusing ways.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from pyunto_robotics.sim.robot import Robot

HANDLE_HEIGHT = 0.90  # office door handles; the arm geometry was designed around this


@pytest.fixture(scope="module")
def robot():
    r = Robot("office.xml", keyframe="start")
    yield r
    r.close()


def _tilt_degrees(quat: np.ndarray) -> float:
    """Lean away from vertical, ignoring yaw (turning is not falling over)."""
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    up_z = rot.reshape(3, 3)[2, 2]
    return math.degrees(math.acos(np.clip(up_z, -1.0, 1.0)))


def test_office_loads_with_expected_structure():
    r = Robot("office.xml", keyframe="start")
    try:
        names = {
            mujoco.mj_id2name(r.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(r.model.nbody)
        }
        assert {"door_workspace", "door_meeting", "door_pantry"} <= names
        assert r.model.nu == 24  # 12 leg + 1 waist + 1 neck + 8 arm + 2 gripper
    finally:
        r.close()


def test_robot_stands_without_falling(robot):
    robot.reset("start")
    robot.stand(3.0)
    assert robot.position[2] > 0.7, "robot sank or fell through the floor"
    assert _tilt_degrees(robot.data.qpos[3:7]) < 15.0, "robot toppled"


def test_forward_command_moves_forward(robot):
    """A +vx command must move the robot along its own heading, not a world axis."""
    robot.reset("lobby")
    start = robot.position.copy()
    yaw = robot.yaw
    for _ in range(100):
        robot.step(vx=0.6)
    moved = robot.position - start

    travelled = math.hypot(moved[0], moved[1])
    assert travelled > 0.8, f"barely moved: {travelled:.2f} m"
    # Displacement should line up with the heading it started with.
    heading = np.array([math.cos(yaw), math.sin(yaw)])
    along = float(np.dot(moved[:2], heading))
    assert along > 0.8 * travelled, "moved sideways instead of forward"


def test_yaw_command_turns(robot):
    robot.reset("start")
    yaw0 = robot.yaw
    for _ in range(50):
        robot.step(wz=0.8)
    delta = (robot.yaw - yaw0 + math.pi) % (2 * math.pi) - math.pi
    assert delta > 0.3, f"did not turn left: {math.degrees(delta):.1f} deg"


def test_zero_command_holds_position(robot):
    robot.reset("start")
    start = robot.position.copy()
    for _ in range(100):
        robot.step()
    assert np.linalg.norm(robot.position - start) < 0.05, "drifted while commanded to hold"


def test_camera_returns_rgb_and_metric_depth(robot):
    robot.reset("start")
    robot.stand(0.2)
    obs = robot.look()

    assert obs.rgb.dtype == np.uint8 and obs.rgb.shape[2] == 3
    assert obs.depth.shape == obs.rgb.shape[:2]
    assert np.isfinite(obs.depth).all(), "depth must never contain NaN/inf"
    # Standing in the corridor there is always a wall within a few metres.
    assert 0.05 < float(obs.depth.min()) < 5.0


def test_bearing_sign_convention(robot):
    """Left of centre must be a positive bearing; the nav code relies on this."""
    obs = robot.look()
    width = obs.rgb.shape[1]
    assert robot.bearing_to_pixel(0) > 0
    assert robot.bearing_to_pixel(width) < 0
    assert abs(robot.bearing_to_pixel(width / 2)) < 1e-6


def test_hand_can_reach_door_handle_height(robot):
    """The whole demo depends on this: the gripper must get to ~0.90 m out in front."""
    robot.reset("start")
    robot.stand(0.2)

    best_forward = -1.0
    for pitch in np.linspace(-1.6, 0.2, 13):
        for elbow in np.linspace(-1.8, 0.0, 13):
            robot.set_arm("r", shoulder_pitch=float(pitch), shoulder_roll=0.0,
                          shoulder_yaw=0.0, elbow=float(elbow))
            for _ in range(20):
                robot.step()
            hand = robot.hand_position("r")
            if abs(hand[2] - HANDLE_HEIGHT) < 0.06:
                forward = np.dot((hand - robot.position)[:2],
                                 [math.cos(robot.yaw), math.sin(robot.yaw)])
                best_forward = max(best_forward, float(forward))

    assert best_forward > 0.20, (
        f"cannot reach a {HANDLE_HEIGHT} m handle far enough in front (best {best_forward:.3f} m)"
    )


def test_gripper_opens_and_closes(robot):
    robot.reset("start")
    joint = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, "grip_r1")
    adr = robot.model.jnt_qposadr[joint]

    robot.grip("r", 0.0)
    for _ in range(60):
        robot.step()
    opened = float(robot.data.qpos[adr])

    robot.grip("r", 1.0)
    for _ in range(60):
        robot.step()
    closed = float(robot.data.qpos[adr])

    assert closed > opened + 0.1, f"gripper did not close (open {opened:.3f}, closed {closed:.3f})"


def test_robot_cannot_pass_a_door_without_moving_it(robot):
    """The leaf is solid: getting to the far side requires actually swinging it.

    Walking into it torso-first does open it - the door is unlatched and light, so that is
    correct physics, not a bug. What must never happen is passing through while the hinge
    stays at zero, which would mean the collision was skipped entirely.
    """
    robot.reset("start")
    robot.arm_home("r")
    robot.arm_home("l")
    joint = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, "door_2")
    adr = robot.model.jnt_qposadr[joint]

    # Track the peak, not the final angle: the spring starts closing the door the moment the
    # robot stops leaning on it, so by the end of the run it has already swung back.
    peak_swing = 0.0
    for _ in range(200):
        robot.step(vx=0.8)
        peak_swing = max(peak_swing, abs(math.degrees(float(robot.data.qpos[adr]))))

    if robot.position[1] > 1.15:  # it got through
        assert peak_swing > 20.0, (
            f"passed the doorway (y={robot.position[1]:.2f}) without the door ever opening "
            f"(peak {peak_swing:.1f} deg) - collision was missed"
        )


def test_door_opens_when_pushed(robot):
    """A modest torque must swing the door; too stiff and the arm could never do it."""
    robot.reset("start")
    joint = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, "door_2")
    adr = robot.model.jnt_qposadr[joint]
    dof = robot.model.jnt_dofadr[joint]

    # Positive torque swings the leaf into the room; the range is 0..+1.9.
    for _ in range(100):
        robot.data.qfrc_applied[dof] = 5.0
        robot.step()
    angle = abs(math.degrees(float(robot.data.qpos[adr])))
    robot.data.qfrc_applied[dof] = 0.0
    assert angle > 20.0, f"door barely moved under 5 Nm: {angle:.1f} deg"


def test_grasp_does_not_teleport_the_target(robot):
    """Welding must freeze the CURRENT relative pose, not the one compiled into the model.

    Without writing eq_data at the moment of contact, the solver enforces the compiled offset
    and the door snaps into the robot's hand.
    """
    body = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "door_meeting")
    before = robot.data.xpos[body].copy()

    assert robot.grasp("door_meeting")
    after = robot.data.xpos[body].copy()

    assert np.linalg.norm(after - before) < 0.005, "door jumped when grasped"
    robot.release()


def test_grasp_rejects_unknown_bodies(robot):
    assert not robot.grasp("no_such_body")


def test_release_clears_every_grasp(robot):
    robot.grasp("door_meeting")
    robot.release()
    assert not robot.data.eq_active.any() or all(
        robot.model.eq_type[i] != mujoco.mjtEq.mjEQ_WELD or not robot.data.eq_active[i]
        for i in range(robot.model.neq)
    )


def test_doors_swing_both_ways(robot):
    """A door that only opens one way traps a robot that can only push."""
    joint = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, "door_2")
    low, high = robot.model.jnt_range[joint]
    assert low < -1.0, "door cannot open toward the corridor, so it can never be pulled"
    assert high > 1.0, "door cannot open into the room"


def test_side_cameras_see_what_the_front_one_cannot(robot):
    """A wall the robot walks alongside sits at 90 degrees, outside a 75-degree front view.

    This is the whole reason the side cameras exist: without them, scraping along a corridor
    reads as perfectly clear ahead.
    """
    robot.reset("start")
    # Stand close to the north wall (y=1.0), facing west along it.
    robot.data.qpos[0] = -2.0
    robot.data.qpos[1] = 0.75
    heading = math.pi
    robot.data.qpos[3:7] = [math.cos(heading / 2), 0, 0, math.sin(heading / 2)]
    mujoco.mj_forward(robot.model, robot.data)
    robot.gait.reset(robot.model, robot.data)
    robot.stand(0.3)

    left, right = robot.side_clearance()

    # Facing west with the wall to the north, the wall is on the robot's right.
    assert right < 0.8, f"side camera did not see the wall it is beside (right={right:.2f} m)"
    assert left > right, "the open side should read further than the wall side"


def test_side_clearance_is_symmetric_in_open_space(robot):
    """Standing in the middle of the lobby, both sides should read similar."""
    robot.reset("lobby")
    robot.stand(0.3)
    left, right = robot.side_clearance()
    assert abs(left - right) < 1.0, f"lopsided in open space: {left:.2f} vs {right:.2f}"
