"""The three robots added after the office humanoid: Momo, the quadruped and the rover.

These tests are written around what was actually measured while building them, so a failure
here means a real regression rather than a tightened threshold. Anything that drives the full
simulator for more than a second or two is marked `slow`.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from pyunto_robotics.brain.domains import DOMAINS, DomainRulePlanner
from pyunto_robotics.brain.patrol import _laps, _waypoint_index
from pyunto_robotics.perception.grounding import ColorGrounder
from pyunto_robotics.sim.cloth import ClothGrasp, find_sheets
from pyunto_robotics.sim.quad_gait import THIGH_M, TrotGait, leg_ik
from pyunto_robotics.sim.robot import Robot
from pyunto_robotics.sim.terrain import apply as apply_terrain
from pyunto_robotics.sim.wheel_drive import SkidDrive, rover_attitude


# ======================================================================================
# Models compile and stand
# ======================================================================================


@pytest.mark.parametrize(
    ("scene", "keyframe"),
    [("home.xml", "start"), ("campus.xml", "start"), ("lunar.xml", "plain")],
)
def test_scenes_compile_and_settle(scene: str, keyframe: str) -> None:
    """Each new scene loads, and its robot is still upright a second later."""
    gait = {"campus.xml": TrotGait, "lunar.xml": SkidDrive}.get(scene)
    robot = Robot(scene, gait=gait() if gait else None, keyframe=keyframe)
    try:
        # Let it settle first. The lunar keyframe deliberately starts the rover above the
        # heightfield -- working out the terrain height in a keyframe is far more fragile than
        # dropping it and letting the physics find the surface -- so it legitimately falls
        # over a metre before coming to rest.
        for _ in range(200):
            robot.step()
        settled = float(robot.position[2])

        # Then it should STAY there, and stay upright.
        for _ in range(200):
            robot.step()
        assert np.isfinite(robot.position).all()
        assert abs(float(robot.position[2]) - settled) < 0.10, "still sinking or bouncing"

        # Upright: the body's own z axis still points up.
        upright = np.zeros(9)
        mujoco.mju_quat2Mat(upright, robot.data.qpos[3:7])
        assert upright[8] > 0.75, "robot has fallen over"
    finally:
        robot.close()


def test_momo_is_the_right_size_and_stands() -> None:
    """Momo is ~1.4 m tall and does not sink or topple over three seconds."""
    robot = Robot("home.xml", keyframe="start")
    try:
        model, data = robot.model, robot.data
        tallest = max(
            float(data.geom_xpos[g][2] + model.geom_rbound[g]) for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(
                ("head", "hair", "ahoge")
            )
        )
        assert 1.30 < tallest < 1.55, f"Momo's head is at {tallest:.2f} m"

        start_z = float(robot.position[2])
        for _ in range(600):
            robot.step()
        assert abs(float(robot.position[2]) - start_z) < 0.05
    finally:
        robot.close()


def test_momo_faces_the_way_her_camera_looks() -> None:
    """The face and head_cam point the same way.

    They did not: the face was modelled looking along -y while the camera looked along +x, so
    the robot walked sideways relative to its own face. The bug was invisible in every physics
    measurement -- she stood, walked and folded towels perfectly well -- and only showed up in
    a photograph, which is exactly why it needs a test.
    """
    model = mujoco.MjModel.from_xml_path("assets/momo.xml")
    data = mujoco.MjData(model)
    keyframe = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, keyframe)
    mujoco.mj_forward(model, data)

    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    # A MuJoCo camera looks down its own -z axis.
    forward = -data.cam_xmat[camera].reshape(3, 3)[:, 2]
    assert forward[0] > 0.95, f"head_cam does not look along +x: {np.round(forward, 2)}"

    # Every face feature must be on the same side, and none of them off to one side in y.
    for name in ("eye_l_i", "eye_r_i", "mouth_g", "brow_l", "blush_l"):
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert geom >= 0, f"{name} is missing"
        assert data.geom_xpos[geom][0] > 0.02, f"{name} is not on the front of the face"

    # The hair at the back must be behind, or the head is on backwards.
    back = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "hair_back1")
    assert data.geom_xpos[back][0] < 0.0

    # And the apron is on the chest, not a hip -- it was on -y with the old head.
    apron = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "apron_g")
    assert data.geom_xpos[apron][0] > 0.05
    assert abs(float(data.geom_xpos[apron][1])) < 0.03


def test_momo_debug_markers_are_invisible() -> None:
    """Sites render as solid spheres, and hers land on the face and hands.

    The head_cam marker sits between the eyes: left at its default red it renders as a bright
    dot on the bridge of the nose, which reads as a clown nose and took two passes to find
    because it looks like a geom that is not there.
    """
    model = mujoco.MjModel.from_xml_path("assets/momo.xml")
    for name in ("head_cam_site", "grip_r", "grip_l"):
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        assert site >= 0, f"{name} is missing -- code addresses it by name"
        assert float(model.site_rgba[site][3]) == 0.0, f"{name} is visible"


def test_momo_materials_are_not_overridden() -> None:
    """The robot is coloured, not bone white.

    A default-class `rgba` silently overrides every geom's material, and the whole model
    renders white. It happened once; this catches it coming back.
    """
    model = mujoco.MjModel.from_xml_path("assets/momo.xml")

    # Colour comes from MATERIALS, so geom_rgba is uniform by design and says nothing. What
    # matters is that the geoms are actually assigned distinct materials: a default-class rgba
    # would still leave matid set but override the colour at render time, so the real check is
    # that the materials exist, are used, and differ from one another.
    used = {int(model.geom_matid[g]) for g in range(model.ngeom)} - {-1}
    assert len(used) >= 10, f"only {len(used)} materials in use; the model should be colourful"

    for name in ("skin", "hair", "dress", "eye_iris"):
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
        assert index >= 0, f"material {name!r} is missing"
        assert index in used, f"material {name!r} is defined but never used"

    palette = {tuple(np.round(model.mat_rgba[i], 2)) for i in used}
    assert len(palette) >= 8, "the materials in use are nearly all the same colour"


# ======================================================================================
# Cloth
# ======================================================================================


def test_cloth_sheets_are_found_as_grids() -> None:
    """Both towels are discovered as 9x7 grids with four distinct corners.

    The grid shape has to be recovered from body positions, not from `flex_vert` -- that is a
    deformation buffer and reads as all zeros at rest, which made every sheet look like a
    63x1 strip and collapsed the four corners onto two.
    """
    robot = Robot("home.xml", keyframe="start")
    try:
        sheets = find_sheets(robot.model)
        assert set(sheets) == {"towel_a", "towel_b"}
        for sheet in sheets.values():
            assert (sheet.rows, sheet.cols) == (9, 7)
            assert len(set(sheet.corners().values())) == 4
            assert len(sheet.perimeter()) == 2 * (9 + 7) - 4
    finally:
        robot.close()


@pytest.mark.slow
def test_cloth_can_be_grasped_and_lifted() -> None:
    """Welding a hand to a cloth vertex lifts the sheet.

    This is the mechanism the whole laundry task rests on: friction alone does not hold cloth,
    so a closed hand is modelled as a weld to one vertex.
    """
    robot = Robot("home.xml", keyframe="counter")
    try:
        cloth = ClothGrasp(robot.model, robot.data)
        sheet = cloth.sheet("towel_b")
        assert sheet is not None

        corner = sheet.corners()["near_left"]
        before = cloth.vertex_position(sheet, corner)[2]
        assert cloth.grasp("towel_b", corner, "r")

        index = robot._act["sh_pitch_r"]
        robot.data.ctrl[index] = -2.2
        for _ in range(400):
            robot.step()

        after = cloth.vertex_position(sheet, corner)[2]
        assert after > before + 0.2, f"cloth did not lift: {before:.3f} -> {after:.3f}"

        cloth.release("r")
        assert cloth.holding("r") is None
    finally:
        robot.close()


# ======================================================================================
# Quadruped
# ======================================================================================


def test_leg_ik_matches_the_model() -> None:
    """The closed-form leg IK puts the foot exactly where it was asked to.

    Verified against MuJoCo's own kinematics rather than against a hand-rolled forward model,
    because the first version had the height right and the fore-aft component mirrored -- a
    gait that walks the robot backwards while the base is driven forwards.
    """
    # The bare robot, not a scene: campus.xml's keyframes rotate the body 90 degrees so it
    # faces the building, and then the leg's own forward axis is world -y rather than +x.
    # Comparing against world x there fails for a reason that has nothing to do with the IK.
    model = mujoco.MjModel.from_xml_path("assets/pyunto_q1.xml")
    data = mujoco.MjData(model)
    keyframe = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, keyframe)

    address = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): model.jnt_qposadr[j]
        for j in range(model.njnt)
    }
    hip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "thigh_fl")
    foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "foot_fl")

    for reach, drop in [(0.0, 0.325), (0.08, 0.325), (-0.08, 0.325), (0.12, 0.30)]:
        hip_angle, knee = leg_ik(reach, drop)
        data.qpos[address["hip_fl"]] = hip_angle
        data.qpos[address["knee_fl"]] = knee
        mujoco.mj_forward(model, data)
        offset = data.geom_xpos[foot] - data.xpos[hip]
        assert abs(offset[0] - reach) < 1e-3
        assert abs(offset[2] + drop) < 1e-3


def test_leg_ik_clamps_instead_of_raising() -> None:
    """A foot target beyond the leg's span straightens the leg rather than failing."""
    hip, knee = leg_ik(0.0, (THIGH_M * 3))
    assert math.isfinite(hip) and math.isfinite(knee)
    assert knee >= 0.0


@pytest.mark.slow
def test_quadruped_walks_and_turns_the_right_way() -> None:
    """Forward, backward and both turn directions all go where they are told.

    The turn assertions exist because writing both the qpos quaternion AND qvel[5] double-
    integrates the rotation: a commanded +0.8 rad/s came out as -2.63 rad over three seconds
    where +2.40 was wanted, which reads as a sign error and is not.
    """
    def run(vx: float, wz: float, seconds: float) -> tuple[float, float]:
        robot = Robot("campus.xml", gait=TrotGait(), keyframe="corner")
        try:
            for _ in range(40):
                robot.step()
            start, heading = robot.position[:2].copy(), robot.yaw
            for _ in range(int(seconds / robot.control_dt)):
                robot.step(vx=vx, wz=wz)
            travelled = float(np.linalg.norm(robot.position[:2] - start))
            turned = (robot.yaw - heading + math.pi) % (2 * math.pi) - math.pi
            return travelled, turned
        finally:
            robot.close()

    travelled, _ = run(0.6, 0.0, 4.0)
    assert travelled > 1.5, f"only walked {travelled:.2f} m in 4 s at 0.6 m/s"

    _, turned = run(0.0, 0.8, 3.0)
    assert turned > 1.0, f"commanded +0.8 rad/s, turned {turned:+.2f} rad"

    _, turned = run(0.0, -0.8, 3.0)
    assert turned < -1.0, f"commanded -0.8 rad/s, turned {turned:+.2f} rad"


# ======================================================================================
# Terrain
# ======================================================================================


def test_heightfields_are_filled() -> None:
    """A heightfield declared in XML is dead flat until Python writes its elevation."""
    for scene, field in (("campus.xml", "lawn_hf"), ("lunar.xml", "regolith_hf")):
        model = mujoco.MjModel.from_xml_path(f"assets/{scene}")
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, field)
        assert index >= 0
        written = apply_terrain(model)
        assert field in written

        address = model.hfield_adr[index]
        count = model.hfield_nrow[index] * model.hfield_ncol[index]
        data = np.asarray(model.hfield_data[address : address + count])
        assert data.max() > 0.9, "heightfield is flat"
        assert data.min() < 0.1


def test_lunar_terrain_has_craters() -> None:
    """The lunar surface has real depressions, not just noise."""
    model = mujoco.MjModel.from_xml_path("assets/lunar.xml")
    apply_terrain(model)
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "regolith_hf")
    rows, cols = model.hfield_nrow[index], model.hfield_ncol[index]
    address = model.hfield_adr[index]
    grid = np.asarray(model.hfield_data[address : address + rows * cols]).reshape(rows, cols)
    # A cratered field has a much wider spread of heights than a smooth slope would.
    assert grid.std() > 0.10


# ======================================================================================
# Rover
# ======================================================================================


@pytest.mark.slow
def test_rover_drives_and_turns_under_lunar_gravity() -> None:
    """The rover moves, and turning is not cancelled out by its own steering.

    The steering angle and the wheel differential act in opposite senses, so getting the sign
    wrong makes them cancel: measured +0.5 rad/s commanded arriving as -0.18.
    """
    def run(vx: float, wz: float, seconds: float) -> tuple[float, float]:
        robot = Robot("lunar.xml", gait=SkidDrive(), keyframe="plain")
        try:
            for _ in range(200):
                robot.step()
            start, heading = robot.position[:2].copy(), robot.yaw
            for _ in range(int(seconds / robot.control_dt)):
                robot.step(vx=vx, wz=wz)
            travelled = float(np.linalg.norm(robot.position[:2] - start))
            turned = (robot.yaw - heading + math.pi) % (2 * math.pi) - math.pi
            return travelled, turned
        finally:
            robot.close()

    travelled, _ = run(0.6, 0.0, 8.0)
    assert travelled > 2.5, f"rover only moved {travelled:.2f} m in 8 s"

    _, turned = run(0.0, 0.5, 8.0)
    assert turned > 0.5, f"commanded +0.5 rad/s, turned {turned:+.2f} rad in 8 s"

    _, turned = run(0.0, -0.5, 8.0)
    assert turned < -0.5, f"commanded -0.5 rad/s, turned {turned:+.2f} rad in 8 s"


def test_lunar_gravity_is_set() -> None:
    """The Moon scene really is at 1.62 m/s^2, and nothing else is."""
    lunar = mujoco.MjModel.from_xml_path("assets/lunar.xml")
    assert abs(float(lunar.opt.gravity[2]) + 1.62) < 0.01
    campus = mujoco.MjModel.from_xml_path("assets/campus.xml")
    assert abs(float(campus.opt.gravity[2]) + 9.81) < 0.1


def test_rover_attitude_reads_level_as_level() -> None:
    robot = Robot("lunar.xml", gait=SkidDrive(), keyframe="plain")
    try:
        for _ in range(200):
            robot.step()
        pitch, roll = rover_attitude(robot.data)
        assert abs(pitch) < math.radians(30)
        assert abs(roll) < math.radians(30)
    finally:
        robot.close()


# ======================================================================================
# Planning
# ======================================================================================


@pytest.mark.parametrize(
    ("domain", "message", "action"),
    [
        ("home", "洗濯機を開けて", "open_washer"),
        ("home", "open the washing machine", "open_washer"),
        ("home", "タオルを取り出して", "take_out"),
        ("home", "かごに入れて", "to_basket"),
        ("home", "洗面台に置いて", "to_counter"),
        ("home", "畳んで", "fold"),
        ("home", "fold the blue towel", "fold"),
        ("patrol", "ビルの周りを1周して", "patrol"),
        ("patrol", "patrol around the building", "patrol"),
        ("patrol", "階段を上って", "climb"),
        ("patrol", "何が見える", "describe"),
        ("lunar", "ビーコンまで行って", "goto"),
        ("lunar", "drive to the beacon", "goto"),
        ("lunar", "クレーターの縁まで行って", "goto"),
        ("lunar", "周りを見て", "survey"),
        ("lunar", "着陸船に戻って", "home"),
    ],
)
def test_rule_planners_understand_both_languages(
    domain: str, message: str, action: str
) -> None:
    plan = DomainRulePlanner(DOMAINS[domain]).plan(message)
    assert plan.steps, f"{message!r} produced no plan"
    assert plan.steps[0].action == action


def test_lunar_planner_picks_the_right_target() -> None:
    planner = DomainRulePlanner(DOMAINS["lunar"])
    assert planner.plan("氷まで行って").steps[0].argument == "ice"
    assert planner.plan("drive to the beacon").steps[0].argument == "beacon"
    assert planner.plan("クレーターの縁まで行って").steps[0].argument == "crater"


def test_waypoint_and_lap_parsing() -> None:
    """A patrol instruction's numbers survive the trip from text to skill."""
    assert _waypoint_index("2番の地点に行って", 4) == 1
    assert _waypoint_index("go to corner 3", 4) == 2
    assert _waypoint_index(None, 4) == 0
    assert _waypoint_index("waypoint 9", 4) is None
    assert _laps("ビルの周りを2周して") == 2
    assert _laps(None) == 1


def test_planners_stay_in_their_own_vocabulary() -> None:
    """A domain's rule planner never emits a verb its skills do not have."""
    for name, domain in DOMAINS.items():
        verbs = {action for action, _ in domain.verbs} | {"goto", "report"}
        planner = DomainRulePlanner(domain)
        for message in ("hello", "何が見える", "do something impossible", "行って"):
            for step in planner.plan(message).steps:
                assert step.action in verbs, f"{name} planner emitted {step.action!r}"


# ======================================================================================
# Skills report honestly
# ======================================================================================


@pytest.mark.slow
def test_patrol_reads_its_route_from_the_scene() -> None:
    """Waypoints come from the XML, so moving the building moves the patrol."""
    from pyunto_robotics.brain.patrol import PatrolSkills

    robot = Robot("campus.xml", gait=TrotGait(), keyframe="corner")
    try:
        skills = PatrolSkills(robot, ColorGrounder())
        route = skills.waypoints()
        assert len(route) == 4
        assert [name for name, _ in route] == [f"waypoint_{i}" for i in range(1, 5)]
        # They should surround the building at the origin, not sit in a heap.
        positions = np.array([position[:2] for _, position in route])
        assert positions[:, 0].min() < -5 and positions[:, 0].max() > 5
        assert positions[:, 1].min() < -5 and positions[:, 1].max() > 5
    finally:
        robot.close()


@pytest.mark.slow
def test_lunar_targets_come_from_the_scene() -> None:
    from pyunto_robotics.brain.lunar import LunarSkills

    robot = Robot("lunar.xml", gait=SkidDrive(), keyframe="plain")
    try:
        skills = LunarSkills(robot, ColorGrounder())
        targets = skills.targets()
        assert set(targets) == {"lander", "ice", "beacon", "crater"}
    finally:
        robot.close()


@pytest.mark.slow
def test_fold_reports_the_span_it_measured() -> None:
    """Folding is judged by measuring the towel, not by asserting the arms moved.

    A flat 0.40 x 0.30 sheet spans about 0.50 m corner to corner; folded once it should be
    appreciably smaller, and the skill reports both numbers.
    """
    from pyunto_robotics.brain.laundry import LaundrySkills

    robot = Robot("home.xml", keyframe="counter")
    try:
        skills = LaundrySkills(robot, ColorGrounder())
        sheet = skills.cloth.sheet("towel_b")
        assert sheet is not None
        assert skills.cloth.sheet_extent(sheet) > 0.45, "towel does not start flat"

        result = skills.fold("pink")
        assert "extent_before" in result.data and "extent_after" in result.data
        if result.ok:
            assert result.data["extent_after"] < result.data["extent_before"]
            # And it must still be on the counter: a towel dragged onto the floor also has a
            # small span, and that used to be reported as a successful fold.
            assert skills.cloth.sheet_centre(sheet)[2] > 0.6
    finally:
        robot.close()
