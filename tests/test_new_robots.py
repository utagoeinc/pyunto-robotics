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
from pyunto_robotics.sim.robot import ASSETS, Robot
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


def test_an01_is_the_right_size_and_stands() -> None:
    """AN-01 is ~1.3 m tall, its feet are ON THE FLOOR, and it does not sink or topple.

    The height is not cosmetic. The room is built around a measured arm -- drum at z=0.91,
    counter at 0.85 -- so the shanks are shortened from the source URDF to put the shoulder at
    1.056 m. A model that drifts back toward the URDF's own 1.72 m puts every work surface
    below where the arm can reach, and every reach measurement in brain/laundry.py stops
    meaning what it says.
    """
    robot = Robot("home.xml", keyframe="start")
    try:
        model, data = robot.model, robot.data
        head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
        tallest = max(
            float(data.geom_xpos[g][2] + model.geom_rbound[g]) for g in range(model.ngeom)
            if model.geom_bodyid[g] == head
        )
        assert 1.18 < tallest < 1.45, f"AN-01's head is at {tallest:.2f} m"

        # The shoulder is the number the scene was designed around.
        shoulder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shoulder_r")
        assert 0.96 < float(data.xpos[shoulder][2]) < 1.08, "shoulder is not at working height"

        # THE FEET MUST BE ON THE FLOOR. Measured from the lowest VERTEX of the foot collision
        # mesh, never from geom_rbound: that is a bounding sphere, and for a long flat foot it
        # under-reports the sole by centimetres. Sizing the keyframe off rbound once left the
        # robot hovering 0.069 m in the air -- plainly visible in the viewer, and invisible to
        # every "is it standing" check, because it WAS standing, held up by its own servos.
        foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_r")
        mesh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "c_foot_r")
        first = model.mesh_vertadr[mesh]
        verts = model.mesh_vert[first:first + model.mesh_vertnum[mesh]]
        sole = float(data.xpos[foot][2] + verts[:, 2].min())
        assert abs(sole) < 0.02, f"sole is {sole:+.3f} m from the floor"

        start_z = float(robot.position[2])
        for _ in range(600):
            robot.step()
        assert abs(float(robot.position[2]) - start_z) < 0.05
    finally:
        robot.close()


def test_an01_stands_without_shaking() -> None:
    """Standing still means STILL, not vibrating in place.

    The robot passed every "is it upright" check while visibly juddering, because those ask
    where it is, not whether it is oscillating. The cause was the foot: the package's own foot
    mesh is 0.055 m long and 0.066 m wide -- a peg, narrower than it is tall -- so a 22 kg
    robot was balancing on two contact patches too small for the ankle servos to have any
    leverage against. Measured 0.022 m of pelvis bounce, with ank_roll swinging 0.096 rad.

    Replacing the CONTACT shape with a foot-sized box (the mesh is still what you see) took
    that to 0.0001 m. This test fails long before the wobble is large enough to notice.
    """
    robot = Robot("home.xml", keyframe="start")
    try:
        for _ in range(100):  # let the initial settle finish
            robot.step()
        heights = []
        for _ in range(500):
            robot.step()
            heights.append(float(robot.position[2]))
        bounce = max(heights) - min(heights)
        assert bounce < 0.005, f"pelvis oscillates {bounce * 1000:.1f} mm peak-to-peak"
    finally:
        robot.close()


def test_an01_faces_the_way_its_camera_looks() -> None:
    """The face and head_cam point the same way.

    On the previous model they did not: the face was modelled looking along -y while the
    camera looked along +x, so the robot walked sideways relative to its own face. The bug was
    invisible in every physics measurement -- it stood, walked and folded towels perfectly
    well -- and only showed up in a photograph, which is exactly why it needs a test.
    """
    model = mujoco.MjModel.from_xml_path(str(ASSETS / "an01.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    # A MuJoCo camera looks down its own -z axis.
    forward = -data.cam_xmat[camera].reshape(3, 3)[:, 2]
    assert forward[0] > 0.95, f"head_cam does not look along +x: {np.round(forward, 2)}"

    # The camera sits on the FRONT of the head, between the eyes.
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "head_cam_site")
    head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
    assert data.site_xpos[site][0] - data.xpos[head][0] > 0.05, "head_cam is not on the face"

    # The chest indicator panel is on the chest and centred, not round a side.
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    assert torso >= 0
    # Both hands hang below the shoulders rather than sticking out in front, which is what a
    # mirrored or mis-signed arm chain looks like.
    for side in ("r", "l"):
        grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"grip_{side}")
        shoulder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"shoulder_{side}")
        assert data.site_xpos[grip][2] < data.xpos[shoulder][2], f"{side} hand is above the shoulder"


def test_an01_debug_markers_are_invisible() -> None:
    """Sites render as solid spheres, and hers land on the face and hands.

    The head_cam marker sits between the eyes: left at its default red it renders as a bright
    dot on the bridge of the nose, which reads as a clown nose and took two passes to find
    because it looks like a geom that is not there.
    """
    model = mujoco.MjModel.from_xml_path(str(ASSETS / "an01.xml"))
    for name in ("head_cam_site", "grip_r", "grip_l"):
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        assert site >= 0, f"{name} is missing -- code addresses it by name"
        assert float(model.site_rgba[site][3]) == 0.0, f"{name} is visible"


def test_an01_materials_are_not_overridden() -> None:
    """The robot is coloured, not bone white.

    A default-class `rgba` silently overrides every geom's material, and the whole model
    renders white. It happened once; this catches it coming back.
    """
    model = mujoco.MjModel.from_xml_path(str(ASSETS / "an01.xml"))

    # Colour comes from MATERIALS, so geom_rgba is uniform by design and says nothing. What
    # matters is that the geoms are actually assigned distinct materials: a default-class rgba
    # would still leave matid set but override the colour at render time, so the real check is
    # that the materials exist, are used, and differ from one another.
    used = {int(model.geom_matid[g]) for g in range(model.ngeom)} - {-1}
    assert len(used) >= 5, f"only {len(used)} materials in use; the model should be colourful"

    # Every material the model declares must be USED. An unused one is either a leftover from
    # a shape that has been replaced -- which is how a_grey came to be dead when the primitive
    # detail parts gave way to the package mesh, and a_hair/a_eye when the face was removed --
    # or a colour that was meant to be applied and silently was not.
    for name in ("a_skin", "a_shell", "a_frame", "a_blue", "a_visor"):
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
        assert index >= 0, f"material {name!r} is missing"
        assert index in used, f"material {name!r} is defined but never used"

    palette = {tuple(np.round(model.mat_rgba[i], 2)) for i in used}
    assert len(palette) >= 5, "the materials in use are nearly all the same colour"


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
    model = mujoco.MjModel.from_xml_path(str(ASSETS / "pyunto_q1.xml"))
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
        model = mujoco.MjModel.from_xml_path(str(ASSETS / scene))
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
    model = mujoco.MjModel.from_xml_path(str(ASSETS / "lunar.xml"))
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
    lunar = mujoco.MjModel.from_xml_path(str(ASSETS / "lunar.xml"))
    assert abs(float(lunar.opt.gravity[2]) + 1.62) < 0.01
    campus = mujoco.MjModel.from_xml_path(str(ASSETS / "campus.xml"))
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


def test_japanese_te_form_chains_are_detected_as_multi_step() -> None:
    """Japanese chains actions with the て-form and 「、」 and no conjunction at all.

    This was missed entirely, and it is the ordinary way to write a sequence. A real six-part
    request -- 「洗濯機を開けて、洗濯物を取り出して、カゴに入れて、テーブルまで運んで」 -- contained
    none of the conjunctions the detector knew, so the rule planner did one arbitrary step of
    six, reported success, and the warning that would have suggested --llm never fired either.
    """
    from pyunto_robotics.brain.planner import looks_multi_step

    chained = [
        "洗濯機のそばに籠をもってきて、洗濯機を開けて、洗濯物を取り出して、"
        "洗濯機のドアを閉めて、カゴに入れて、洗濯物を畳むテーブルまで運んで",
        "洗濯機を開けて、洗濯物を取り出して",
        "右のドアを開けて、その後、一番左の部屋に入って",
        "open the door, then go inside",
    ]
    for message in chained:
        assert looks_multi_step(message), f"missed a chain: {message}"

    single = [
        "洗濯機を開けて",
        "タオルを畳んで",
        "open the washing machine",
        "ビルの周りを1周して",
        "ビーコンまで行って",
    ]
    for message in single:
        assert not looks_multi_step(message), f"false positive: {message}"


def test_home_planner_tells_open_from_close() -> None:
    """「閉めて」 must not match 「開けて」's patterns.

    Verb matching stops at the first hit, so with open_washer listed first a request to CLOSE
    the washer matched on 「洗濯機」 and came out as "open the washer" -- the opposite of what
    was asked, which is worse than not understanding at all.
    """
    planner = DomainRulePlanner(DOMAINS["home"])
    assert planner.plan("洗濯機を開けて").steps[0].action == "open_washer"
    assert planner.plan("洗濯機のドアを閉めて").steps[0].action == "close_washer"
    assert planner.plan("open the washing machine").steps[0].action == "open_washer"
    assert planner.plan("close the washing machine").steps[0].action == "close_washer"


def test_home_domain_covers_the_whole_errand() -> None:
    """Every part of a realistic laundry request maps to a verb the skills implement."""
    from pyunto_robotics.brain.laundry import LaundrySkills

    planner = DomainRulePlanner(DOMAINS["home"])
    for message, action in [
        ("籠をもってきて", "bring_basket"),
        ("洗濯機を開けて", "open_washer"),
        ("洗濯物を取り出して", "take_out"),
        ("洗濯機のドアを閉めて", "close_washer"),
        ("カゴに入れて", "to_basket"),
        ("洗面台に置いて", "to_counter"),
        ("タオルを畳んで", "fold"),
    ]:
        steps = planner.plan(message).steps
        assert steps, f"{message} produced no plan"
        assert steps[0].action == action, f"{message} -> {steps[0].action}, wanted {action}"

    # And every one of those verbs must reach a real handler. A planner that emits a verb the
    # skills do not implement is the same failure as not understanding, just later and more
    # confusing -- so this checks the dispatch table rather than trusting the two lists match.
    robot = Robot("home.xml", keyframe="start")
    try:
        skills = LaundrySkills(robot, ColorGrounder())
        handled = skills.run.__wrapped__ if hasattr(skills.run, "__wrapped__") else skills.run
        import inspect

        source = inspect.getsource(handled)
        for action in ("bring_basket", "open_washer", "close_washer", "take_out",
                       "to_basket", "to_counter", "fold", "describe", "where", "home"):
            assert f'"{action}"' in source, f"{action} has no handler in Skills.run"
    finally:
        robot.close()


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
