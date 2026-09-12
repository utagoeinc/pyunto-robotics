"""The robots that ship with the SDK. Importing this registers them."""

from ..brain.domains import DOMAINS
from ..registry import RobotSetup, register
from ..viewer import Camera


def _office_planner(use_llm: bool):
    """The office humanoid has its own planner (it predates the Domain abstraction)."""
    from ..brain.planner import LLMPlanner, RulePlanner

    return LLMPlanner() if use_llm else RulePlanner()


def _office() -> RobotSetup:
    from ..brain.skills import Skills

    return RobotSetup(
        name="H1 (office humanoid)",
        scene="office.xml",
        domain=None,
        planner=_office_planner,
        skills=lambda robot, grounder, planner=None: Skills(robot, grounder, planner=planner),
        default_keyframe="lobby",
        keyframe_help="lobby (by the entrance)",
        examples=("walk to the door", "オフィスのドアを開けて"),
        camera=Camera(distance=4.5, elevation=-18, azimuth=140),
    )


def _home() -> RobotSetup:
    from ..brain.laundry import LaundrySkills

    return RobotSetup(
        name="Momo (home assistant)",
        scene="home.xml",
        domain=DOMAINS["home"],
        skills=lambda robot, grounder: LaundrySkills(robot, grounder),
        default_keyframe="start",
        keyframe_help="start (at the washer), middle (centre of room), counter (at the counter)",
        examples=("タオルを洗濯機から出して畳んで", "open the washing machine"),
        camera=Camera(distance=4.0, elevation=-20, azimuth=135),
    )


def _patrol() -> RobotSetup:
    from ..brain.patrol import PatrolSkills
    from ..sim.quad_gait import TrotGait

    return RobotSetup(
        name="Q1 (patrol quadruped)",
        scene="campus.xml",
        domain=DOMAINS["patrol"],
        skills=lambda robot, grounder: PatrolSkills(robot, grounder),
        gait=TrotGait,
        default_keyframe="start",
        keyframe_help="start (south of the building), corner (SE corner), steps (at the stairs)",
        examples=("ビルの周りを1周して", "patrol around the building"),
        camera=Camera(distance=8.0, elevation=-22, azimuth=135),
    )


def _lunar() -> RobotSetup:
    from ..brain.lunar import LunarSkills
    from ..sim.wheel_drive import SkidDrive

    return RobotSetup(
        name="R1 (lunar rover)",
        scene="lunar.xml",
        domain=DOMAINS["lunar"],
        skills=lambda robot, grounder: LunarSkills(robot, grounder),
        gait=SkidDrive,
        default_keyframe="plain",
        keyframe_help="plain (open surface), start (beside the lander)",
        examples=("クレーターの縁まで行って", "drive to the beacon"),
        camera=Camera(distance=10.0, elevation=-25, azimuth=135),
    )


register("office", _office())
register("home", _home())
register("patrol", _patrol())
register("lunar", _lunar())
