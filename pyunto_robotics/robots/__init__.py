"""The robots that ship with the SDK. Importing this registers them."""

from ..brain.domains import DOMAINS
from ..registry import RobotSetup, register
from ..viewer import Camera


def _solar() -> RobotSetup:
    """A robot that goes out and fetches its own energy.

    This is the demonstration the SDK leads with. Everything else here shows a robot doing
    what it was told; this one shows a robot going out, finding something by measurement,
    and coming back with a result that turns the house lights on.
    """
    from ..brain.solar_errand import SolarErrandSkills
    from ..sim.wheel_drive import LEFT_WHEELS_4, RIGHT_WHEELS_4, SkidDrive

    return RobotSetup(
        name="S1 (solar errand robot)",
        scene="solar.xml",
        domain=DOMAINS["solar"],
        skills=lambda robot, grounder: SolarErrandSkills(robot, grounder),
        # Four wheels on pavement, so none of the rover's lunar slip compensation applies.
        gait=lambda: SkidDrive(
            left_wheels=LEFT_WHEELS_4, right_wheels=RIGHT_WHEELS_4, slip_factor=1.4
        ),
        default_keyframe="carport",
        keyframe_help="carport (at home, in shade), street (out on the road), park (in the sun)",
        examples=(
            "日光が当たる場所まで移動して、電力を取得してきて",
            "go and fetch some power",
        ),
        camera=Camera(distance=7.0, elevation=-20, azimuth=125),
    )


register("solar", _solar())
