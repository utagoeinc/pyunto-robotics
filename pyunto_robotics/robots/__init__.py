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
        # The park is 20 m from the carport, so this robot needs the far field too.
        max_depth=45.0,
        camera=Camera(distance=7.0, elevation=-20, azimuth=125),
    )


def _mars() -> RobotSetup:
    """A rover on Mars, driving to named places by camera alone.

    The clearest demonstration of mapless navigation: there is no map of Mars in the robot,
    and the targets are found by looking. What differs from the lunar scene it replaces is
    the landscape -- a channel to follow rather than craters to avoid -- and the light, which
    on Mars is diffuse enough that a camera can see into shadow.
    """
    from ..brain.rover import RoverSkills
    from ..sim.wheel_drive import SkidDrive

    return RobotSetup(
        name="R1 (Mars rover)",
        scene="mars.xml",
        domain=DOMAINS["mars"],
        skills=lambda robot, grounder: RoverSkills(robot, grounder),
        gait=SkidDrive,
        default_keyframe="lander",
        keyframe_help="lander (beside the lander), channel (out on the channel floor)",
        examples=("サンプルまで行って", "drive to the beacon"),
        camera=Camera(distance=9.0, elevation=-22, azimuth=135),
        # 60 m, not the indoor 12. The beacon is 19 m off and the lander 21 m, and at 12 both
        # read as exactly 12 -- so the rover drove to a phantom and circled it.
        max_depth=60.0,
    )


register("solar", _solar())
register("mars", _mars())
