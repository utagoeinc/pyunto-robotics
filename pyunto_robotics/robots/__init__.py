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
        greeting=(
            "🔆 カーポートにいます。日の当たる場所を探して充電し、"
            "戻って家に電気を届けます。\n"
            "  • {example_a}\n  • {example_b}\n"
            "出発前にどう理解したかをお伝えし、道中の様子と充電の進み具合を報告します。"
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
        greeting=(
            "🛻 火星の地表にいます。カメラと距離センサーで周りが見えています。\n"
            "  • {example_a}\n  • {example_b}\n"
            "行き先を伝えてください。出発前にどう理解したかをお伝えし、"
            "走りながら経過を報告して、着いたら写真を送ります。"
        ),
        camera=Camera(distance=9.0, elevation=-22, azimuth=135),
        # 60 m, not the indoor 12. The beacon is 19 m off and the lander 21 m, and at 12 both
        # read as exactly 12 -- so the rover drove to a phantom and circled it.
        max_depth=60.0,
    )


def _orchard() -> RobotSetup:
    """A four-legged robot carrying fruit out of an orchard.

    Legs earn their place here: orchard ground between the rows is soft and rutted, which is
    where the fruit is and where a wheeled machine either sinks or has to stay on a track.
    """
    from ..brain.orchard import OrchardSkills
    from ..sim.quad_gait import TrotGait

    return RobotSetup(
        name="Q1 (orchard quadruped)",
        scene="orchard.xml",
        domain=DOMAINS["orchard"],
        skills=lambda robot, grounder: OrchardSkills(robot, grounder),
        gait=TrotGait,
        default_keyframe="shed",
        keyframe_help="shed (at the packing shed), lane (halfway up the row)",
        examples=("リンゴのコンテナを取ってきて", "fetch the apples"),
        greeting=(
            "🐕 果樹園にいます。四脚で畝の間を歩き、荷物を運べます。\n"
            "  • {example_a}\n  • {example_b}\n"
            "運ぶものと行き先を伝えてください。歩きながら経過を報告し、"
            "着いたら写真を送ります。"
        ),
        camera=Camera(distance=8.0, elevation=-20, azimuth=110),
        # The crates are 16 m up the lane from the shed.
        max_depth=40.0,
    )


def _hotel() -> RobotSetup:
    """A cleaning robot that takes the lift between two corridors.

    The lift is the point. A cleaner that works one floor is a novelty; one that moves between
    them is a machine a hotel can staff a building with.
    """
    from ..brain.hotel import HotelSkills

    return RobotSetup(
        name="H1 (hotel cleaner)",
        scene="hotel.xml",
        domain=DOMAINS["hotel"],
        skills=lambda robot, grounder: HotelSkills(robot, grounder),
        default_keyframe="corridor",
        keyframe_help="corridor (ground floor, by the rooms), in_lift (standing in the car)",
        examples=("両方のフロアを掃除して", "clean both floors"),
        greeting=(
            "🧹 ホテルにいます。客室を掃除し、リフトで階を移動できます。\n"
            "  • {example_a}\n  • {example_b}\n"
            "作業前にどう理解したかをお伝えし、部屋ごとに進み具合を報告します。"
        ),
        camera=Camera(distance=9.0, elevation=-18, azimuth=150),
        max_depth=30.0,
    )


def _watch() -> RobotSetup:
    """A flat that watches an older person living alone.

    The only demonstration here with no robot in it. Nothing is commanded: the house watches,
    and what it sees goes into a diary that a family member reads from another city. That is
    what a diary is for, and it is the case where the record is the product rather than the
    by-product of an errand.

    Replaces the earlier `house` robot, whose devices it keeps -- noticing a room is 29°C is
    worth something, and turning the air conditioning on is worth more.
    """
    from ..brain.watching import WatchingSkills

    return RobotSetup(
        name="スマートハウス",
        scene="watch.xml",
        domain=DOMAINS["watch"],
        skills=lambda robot, grounder: WatchingSkills(robot, grounder),
        default_keyframe="asleep",
        keyframe_help="asleep (in bed, start of the day), up (out of bed)",
        examples=("母の様子はどう？", "how is she today?"),
        greeting=(
            "🏠 この家のスマートハウスです。各部屋のセンサーでお母さまの様子が分かり、"
            "エアコン・照明・玄関の鍵を操作できます。\n"
            "  • {example_a}\n  • {example_b}\n  • エアコンをつけて\n  • 今何時？\n"
            "気になることがあれば、聞かれなくてもこちらからお伝えします。"
        ),
        camera=Camera(distance=11.0, elevation=-55, azimuth=90),
    )


register("solar", _solar())
register("watch", _watch())
register("hotel", _hotel())
register("orchard", _orchard())
register("mars", _mars())
