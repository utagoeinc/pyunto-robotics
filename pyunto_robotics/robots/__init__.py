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
            "go and find some sunlight, and bring back power",
            "go and fetch some power",
        ),
        greeting=(
            "🔆 I am parked in the carport. I can go and find sunlight, charge there, come "
            "home, and put the power into the house.\n"
            "  • {example_a}\n  • {example_b}\n"
            "I will say how I understood you before I set off, and report where I am and how "
            "the charging is going as I work."
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
        examples=("drive to the sample", "drive to the beacon"),
        greeting=(
            "🛻 I am on the surface of Mars, and I can see around me with a camera and a "
            "depth sensor.\n"
            "  • {example_a}\n  • {example_b}\n"
            "Tell me where to go. I will say how I understood you before I set off, report "
            "as I drive, and send a photograph when I arrive."
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
        examples=("fetch the crate of apples", "fetch the apples"),
        greeting=(
            "🐕 I am in the orchard. I walk the rows on four legs and can carry things.\n"
            "  • {example_a}\n  • {example_b}\n"
            "Tell me what to carry and where to take it. I will report as I walk, and send a "
            "photograph when I get there."
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
        examples=("clean the rooms on both floors", "clean both floors"),
        greeting=(
            "🧹 I am in the hotel. I clean the guest rooms and ride the lift between "
            "floors.\n"
            "  • {example_a}\n  • {example_b}\n"
            "I will say how I understood you before I start, and report room by room."
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
        name="Smart house",
        scene="watch.xml",
        domain=DOMAINS["watch"],
        skills=lambda robot, grounder: WatchingSkills(robot, grounder),
        default_keyframe="asleep",
        keyframe_help="asleep (in bed, start of the day), up (out of bed)",
        examples=("how is she doing?", "how is she today?"),
        greeting=(
            "🏠 I am the smart house here. I have motion sensors in the floor of every room, "
            "a thermometer and hygrometer, the front door lock, and the doorphone.\n"
            "  • {example_a}\n  • {example_b}\n"
            "  • has anyone been to the door?\n  • how humid is it?\n"
            "  • turn the air conditioning on\n"
            "There are no cameras inside the flat. The only picture is the doorphone, and it "
            "faces the street.\n"
            "If something concerns me, I will tell you without being asked."
        ),
        camera=Camera(distance=11.0, elevation=-55, azimuth=90),
    )


def _pet() -> RobotSetup:
    """A small camera robot looking after a cat while the owner is out.

    The counterpart to the watching flat, and deliberately its opposite. There, the person
    observed had not asked to be, and the answer was to use no indoor camera at all. Here the
    only human is the one holding the phone, in their own home, looking for their own cat --
    so a camera is the right instrument, and the demonstration is about aiming it.

    Which is the point: the cat's four usual places are at four different heights, and none of
    them can be reached by driving alone. A robot that only moved would find a cat on the
    floor and nothing else.
    """
    from ..brain.pet_watch import PetWatchSkills
    from ..sim.wheel_drive import LEFT_WHEELS_4, RIGHT_WHEELS_4, SkidDrive

    return RobotSetup(
        name="P1 (pet camera)",
        scene="pet.xml",
        domain=DOMAINS["pet"],
        skills=lambda robot, grounder: PetWatchSkills(robot, grounder),
        gait=lambda: SkidDrive(
            left_wheels=LEFT_WHEELS_4, right_wheels=RIGHT_WHEELS_4, slip_factor=1.2
        ),
        default_keyframe="dock",
        keyframe_help="dock (cat under the sofa), sill, tree, shelf (cat in each place)",
        examples=("where is the cat?", "find the cat"),
        greeting=(
            "🐱 I am the camera looking after the flat while you are out. I can drive around "
            "and aim my camera up, down and sideways to find the cat.\n"
            "  • {example_a}\n  • {example_b}\n"
            "  • look around\n  • send me a photo\n  • look up\n"
            "If I cannot find her, I will tell you that rather than guess."
        ),
        # Indoors: nothing is more than about 9 m away.
        max_depth=12.0,
        camera=Camera(distance=7.6, elevation=-52, azimuth=215),
    )


register("solar", _solar())
register("watch", _watch())
register("pet", _pet())
register("hotel", _hotel())
register("orchard", _orchard())
register("mars", _mars())
