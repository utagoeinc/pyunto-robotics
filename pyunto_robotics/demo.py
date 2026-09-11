"""The one-command demonstration.

    pyunto-robotics demo --pair K3F9QZ

A person who has never seen this code should be able to run that line and, a few seconds
later, watch a robot in a window do what they typed into a diary on their phone. Everything
here exists to make that happen without a stumble: the four things that can go wrong are each
explained in plain language rather than raised as a traceback.
"""

from __future__ import annotations

import logging
import time

from pyunto_agent.auth import AuthError
from pyunto_agent.bridge import Bridge

from . import registry
from .agent import RobotAgent
from .backend import RobotBackend
from .connect import Connection, connect
from .perception.grounding import ColorGrounder
from .sim.robot import Robot
from .viewer import open_viewer

log = logging.getLogger(__name__)

GREETING = (
    "🤖 I am here and I can see the room. Tell me what to do — try one of these:\n"
    "  • {example_a}\n  • {example_b}\n  • raise your right hand\n"
    "I will say how I understood you before I move, report each step as I go, "
    "and send a photograph when I am done."
)


def _already_paired(connection: Connection) -> str | None:
    """The shared space this robot is already in, if any.

    Pairing codes are consumed on use. Restarting a robot -- which happens constantly during
    a demo -- therefore presents a spent code, and the honest answer to "this code is used
    up" is usually "because you already did this".
    """
    try:
        spaces = connection.client.list_spaces()
    except Exception:  # noqa: BLE001
        return None
    shared = [s for s in spaces if not (s.get("is_self") or s.get("isSelf"))]
    return str(shared[0].get("uuid")) if shared else None


def _wait_for_key(connection: Connection, space_id: str, timeout: float = 300.0) -> bool:
    """Wait until a member's app has sealed the space key for this robot.

    Pyunto is end-to-end encrypted: the server cannot hand out a key, so the robot cannot read
    anything until somebody opens the app after it joins. Without this wait the robot looks
    like it is ignoring you, which is the worst possible first impression.
    """
    if connection.keys.has_key(space_id):
        return True
    print("keys    : waiting for a member to open the Pyunto app so the robot can be let in…")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            connection.keys.get_key(space_id)
            print("keys    : received — the robot can read this diary now")
            return True
        except Exception:  # noqa: BLE001 - "not yet" is the expected answer here
            time.sleep(2.0)
    print("keys    : still waiting. Open the space in the Pyunto app, then message the robot.")
    return False


def run_demo(
    robot_name: str = "office",
    pair: str | None = None,
    use_llm: bool = False,
    view: bool = True,
    speed: float = 1.0,
    send_images: bool = True,
) -> int:
    setup = registry.get(robot_name)
    print(f"Pyunto Robot SDK\nrobot   : {setup.name}")

    print("account : creating a robot account…", end=" ", flush=True)
    try:
        connection = connect(display_name=setup.name.split(" (")[0])
    except AuthError as e:
        print(f"\nERROR: {e}")
        return 1
    print(f"done ({connection.identity.display_name})")

    space_id: str | None = None
    if pair:
        print(f"pairing : joining with code {pair}…", end=" ", flush=True)
        try:
            # The client logs a failed join at ERROR. Here a failure is usually just "you
            # already did this", so a red ERROR line is a false alarm; we decide below what
            # this actually means and say so.
            client_log = logging.getLogger("pyunto_agent.client")
            previous_level = client_log.level
            client_log.setLevel(logging.CRITICAL)
            try:
                space_id = connection.client.join(pair)
            finally:
                client_log.setLevel(previous_level)
            print("joined")
        except Exception as e:  # noqa: BLE001 - a wrong code is a user error, not a crash
            # A pairing code is single-use, so the second run of the same robot always fails
            # here -- and it fails with "invalid or expired", which sends the person back to
            # the app to make a code they did not need. If the robot is already in a shared
            # space, that is what the code was for: carry on.
            space_id = _already_paired(connection)
            if space_id is None:
                print(f"\nERROR: could not join with that code ({e}).")
                print("        Codes expire after 10 minutes — make a new one in the app.")
                return 1
            print("already a member (that code was already used)")
    else:
        spaces = [s for s in connection.client.list_spaces() if not (s.get("is_self") or s.get("isSelf"))]
        if spaces:
            space_id = str(spaces[0].get("uuid"))
            print(f"pairing : already a member of \"{spaces[0].get('name')}\"")
        else:
            print("pairing : not in any shared space yet.")
            print("          In the Pyunto app open a premium space, choose \"Invite a robot\",")
            print("          then run:  pyunto-robotics demo --pair <code>")

    print("window  : opening the simulator…")
    robot = Robot(setup.scene, gait=setup.gait() if setup.gait else None, keyframe=setup.default_keyframe)
    viewer = open_viewer(robot, speed, setup.camera) if view else None
    if view and viewer is None:
        robot.close()
        return 1

    grounder = ColorGrounder()
    planner = (
        setup.planner(use_llm)
        if setup.planner is not None
        else _domain_planner(setup, use_llm)
    )
    try:
        skills = setup.skills(robot, grounder, planner=planner)
    except TypeError:
        skills = setup.skills(robot, grounder)
    agent = RobotAgent(robot, grounder, planner=planner, skills=skills)

    if space_id and _wait_for_key(connection, space_id):
        examples = setup.examples or ("walk to the door", "where are you?")
        try:
            connection.client.send(
                space_id,
                GREETING.format(example_a=examples[0], example_b=examples[-1]),
            )
        except Exception:  # noqa: BLE001 - a greeting is nice to have, not required
            log.debug("could not post the greeting", exc_info=True)

    bridge = Bridge(
        connection.client,
        # The client and the robot go in so the robot can narrate into the thread as it
        # works -- what it understood, each step, and a picture at the end -- rather than
        # going quiet for a minute and posting one sentence.
        RobotBackend(agent, client=connection.client, robot=robot,
                     send_images=send_images),
        persona="",  # the robot acts; it does not role-play
        space_ids={space_id} if space_id else None,
        history=4,
        # Redraw the window while waiting. The reply loop owns this thread, so nothing else can.
        on_idle=viewer.sync if viewer else None,
    )
    print("\nlistening — message the robot from the Pyunto app. Ctrl-C to stop.\n")
    try:
        bridge.run()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()
        if viewer:
            viewer.close()
        robot.close()
    return 0


def _domain_planner(setup, use_llm: bool):
    from .brain.domains import DomainLLMPlanner, DomainRulePlanner

    return DomainLLMPlanner(setup.domain) if use_llm else DomainRulePlanner(setup.domain)
