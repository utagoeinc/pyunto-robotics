"""The robot narrates: what it understood, each step, and a picture at the end.

These are the three things a person waiting on a robot actually needs, and all three are easy
to lose in a refactor without anything else breaking -- the errand still runs, it just stops
saying anything. So they are pinned here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from pyunto_agent.backends import Context, Turn
from pyunto_robotics.agent import RobotAgent
from pyunto_robotics.backend import RobotBackend


@dataclass
class FakeStep:
    action: str
    argument: str = ""
    where: str = ""
    expect: str = ""

    def __str__(self) -> str:
        return f"{self.action} {self.argument}".strip()


@dataclass
class FakeResult:
    ok: bool
    message: str
    data: dict = field(default_factory=dict)
    fatal: bool = True


class FakePlan:
    def __init__(self, steps, reply=None):
        self.steps = steps
        self.reply = reply


class FakePlanner:
    """Understands "door"; understands nothing else."""

    def plan(self, text: str) -> FakePlan:
        if "door" in text or "ドア" in text:
            return FakePlan([FakeStep("goto", "door"), FakeStep("open", "door")])
        return FakePlan([])


class FakeSkills:
    actions = ["goto", "open", "report"]

    def run(self, action, argument, where, expect):  # noqa: ANN001
        if action == "open":
            return FakeResult(False, f"I cannot reach the {argument} handle.")
        return FakeResult(True, f"Walked to the {argument}.")


class FakeRobot:
    """A camera whose view changes every frame, as a real one does."""

    def __init__(self) -> None:
        self._frame = 0

    def look(self, camera: str = "head_cam"):  # noqa: ANN001
        self._frame += 1
        pixels = np.full((8, 8, 3), self._frame % 251, dtype=np.uint8)

        class Observation:
            rgb = pixels

        return Observation()


class RecordingClient:
    def __init__(self) -> None:
        self.posts: list[tuple] = []

    def send(self, space, text, thread_id=None, **kw):  # noqa: ANN001
        self.posts.append(("text", text, thread_id))

    def send_image(self, space, png, thread_id=None, caption=None):  # noqa: ANN001
        self.posts.append(("image", caption, thread_id))


def build(client=None):
    agent = RobotAgent.__new__(RobotAgent)
    agent.robot = FakeRobot()
    agent.planner = FakePlanner()
    agent.skills = FakeSkills()
    agent.max_steps_per_message = 4
    agent.max_replans = 0
    agent.client = None
    agent.on_idle = None
    return RobotBackend(agent, client=client, robot=FakeRobot())


def run(backend, text, thread_id="thread-1"):
    ctx = Context(
        space_name="diary",
        thread_id=thread_id,
        turns=[Turn("user", "someone", text)],
        persona="",
        chat_space_id="space-1",
    )
    return backend.reply(ctx)


def texts(client) -> list[str]:
    return [p[1] for p in client.posts if p[0] == "text"]


def test_says_what_it_understood_before_it_moves():
    client = RecordingClient()
    run(build(client), "open the door")
    first = texts(client)[0]
    assert "open the door" in first
    assert "goto door" in first and "open door" in first


def test_reports_each_step_as_it_finishes():
    client = RecordingClient()
    run(build(client), "open the door")
    lines = texts(client)
    assert any(line.startswith("✅") and "Walked to the door" in line for line in lines)
    # A step that failed must say so rather than be folded into a cheerful summary.
    assert any(line.startswith("⚠️") and "cannot reach" in line for line in lines)


def test_pictures_frame_the_whole_errand():
    """Before it moves, after each step, and at the end -- the sequence a person can follow."""
    client = RecordingClient()
    run(build(client), "open the door")
    images = [p for p in client.posts if p[0] == "image"]
    # Two steps: before + one per step + final.
    assert len(images) == 4, [p[1] for p in images]
    assert client.posts[0][0] == "image", "the first thing posted must be the scene as found"
    assert client.posts[-1][0] == "image", "the last thing posted must be the scene as left"


def test_the_closing_report_stands_on_its_own():
    """Someone reading the diary tomorrow gets one entry, not eight fragments."""
    client = RecordingClient()
    run(build(client), "open the door")
    summary = texts(client)[-1]
    assert summary.startswith("⚠️ Not finished"), summary
    assert "open the door" in summary
    # Every step accounted for, in order.
    assert "1. goto door" in summary and "2. open door" in summary


def test_the_closing_report_says_done_when_it_worked():
    class AlwaysWorks(FakeSkills):
        def run(self, action, argument, where, expect):  # noqa: ANN001
            return FakeResult(True, f"Did {action}.")

    client = RecordingClient()
    backend = build(client)
    backend.agent.skills = AlwaysWorks()
    run(backend, "open the door")
    assert texts(client)[-1].startswith("✅ Done")


def test_sensor_readings_are_attached_to_the_step():
    class Measuring(FakeSkills):
        def run(self, action, argument, where, expect):  # noqa: ANN001
            return FakeResult(True, "Walked.", {"distance_m": 0.42, "angle_deg": 17.0})

    client = RecordingClient()
    backend = build(client)
    backend.agent.skills = Measuring()
    run(backend, "open the door")
    measured = [line for line in texts(client) if "📡" in line]
    assert measured, texts(client)
    # Rendered as readable quantities, not a dict.
    assert "0.42 m" in measured[0] and "17°" in measured[0]


def test_an_unknown_instruction_is_answered_not_ignored():
    """The commonest outcome. Silence here is what makes a robot feel broken."""
    client = RecordingClient()
    run(build(client), "raise your right hand")
    lines = texts(client)
    assert len(lines) == 1
    assert "raise your right hand" in lines[0]
    # And it says what would work instead.
    assert "goto" in lines[0]


def test_everything_lands_in_the_originating_thread():
    client = RecordingClient()
    run(build(client), "open the door", thread_id="thread-42")
    assert {p[2] for p in client.posts} == {"thread-42"}


def test_no_duplicate_summary_when_narrating():
    """Bridge posts whatever reply() returns; returning text too would say it all twice."""
    assert run(build(RecordingClient()), "open the door") is None


def test_without_a_client_it_behaves_as_before():
    backend = build(client=None)
    reply = run(backend, "open the door")
    assert reply and "Walked to the door" in reply


def test_a_failing_post_does_not_abandon_the_errand():
    class BrokenClient(RecordingClient):
        def send(self, space, text, thread_id=None, **kw):  # noqa: ANN001
            raise RuntimeError("network down")

    # The instruction must still run to completion with the network gone.
    run(build(BrokenClient()), "open the door")


@pytest.mark.parametrize("send_images", [True, False])
def test_pictures_can_be_turned_off(send_images):
    client = RecordingClient()
    agent = build(client).agent
    backend = RobotBackend(agent, client=client, robot=FakeRobot(), send_images=send_images)
    run(backend, "open the door")
    assert bool([p for p in client.posts if p[0] == "image"]) is send_images


# -- the gestures themselves ------------------------------------------------------
#
# "Raise your right hand" is the first thing anyone asks a humanoid, and for a while it was
# the one thing this robot answered with "I do not know how to do that".


@pytest.mark.parametrize(
    ("text", "action", "side"),
    [
        ("右手を挙げて", "raise_arm", "r"),
        ("左手を振って", "wave", "l"),
        ("raise your right hand", "raise_arm", "r"),
        ("wave your left hand", "wave", "l"),
        ("put your hand down", "lower_arm", "r"),
        ("手を下ろして", "lower_arm", "r"),
    ],
)
def test_gestures_are_understood(text, action, side):
    from pyunto_robotics.brain.planner import RulePlanner

    plan = RulePlanner().plan(text)
    assert len(plan.steps) == 1, f"{text!r} produced {plan}"
    assert plan.steps[0].action == action
    assert plan.steps[0].argument == side


def test_waving_reads_as_a_wave_not_a_raise():
    """Both sentences name a hand; only the verb separates them, so order matters."""
    from pyunto_robotics.brain.planner import RulePlanner

    assert RulePlanner().plan("手を振って").steps[0].action == "wave"


@pytest.mark.slow
def test_raising_an_arm_actually_lifts_the_hand():
    """The claim in the reply has to correspond to the physics."""
    import mujoco

    from pyunto_robotics.brain.skills import Skills
    from pyunto_robotics.perception.grounding import ColorGrounder
    from pyunto_robotics.sim.robot import Robot

    robot = Robot("office.xml", keyframe="start")
    try:
        skills = Skills(robot, ColorGrounder())

        def hand_height() -> float:
            body = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "palm_r")
            return float(robot.data.xpos[body][2])

        resting = hand_height()
        assert skills.raise_arm("r").ok
        raised = hand_height()
        assert raised > resting + 0.4, f"hand went from {resting:.2f} to {raised:.2f}"

        assert skills.lower_arm("r").ok
        assert hand_height() < raised - 0.4, "the arm did not come back down"
    finally:
        robot.close()


def test_a_one_step_instruction_stays_short():
    """Volume is a feature with a cost. Twelve messages for "raise your hand" is spam.

    A single-step instruction gets at most six posts: the scene before, the plan, the step,
    the step's picture, the closing report, the scene after. In practice a gesture leaves the
    robot where it stood, so the last two frames are identical and one is dropped -- five.
    If a change pushes this up, that is a decision to make deliberately rather than discover
    in a screenshot.
    """

    class OneStep(FakeSkills):
        def run(self, action, argument, where, expect):  # noqa: ANN001
            return FakeResult(True, "Raised.")

    class SingleStepPlanner:
        def plan(self, text):  # noqa: ANN001
            return FakePlan([FakeStep("raise_arm", "r")])

    client = RecordingClient()
    backend = build(client)
    backend.agent.skills = OneStep()
    backend.agent.planner = SingleStepPlanner()
    run(backend, "右手を挙げて")
    assert len(client.posts) == 6, [p[1] for p in client.posts]


def test_the_robot_does_not_take_its_own_narration_as_an_instruction():
    """The bug that filled a thread: narrate, then obey your own narration, forever.

    Progress notes are posted into the same thread, so they come back as history. Acting on
    the last turn overall hands the robot its own "Understood: ..." to carry out, which it
    re-plans and re-runs -- and each run posts more notes to trip over next time.
    """
    from pyunto_agent.backends import Context, Turn

    client = RecordingClient()
    backend = build(client)
    ctx = Context(
        space_name="diary",
        thread_id="thread-1",
        turns=[
            Turn("user", "someone", "open the door"),
            # Everything below is the robot's own commentary from the previous run.
            Turn("assistant", "H1", "🤖 Understood: “open the door”\nI will: goto door"),
            Turn("assistant", "H1", "✅ 1. goto door — Walked to the door."),
        ],
        persona="",
        chat_space_id="space-1",
    )
    backend.reply(ctx)
    plan = texts(client)[0]
    # It must act on what the person wrote, not on its own last sentence.
    assert "open the door" in plan
    assert "Understood: “🤖" not in plan and "✅" not in plan.splitlines()[0]


def test_an_instruction_is_found_even_behind_its_own_chatter():
    """History can end with many robot posts; the person's words are still the instruction."""
    from pyunto_agent.backends import Context, Turn

    client = RecordingClient()
    backend = build(client)
    ctx = Context(
        space_name="diary",
        thread_id="thread-1",
        turns=[
            Turn("user", "someone", "open the door"),
            *[Turn("assistant", "H1", f"✅ {i}. step") for i in range(6)],
        ],
        persona="",
        chat_space_id="space-1",
    )
    backend.reply(ctx)
    assert "open the door" in texts(client)[0]


def test_a_thread_with_only_robot_posts_is_left_alone():
    """Nothing was asked, so nothing should happen."""
    from pyunto_agent.backends import Context, Turn

    client = RecordingClient()
    backend = build(client)
    ctx = Context(
        space_name="diary",
        thread_id="thread-1",
        turns=[Turn("assistant", "H1", "✅ 1. goto door")],
        persona="",
        chat_space_id="space-1",
    )
    assert backend.reply(ctx) is None
    assert client.posts == []


def test_the_same_picture_is_never_posted_twice_in_a_row():
    """On a one-step errand the step frame and the closing frame are the same pose.

    Two identical photographs in a row read as the robot malfunctioning -- exactly the
    impression these pictures exist to dispel.
    """

    class StillCamera:
        """A robot that never moves, so every frame is byte-identical."""

        def look(self, camera: str = "head_cam"):  # noqa: ANN001
            class Observation:
                rgb = np.zeros((8, 8, 3), dtype=np.uint8)

            return Observation()

    class OneStep(FakeSkills):
        def run(self, action, argument, where, expect):  # noqa: ANN001
            return FakeResult(True, "Raised.")

    class SingleStepPlanner:
        def plan(self, text):  # noqa: ANN001
            return FakePlan([FakeStep("raise_arm", "r")])

    client = RecordingClient()
    backend = build(client)
    backend.robot = StillCamera()
    backend.agent.skills = OneStep()
    backend.agent.planner = SingleStepPlanner()
    run(backend, "右手を挙げて")
    images = [p for p in client.posts if p[0] == "image"]
    assert len(images) == 1, [p[1] for p in images]
