"""Planner and skill tests.

The planner tests are pure logic and run instantly. The skill tests drive the simulator, and
the important one is `test_open_door_end_to_end`: message-shaped instruction in, door open and
robot through the doorway out.
"""

from __future__ import annotations

import pytest

from pyunto_robotics.brain.planner import LLMPlanner, Plan, RulePlanner, Step, parse_plan
from pyunto_robotics.brain.skills import Skills
from pyunto_robotics.perception.grounding import ColorGrounder
from pyunto_robotics.sim.robot import Robot


# -- planning ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def planner() -> RulePlanner:
    return RulePlanner()


@pytest.mark.parametrize(
    "message,action,argument",
    [
        # The demo's headline instruction, in both languages.
        ("オフィスのドアを開けて", "open", "door"),
        ("open the office door", "open", "door"),
        ("ドアを開けて", "open", "door"),
        ("push open the door", "open", "door"),
        # Navigation
        ("go to the whiteboard", "goto", "whiteboard"),
        ("ドアまで行って", "goto", "door"),
        ("walk to the plant", "goto", "plant"),
        # Attention
        ("look at the door", "face", "door"),
        ("point at the plant", "point_at", "plant"),
    ],
)
def test_instructions_map_to_the_right_action(planner, message, action, argument):
    plan = planner.plan(message)
    assert plan.steps, f"no plan for {message!r}"
    assert plan.steps[0].action == action
    assert plan.steps[0].argument == argument


@pytest.mark.parametrize(
    "message,action",
    [
        ("look around", "look_around"),
        ("周りを見て", "look_around"),
        ("まわりを見て", "look_around"),
        ("what do you see?", "describe"),
        ("何が見える？", "describe"),
        ("where are you", "where"),
        ("どこにいる", "where"),
    ],
)
def test_argumentless_instructions(planner, message, action):
    plan = planner.plan(message)
    assert plan.steps and plan.steps[0].action == action


def test_look_around_beats_look_at(planner):
    """「見て」 is a substring of 「周りを見て」; the more specific phrase must win."""
    assert planner.plan("周りを見て").steps[0].action == "look_around"
    assert planner.plan("ドアを見て").steps[0].action == "face"


def test_open_beats_go_when_both_appear(planner):
    """"go open the door" is an instruction to open, not merely to walk over."""
    assert planner.plan("go open the door").steps[0].action == "open"


def test_bare_object_is_treated_as_go_there(planner):
    plan = planner.plan("the door")
    assert plan.steps and plan.steps[0] == Step("goto", "door")


def test_greeting_gets_a_reply_not_an_action(planner):
    plan = planner.plan("こんにちは")
    assert not plan.steps
    assert plan.reply and "Hello" in plan.reply


def test_unintelligible_message_explains_capabilities(planner):
    plan = planner.plan("qwertyuiop")
    assert not plan.steps
    assert plan.reply and "open" in plan.reply.lower()


def test_empty_message(planner):
    plan = planner.plan("   ")
    assert not plan.steps and plan.reply


# -- parsing LLM output --------------------------------------------------------------


def test_parse_plan_accepts_plain_json():
    steps = parse_plan('[{"action": "open", "argument": "door"}]')
    assert steps == [Step("open", "door")]


def test_parse_plan_accepts_fenced_json():
    """Models wrap JSON in prose and code fences constantly."""
    reply = 'Sure!\n```json\n[{"action": "goto", "argument": "desk"}]\n```\nOn my way.'
    assert parse_plan(reply) == [Step("goto", "desk")]


def test_parse_plan_normalises_open_door_alias():
    assert parse_plan('[{"action": "open_door", "argument": "door"}]') == [Step("open", "door")]


def test_parse_plan_drops_unknown_actions():
    """A hallucinated verb must be skipped, not passed to the skill layer."""
    steps = parse_plan('[{"action": "teleport", "argument": "moon"}, {"action": "where"}]')
    assert steps == [Step("where", None)]


@pytest.mark.parametrize("reply", ["", "I cannot help", "[]", "[1, 2, 3]", "{not json}"])
def test_parse_plan_handles_junk(reply):
    assert parse_plan(reply) == []


def test_llm_planner_falls_back_when_model_missing():
    """A missing or broken model must degrade to rules, not raise."""
    planner = LLMPlanner(model_id="definitely/not-a-real-model")
    plan = planner.plan("オフィスのドアを開けて")
    assert plan.steps and plan.steps[0] == Step("open", "door")


def test_plan_truthiness():
    assert not Plan([])
    assert Plan([Step("where")])


# -- skills --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def skills():
    robot = Robot("office.xml", keyframe="lobby")
    yield Skills(robot, ColorGrounder())
    robot.close()


def test_report_position_names_the_room(skills):
    skills.robot.reset("lobby")
    result = skills.report_position()
    assert result.ok
    assert result.data["room"] == "in the lobby"


def test_describe_view_lists_visible_objects(skills):
    skills.robot.reset("lobby")
    skills.robot.stand(0.3)
    result = skills.describe_view()
    assert result.ok
    assert "door" in result.message


def test_unknown_action_is_reported_not_raised(skills):
    result = skills.run("dance")
    assert not result.ok
    assert "do not know how" in result.message


def test_goto_reports_failure_for_missing_target(skills):
    skills.robot.reset("lobby")
    skills.nav.grounder = ColorGrounder()
    result = skills.run("goto", "a purple giraffe")
    assert not result.ok


@pytest.mark.slow
def test_open_door_end_to_end(skills):
    """The demo, minus the messaging: walk across the office and get through a door."""
    skills.robot.reset("lobby")
    start_y = skills.robot.position[1]

    result = skills.run("open", "door")

    assert result.ok, f"door skill failed: {result.message}"
    assert result.data["swing_degrees"] > 20, "door barely moved"
    # The doorway is at y=+1.0; being past it proves it actually went through.
    assert skills.robot.position[1] > 1.0, (
        f"did not pass through the doorway (y={skills.robot.position[1]:.2f}, started {start_y:.2f})"
    )
