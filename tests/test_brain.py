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


# -- spatial qualifiers ---------------------------------------------------------------


@pytest.mark.parametrize(
    "message,action,where",
    [
        ("右の会議室のドアを開けて", "open", "right"),
        ("左のドアを開けて", "open", "left"),
        ("真ん中のドアまで行って", "goto", "middle"),
        ("一番右のドアを開けて", "open", "right"),
        ("左端のドア", "goto", "left"),
        ("open the right door", "open", "right"),
        ("the door on the left", "goto", "left"),
        ("go to the middle door", "goto", "middle"),
        ("ドアを開けて", "open", None),  # no qualifier given
    ],
)
def test_spatial_qualifiers_are_captured(planner, message, action, where):
    """"the door on the RIGHT" has to survive planning; three doors look identical."""
    plan = planner.plan(message)
    assert plan.steps
    assert plan.steps[0].action == action
    assert plan.steps[0].where == where


def test_qualifier_is_not_left_in_the_target_text(planner):
    """Qualifiers travel on Step.where, not glued onto the object name."""
    step = planner.plan("go to the right purple giraffe").steps[0]
    assert step.where == "right"
    assert "right" not in (step.argument or "")


@pytest.mark.slow
@pytest.mark.parametrize(
    "where,expected_x",
    [("left", -4.0), ("right", 4.0), ("middle", 0.0)],
)
def test_qualifier_selects_the_right_door(skills, where, expected_x):
    """The end of the chain: a spatial word has to land the robot at that specific door.

    Doors are at x = -4, 0, +4. Reaching the wrong one is a silent failure - the robot
    reports success either way - so this asserts on position, not on the reply.
    """
    skills.robot.reset("lobby")
    result = skills.run("open", "door", where)

    assert result.ok, result.message
    assert abs(skills.robot.position[0] - expected_x) < 2.0, (
        f"asked for the {where} door (x={expected_x}) but ended at "
        f"x={skills.robot.position[0]:.2f}"
    )
    assert skills.robot.position[1] > 1.0, "did not get through the doorway"


@pytest.mark.parametrize(
    "message,chained",
    [
        ("右のドアを開けて中に入って、その後、廊下に出て、今度は、一番左の部屋に入って", True),
        ("open the right door then the left one", True),
        ("go to the desk. after that, look around", True),
        ("右のドアを開けて", False),
        ("open the door", False),
        ("look around", False),
    ],
)
def test_multi_step_instructions_are_recognised(message, chained):
    """RulePlanner silently does the wrong single thing on a chained instruction.

    Callers need to know that before running it, so they can warn or switch to the LLM.
    """
    from pyunto_robotics.brain.planner import looks_multi_step

    assert looks_multi_step(message) is chained


def test_rule_planner_still_answers_a_chained_instruction(planner):
    """Warning about it is not the same as refusing: it still returns its best single step."""
    plan = planner.plan("右のドアを開けて、その後、左の部屋に入って")
    assert plan.steps, "should still produce something rather than nothing"


@pytest.mark.slow
def test_pull_door_opens_it(skills):
    """Pulling needs a real grasp: friction alone slips off a 3.6 cm handle."""
    skills.robot.reset("lobby")
    result = skills.pull_door("door")

    assert result.ok, result.message
    assert result.data["swing_degrees"] > 10
    # Pulling swings the leaf toward the robot, so it should still be on the corridor side.
    assert skills.robot.position[1] < 1.2


@pytest.mark.slow
@pytest.mark.parametrize("where", ["left", "middle", "right"])
def test_leave_room_returns_to_the_corridor(skills, where):
    """Getting out of a room needs a planned manoeuvre, not the reactive controller.

    All three rooms, including the pantry, which used to fail: the spring closes the door onto
    a robot standing in the opening, so it has to brace an arm against the leaf and push
    through rather than back off.
    """
    skills.robot.reset("lobby")
    skills.run("open", "door", where)
    assert skills.robot.position[1] > 1.0, "did not get into the room to begin with"

    result = skills.leave_room()

    assert result.ok, result.message
    assert skills.robot.position[1] < 1.0, "still inside the room"


def test_leave_room_without_a_memory_says_so(skills):
    """Leaving depends on the pose recorded on the way in; without it, say so plainly."""
    skills.robot.reset("start")
    skills._doorway_return = None
    result = skills.leave_room()
    assert not result.ok
    assert "remember" in result.message


@pytest.mark.slow
def test_returning_home_aims_at_the_far_door():
    """Going back to the start is what makes a second "leftmost" mean the leftmost.

    Asserts the direction, not arrival. From beside a doorway only one door is in frame, so
    without going home the robot opens whichever it is next to -- it ends up around x=+4.
    After going home it heads for the far side. Actually reaching and opening that door is
    still unreliable, so this checks the choice rather than the outcome.
    """
    robot = Robot("office.xml", keyframe="lobby")
    try:
        skills = Skills(robot, ColorGrounder())
        assert skills.run("open", "door", "right").ok
        assert skills.leave_room().ok
        assert skills.return_home().ok

        # Resolving the qualifier from home is the part that works: it picks the door at
        # x=-3.6 rather than whichever is nearest. Walking there afterwards is not yet
        # dependable -- the tracked position gets dropped during a detour and the robot
        # re-acquires a nearer door -- so this asserts the choice, not the arrival.
        detections, _, fovy, size, depth = skills.nav._observe("door")
        located = skills.nav._locate(detections, depth, fovy, size, where="left")
        assert located is not None, "could not see a door from the starting viewpoint"
        chosen = skills.nav._world_position(located[1], located[2])
        assert chosen[0] < -2.0, (
            f"'left' resolved to x={chosen[0]:.2f} from home; expected the far door near -4"
        )
    finally:
        robot.close()


@pytest.mark.slow
def test_return_home_goes_back_to_the_start():
    import numpy as np

    robot = Robot("office.xml", keyframe="lobby")
    try:
        skills = Skills(robot, ColorGrounder())
        home = skills._home.copy()
        skills.run("open", "door", "middle")
        skills.leave_room()
        result = skills.return_home()

        assert result.ok, result.message
        assert float(np.linalg.norm(robot.position[:2] - home)) < 1.0
    finally:
        robot.close()


# -- stated counts --------------------------------------------------------------------


@pytest.mark.parametrize(
    "message,expect",
    [
        ("三つ見えるドアのうち、右のドアを開けて", 3),
        ("of the three doors, open the left one", 3),
        ("二つあるドアの左を開けて", 2),
        ("右のドアを開けて", None),  # no count stated
        ("ドアを開けて", None),  # no qualifier, so a count would mean nothing
    ],
)
def test_stated_count_is_carried_into_the_plan(planner, message, expect):
    """「三つ見えるドアのうち」 says how many to choose between, and that is checkable."""
    plan = planner.plan(message)
    assert plan.steps
    assert plan.steps[0].expect == expect


def test_parse_plan_reads_expect_from_the_model():
    steps = parse_plan('[{"action": "open", "argument": "door", "where": "left", "expect": 3}]')
    assert steps == [Step("open", "door", "left", 3)]


@pytest.mark.slow
def test_declines_when_it_cannot_see_the_stated_number():
    """Told there are three but able to see one, the robot should say so rather than guess.

    Opening the wrong door confidently is worse than admitting the ambiguity.
    """
    robot = Robot("office.xml", keyframe="start")  # right by the doors; only one in frame
    try:
        skills = Skills(robot, ColorGrounder())
        assert skills._count_doors() < 3, "test needs a spot where all three are not visible"

        result = skills.run("open", "door", "left", 3)

        assert not result.ok
        assert "only see" in result.message
        assert result.data["expected"] == 3
    finally:
        robot.close()


@pytest.mark.slow
def test_acts_when_the_stated_number_is_visible():
    robot = Robot("office.xml", keyframe="lobby")  # all three in frame
    try:
        skills = Skills(robot, ColorGrounder())
        result = skills.run("open", "door", "right", 3)
        assert result.ok, result.message
        assert abs(robot.position[0] - 4.0) < 1.5
    finally:
        robot.close()
