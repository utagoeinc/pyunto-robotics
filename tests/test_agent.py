"""Agent tests: message in, action out, reply back.

These use a fake Pyunto client so they run offline. The real round trip through
api.pyunto.com is exercised by scripts/run_robot.py.
"""

from __future__ import annotations

import pytest

from pyunto_robotics.agent import RobotAgent
from pyunto_robotics.brain.planner import Plan, RulePlanner, Step
from pyunto_robotics.comms.client import IncomingMessage
from pyunto_robotics.perception.grounding import ColorGrounder
from pyunto_robotics.sim.robot import Robot


class FakeClient:
    """Records what would have been sent instead of talking to a server."""

    def __init__(self):
        self.sent: list[tuple[str, str, str | None]] = []

    def send(self, chat_space_id: str, text: str, thread_id: str | None = None, **kw):
        self.sent.append((chat_space_id, text, thread_id))
        return {"uuid": "fake"}

    def stop(self) -> None:
        pass


def _message(text: str) -> IncomingMessage:
    return IncomingMessage(
        uuid="m1",
        text=text,
        thread_id="t1",
        chat_space_id="s1",
        sender_uuid="human",
        sender_name="Tom",
    )


@pytest.fixture(scope="module")
def agent():
    robot = Robot("office.xml", keyframe="lobby")
    yield RobotAgent(robot, ColorGrounder(), client=FakeClient(), planner=RulePlanner())
    robot.close()


def test_conversational_message_gets_a_reply_without_moving(agent):
    agent.robot.reset("lobby")
    before = agent.robot.position.copy()

    execution = agent.execute("こんにちは")

    assert execution.ok
    assert "Hello" in execution.reply()
    assert (agent.robot.position == before).all(), "greeting should not move the robot"


def test_where_am_i(agent):
    agent.robot.reset("lobby")
    execution = agent.execute("where are you")
    assert execution.ok
    assert "lobby" in execution.reply()


def test_unknown_instruction_explains_itself(agent):
    execution = agent.execute("qwertyuiop")
    assert execution.ok  # not understanding is not a failure
    assert "open" in execution.reply().lower()


def test_failed_skill_is_reported_not_raised(agent):
    agent.robot.reset("lobby")
    execution = agent.execute("go to the purple giraffe")
    assert not execution.ok
    assert execution.reply()


def test_plan_length_is_capped():
    """A model that emits a wall of steps has misunderstood; do not run them all."""

    class Runaway:
        def plan(self, message: str) -> Plan:
            return Plan([Step("where") for _ in range(20)])

    robot = Robot("office.xml", keyframe="lobby")
    try:
        a = RobotAgent(robot, ColorGrounder(), planner=Runaway(), max_steps_per_message=3)
        execution = a.execute("anything")
        assert len(execution.messages) == 3
    finally:
        robot.close()


def test_reply_goes_back_to_the_same_thread(agent):
    agent.client.sent.clear()
    agent._handle(_message("where are you"))
    assert agent.pump_once(timeout=1.0)

    assert agent.client.sent, "no reply was sent"
    space_id, text, thread_id = agent.client.sent[-1]
    assert space_id == "s1"
    assert thread_id == "t1", "reply must land in the thread the instruction came from"
    assert text


def test_busy_robot_acknowledges_rather_than_dropping(agent):
    """A second instruction mid-task should be answered, then queued."""
    agent.client.sent.clear()
    agent._busy.set()
    try:
        agent._handle(_message("open the door"))
    finally:
        agent._busy.clear()

    assert agent.client.sent, "busy robot said nothing"
    assert "middle of something" in agent.client.sent[-1][1]
    assert agent._work.qsize() == 1, "the instruction should still be queued"
    agent._work.get_nowait()  # drain so later tests start clean


@pytest.mark.slow
def test_full_instruction_to_door_open(agent):
    """The demo path, without the network: instruction in, door open, sentence out."""
    agent.robot.reset("lobby")
    agent.client.sent.clear()

    execution = agent.execute("オフィスのドアを開けて")

    assert execution.ok, execution.reply()
    assert "opened the door" in execution.reply()
    assert agent.robot.position[1] > 1.0, "robot did not get through the doorway"
