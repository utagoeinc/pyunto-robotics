"""Agent tests: message in, action out, reply back.

These use a fake Pyunto client so they run offline, and a stub skill layer so they test the
agent's own behaviour -- planning, step limits, error handling -- rather than any one robot's
abilities. The real round trip through api.pyunto.com is exercised by `pyunto-robotics demo`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from pyunto_robotics.agent import RobotAgent
from pyunto_robotics.brain.result import SkillResult


class FakeClient:
    """Records what would have been sent instead of talking to a server."""

    def __init__(self):
        self.sent: list[tuple[str, str, str | None]] = []

    def send(self, chat_space_id: str, text: str, thread_id: str | None = None, **kw):
        self.sent.append((chat_space_id, text, thread_id))
        return {"uuid": "fake"}

    def stop(self) -> None:
        pass


@dataclass
class Step:
    action: str
    argument: str = ""
    where: str = ""
    expect: Any = None

    def __str__(self) -> str:
        return f"{self.action} {self.argument}".strip()


@dataclass
class Plan:
    steps: list[Step] = field(default_factory=list)
    reply: str | None = None


class StubSkills:
    """Succeeds at everything, and remembers what it was asked to do."""

    actions = ("goto", "charge", "report")

    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, action, argument=None, where=None, expect=None):  # noqa: ANN001
        self.calls.append(action)
        return SkillResult(True, f"Did {action}.")


def agent_with(planner, skills=None, **kw) -> RobotAgent:
    """A RobotAgent with no simulator behind it: these tests are about the agent."""
    return RobotAgent(
        robot=None, grounder=None, planner=planner, skills=skills or StubSkills(), **kw
    )


def test_skills_are_required_rather_than_guessed():
    """There is no sensible default: what a robot can do is what differs between robots."""
    with pytest.raises(ValueError, match="skills"):
        RobotAgent(robot=None, grounder=None, planner=_Planner([]), skills=None)


class _Planner:
    def __init__(self, steps, reply=None):
        self._steps, self._reply = steps, reply

    def plan(self, text):  # noqa: ANN001
        return Plan(list(self._steps), self._reply)


def test_an_instruction_becomes_actions():
    skills = StubSkills()
    agent = agent_with(_Planner([Step("goto", "park"), Step("charge")]), skills)
    execution = agent.execute("go to the park and charge")
    assert execution.ok
    assert skills.calls == ["goto", "charge"]


def test_an_unknown_instruction_explains_itself():
    """Silence is the worst answer: the person cannot tell if it landed."""
    agent = agent_with(_Planner([]))
    execution = agent.execute("make me a coffee")
    assert execution.ok
    reply = execution.reply()
    assert "make me a coffee" in reply
    # And it says what would have worked.
    assert "goto" in reply


def test_a_failed_skill_is_reported_not_raised():
    class Failing(StubSkills):
        def run(self, action, argument=None, where=None, expect=None):  # noqa: ANN001
            return SkillResult(False, "The way was blocked.")

    agent = agent_with(_Planner([Step("goto", "park")]), Failing(), max_replans=0)
    execution = agent.execute("go to the park")
    assert execution.ok is False
    assert "blocked" in execution.reply()


def test_plan_length_is_capped_and_said_so():
    """Silently doing less than asked is the failure worth being loud about."""
    steps = [Step("goto", f"place{i}") for i in range(6)]
    agent = agent_with(_Planner(steps), max_steps_per_message=3, max_replans=0)
    execution = agent.execute("do six things")
    assert execution.ok is False
    assert "6 steps" in execution.reply() or "not done" in execution.reply()


def test_measurements_are_kept_per_step():
    """A reply can say what was measured, not only whether it worked."""
    class Measuring(StubSkills):
        def run(self, action, argument=None, where=None, expect=None):  # noqa: ANN001
            return SkillResult(True, "Arrived.", {"distance_m": 4.2})

    agent = agent_with(_Planner([Step("goto", "park")]), Measuring())
    execution = agent.execute("go to the park")
    assert execution.data[0]["distance_m"] == 4.2
