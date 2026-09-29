"""The examples are the first code a robot maker runs. They must work as written."""
from __future__ import annotations

import importlib.util
import pathlib

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_urdf_arm_moves_on_raise_and_returns_on_lower():
    ex = _load("urdf_robot")
    robot = ex.URDFRobot(ex.load_urdf(ex.DEFAULT_URDF))
    assert robot.joints == ["shoulder", "elbow"]
    assert robot.run("raise").ok
    shoulder, _ = robot.angles()
    assert shoulder > 1.0, "the shoulder should have lifted towards its upper limit"
    assert robot.run("lower").ok
    assert abs(robot.angles()[0]) < 0.1
    assert not robot.run("fly").ok


def test_urdf_agent_is_called_the_way_robotbackend_calls_it():
    ex = _load("urdf_robot")
    agent = ex.DirectAgent(ex.URDFRobot(ex.load_urdf(ex.DEFAULT_URDF)), "🤖 Arm")
    # RobotBackend: `self.agent.execute(instruction, report=reporter)`
    assert agent.execute("@Arm wave", report=None).reply() == "I waved."


def test_my_robot_agent_accepts_the_report_argument():
    src = (EXAMPLES / "my_robot.py").read_text(encoding="utf-8")
    assert "def execute(self, text: str, report=None)" in src
