"""A robot from somebody else's package must appear here with no change to this one.

"Bringing your own robot" promises that declaring an entry point is enough. Nothing tested
it, so the promise rested on the code being right rather than on it being checked -- and the
README's example used `acme`, a placeholder that explained none of the three parts.

Rather than build and install a package, this registers an entry point in-process, which is
the same thing `registry._load()` reads.

Clearing `_REGISTRY` is not enough to start over: `_load` registers the bundled robots with
`from . import robots`, which is a no-op once that module has been imported, so a second pass
would find an empty registry. `reset()` re-runs the registration instead.
"""

from __future__ import annotations

import importlib.metadata
import unittest.mock as mock

import pytest

from pyunto_robotics import registry
from pyunto_robotics.api import SkillResult
from pyunto_robotics.brain.domains import DOMAINS
from pyunto_robotics.registry import RobotSetup


def reset() -> None:
    """Make `registry._load()` run again from scratch, bundled robots included."""
    import importlib

    from pyunto_robotics import robots

    registry._LOADED = False
    registry._REGISTRY = {}
    importlib.reload(robots)      # re-registers solar, pet, watch and the rest
    registry._LOADED = False      # reload() set it while registering


class WarehouseAGV:
    """Exactly the shape the README tells somebody to write."""

    def run(self, action, argument=None, where=None, expect=None) -> SkillResult:  # noqa: ANN001
        if action == "goto":
            return SkillResult(True, f"I went to {argument}.")
        return SkillResult(False, f"I do not know how to '{action}' yet.")


def _setup() -> RobotSetup:
    return RobotSetup(
        name="Warehouse AGV (third-party)",
        scene="",                        # no simulator: real hardware
        domain=DOMAINS["mars"],
        skills=lambda robot, grounder: WarehouseAGV(),
        examples=("go to bay 4",),
    )


class FakeEntryPoint:
    name = "warehouse-agv"

    def load(self):
        return _setup


@pytest.fixture
def third_party(monkeypatch):
    """Register a plugin the way an installed package would, for one test."""
    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda *a, **k: [FakeEntryPoint()]
    )
    reset()
    yield
    reset()


def test_it_appears_alongside_the_bundled_robots(third_party):
    names = registry.names()
    assert "warehouse-agv" in names
    for bundled in ("solar", "pet", "watch"):
        assert bundled in names, "a plugin must not displace the bundled robots"


def test_its_skills_run(third_party):
    setup = registry.get("warehouse-agv")
    skills = setup.skills(None, None)
    assert skills.run("goto", "bay 4").message == "I went to bay 4."
    assert not skills.run("fly").ok


def test_a_robot_with_no_scene_is_allowed(third_party):
    """Real hardware has no MuJoCo file. `watch` proves it in-tree; this proves it out."""
    assert registry.get("warehouse-agv").scene == ""


def test_a_broken_plugin_does_not_take_the_others_down(monkeypatch):
    """One bad package must not make `pyunto-robotics robots` fail for everybody."""

    class Exploding:
        name = "broken"

        def load(self):
            raise ImportError("this plugin is not installed properly")

    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda *a, **k: [Exploding()]
    )
    reset()
    try:
        names = registry.names()
        assert "broken" not in names
        assert "solar" in names, "a failing plugin took the bundled robots with it"
    finally:
        reset()
