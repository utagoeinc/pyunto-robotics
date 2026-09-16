"""Which robots exist, and how a customer adds their own.

The bundled four are registered on import. A third party ships their machine by declaring an
entry point in their own package::

    # their pyproject.toml
    [project.entry-points."pyunto_robotics.robots"]
    acme = "acme_robot:setup"

After `pip install acme-robot`, `pyunto-robotics demo --robot acme` works with no change here.
That is the difference between a demo with four robots in it and an SDK.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

__all__ = ["RobotSetup", "get", "names", "register", "setups"]


@dataclass
class RobotSetup:
    """What one robot needs that the others do not."""

    name: str
    # The MuJoCo scene, or "" for a robot with no body to simulate (see demo.py).
    scene: str
    domain: object
    # Builds the skills object. Takes (robot, grounder) and returns anything with a
    # `run(action, argument, where, expect) -> SkillResult` method -- see api.RobotSkills.
    skills: Callable[..., object]
    # Builds the gait, or None for the humanoid default.
    gait: Callable[[], object] | None = None
    default_keyframe: str = "start"
    keyframe_help: str = ""
    examples: tuple[str, ...] = ()
    # How the window frames this machine. A rover wants a wider shot than a humanoid.
    camera: object | None = None
    # How far this robot's depth camera reports. Outdoor scenes need much more than the
    # indoor default: past the limit every target collapses onto it, and the robot drives
    # confidently to a point short of the real one.
    max_depth: float = 12.0
    # Builds the planner. `None` means "build a Domain planner from `domain`", which is what
    # every robot here does. The seam is kept for a customer whose robot needs its own.
    planner: Callable[[bool], object] | None = None
    # What this machine says when it joins the space, in its own voice.
    #
    # There was one greeting for everything, and it described a humanoid: it offered to raise
    # its right hand and to photograph the room when it was done. A rover has no hands, and
    # the watching flat has no body at all -- a person messaging it is addressing a house.
    # Telling somebody they are talking to the wrong kind of thing is a bad first sentence,
    # so each robot brings its own. `{example_a}` and `{example_b}` are filled from `examples`.
    greeting: str = ""


_REGISTRY: dict[str, RobotSetup] = {}
_LOADED = False


def register(key: str, setup: RobotSetup) -> None:
    """Add a robot under a short name (the one `--robot` takes)."""
    _REGISTRY[key] = setup


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    from . import robots  # noqa: PLC0415, F401 - registers the bundled machines

    # Third-party robots, if any are installed.
    try:
        from importlib.metadata import entry_points  # noqa: PLC0415

        for ep in entry_points(group="pyunto_robotics.robots"):
            try:
                setup = ep.load()
                register(ep.name, setup() if callable(setup) else setup)
                log.info("registered third-party robot %r", ep.name)
            except Exception:  # noqa: BLE001 - one bad plugin must not break the others
                log.warning("could not load robot plugin %r", ep.name, exc_info=True)
    except Exception:  # noqa: BLE001
        log.debug("entry point discovery unavailable", exc_info=True)


def names() -> list[str]:
    _load()
    return sorted(_REGISTRY)


def setups() -> dict[str, RobotSetup]:
    _load()
    return dict(_REGISTRY)


def get(name: str) -> RobotSetup:
    _load()
    if name not in _REGISTRY:
        raise KeyError(f"unknown robot {name!r}; known: {', '.join(names())}")
    return _REGISTRY[name]
