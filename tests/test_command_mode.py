"""Command mode: a closed vocabulary a customer can supply.

Reading sentences is the default and the reason the product exists. Command mode is the
deliberate opposite, for a site that WANTS a fixed list -- equipment with its own command set,
an operator typing the same six instructions, a safety case that will not accept a model
deciding what was meant. That list is site-specific, so it is data rather than a table
somebody would have to fork the SDK to change.
"""

from __future__ import annotations

import json

import pytest

from pyunto_robotics.brain.domains import DOMAINS, DomainRulePlanner


def site(tmp_path, data: dict):
    path = tmp_path / "commands.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return DOMAINS["pet"].with_commands(path)


def test_equipment_codes_reach_the_action(tmp_path):
    domain = site(tmp_path, {"verbs": {"find": ["FIND-TGT"], "photo": ["CAM-SNAP"]}})
    planner = DomainRulePlanner(domain)
    assert [s.action for s in planner.plan("FIND-TGT").steps] == ["find"]
    assert [s.action for s in planner.plan("CAM-SNAP").steps] == ["photo"]


def test_codes_match_whatever_case_they_are_typed_in(tmp_path):
    """A site writes "FIND-TGT" because that is how the manual spells it."""
    planner = DomainRulePlanner(site(tmp_path, {"verbs": {"find": ["FIND-TGT"]}}))
    assert [s.action for s in planner.plan("find-tgt").steps] == ["find"]
    assert [s.action for s in planner.plan("FIND-TGT").steps] == ["find"]


def test_actions_not_named_keep_their_built_in_words(tmp_path):
    """Override the two verbs your equipment words differently; inherit the rest."""
    planner = DomainRulePlanner(site(tmp_path, {"verbs": {"find": ["FIND-TGT"]}}))
    assert [s.action for s in planner.plan("look around").steps] == ["patrol"]


def test_an_override_replaces_rather_than_adds(tmp_path):
    """A closed vocabulary that still answered to everything would not be closed."""
    planner = DomainRulePlanner(site(tmp_path, {"verbs": {"find": ["FIND-TGT"]}}))
    assert planner.plan("where is the cat").steps == []


def test_objects_can_be_site_positions(tmp_path):
    domain = site(tmp_path, {"objects": {"sill": ["POS-03"]}})
    assert domain.object_in("go to pos-03") == "sill"


def test_an_action_the_robot_does_not_have_is_refused(tmp_path):
    """A command that can never fire is a fault in the file, and startup is when to say so."""
    with pytest.raises(ValueError) as e:
        site(tmp_path, {"verbs": {"teleport": ["BEAM-ME-UP"]}})
    assert "teleport" in str(e.value)
    assert "find" in str(e.value)          # says what the robot can actually do


def test_the_shipped_example_loads_and_works():
    """A broken example is worse than none: it is the first thing a customer copies."""
    planner = DomainRulePlanner(DOMAINS["pet"].with_commands("examples/commands.example.json"))
    assert [s.action for s in planner.plan("RTB").steps] == ["home"]
    assert [s.action for s in planner.plan("PATROL-ALL").steps] == ["patrol"]
