"""The example is the SDK's selling point, so it has to actually run.

`examples/my_robot.py` is what somebody reads to decide whether attaching their own machine is
plausible. It still told them to pass a six-character pairing code, which no longer exists --
and nothing here ever executed it, so the file could rot unnoticed.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest.mock as mock

import pytest

EXAMPLE = pathlib.Path("examples/my_robot.py")


def load():
    spec = importlib.util.spec_from_file_location("my_robot_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_connection():
    connection = mock.Mock()
    connection.identity.display_name = "My Robot"
    connection.user_id = "00000000-0000-0000-0000-000000000000"
    connection.identity_store.public_key_b64 = "AAAA"
    return connection


def test_it_mentions_no_pairing_code():
    """Pairing is a QR code somebody scans; there is no code to type."""
    text = EXAMPLE.read_text(encoding="utf-8")
    for gone in ("--pair", "K3F9QZ", "pairing code"):
        assert gone not in text, f"the example still refers to {gone!r}"


def test_the_skills_answer():
    robot = load().MyRobot()
    assert robot.run("goto", "the door").ok
    assert "the door" in robot.run("goto", "the door").message
    assert not robot.run("fly").ok, "an unknown verb must be reported, not raised"


def test_it_draws_a_square_pairs_and_listens(capsys, monkeypatch):
    """The whole path, with the network and the bridge stubbed."""
    monkeypatch.setattr(sys, "argv", ["my_robot.py"])
    module = load()
    with mock.patch.object(module, "connect", return_value=fake_connection()), \
         mock.patch.object(module, "wait_for_scan", return_value="space-1"), \
         mock.patch.object(module, "Bridge") as bridge:
        bridge.return_value.run.side_effect = KeyboardInterrupt
        assert module.main() == 0

    printed = capsys.readouterr().out
    assert "█" in printed or "▄" in printed, "no QR code was drawn"
    assert "paired ✓" in printed
    assert bridge.return_value.run.called, "it never started listening"


def test_ctrl_c_while_waiting_is_not_a_crash(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["my_robot.py"])
    module = load()
    with mock.patch.object(module, "connect", return_value=fake_connection()), \
         mock.patch.object(module, "wait_for_scan", side_effect=KeyboardInterrupt):
        assert module.main() == 0
    assert "Stopped" in capsys.readouterr().out


def test_nobody_scanning_is_reported(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["my_robot.py"])
    module = load()
    with mock.patch.object(module, "connect", return_value=fake_connection()), \
         mock.patch.object(module, "wait_for_scan", return_value=None):
        assert module.main() == 1
    assert "Nobody scanned" in capsys.readouterr().out
