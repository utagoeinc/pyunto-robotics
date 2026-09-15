"""A robot must not sign in as the person it is meant to answer.

This cost an afternoon to diagnose, twice, from opposite ends: the robot sits there reporting
"listening", the app reports the message sent, the server reports it delivered -- and nothing
happens, because the agent drops its own posts and the robot and the person are the same
account. Nothing in any log says so. Hence a refusal rather than a warning.
"""

from __future__ import annotations

import pytest

from pyunto_agent.auth import AuthError
from pyunto_robotics.connect import connect


def test_signing_in_as_a_person_is_refused(monkeypatch):
    monkeypatch.setenv("PYUNTO_EMAIL", "someone@example.com")
    monkeypatch.setenv("PYUNTO_PASSWORD", "secret")
    monkeypatch.delenv("PYUNTO_ROBOT_ACCOUNT", raising=False)

    with pytest.raises(AuthError) as excinfo:
        connect(display_name="S1")

    message = str(excinfo.value)
    # The message has to name the consequence, not the cosmetic symptom. An earlier version
    # warned that the participant list would miss its 🤖 label, which is true and useless.
    assert "ignores its own posts" in message
    assert "someone@example.com" in message
    # And it has to say what to do instead.
    assert "PYUNTO_EMAIL=" in message


def test_the_escape_hatch_is_honoured(monkeypatch):
    """A registered account genuinely separate from the person is a real setup."""
    monkeypatch.setenv("PYUNTO_EMAIL", "robot@example.com")
    monkeypatch.setenv("PYUNTO_PASSWORD", "secret")
    monkeypatch.setenv("PYUNTO_ROBOT_ACCOUNT", "1")

    with pytest.raises(AuthError) as excinfo:
        connect(display_name="S1")

    # It gets past the guard and fails at the login instead, which is the honest outcome for
    # a made-up account.
    assert "ignores its own posts" not in str(excinfo.value)


@pytest.mark.parametrize("value", ["", "   "])
def test_a_blank_credential_means_no_account(monkeypatch, value):
    """`PYUNTO_EMAIL= pyunto-robotics demo` is the documented way to override a .env."""
    monkeypatch.setenv("PYUNTO_EMAIL", value)
    monkeypatch.setenv("PYUNTO_PASSWORD", value)
    monkeypatch.delenv("PYUNTO_ROBOT_ACCOUNT", raising=False)

    # No AuthError about self-answering: a blank value is "no account", so the robot makes
    # one of its own. This reaches the network, so only assert on what we can here.
    try:
        connect(display_name="S1")
    except AuthError as e:
        assert "ignores its own posts" not in str(e)
