"""Every command path must at least be executed once.

`showqr` gained the ability to open a robot, but `run_demo` was imported inside the `demo`
branch -- so the new path called a name that did not exist, and nothing caught it until
somebody paired a robot for real. The tests covered the pieces and not the route through
them.

These stub out the network and the simulator, so they check the plumbing rather than the
behaviour: does each path reach the code it means to reach, with its imports in scope.
"""

from __future__ import annotations

import unittest.mock as mock

import pytest

from pyunto_robotics import cli


def test_showqr_reaches_run_demo_after_pairing(capsys):
    """The reported crash: UnboundLocalError on `run_demo`."""
    fake_connection = mock.Mock()
    fake_connection.identity.display_name = "🤖 Test"
    fake_connection.identity_store.public_key_b64 = "AAAA"
    fake_connection.user_id = "00000000-0000-0000-0000-000000000000"

    with mock.patch.object(cli, "connect", return_value=fake_connection), \
         mock.patch("pyunto_agent.pairing.wait_for_scan", return_value="space-1"), \
         mock.patch.object(cli, "_reexec_under_mjpython_if_needed"), \
         mock.patch("pyunto_robotics.demo.run_demo", return_value=0) as run_demo:
        assert cli.main(["showqr", "--no-window"]) == 0

    assert run_demo.called, "showqr paired and then never opened the robot"
    assert run_demo.call_args.kwargs["robot_name"] == "solar"


def test_showqr_reexecs_under_mjpython_when_a_window_is_wanted():
    """Without this the square is scanned and no window ever appears on macOS."""
    fake_connection = mock.Mock()
    fake_connection.identity.display_name = "🤖 Test"
    fake_connection.identity_store.public_key_b64 = "AAAA"
    fake_connection.user_id = "00000000-0000-0000-0000-000000000000"

    with mock.patch.object(cli, "connect", return_value=fake_connection), \
         mock.patch("pyunto_agent.pairing.wait_for_scan", return_value="space-1"), \
         mock.patch.object(cli, "_reexec_under_mjpython_if_needed") as reexec, \
         mock.patch("pyunto_robotics.demo.run_demo", return_value=0):
        cli.main(["showqr"])

    reexec.assert_called_once_with(True)


def test_ctrl_c_while_waiting_is_not_a_crash(capsys):
    fake_connection = mock.Mock()
    fake_connection.identity.display_name = "🤖 Test"
    fake_connection.identity_store.public_key_b64 = "AAAA"
    fake_connection.user_id = "00000000-0000-0000-0000-000000000000"

    with mock.patch.object(cli, "connect", return_value=fake_connection), \
         mock.patch("pyunto_agent.pairing.wait_for_scan", side_effect=KeyboardInterrupt):
        assert cli.main(["showqr"]) == 0
    assert "Stopped" in capsys.readouterr().out


def test_nobody_scanning_is_reported_not_crashed(capsys):
    fake_connection = mock.Mock()
    fake_connection.identity.display_name = "🤖 Test"
    fake_connection.identity_store.public_key_b64 = "AAAA"
    fake_connection.user_id = "00000000-0000-0000-0000-000000000000"

    with mock.patch.object(cli, "connect", return_value=fake_connection), \
         mock.patch("pyunto_agent.pairing.wait_for_scan", return_value=None):
        assert cli.main(["showqr"]) == 1
    assert "Nobody scanned" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["robots"], ["--help"]])
def test_the_simple_commands_run(argv):
    try:
        cli.main(argv)
    except SystemExit as exit_:          # --help exits 0
        assert exit_.code == 0
