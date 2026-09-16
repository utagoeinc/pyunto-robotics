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


class TestChoosingARobot:
    """`showqr` used to pick one on its own and open it without saying which.

    The terminal printed a list of "other robots" that did not include the one actually
    running, and nothing connected the square just scanned to a window appearing.
    """

    def test_a_number_picks_that_robot(self):
        with mock.patch("sys.stdin.isatty", return_value=True), \
             mock.patch("builtins.input", return_value="3"):
            from pyunto_robotics import registry
            expected = sorted(registry.names(), key=lambda k: (k != cli.DEFAULT_ROBOT, k))[2]
            assert cli._choose_robot(None) == expected

    def test_a_name_picks_that_robot(self):
        with mock.patch("sys.stdin.isatty", return_value=True), \
             mock.patch("builtins.input", return_value="pet"):
            assert cli._choose_robot(None) == "pet"

    def test_enter_takes_the_first_one(self):
        with mock.patch("sys.stdin.isatty", return_value=True), \
             mock.patch("builtins.input", return_value=""):
            assert cli._choose_robot(None) == cli.DEFAULT_ROBOT

    def test_nonsense_asks_again(self, capsys):
        with mock.patch("sys.stdin.isatty", return_value=True), \
             mock.patch("builtins.input", side_effect=["banana", "99", "mars"]):
            assert cli._choose_robot(None) == "mars"
        assert "Not one of" in capsys.readouterr().out

    def test_ctrl_c_at_the_question_opens_nothing(self):
        with mock.patch("sys.stdin.isatty", return_value=True), \
             mock.patch("builtins.input", side_effect=KeyboardInterrupt):
            assert cli._choose_robot(None) is None

    def test_a_pipe_is_not_asked(self, capsys):
        """A service or CI job has nobody to answer; blocking on input would hang it."""
        with mock.patch("sys.stdin.isatty", return_value=False):
            assert cli._choose_robot(None) == cli.DEFAULT_ROBOT
        assert "not a terminal" in capsys.readouterr().out

    def test_robot_flag_skips_the_question(self):
        with mock.patch("builtins.input", side_effect=AssertionError("should not ask")):
            assert cli._choose_robot("hotel") == "hotel"

    def test_the_errand_robot_is_offered_first(self, capsys):
        """Alphabetical order would lead with the hotel cleaner."""
        with mock.patch("sys.stdin.isatty", return_value=True), \
             mock.patch("builtins.input", return_value=""):
            cli._choose_robot(None)
        assert f"1. {cli.DEFAULT_ROBOT}" in capsys.readouterr().out


def test_what_is_running_is_named_with_its_own_command(capsys):
    """The old list showed every robot EXCEPT the one that had just started."""
    cli._print_what_to_try("pet")
    printed = capsys.readouterr().out
    assert "--robot pet" in printed, "the running robot's own command is missing"
    running = printed.index("--robot pet")
    others = printed.index("To open a different robot afterwards")
    assert running < others, "it should be named before the alternatives, not among them"


def test_declining_at_the_question_leaves_the_robot_paired(capsys):
    fake_connection = mock.Mock()
    fake_connection.identity.display_name = "🤖 Test"
    fake_connection.identity_store.public_key_b64 = "AAAA"
    fake_connection.user_id = "00000000-0000-0000-0000-000000000000"

    with mock.patch.object(cli, "connect", return_value=fake_connection), \
         mock.patch("pyunto_agent.pairing.wait_for_scan", return_value="space-1"), \
         mock.patch.object(cli, "_choose_robot", return_value=None), \
         mock.patch("pyunto_robotics.demo.run_demo") as run_demo:
        assert cli.main(["showqr"]) == 0
    assert not run_demo.called
    assert "stays paired" in capsys.readouterr().out
