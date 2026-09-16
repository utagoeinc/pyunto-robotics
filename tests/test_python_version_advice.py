"""Never send somebody to a command that cannot help them.

`mlx-vlm` has no build for Python 3.13+. On those versions the `[llm]` extra's environment
markers resolve to "nothing to do", so `pip install` reports success and the first sign of
trouble is `download_model` refusing to run -- which is exactly how this was reported.

Both messages must therefore say what to DO, and neither may point a 3.13+ user at
`download_model`, which will only show them the same refusal again.
"""

from __future__ import annotations

import sys
import unittest.mock as mock

import pytest

from pyunto_robotics import cli, download_model


class FakeVersion(tuple):
    """A version_info stand-in: tuple comparisons and .major/.minor both work."""

    def __new__(cls, major: int, minor: int):
        self = super().__new__(cls, (major, minor, 0))
        self.major, self.minor = major, minor
        return self


@pytest.mark.parametrize("minor", [13, 14, 15])
def test_download_model_refuses_new_pythons_with_a_way_forward(minor: int):
    with mock.patch.object(download_model.sys, "version_info", FakeVersion(3, minor)):
        ok, message = download_model.supported()
    assert not ok
    assert f"3.{minor}" in message
    assert "3.12" in message, "the message must name a version that works"
    assert "venv" in message, "the fix is a new environment; say so"


def test_the_cli_does_not_send_a_314_user_to_a_command_that_refuses(capsys):
    """On 3.14 `download_model` cannot help, so pointing at it is a dead end."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def without_mlx(name, *args, **kwargs):
        if name == "mlx_vlm":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    with mock.patch.object(cli.sys, "version_info", FakeVersion(3, 14)), \
         mock.patch("builtins.__import__", without_mlx):
        assert cli._understanding(False) is False
    printed = capsys.readouterr().out
    assert "download_model" not in printed, "3.14 users must not be sent to download_model"
    assert "3.11 or 3.12" in printed


def test_the_cli_does_point_a_312_user_at_the_download(capsys):
    """Where the model CAN be installed, say how."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def without_mlx(name, *args, **kwargs):
        if name == "mlx_vlm":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    with mock.patch.object(cli.sys, "version_info", FakeVersion(3, 12)), \
         mock.patch.object(cli.sys, "platform", "darwin"), \
         mock.patch("builtins.__import__", without_mlx):
        assert cli._understanding(False) is False
    assert "download_model" in capsys.readouterr().out


def test_the_readme_warns_before_the_commands_not_after():
    """The version constraint is useless below the command that depends on it."""
    import pathlib

    text = pathlib.Path("README.md").read_text(encoding="utf-8")
    quick_start = text.index("## Quick start")
    install = text.index("pip install 'pyunto-robotics[llm]", quick_start)
    warning = text.index("3.11 or 3.12", quick_start)
    assert warning < install, "the Python version must be stated before the install command"
