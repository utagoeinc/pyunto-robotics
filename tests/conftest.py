"""Test-run settings shared by every test module.

PYUNTO_TEST_NO_GL=1 is for machines that cannot create an OpenGL context at all -- GitHub's
macOS runners are virtual machines with no GPU, and CGL refuses them a pixel format. There a
test that fails *because the context could not be created* is reported as skipped, with the
reason, rather than as a failure of the robot. Anything else still fails. Without the variable
nothing changes: on a real machine a GL error is a real failure.
"""
from __future__ import annotations

import os

import pytest

_GL_ERRORS = ("gladLoadGL", "invalid pixel format", "CGLError", "OSMesa", "eglInitialize",
              "EGL_", "GLFW", "OpenGL context")


def _is_gl_context_error(exc: BaseException | None) -> bool:
    seen = 0
    while exc is not None and seen < 10:
        text = f"{type(exc).__name__}: {exc}"
        if any(marker in text for marker in _GL_ERRORS):
            return True
        exc = exc.__cause__ or exc.__context__
        seen += 1
    return False


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):  # noqa: ANN001, ANN201
    try:
        return (yield)
    except Exception as exc:
        if os.environ.get("PYUNTO_TEST_NO_GL") and _is_gl_context_error(exc):
            pytest.skip(f"no OpenGL context on this machine: {exc}")
        raise
