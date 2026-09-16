"""The square has to draw itself.

`showqr` is the documented first step, and the whole flow is "scan this". Without `qrcode`
it printed a JSON payload instead -- correct, unscannable, and a dead end for somebody
following the README. qrcode was an optional extra of pyunto-agent, which is right for the
agent (pairing is one feature among several) and wrong here (it is the way in).
"""

from __future__ import annotations

import pathlib
import re

import pytest


def test_qrcode_is_importable():
    """A hard dependency, not an extra. If this fails, `showqr` cannot draw anything."""
    import qrcode  # noqa: F401, PLC0415


def test_a_square_is_actually_rendered():
    from pyunto_agent.pairing import encode_payload, pairing_payload, render_qr

    payload = pairing_payload(
        user_id="00000000-0000-0000-0000-000000000000",
        display_name="🤖 Test",
        public_key="AAAA",
        operator="",
        runtime="self_hosted",
    )
    square = render_qr(encode_payload(payload))
    assert square, "render_qr returned nothing; the payload would be printed as raw JSON"
    # A QR code drawn as text is blocks, not braces.
    assert "█" in square or "▄" in square
    assert "{" not in square


def test_the_dependency_is_declared_with_the_qr_extra():
    """Guard the declaration too: an import test passes in a dev venv that has it anyway."""
    text = pathlib.Path("pyproject.toml").read_text(encoding="utf-8")
    dependencies = text[text.index("dependencies = ["):text.index("[project.optional-dependencies]")]
    assert re.search(r'"pyunto-agent\[qr\] @', dependencies), (
        "pyunto-agent must be depended on with its [qr] extra, or `showqr` prints JSON"
    )
