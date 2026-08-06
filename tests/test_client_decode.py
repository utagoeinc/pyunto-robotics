"""Message-decoding tests.

REST responses are snake_case; Socket.IO payloads are camelCase inside `data`. Both shapes must
produce the same IncomingMessage. These tests use a stub key provider so they run offline.
"""

from __future__ import annotations

import os

import pytest

from pyunto_robotics.comms import crypto as c
from pyunto_robotics.comms.client import PyuntoClient

SPACE = "7b49afe6-2f4b-4cad-a08c-f36cc3acb688"
KEY = os.urandom(32)


class StubKeys:
    """SpaceKeyProvider that always returns the same key."""

    def get_key(self, chat_space_id: str) -> bytes:  # noqa: ARG002
        return KEY


class StubSession:
    base_url = "https://example.invalid"
    identity = None

    def auth_header(self) -> dict[str, str]:
        return {}


@pytest.fixture
def client() -> PyuntoClient:
    return PyuntoClient(StubSession(), StubKeys())


def _encrypted(text: str) -> dict:
    return c.encrypt_message(text, KEY).to_wire()


def test_decode_rest_message_snake_case(client):
    text = "オフィスのドアを開けて"
    msg = client._decode_rest_message(
        {
            "uuid": "m1",
            "content": c.ENCRYPTED_PLACEHOLDER,
            "encrypted_content": _encrypted(text),
            "encryption_metadata": c.encryption_metadata(SPACE),
            "chat_thread_id": "t1",
            "sender": {"uuid": "ABC-DEF", "display_name": "Tom"},
        }
    )
    assert msg is not None
    assert msg.text == text
    assert msg.thread_id == "t1"
    assert msg.sender_uuid == "abc-def"  # normalized to lowercase for self-comparison


def test_decode_ws_message_camel_case(client):
    """Socket.IO uses encryptedContent / encryptionMetadata / chatSpaceId."""
    text = "ミーティングルームに行って"
    msg = client._decode_ws_message(
        {
            "uuid": "m2",
            "content": c.ENCRYPTED_PLACEHOLDER,
            "encryptedContent": _encrypted(text),
            "encryptionMetadata": c.encryption_metadata(SPACE),
            "chatSpaceId": SPACE,
            "threadId": "t2",
            "sender": {"uuid": "XYZ", "display_name": "Tom"},
        }
    )
    assert msg is not None
    assert msg.text == text
    assert msg.thread_id == "t2"
    assert msg.sender_uuid == "xyz"


def test_space_id_recovered_from_metadata(client):
    """GET /api/messages/:threadId omits chat_space_id; it must come from encryption_metadata."""
    msg = client._decode_rest_message(
        {
            "uuid": "m3",
            "content": c.ENCRYPTED_PLACEHOLDER,
            "encrypted_content": _encrypted("hi"),
            "encryption_metadata": c.encryption_metadata(SPACE),
            "chat_thread_id": "t3",
            "sender": {"uuid": "a", "display_name": "Tom"},
            # note: no chat_space_id field
        }
    )
    assert msg is not None
    assert msg.chat_space_id.lower() == SPACE.lower()
    assert msg.text == "hi"


def test_plaintext_message_passes_through(client):
    """Unencrypted messages still have a readable body."""
    msg = client._decode_ws_message(
        {
            "uuid": "m4",
            "content": "plain text",
            "chatSpaceId": SPACE,
            "threadId": "t4",
            "sender": {"uuid": "a", "display_name": "Tom"},
        }
    )
    assert msg is not None
    assert msg.text == "plain text"


def test_undecryptable_message_does_not_raise(client):
    """A message encrypted with a different key must degrade, not crash the listener."""
    other = c.encrypt_message("secret", os.urandom(32)).to_wire()
    msg = client._decode_ws_message(
        {
            "uuid": "m5",
            "content": c.ENCRYPTED_PLACEHOLDER,
            "encryptedContent": other,
            "encryptionMetadata": c.encryption_metadata(SPACE),
            "chatSpaceId": SPACE,
            "threadId": "t5",
            "sender": {"uuid": "a", "display_name": "Tom"},
        }
    )
    assert msg is not None
    assert msg.text == c.ENCRYPTED_PLACEHOLDER  # falls back rather than raising


def test_message_without_space_id_is_skipped(client):
    assert client._decode_ws_message({"uuid": "m6", "content": "x"}) is None
