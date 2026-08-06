"""Wire-format tests for the AES-256-GCM layer.

These lock in compatibility with the iOS/Android clients. If any of these fail, the robot's
messages will be unreadable in the app (or vice versa).
"""

from __future__ import annotations

import base64
import os

import pytest

from pyunto_robotics.comms import crypto as c


def test_roundtrip_text():
    key = os.urandom(32)
    for text in ["hello", "オフィスのドアを開けて", "emoji 🤖 and\nnewline", ""]:
        assert c.decrypt_message(c.encrypt_message(text, key), key) == text


def test_wire_format_matches_ios():
    """ciphertext must be base64(ct || tag16) and nonce must be 12 bytes.

    Mirrors SymmetricCryptoService.swift, which appends sealedBox.tag to sealedBox.ciphertext.
    """
    key = os.urandom(32)
    plaintext = b"x" * 100
    enc = c.encrypt(plaintext, key)
    combined = base64.b64decode(enc.ciphertext)
    nonce = base64.b64decode(enc.nonce)

    assert len(nonce) == c.NONCE_SIZE == 12
    assert len(combined) == len(plaintext) + c.TAG_SIZE  # tag appended, not prepended


def test_decrypt_accepts_line_wrapped_base64():
    """Android historically emitted wrapped base64; decoding must tolerate it."""
    key = os.urandom(32)
    enc = c.encrypt_message("wrapped", key)
    wrapped = c.EncryptedData(
        ciphertext="\n".join(enc.ciphertext[i : i + 20] for i in range(0, len(enc.ciphertext), 20)),
        nonce=enc.nonce,
    )
    assert c.decrypt_message(wrapped, key) == "wrapped"


def test_tampered_ciphertext_is_rejected():
    key = os.urandom(32)
    enc = c.encrypt_message("secret", key)
    raw = bytearray(base64.b64decode(enc.ciphertext))
    raw[0] ^= 0x01
    tampered = c.EncryptedData(base64.b64encode(bytes(raw)).decode(), enc.nonce)
    with pytest.raises(c.CryptoError):
        c.decrypt_message(tampered, key)


def test_wrong_key_is_rejected():
    enc = c.encrypt_message("secret", os.urandom(32))
    with pytest.raises(c.CryptoError):
        c.decrypt_message(enc, os.urandom(32))


def test_bad_key_size():
    with pytest.raises(c.CryptoError):
        c.encrypt(b"data", os.urandom(16))


@pytest.mark.parametrize(
    "value,expected_nonce",
    [
        ({"ciphertext": "a", "nonce": "b"}, "b"),  # object (REST)
        ('{"ciphertext":"a","nonce":"b"}', "b"),  # JSON string (server fallback path)
    ],
)
def test_parse_encrypted_content_both_shapes(value, expected_nonce):
    """The server may return encrypted_content as an object OR a JSON string."""
    parsed = c.parse_encrypted_content(value)
    assert parsed is not None
    assert parsed.nonce == expected_nonce


@pytest.mark.parametrize("value", [None, "not json", 42, {}, {"ciphertext": "a"}])
def test_parse_encrypted_content_rejects_junk(value):
    assert c.parse_encrypted_content(value) is None


def test_is_encrypted():
    assert c.is_encrypted({"algorithm": "symmetric"})
    assert c.is_encrypted('{"algorithm":"symmetric"}')
    assert not c.is_encrypted({"algorithm": "sealed"})
    assert not c.is_encrypted(None)


def test_encryption_metadata_uppercases_space_id():
    """Clients send the space id uppercased; the server matches Socket.IO rooms on that form."""
    meta = c.encryption_metadata("7b49afe6-2f4b-4cad-a08c-f36cc3acb688")
    assert meta["chat_space_id"] == "7B49AFE6-2F4B-4CAD-A08C-F36CC3ACB688"
    assert meta["algorithm"] == "symmetric"


def test_space_key_roundtrip():
    key = os.urandom(32)
    assert c.string_to_key(c.key_to_string(key)) == key


def test_space_key_wrong_length_rejected():
    with pytest.raises(c.CryptoError):
        c.string_to_key(base64.b64encode(os.urandom(16)).decode())
