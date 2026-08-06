"""AES-256-GCM message encryption, wire-compatible with the Pyunto iOS/Android clients.

The reference implementation is pyunto-ios SymmetricCryptoService.swift:39-109. The wire format is:

    ciphertext = base64( ct || tag16 )    # 16-byte GCM tag APPENDED to the ciphertext
    nonce      = base64( 12 random bytes )

Python's AESGCM.encrypt() already returns ct||tag, so it maps 1:1 onto the Swift
`sealedBox.ciphertext + sealedBox.tag`.

Base64 is standard-alphabet and padded. Android historically emitted line-wrapped base64,
so decoding strips \\n and \\r first (same defensive step as the iOS client).
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# GCM tag length in bytes. Fixed by the format; the iOS client hardcodes the same value.
TAG_SIZE = 16
NONCE_SIZE = 12
KEY_SIZE = 32

# Placeholder the clients put in the plaintext `content` field when the real body is encrypted.
ENCRYPTED_PLACEHOLDER = "[Encrypted Message]"


class CryptoError(Exception):
    """Raised when encryption or decryption fails."""


@dataclass(frozen=True)
class EncryptedData:
    """An encrypted payload in the shape the Pyunto API expects."""

    ciphertext: str  # base64(ct || tag)
    nonce: str  # base64(12 bytes)

    def to_wire(self) -> dict[str, str]:
        """The `encrypted_content` object sent to POST /api/messages."""
        return {
            "algorithm": "AES-256-GCM",
            "ciphertext": self.ciphertext,
            "nonce": self.nonce,
        }


def _normalize_b64(s: str) -> str:
    """Strip newlines Android's Base64 may have inserted (iOS does the same)."""
    return s.replace("\n", "").replace("\r", "")


def key_to_string(key: bytes) -> str:
    """Serialize a raw 32-byte space key the way the server stores it."""
    return base64.b64encode(key).decode("ascii")


def string_to_key(s: str) -> bytes:
    """Parse a base64 space key as returned by GET /api/chat-spaces/:uuid/key."""
    key = base64.b64decode(_normalize_b64(s))
    if len(key) != KEY_SIZE:
        raise CryptoError(f"space key must be {KEY_SIZE} bytes, got {len(key)}")
    return key


def encrypt(data: bytes, key: bytes) -> EncryptedData:
    """Encrypt raw bytes. Used for both message bodies and image payloads."""
    if len(key) != KEY_SIZE:
        raise CryptoError(f"key must be {KEY_SIZE} bytes, got {len(key)}")
    nonce = os.urandom(NONCE_SIZE)
    # AESGCM.encrypt returns ct||tag, exactly the layout the Swift client builds by hand.
    combined = AESGCM(key).encrypt(nonce, data, None)
    return EncryptedData(
        ciphertext=base64.b64encode(combined).decode("ascii"),
        nonce=base64.b64encode(nonce).decode("ascii"),
    )


def decrypt(enc: EncryptedData, key: bytes) -> bytes:
    """Decrypt to raw bytes."""
    if len(key) != KEY_SIZE:
        raise CryptoError(f"key must be {KEY_SIZE} bytes, got {len(key)}")
    try:
        combined = base64.b64decode(_normalize_b64(enc.ciphertext))
        nonce = base64.b64decode(_normalize_b64(enc.nonce))
    except Exception as e:  # noqa: BLE001 - surface any base64 problem the same way
        raise CryptoError(f"invalid base64: {e}") from e

    if len(combined) < TAG_SIZE:
        raise CryptoError("ciphertext shorter than the GCM tag")
    if len(nonce) != NONCE_SIZE:
        raise CryptoError(f"nonce must be {NONCE_SIZE} bytes, got {len(nonce)}")

    try:
        return AESGCM(key).decrypt(nonce, combined, None)
    except Exception as e:  # noqa: BLE001 - InvalidTag and friends
        raise CryptoError(f"decryption failed: {e}") from e


def encrypt_message(plaintext: str, key: bytes) -> EncryptedData:
    """Encrypt a UTF-8 message body."""
    return encrypt(plaintext.encode("utf-8"), key)


def decrypt_message(enc: EncryptedData, key: bytes) -> str:
    """Decrypt a message body to text."""
    raw = decrypt(enc, key)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise CryptoError(f"decrypted bytes are not UTF-8: {e}") from e


def parse_encrypted_content(value: Any) -> EncryptedData | None:
    """Read an `encrypted_content` field into EncryptedData.

    The server may hand this back as either a JSON object or a JSON *string* -- it tries
    JSON.parse and falls back to the raw string (messageController.ts:434-442). Socket.IO
    payloads also use camelCase inside `data`, unlike the snake_case REST responses, so
    both spellings are accepted.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None

    ciphertext = value.get("ciphertext")
    nonce = value.get("nonce")
    if not ciphertext or not nonce:
        return None
    return EncryptedData(ciphertext=ciphertext, nonce=nonce)


def is_encrypted(metadata: Any) -> bool:
    """True when encryption_metadata marks the message as space-key encrypted."""
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return False
    return isinstance(metadata, dict) and metadata.get("algorithm") == "symmetric"


def encryption_metadata(chat_space_id: str) -> dict[str, str]:
    """The `encryption_metadata` object accompanying an encrypted message.

    chat_space_id is uppercased to match what the clients send and what the server uses
    for Socket.IO room names.
    """
    return {
        "version": "3.0",
        "algorithm": "symmetric",
        "chat_space_id": chat_space_id.upper(),
    }
