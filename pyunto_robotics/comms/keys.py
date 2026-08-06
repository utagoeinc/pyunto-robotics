"""Chat-space key resolution.

Every message body is encrypted with a per-space AES-256 key. The iOS client resolves that key
in this order (ChatSpaceKeyManager.swift:66-99):

    local store -> wrapped key (unwrap with X25519 identity) -> legacy raw GET -> generate

The robot implements only the *raw* path. `GET /api/chat-spaces/:uuid/key` returns the space key
in the clear despite the `encrypted_key` field name -- chatSpaceKeyService.ts:20-28 does
`crypto.randomBytes(32).toString('base64')` and stores it as-is. Membership is the only gate.
That path is kept alive for old-client compatibility and is documented as being removed in
Phase 3 (E2EE_IMPLEMENTATION.md:20-21).

`SpaceKeyProvider` exists so that when Phase 3 lands, a wrapped-key implementation can be
dropped in without touching any calling code.
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol

import requests

from .auth import Session
from .crypto import string_to_key

log = logging.getLogger(__name__)


class KeyError_(Exception):
    """Could not obtain a space key."""


class SpaceKeyProvider(Protocol):
    """Supplies the symmetric key for a chat space."""

    def get_key(self, chat_space_id: str) -> bytes:
        """Return the 32-byte key for this space, or raise KeyError_."""
        ...


class RawSpaceKeyProvider:
    """Fetches space keys via the legacy raw endpoint, caching them in memory.

    Phase 3 will delete this endpoint; swap in a wrapped-key provider then.
    """

    def __init__(self, session: Session, timeout: float = 15.0):
        self._session = session
        self._timeout = timeout
        self._cache: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def get_key(self, chat_space_id: str) -> bytes:
        # Space UUIDs come back from the API in mixed case; cache on a stable form.
        cache_key = chat_space_id.lower()
        with self._lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        key = self._fetch(chat_space_id)
        with self._lock:
            self._cache[cache_key] = key
        return key

    def _fetch(self, chat_space_id: str) -> bytes:
        url = f"{self._session.base_url}/api/chat-spaces/{chat_space_id}/key"
        try:
            resp = requests.get(url, headers=self._session.auth_header(), timeout=self._timeout)
            if resp.status_code == 401:
                self._session.invalidate()
                resp = requests.get(
                    url, headers=self._session.auth_header(), timeout=self._timeout
                )
        except requests.RequestException as e:
            raise KeyError_(f"could not fetch space key: {e}") from e

        if resp.status_code == 403:
            raise KeyError_(
                f"not a member of chat space {chat_space_id} -- the robot must join it first"
            )
        if resp.status_code != 200:
            raise KeyError_(f"space key request failed: HTTP {resp.status_code} {resp.text[:200]}")

        data = resp.json().get("data") or {}
        raw = data.get("encrypted_key")  # misnomer: this is the plain key
        if not raw:
            raise KeyError_(f"no key in response for space {chat_space_id}")

        key = string_to_key(raw)
        log.info("resolved space key for %s", chat_space_id)
        return key
