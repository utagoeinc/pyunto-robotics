"""Pyunto client: REST calls plus a Socket.IO listener for incoming messages.

Server-behaviour notes that shape this code (all verified against pyunto-server/server/src):

* Login returns `token`, not `access_token`; there is no refresh endpoint (authController.ts:441).
* REST responses are snake_case, but Socket.IO payloads are camelCase *inside* `data`
  (messageController.ts:459-474). Two different shapes for the same message.
* The server broadcasts `new_message` to every thread member's room -- including the sender.
  Without a self-filter the robot answers itself forever (messageController.ts:498-508).
* Thread membership, not space membership, controls visibility (messageController.ts:295-302).
* `chat_space_id` is uppercased server-side; user UUIDs are lowercased. Compare case-insensitively.
* GET threads without `limit` returns everything and silently changes the sort order
  (chatSpaceController.ts:1110-1177), so `limit` is always passed.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import requests
import socketio

from .auth import Session
from .crypto import (
    ENCRYPTED_PLACEHOLDER,
    encrypt_message,
    encryption_metadata,
    is_encrypted,
    parse_encrypted_content,
)
from .crypto import CryptoError, decrypt_message
from .keys import SpaceKeyProvider

log = logging.getLogger(__name__)


@dataclass
class IncomingMessage:
    """A message addressed to the robot, already decrypted."""

    uuid: str
    text: str
    thread_id: str
    chat_space_id: str
    sender_uuid: str
    sender_name: str
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.sender_name}] {self.text}"


class PyuntoClient:
    """Talks to Pyunto as the robot's account."""

    def __init__(
        self,
        session: Session,
        key_provider: SpaceKeyProvider,
        timeout: float = 20.0,
    ):
        self.session = session
        self.keys = key_provider
        self.timeout = timeout
        self._sio: socketio.Client | None = None
        self._on_message: Callable[[IncomingMessage], None] | None = None
        self._stop = threading.Event()

    # -- identity -----------------------------------------------------------------

    @property
    def uuid(self) -> str:
        """The robot's own user UUID (lowercased)."""
        if self.session.identity is None:
            self.session.login()
        assert self.session.identity is not None
        return self.session.identity.uuid

    # -- REST ---------------------------------------------------------------------

    def _request(self, method: str, path: str, **kw) -> requests.Response:
        """Authenticated request that retries once after re-login on a 401."""
        url = f"{self.session.base_url}{path}"
        kw.setdefault("timeout", self.timeout)
        headers = {**kw.pop("headers", {}), **self.session.auth_header()}
        resp = requests.request(method, url, headers=headers, **kw)
        if resp.status_code == 401:
            self.session.invalidate()
            headers = {**headers, **self.session.auth_header()}
            resp = requests.request(method, url, headers=headers, **kw)
        return resp

    def list_spaces(self) -> list[dict[str, Any]]:
        """All chat spaces the robot belongs to."""
        r = self._request("GET", "/api/chat-spaces")
        r.raise_for_status()
        return r.json().get("data", [])

    def list_threads(self, chat_space_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
        """Threads in a space. `limit` is always sent -- omitting it changes the sort order."""
        r = self._request(
            "GET",
            f"/api/chat-spaces/{chat_space_id}/threads",
            params={"limit": limit, "offset": offset},
        )
        r.raise_for_status()
        return r.json().get("data", [])

    def get_messages(
        self, thread_id: str, chat_space_id: str | None = None
    ) -> list[IncomingMessage]:
        """Full message history of a thread, decrypted.

        chat_space_id is a fallback for messages that carry no encryption metadata to derive it
        from (e.g. plaintext ones).
        """
        r = self._request("GET", f"/api/messages/{thread_id}")
        r.raise_for_status()
        out = []
        for m in r.json().get("data", []):
            msg = self._decode_rest_message(m, chat_space_id)
            if msg is not None:
                out.append(msg)
        return out

    def join_with_code(self, invite_code: str) -> str | None:
        """Join a human's chat space using an invite code from the app."""
        r = self._request("POST", "/api/chat-spaces/join-with-code", json={"invite_code": invite_code})
        if r.status_code >= 400:
            log.error("join failed: HTTP %s %s", r.status_code, r.text[:300])
            r.raise_for_status()
        body = r.json()
        space_id = (body.get("data") or {}).get("uuid") or body.get("chat_space_id")
        log.info("joined chat space %s", space_id)
        return space_id

    def send(
        self,
        chat_space_id: str,
        text: str,
        thread_id: str | None = None,
        mentioned_users: list[str] | None = None,
    ) -> dict[str, Any]:
        """Post an encrypted message. thread_id=None creates a new thread.

        Replies into an existing thread notify every thread member, so the robot does not
        need to mention anyone to be seen (messageController.ts:607).
        """
        key = self.keys.get_key(chat_space_id)
        enc = encrypt_message(text, key)
        payload: dict[str, Any] = {
            "content": ENCRYPTED_PLACEHOLDER,
            "encrypted_content": enc.to_wire(),
            "encryption_metadata": encryption_metadata(chat_space_id),
            "chat_space_id": chat_space_id,
        }
        if thread_id:
            payload["thread_id"] = thread_id
        if mentioned_users:
            payload["mentioned_users"] = [u.lower() for u in mentioned_users]

        r = self._request("POST", "/api/messages", json=payload)
        if r.status_code >= 400:
            log.error("send failed: HTTP %s %s", r.status_code, r.text[:300])
            r.raise_for_status()
        return r.json()

    # -- decoding -----------------------------------------------------------------

    def _decrypt(self, chat_space_id: str, encrypted: Any, metadata: Any, fallback: str) -> str:
        """Decrypt a body, falling back to the plaintext field when not encrypted."""
        if not is_encrypted(metadata):
            return fallback or ""
        enc = parse_encrypted_content(encrypted)
        if enc is None:
            return fallback or ""
        try:
            return decrypt_message(enc, self.keys.get_key(chat_space_id))
        except (CryptoError, Exception) as e:  # noqa: BLE001
            log.warning("could not decrypt message in space %s: %s", chat_space_id, e)
            return fallback or ""

    def _decode_rest_message(
        self, m: dict[str, Any], chat_space_id: str | None = None
    ) -> IncomingMessage | None:
        """REST shape: snake_case.

        GET /api/messages/:threadId does not include `chat_space_id` on each message, so it is
        recovered from encryption_metadata (which carries it uppercased) or supplied by the caller.
        """
        sender = m.get("sender") or m.get("User") or {}
        meta = m.get("encryption_metadata")
        if isinstance(meta, str):
            try:
                import json as _json

                meta = _json.loads(meta)
            except ValueError:
                meta = None
        space_id = (
            m.get("chat_space_id")
            or (meta or {}).get("chat_space_id")
            or chat_space_id
            or ""
        )
        if not space_id:
            return None
        text = self._decrypt(
            space_id, m.get("encrypted_content"), m.get("encryption_metadata"), m.get("content", "")
        )
        return IncomingMessage(
            uuid=m.get("uuid", ""),
            text=text,
            thread_id=m.get("chat_thread_id", ""),
            chat_space_id=space_id,
            sender_uuid=str(sender.get("uuid", "")).lower(),
            sender_name=sender.get("display_name", "?"),
            raw=m,
        )

    def _decode_ws_message(self, data: dict[str, Any]) -> IncomingMessage | None:
        """Socket.IO shape: camelCase inside `data`."""
        sender = data.get("sender") or {}
        space_id = data.get("chatSpaceId") or ""
        if not space_id:
            return None
        text = self._decrypt(
            space_id,
            data.get("encryptedContent"),
            data.get("encryptionMetadata"),
            data.get("content", ""),
        )
        return IncomingMessage(
            uuid=data.get("uuid", ""),
            text=text,
            thread_id=data.get("threadId") or data.get("chatThreadId") or "",
            chat_space_id=space_id,
            sender_uuid=str(sender.get("uuid", "")).lower(),
            sender_name=sender.get("display_name", "?"),
            raw=data,
        )

    # -- realtime -----------------------------------------------------------------

    def listen(self, on_message: Callable[[IncomingMessage], None]) -> None:
        """Connect to Socket.IO and dispatch incoming messages until stop() is called."""
        self._on_message = on_message
        self._stop.clear()
        me = self.uuid

        sio = socketio.Client(logger=False, engineio_logger=False, reconnection=True)
        self._sio = sio

        @sio.event
        def connect() -> None:  # noqa: ANN202
            log.info("socket.io connected")

        @sio.event
        def connect_error(err: Any) -> None:  # noqa: ANN202
            log.error("socket.io connect error: %s", err)

        @sio.event
        def disconnect() -> None:  # noqa: ANN202
            log.info("socket.io disconnected")

        @sio.on("connection_established")
        def _established(payload: Any) -> None:  # noqa: ANN202
            spaces = (payload or {}).get("connectedChatSpaces", [])
            log.info("listening on %d chat space(s)", len(spaces))

        def _handle(event: str, payload: Any) -> None:
            data = (payload or {}).get("data") or {}
            msg = self._decode_ws_message(data)
            if msg is None:
                return
            # The server echoes our own posts back to us. Without this the robot loops forever.
            if msg.sender_uuid == me:
                return
            if not msg.text.strip():
                return
            log.info("<- %s (%s)", msg, event)
            if self._on_message:
                try:
                    self._on_message(msg)
                except Exception:  # noqa: BLE001 - a handler crash must not kill the listener
                    log.exception("message handler raised")

        @sio.on("new_message")
        def _new_message(payload: Any) -> None:  # noqa: ANN202
            _handle("new_message", payload)

        @sio.on("thread_created")
        def _thread_created(payload: Any) -> None:  # noqa: ANN202
            # A new thread carries its first message inline (null for image-only threads).
            data = (payload or {}).get("data") or {}
            first = data.get("firstMessage")
            if not first:
                return
            merged = {**first, "chatSpaceId": data.get("chatSpaceId"), "threadId": data.get("threadId")}
            _handle("thread_created", {"data": merged})

        sio.connect(
            self.session.base_url,
            auth={"token": self.session.token},
            socketio_path="/socket.io",
            transports=["websocket"],
        )
        try:
            while not self._stop.is_set():
                self._stop.wait(1.0)
        finally:
            try:
                sio.disconnect()
            except Exception:  # noqa: BLE001
                pass

    def stop(self) -> None:
        """Break out of listen()."""
        self._stop.set()
