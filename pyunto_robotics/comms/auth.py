"""Pyunto authentication.

POST /api/auth/login takes {email, password} and returns {"token": "<JWT>"} -- the field is
`token`, NOT `access_token` (authController.ts:441-446). Tokens last 30 days and there is no
refresh endpoint, so the only recovery is to log in again. We decode `exp` locally to renew
early, and callers re-login on any 401.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)

# Renew this long before `exp` rather than waiting for a 401 mid-task.
RENEW_MARGIN_S = 24 * 60 * 60  # 1 day


class AuthError(Exception):
    """Login failed or the session could not be established."""


@dataclass(frozen=True)
class Identity:
    """Who the robot is, as the server sees it."""

    uuid: str
    email: str
    display_name: str

    def __str__(self) -> str:
        return f"{self.display_name} <{self.email}> ({self.uuid})"


def _decode_jwt_exp(token: str) -> float | None:
    """Read `exp` out of a JWT without verifying it (we only need the expiry hint)."""
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)  # restore stripped padding
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        exp = payload.get("exp")
        return float(exp) if exp is not None else None
    except Exception:  # noqa: BLE001 - a malformed token just means "no hint"
        return None


class Session:
    """Holds the JWT and re-logs-in when it expires.

    Thread-safe: the Socket.IO listener and the REST caller live on different threads.
    """

    def __init__(self, base_url: str, email: str, password: str, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self._email = email
        self._password = password
        self.timeout = timeout

        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at: float | None = None
        self.identity: Identity | None = None

    # -- public API ---------------------------------------------------------------

    @property
    def token(self) -> str:
        """A valid JWT, logging in or renewing as needed."""
        with self._lock:
            if self._token is None or self._is_expiring():
                self._login_locked()
            assert self._token is not None
            return self._token

    def auth_header(self) -> dict[str, str]:
        """Authorization header. The server requires exactly 'Bearer <3-segment JWT>'."""
        return {"Authorization": f"Bearer {self.token}"}

    def invalidate(self) -> None:
        """Drop the cached token so the next use forces a fresh login (call this on a 401)."""
        with self._lock:
            log.info("token invalidated; will re-login on next use")
            self._token = None
            self._expires_at = None

    def login(self) -> Identity:
        """Log in now and return the robot's identity."""
        with self._lock:
            return self._login_locked()

    # -- internals ----------------------------------------------------------------

    def _is_expiring(self) -> bool:
        if self._expires_at is None:
            return False  # no expiry hint -> trust it until a 401 says otherwise
        return time.time() >= (self._expires_at - RENEW_MARGIN_S)

    def _login_locked(self) -> Identity:
        """Caller must hold self._lock."""
        url = f"{self.base_url}/api/auth/login"
        try:
            resp = requests.post(
                url,
                json={"email": self._email, "password": self._password},
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise AuthError(f"could not reach {url}: {e}") from e

        if resp.status_code == 401:
            # The server returns 401 for every failure mode (wrong password, unknown user,
            # inactive or deleted account) -- authService.ts:351-357.
            raise AuthError(
                "login rejected (401). Check PYUNTO_EMAIL / PYUNTO_PASSWORD, and that the "
                "account is verified and active."
            )
        if resp.status_code != 200:
            raise AuthError(f"login failed: HTTP {resp.status_code} {resp.text[:200]}")

        body = resp.json()
        token = body.get("token")  # NOT access_token
        if not token:
            raise AuthError(f"login response had no 'token' field: {body}")

        user = body.get("user") or {}
        identity = Identity(
            # UUIDs are lowercased server-side for user ids; normalize so self-comparison works.
            uuid=str(user.get("uuid", "")).lower(),
            email=user.get("email", self._email),
            display_name=user.get("display_name") or "Robot",
        )

        self._token = token
        self._expires_at = _decode_jwt_exp(token)
        self.identity = identity

        if self._expires_at:
            days = (self._expires_at - time.time()) / 86400
            log.info("logged in as %s (token valid %.1f days)", identity, days)
        else:
            log.info("logged in as %s", identity)
        return identity
