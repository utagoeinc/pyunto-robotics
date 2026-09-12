"""Connecting the robot to Pyunto.

One place that builds the account, the E2EE identity and the client, so every entry point
(runner, demo, CLI, a customer's own script) reaches the network the same way.

The transport itself comes from `pyunto-agent` -- the robot SDK deliberately does not ship a
copy of it. An earlier copy in `pyunto_robotics/comms/` went stale and could not fetch keys for
spaces created after August 2026; depending on the package instead makes that impossible.

Two ways to authenticate, mirroring pyunto-agent:

* PYUNTO_EMAIL + PYUNTO_PASSWORD -- a registered Pyunto account.
* nothing -- an anonymous robot account is created and remembered in the data directory
  (PYUNTO_ROBOT_DIR, default ~/.pyunto-robot). This is what `pyunto-robotics demo` uses: the
  buyer types one command and a robot account appears.

The display name is prefixed with the robot marker so the app can label it in the participant
list. It is a display convention, never an authorization signal.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from pyunto_agent.auth import AuthError, Identity, Session
from pyunto_agent.client import PyuntoClient
from pyunto_agent.identity import IdentityStore
from pyunto_agent.keys import WrappedSpaceKeyProvider

log = logging.getLogger(__name__)

__all__ = ["AuthError", "Connection", "ROBOT_NAME_PREFIX", "connect", "data_dir"]

#: Display names of robot accounts start with this, so the app can show "Robot (self-hosted)"
#: next to them. Kept in sync with AgentMemberInfo on iOS.
ROBOT_NAME_PREFIX = "🤖 "


@dataclass
class Connection:
    """Everything a robot needs to talk to Pyunto."""

    client: PyuntoClient
    keys: WrappedSpaceKeyProvider
    identity_store: IdentityStore
    identity: Identity

    @property
    def user_id(self) -> str:
        return self.client.uuid


def data_dir() -> Path:
    """Where the robot's account and keys live. Deleting this makes a *different* robot."""
    d = Path(os.environ.get("PYUNTO_ROBOT_DIR") or Path.home() / ".pyunto-robot")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _device_id(directory: Path) -> str:
    """Stable per-installation id. register-anonymous is idempotent on it, so the robot keeps
    the same account (and therefore the same space memberships) across restarts."""
    p = directory / "device_id"
    if p.exists():
        return p.read_text().strip()
    did = str(uuid.uuid4())
    p.write_text(did)
    return did


def connect(display_name: str = "Robot", base_url: str | None = None) -> Connection:
    """Log in (or create an anonymous robot account) and register the E2EE identity key.

    Registering the identity is what lets the members' apps seal the space key for this robot.
    Until one of them opens the app afterwards, the robot can join a space but not read it.
    """
    base = base_url or os.environ.get("PYUNTO_BASE_URL", "https://api.pyunto.com")
    directory = data_dir()

    # `.strip() or None` so an empty PYUNTO_EMAIL= in a .env means "no account", not "".
    email = (os.environ.get("PYUNTO_EMAIL") or "").strip() or None
    password = (os.environ.get("PYUNTO_PASSWORD") or "").strip() or None
    if email and password:
        # A registered account. Its display name is whatever the person set, so the app cannot
        # tell it is a robot -- warn once rather than silently losing the label in the
        # participant list. PYUNTO_ROBOT_ACCOUNT=1 suppresses this for deliberate setups.
        if not os.environ.get("PYUNTO_ROBOT_ACCOUNT"):
            log.warning(
                "using the registered account in PYUNTO_EMAIL. The Pyunto app labels robots by "
                "a display name starting with %r; rename the account, or unset PYUNTO_EMAIL to "
                "let an anonymous robot account be created.",
                ROBOT_NAME_PREFIX.strip(),
            )
        session = Session(base, email, password)
    else:
        name = os.environ.get("PYUNTO_ROBOT_NAME") or display_name
        if not name.startswith(ROBOT_NAME_PREFIX):
            name = ROBOT_NAME_PREFIX + name
        session = Session(base, device_id=_device_id(directory), display_name=name)

    identity_store = IdentityStore(directory)
    keys = WrappedSpaceKeyProvider(session, identity_store, directory)
    client = PyuntoClient(session, keys)
    identity = session.login()
    identity_store.upload(client)
    return Connection(client=client, keys=keys, identity_store=identity_store, identity=identity)
