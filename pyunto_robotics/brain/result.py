"""What one skill did, in a form that can be sent to a person.

Lives here rather than beside any one robot's skills because every robot returns these and
none of them should have to import another robot's module to do it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillResult:
    """Outcome of one skill, in a form that can be messaged to a human."""

    ok: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    # Whether a failure should stop the rest of the plan.
    #
    # Most failures should: later steps assume the earlier ones worked, and a robot that could
    # not reach the sunlight has no business reporting how much it charged. But some are
    # simply "I cannot do that particular thing", and abandoning a whole errand over one
    # impossible aside is the wrong response. Skills set this False to say "note it and
    # carry on".
    fatal: bool = True
