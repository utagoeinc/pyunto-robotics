"""Turning a sentence into a plan — the shared machinery.

`Step`, `Plan` and `parse_plan` are used by every robot, through the per-robot planners in
brain/domains.py. What differs between machines is vocabulary, and vocabulary is data; what
does not differ lives here.

`RulePlanner` is the general fallback: pattern matching over verbs and objects, instant, no
weights, no failure modes. Each robot's `Domain` overrides its vocabulary, and `DomainRulePlanner`
borrows its qualifier parsing ("the leftmost door", "the far crate"), which is the same problem
whatever the robot is.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Verbs the skill layer understands. The planner may only emit these.
ACTIONS = (
    "goto", "face", "open", "leave", "close", "home", "point_at", "look_around", "describe",
    "where", "report", "raise_arm", "wave", "lower_arm",
)


def _which_arm(text: str) -> str:
    """Left or right, from the words. Right when unsaid, which is what a person would do."""
    lowered = text.lower()
    if "left" in lowered:
        return "l"
    return "r"


@dataclass(frozen=True)
class Step:
    """One action in a plan.

    `where` carries a spatial qualifier -- "the door on the RIGHT" -- which matters whenever
    several instances of the same object are in view. A corridor may have three identical doors, so
    dropping it would silently send the robot to whichever one happened to score best.
    """

    action: str
    argument: str | None = None
    where: str | None = None  # "left" | "right" | "middle" | "nearest" | "far"
    # How many of the object the user said were visible. "the right one of the three doors" tells the
    # robot both which door to pick and how many it should be picking from -- and the second
    # half is worth acting on, because "leftmost of three" and "leftmost of one" name different
    # doors. Without it the robot happily resolves a qualifier against whatever it can see.
    expect: int | None = None

    def __str__(self) -> str:
        target = self.argument or ""
        if self.where:
            target = f"{self.where} {target}".strip()
        if self.expect:
            target = f"{target} of {self.expect}"
        return f"{self.action}({target})"


@dataclass
class Plan:
    """What the robot intends to do about a message."""

    steps: list[Step]
    reply: str | None = None  # what to say if there is nothing to do

    def __bool__(self) -> bool:
        return bool(self.steps)

    def __str__(self) -> str:
        return " -> ".join(str(s) for s in self.steps) if self.steps else "(no action)"


# --------------------------------------------------------------------------------------
# Rule-based planning
# --------------------------------------------------------------------------------------

# Objects the robot can be sent to, with the words people use for them.
_OBJECTS: dict[str, tuple[str, ...]] = {
    'door': ('door', 'doorway', 'entrance', 'exit'),
    'whiteboard': ('whiteboard', 'board'),
    'desk': ('desk',),
    'table': ('table',),
    'monitor': ('monitor', 'screen', 'display'),
    'plant': ('plant',),
    'fridge': ('fridge', 'refrigerator'),
}

# Verb patterns, matched in order, so more specific phrasings must come first.
# "look around" has to beat "look at", and "open" has to beat "go"
# in "go open the door", or the robot does the wrong thing for a perfectly clear instruction.
_VERBS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ('wave', ('wave',)),
    ('raise_arm', ('raise your', 'put your hand up', 'lift your', 'hold up your')),
    ('lower_arm', ('lower your', 'put your hand down', 'put your arm down')),
    ('open', ('open', 'push open')),
    ('home', ('go back to where you started', 'back to the start')),
    ('leave', ('leave the room', 'come back out', 'go back out', 'exit the room')),
    ('look_around', ('look around', 'explore', 'scan')),
    ('describe', ('what do you see', 'describe', 'what can you see')),
    ('where', ('where are you', 'your position')),
    ('point_at', ('point at', 'point to')),
    ('face', ('face', 'look at', 'turn to', 'turn toward')),
    ('goto', ('go to', 'walk to', 'goto', 'move to', 'head to', 'approach', 'come to')),
)

# Spatial qualifiers that pick between several instances of the same object.
_QUALIFIERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ('left', ('leftmost', 'left-hand', 'on the left', 'to the left', 'left')),
    ('right', ('rightmost', 'right-hand', 'on the right', 'to the right', 'right')),
    ('middle', ('middle', 'centre', 'center', 'in the middle', 'central')),
    ('far', ('furthest', 'farthest', 'far', 'at the end')),
    ('nearest', ('nearest', 'closest', 'this one')),
)

# Counts a user might state, in the forms they actually write them.
_COUNTS: dict[str, int] = {
    '1': 1,
    'one': 1,
    '2': 2,
    'two': 2,
    '3': 3,
    'three': 3,
    '4': 4,
    'four': 4,
    '5': 5,
    'five': 5,
}


def stated_count(message: str) -> int | None:
    """How many of the thing the user said were there, if they said.

    "of the three doors" is not decoration: it tells the robot how many doors it should be
    choosing between, which is what makes "the leftmost" mean a particular door rather than
    the leftmost of however many happen to be in frame.
    """
    text = message.lower()
    # Longest first, so "three" beats "3" where both appear.
    for word in sorted(_COUNTS, key=len, reverse=True):
        if word in text:
            return _COUNTS[word]
    return None


_GREETINGS = ("hello", "hi", "hey", "good morning", "good evening", "hiya")

# Words to drop when salvaging an unrecognised target from an instruction.
_FILLER = frozenset(
    {"go", "to", "the", "a", "an", "walk", "move", "head", "come", "approach",
     "please", "now", "at", "toward", "towards", "face", "look", "point", "and", "then",
     # Qualifiers travel on Step.where, so leaving them in the target text would produce
     # nonsense like goto("right purple giraffe").
     "left", "right", "middle", "centre", "center", "nearest", "closest", "furthest", "far",
     "leftmost", "rightmost", "on", "in", "of"}
)


def _unknown_target(text: str) -> str | None:
    """Recover the noun from an instruction naming something the robot does not know.

    Keeps "purple giraffe" out of the door-shaped hole, so the failure the user gets back
    names what they actually asked for.
    """
    words = [w.strip(".,!?\"'") for w in text.split()]
    remainder = [w for w in words if w and w.lower() not in _FILLER]
    return " ".join(remainder) if remainder else None


class RulePlanner:
    """Plans by matching verbs and objects. No model, no latency, no surprises.

    One step only: it scans the whole message once for a verb and an object. A chained
    instruction is beyond it by construction -- see `looks_multi_step`.
    """

    def plan(self, message: str) -> Plan:
        text = message.lower().strip()
        if not text:
            return Plan([], reply="I did not catch that.")

        if looks_multi_step(message):
            log.warning(
                "this instruction chains several actions; RulePlanner can only do the first. "
                "Use LLMPlanner (--llm) for multi-step instructions."
            )

        verb = self._verb(text)
        obj = self._object(text)
        where = self._qualifier(text)
        # Only meaningful alongside a qualifier: "three doors" on its own says nothing about
        # which one is wanted.
        expect = stated_count(message) if where else None

        if verb == "open":
            # "open" with no object named is unambiguous in a corridor: it means a door.
            return Plan([Step("open", obj or "door", where, expect)])
        if verb in ("goto", "face", "point_at"):
            if obj is None:
                # Do not silently substitute a door. "go to the purple giraffe" has a clear
                # target that simply is not something the robot knows, and pretending it said
                # "door" would send it walking off on the wrong errand.
                target = _unknown_target(text) or "door"
                return Plan([Step(verb, target, where, expect)])
            return Plan([Step(verb, obj, where, expect)])
        if verb in ("wave", "raise_arm", "lower_arm"):
            return Plan([Step(verb, _which_arm(text))])
        if verb in ("look_around", "describe", "where", "leave", "close", "home"):
            return Plan([Step(verb)])

        # No verb, but a nameable object -- "the meeting room door" almost certainly means go.
        if obj:
            return Plan([Step("goto", obj, where, expect)])

        if any(g in text for g in _GREETINGS):
            return Plan([], reply="Hello. Tell me where to go or what to open.")

        return Plan(
            [],
            reply=(
                "I can walk to things, look around, describe what I see, open doors, "
                "and raise or wave a hand. "
                'Try: "open the door" or "walk to the whiteboard"'
            ),
        )

    @staticmethod
    def _verb(text: str) -> str | None:
        for action, patterns in _VERBS:
            if any(p in text for p in patterns):
                return action
        return None

    @staticmethod
    def _qualifier(text: str) -> str | None:
        """Pull out a spatial qualifier, longest phrase first.

        Ordering matters: "on the left" has to beat the bare "left" inside it, or the match
        is right but for the wrong reason.
        """
        best: tuple[int, str] | None = None
        for name, phrases in _QUALIFIERS:
            for phrase in phrases:
                if phrase in text and (best is None or len(phrase) > best[0]):
                    best = (len(phrase), name)
        return best[1] if best else None

    @staticmethod
    def _object(text: str) -> str | None:
        best: tuple[int, str] | None = None
        for name, words in _OBJECTS.items():
            for word in words:
                if word in text and (best is None or len(word) > best[0]):
                    best = (len(word), name)
        return best[1] if best else None


# --------------------------------------------------------------------------------------
# LLM planning
# --------------------------------------------------------------------------------------


_CHECK_PROMPT = """You are a robot standing in front of one of several doors.

Bearings to the doors you can see, from your own point of view. Positive is to your LEFT,
negative is to your RIGHT, and 0 is straight ahead: {bearings}

You were told to go to the {where} door. The door you are facing is the one nearest 0 degrees.

Is the door you are facing the {where} one? Answer with a single word, yes or no."""


def parse_plan(text: str, allowed: tuple[str, ...] = ACTIONS) -> list[Step]:
    """Extract steps from a model reply.

    Locates the JSON array by bracket matching rather than parsing the whole reply, because
    models routinely wrap it in prose or a code fence.

    `allowed` is the vocabulary the plan is checked against. It defaults to the general
    verbs so existing callers are unaffected; the other robots pass their own, because a plan
    naming an action their skills do not have is a misunderstanding worth dropping rather than
    a step worth attempting. See brain/domains.py.
    """
    match = re.search(r"\[.*]", text, re.DOTALL)
    if not match:
        return []
    try:
        entries = json.loads(match.group(0))
    except json.JSONDecodeError:
        log.warning("planner returned unparseable JSON: %s", text[:200])
        return []

    steps: list[Step] = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        action = str(entry.get("action", "")).strip().lower()
        if action == "open_door":
            action = "open"
        # `report` is always available: every domain uses it to say something back, and a
        # model that answers a greeting with one should not have the step thrown away.
        if action not in allowed and action != "report":
            log.warning("planner emitted unknown action %r, skipping", action)
            continue
        argument = entry.get("argument")
        expect = entry.get("expect") or entry.get("count") or entry.get("of")
        try:
            expect = int(expect) if expect else None
        except (TypeError, ValueError):
            expect = None
        # Accept a spatial qualifier either as its own field or folded into the argument, since
        # models do both no matter how the prompt asks.
        where = entry.get("where") or entry.get("which") or entry.get("position")
        where = str(where).strip().lower() if where else None
        if where not in (None, "left", "right", "middle", "nearest", "far"):
            where = None
        if where is None and argument:
            where = RulePlanner._qualifier(str(argument).lower())
        steps.append(Step(action, str(argument) if argument else None, where, expect))
    return steps
