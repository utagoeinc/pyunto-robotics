"""Turning a sentence into a plan.

Someone messages "オフィスのドアを開けて" and this decides that means
[open(door), report(...)]. Two implementations of the same interface:

  RulePlanner  Pattern matching over verbs and objects, in English and Japanese. Instant, no
               weights, no failure modes. It handles the phrasings a demo actually receives.

  LLMPlanner   Gemma 4 E2B running locally via MLX. Open-vocabulary: it copes with phrasings
               nobody enumerated, at the cost of a model load and ~1 s per plan.

LLMPlanner falls back to RulePlanner when the model is missing or returns something
unparseable, so a bad generation degrades to a worse plan rather than no plan at all.

On the model choice: Gemma 4 E2B is multimodal and small enough to be plausible on a real
robot, which is the point of picking it. Use the 8-bit build - Gemma 4's per-layer embeddings
quantise badly at 4-bit and the community weights are reported to produce garbage. There is
128 GB here, so 8-bit costs nothing worth saving.
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
    "where", "report",
)


@dataclass(frozen=True)
class Step:
    """One action in a plan.

    `where` carries a spatial qualifier -- "the door on the RIGHT" -- which matters whenever
    several instances of the same object are in view. The office has three identical doors, so
    dropping it would silently send the robot to whichever one happened to score best.
    """

    action: str
    argument: str | None = None
    where: str | None = None  # "left" | "right" | "middle" | "nearest" | "far"
    # How many of the object the user said were visible. 「三つ見えるドアのうち、右の」 tells the
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
    "door": ("door", "doorway", "entrance", "exit", "ドア", "扉", "入口", "出口"),
    "whiteboard": ("whiteboard", "board", "ホワイトボード"),
    "desk": ("desk", "机", "デスク"),
    "table": ("table", "テーブル"),
    "monitor": ("monitor", "screen", "display", "モニタ", "モニター", "画面"),
    "plant": ("plant", "植物", "観葉植物"),
    "fridge": ("fridge", "refrigerator", "冷蔵庫"),
}

# Verb patterns, matched in order, so more specific phrasings must come first.
# "look around" / 「周りを見て」 has to beat "look at" / 「見て」, and "open" has to beat "go"
# in "go open the door", or the robot does the wrong thing for a perfectly clear instruction.
_VERBS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("open", ("open", "push open", "開けて", "開けろ", "あけて", "開いて")),
    ("home", ("go back to where you started", "back to the start", "元の位置に戻って",
              "最初の位置に戻って", "スタート地点に戻って")),
    ("leave", ("leave the room", "come back out", "go back out", "exit the room",
               "廊下に出て", "部屋を出て", "出て来て", "戻って")),
    ("look_around", ("look around", "explore", "scan", "見回して", "周りを見て",
                     "まわりを見て", "あたりを見て", "探索")),
    ("describe", ("what do you see", "describe", "what can you see", "何が見える",
                  "何が見えますか", "見えるもの")),
    ("where", ("where are you", "your position", "どこにいる", "現在地", "どこですか")),
    ("point_at", ("point at", "point to", "指さして", "指して")),
    ("face", ("face", "look at", "turn to", "turn toward", "向いて", "見て")),
    ("goto", ("go to", "walk to", "goto", "move to", "head to", "approach", "come to",
              "行って", "移動して", "近づいて", "向かって")),
)

# Spatial qualifiers that pick between several instances of the same object.
_QUALIFIERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("left", ("leftmost", "left-hand", "on the left", "to the left", "left",
              "一番左", "左端", "左側", "左の", "左")),
    ("right", ("rightmost", "right-hand", "on the right", "to the right", "right",
               "一番右", "右端", "右側", "右の", "右")),
    ("middle", ("middle", "centre", "center", "in the middle", "central",
                "真ん中", "中央", "まんなか", "中程")),
    ("far", ("furthest", "farthest", "far", "at the end", "一番奥", "奥の", "奥")),
    ("nearest", ("nearest", "closest", "this one", "一番近い", "手前の", "手前")),
)

# Counts a user might state, in the forms they actually write them.
_COUNTS: dict[str, int] = {
    "1": 1, "one": 1, "一": 1, "一つ": 1, "1つ": 1, "ひとつ": 1,
    "2": 2, "two": 2, "二": 2, "二つ": 2, "2つ": 2, "ふたつ": 2,
    "3": 3, "three": 3, "三": 3, "三つ": 3, "3つ": 3, "みっつ": 3,
    "4": 4, "four": 4, "四": 4, "四つ": 4, "4つ": 4, "よっつ": 4,
    "5": 5, "five": 5, "五": 5, "五つ": 5, "5つ": 5, "いつつ": 5,
}


def stated_count(message: str) -> int | None:
    """How many of the thing the user said were there, if they said.

    「三つ見えるドアのうち」 is not decoration: it tells the robot how many doors it should be
    choosing between, which is what makes "the leftmost" mean a particular door rather than
    the leftmost of however many happen to be in frame.
    """
    text = message.lower()
    # Longest first so 「三つ」 beats 「三」 and "three" beats "3" inside "3つ".
    for word in sorted(_COUNTS, key=len, reverse=True):
        if word in text:
            return _COUNTS[word]
    return None


_GREETINGS = ("hello", "hi", "hey", "こんにちは", "はじめまして", "やあ", "おはよう", "こんばんは")

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


# Words that join one action to the next. Their presence means the instruction has more parts
# than a single verb-object match can represent.
_SEQUENCERS = (
    "その後", "そのあと", "それから", "次に", "つぎに", "今度は", "こんどは", "してから",
    "then", "after that", "afterwards", "next,", "and then",
)


def looks_multi_step(message: str) -> bool:
    """True when an instruction chains several actions together.

    RulePlanner can only ever produce one step, so on a chained instruction it silently
    returns the wrong one -- 「右のドアを開けて…今度は一番左の部屋に」 came out as
    open(left door), having dropped the first half. Callers use this to warn that --llm is
    needed rather than letting the robot confidently do the wrong thing.
    """
    text = message.lower()
    return any(word in text for word in _SEQUENCERS)


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
            # "open" with no object named is unambiguous in this office: it means a door.
            return Plan([Step("open", obj or "door", where, expect)])
        if verb in ("goto", "face", "point_at"):
            if obj is None:
                # Do not silently substitute a door. "go to the purple giraffe" has a clear
                # target that simply is not something the robot knows, and pretending it said
                # "door" would send it walking off on the wrong errand.
                target = _unknown_target(text) or "door"
                return Plan([Step(verb, target, where, expect)])
            return Plan([Step(verb, obj, where, expect)])
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
                "I can walk to things, look around, describe what I see, and open doors. "
                "Try: \"open the door\" / 「オフィスのドアを開けて」"
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

        Ordering matters: "on the left" has to beat the bare "left" inside it, and 「一番左」
        has to beat 「左」, or the match is right but for the wrong reason.
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

_PLANNER_PROMPT = """You control a humanoid robot in an office. A corridor runs along the \
south side. Off it, behind three doors, are three rooms: the workspace (left), the meeting \
room (middle), and the pantry (right).

Available actions:
  goto <object>      walk to something (door, whiteboard, desk, table, monitor, plant, fridge)
  face <object>      turn to look at something
  open <object>      walk to a door, push it open, and go through
  leave              come back out of the room the robot is in, into the corridor
  close              pull the nearest door shut
  home               walk back to where the robot was standing when it was given the task
  point_at <object>  point at something
  look_around        turn in place and report what is visible
  describe           say what is currently in view
  where              report which room the robot is in
  report <text>      say something to the user

Each step may also carry "where" to pick between identical objects. Use it whenever the user
says which one they mean:
  "left" | "right" | "middle" | "nearest" | "far"

If the user says how many there are -- 「三つ見えるドアのうち」, "of the three doors" -- put that
number in "expect". It tells the robot how many it should be choosing between.

Rules:
- Only ever use the object names listed above. There is no "corridor" or "room" object; a room
  is entered by opening its door, so "go into the left room" is {{"action": "open",
  "argument": "door", "where": "left"}}.
- "open" already walks there and goes through, so never follow it with a goto for the same door.
- From inside a room no other door is visible, so ALWAYS use "leave" before opening a
  different door.
- "left" and "right" are judged from where the user is describing, which is where the robot
  started. After leaving a room it is beside one doorway and can only see that one, so ALWAYS
  use "home" before a second qualified door. "go into the right room, then the left room" is:
  open right -> leave -> home -> open left.
- Break a multi-part instruction into one step per action, in the order the user said them.

The user said: "{message}"

Reply with ONLY a JSON array of steps, no other text. Examples:

"open the right door"
[{{"action": "open", "argument": "door", "where": "right"}}]

"of the three doors you can see, open the right one"  (note the count)
[{{"action": "open", "argument": "door", "where": "right", "expect": 3}}]

"open the right door, then go into the leftmost room"
[{{"action": "open", "argument": "door", "where": "right"}}, {{"action": "leave"}}, \
{{"action": "home"}}, {{"action": "open", "argument": "door", "where": "left"}}]

If the request is just conversation, reply with:
[{{"action": "report", "argument": "<your reply>"}}]"""


class LLMPlanner:
    """Plans with a local Gemma 4 model, falling back to rules when it cannot."""

    def __init__(
        self,
        model_id: str = "lmstudio-community/gemma-4-E2B-it-MLX-8bit",
        max_tokens: int = 200,
        fallback: RulePlanner | None = None,
    ):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.fallback = fallback or RulePlanner()
        self._model = None
        self._tokenizer = None
        self._config = None

    def _load(self) -> None:
        """Load the model through mlx_vlm, not mlx_lm.

        Gemma 4 E2B is multimodal, so its weights live under `language_model.*` and mlx_lm's
        text-only loader rejects every tensor. mlx_vlm understands the layout, and using it
        also leaves the door open to handing the planner a camera frame later.
        """
        if self._model is not None:
            return
        try:
            from mlx_vlm import load  # noqa: PLC0415 - optional heavy dependency
            from mlx_vlm.utils import load_config  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "mlx-vlm is not installed. Install the extra: uv pip install -e '.[llm]'"
            ) from e
        log.info("loading planner model %s (first run downloads weights)", self.model_id)
        self._model, self._tokenizer = load(self.model_id)
        self._config = load_config(self.model_id)

    def plan(self, message: str) -> Plan:
        try:
            self._load()
            from mlx_vlm import generate  # noqa: PLC0415
            from mlx_vlm.prompt_utils import apply_chat_template  # noqa: PLC0415

            prompt = apply_chat_template(
                self._tokenizer, self._config, _PLANNER_PROMPT.format(message=message),
                num_images=0,
            )
            reply = generate(
                self._model, self._tokenizer, prompt, [],
                max_tokens=self.max_tokens, verbose=False,
            )
            text = reply if isinstance(reply, str) else getattr(reply, "text", str(reply))
            steps = parse_plan(text)
            if steps:
                return Plan(steps)
            log.warning("planner model produced no usable steps; falling back to rules")
        except Exception as e:  # noqa: BLE001 - a planner failure must not stop the robot
            log.warning("planner model unavailable (%s); falling back to rules", e)
        return self.fallback.plan(message)


def parse_plan(text: str) -> list[Step]:
    """Extract steps from a model reply.

    Locates the JSON array by bracket matching rather than parsing the whole reply, because
    models routinely wrap it in prose or a code fence.
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
        if action not in ACTIONS:
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
