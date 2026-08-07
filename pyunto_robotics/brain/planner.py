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
ACTIONS = ("goto", "face", "open", "point_at", "look_around", "describe", "where", "report")


@dataclass(frozen=True)
class Step:
    """One action in a plan."""

    action: str
    argument: str | None = None

    def __str__(self) -> str:
        return f"{self.action}({self.argument or ''})"


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

_GREETINGS = ("hello", "hi", "hey", "こんにちは", "はじめまして", "やあ", "おはよう", "こんばんは")

# Words to drop when salvaging an unrecognised target from an instruction.
_FILLER = frozenset(
    {"go", "to", "the", "a", "an", "walk", "move", "head", "come", "approach",
     "please", "now", "at", "toward", "towards", "face", "look", "point", "and", "then"}
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
    """Plans by matching verbs and objects. No model, no latency, no surprises."""

    def plan(self, message: str) -> Plan:
        text = message.lower().strip()
        if not text:
            return Plan([], reply="I did not catch that.")

        verb = self._verb(text)
        obj = self._object(text)

        if verb == "open":
            # "open" with no object named is unambiguous in this office: it means a door.
            return Plan([Step("open", obj or "door")])
        if verb in ("goto", "face", "point_at"):
            if obj is None:
                # Do not silently substitute a door. "go to the purple giraffe" has a clear
                # target that simply is not something the robot knows, and pretending it said
                # "door" would send it walking off on the wrong errand.
                target = _unknown_target(text) or "door"
                return Plan([Step(verb, target)])
            return Plan([Step(verb, obj)])
        if verb in ("look_around", "describe", "where"):
            return Plan([Step(verb)])

        # No verb, but a nameable object -- "the meeting room door" almost certainly means go.
        if obj:
            return Plan([Step("goto", obj)])

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

_PLANNER_PROMPT = """You control a humanoid robot in an office with a corridor, a workspace, a \
meeting room and a pantry, connected by doors.

Available actions:
  goto <object>      walk to something (door, whiteboard, desk, table, monitor, plant, fridge)
  face <object>      turn to look at something
  open <object>      walk to a door and push it open
  point_at <object>  point at something
  look_around        turn in place and report what is visible
  describe           say what is currently in view
  where              report which room the robot is in
  report <text>      say something to the user

The user said: "{message}"

Reply with ONLY a JSON array of steps, no other text. Example:
[{{"action": "open", "argument": "door"}}]

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
        steps.append(Step(action, str(argument) if argument else None))
    return steps
