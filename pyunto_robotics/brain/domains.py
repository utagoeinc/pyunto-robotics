"""Turning a sentence into a plan, for each robot.

The office planner in brain/planner.py is written around one robot in one building: its verbs
are `open`/`leave`/`home`, its nouns are doors and whiteboards, and its LLM prompt describes a
corridor with three rooms. None of that transfers to a robot folding laundry or driving on the
Moon.

So each domain gets a Domain: its own verbs, its own words for things, and its own prompt. The
machinery underneath is shared -- the same rule matcher, the same JSON extraction, the same
fallback from model to rules -- because none of that is domain-specific. What changes is the
vocabulary, and vocabulary is data.

A Domain is deliberately small: a list of verbs with the phrasings people use, a list of
objects, and a prompt. Adding a fourth robot means adding one of these, not another planner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .planner import (
    Plan,
    RulePlanner,
    Step,
    parse_plan,
    stated_count,
)

log = logging.getLogger(__name__)

# Greetings are the same in any domain, so they live here rather than in each one.
_GREETINGS = (
    "hello", "hi", "hey", "こんにちは", "はじめまして", "やあ", "おはよう", "こんばんは",
)


@dataclass(frozen=True)
class Domain:
    """Everything that makes one robot's language different from another's."""

    name: str
    # Verb patterns, matched in order, so the more specific phrasing must come first --
    # "put it in the basket" has to beat "put it down", and 「洗濯機を開けて」 has to beat 「開けて」.
    verbs: tuple[tuple[str, tuple[str, ...]], ...]
    # Object words, mapped onto the name a skill expects.
    objects: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Verbs that take no argument, so a bare match is a complete step.
    intransitive: frozenset[str] = frozenset()
    # What to say when nothing matched.
    help_text: str = "I did not understand that."
    # The system prompt handed to the language model. `{message}` is substituted.
    prompt: str = ""

    def verb(self, text: str) -> str | None:
        for action, patterns in self.verbs:
            if any(pattern in text for pattern in patterns):
                return action
        return None

    def object_in(self, text: str) -> str | None:
        """Longest match wins, so 「青いタオル」 beats 「タオル」."""
        best: tuple[int, str] | None = None
        for name, words in self.objects.items():
            for word in words:
                if word in text and (best is None or len(word) > best[0]):
                    best = (len(word), name)
        return best[1] if best else None


class DomainRulePlanner:
    """Plans by matching a domain's verbs and objects. No model, no latency, no surprises.

    One step per message, exactly like the office RulePlanner -- and with the same caveat: a
    chained instruction is beyond it by construction, so callers should warn and suggest --llm.
    """

    def __init__(self, domain: Domain):
        self.domain = domain

    def plan(self, message: str) -> Plan:
        text = message.lower().strip()
        if not text:
            return Plan([], reply="I did not catch that.")

        verb = self.domain.verb(text)
        obj = self.domain.object_in(text)
        where = RulePlanner._qualifier(text)
        expect = stated_count(message) if where else None

        if verb:
            if verb in self.domain.intransitive:
                # Pass the whole message through as the argument. Skills read details out of
                # it that the object table does not capture -- `patrol` wants the lap count
                # from "go round twice", `to_basket` wants nothing at all -- and handing over
                # the original text lets each decide for itself.
                return Plan([Step(verb, obj or message)])
            # A transitive verb with no object still needs its argument: "go to corner 3" and
            # "put it on the counter" both carry the detail in the sentence rather than in a
            # word the object table knows. Falling back to the message keeps it.
            return Plan([Step(verb, obj or message, where, expect)])

        # A nameable object with no verb almost always means "go there".
        if obj:
            return Plan([Step("goto", obj, where, expect)])

        if any(greeting in text for greeting in _GREETINGS):
            return Plan([], reply="Hello. " + self.domain.help_text)

        return Plan([], reply=self.domain.help_text)


class DomainLLMPlanner:
    """Plans with a local Gemma 4 model against a domain's prompt, falling back to rules.

    Shares the office LLMPlanner's design and its reasons: load through mlx_vlm rather than
    mlx_lm because Gemma 4 is multimodal, use the 8-bit build because the per-layer embeddings
    quantise badly at 4-bit, and never let a model failure stop the robot -- a bad generation
    degrades to a worse plan, not to no plan.
    """

    def __init__(
        self,
        domain: Domain,
        model_id: str = "lmstudio-community/gemma-4-E2B-it-MLX-8bit",
        max_tokens: int = 220,
        fallback: DomainRulePlanner | None = None,
    ):
        self.domain = domain
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.fallback = fallback or DomainRulePlanner(domain)
        self._model = None
        self._tokenizer = None
        self._config = None

    def _load(self) -> None:
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
                self._tokenizer,
                self._config,
                self.domain.prompt.format(message=message),
                num_images=0,
            )
            reply = generate(
                self._model, self._tokenizer, prompt, [],
                max_tokens=self.max_tokens, verbose=False,
            )
            text = reply if isinstance(reply, str) else getattr(reply, "text", str(reply))
            steps = parse_plan(text, allowed=tuple(action for action, _ in self.domain.verbs))
            if steps:
                return Plan(steps)
            log.warning("planner model produced no usable steps; falling back to rules")
        except Exception as e:  # noqa: BLE001 - a planner failure must not stop the robot
            log.warning("planner model unavailable (%s); falling back to rules", e)
        return self.fallback.plan(message)


# ======================================================================================
# The home / laundry domain
# ======================================================================================

HOME = Domain(
    name="home",
    verbs=(
        # Specific before general throughout: 「洗濯機を開けて」 must beat 「開けて」, and
        # "put it on the counter" must beat "put it in the basket"'s shared "put".
        # Closing comes FIRST. Matching runs in order and stops at the first hit, so with
        # open_washer ahead of it 「洗濯機のドアを閉めて」 matched 「洗濯機」 and came out as
        # "open the washer" -- the opposite of what was asked.
        ("close_washer", ("close the washer", "close the washing machine", "close the door",
                          "shut the", "洗濯機を閉めて", "ドアを閉めて", "扉を閉めて",
                          "閉めて", "しめて")),
        ("open_washer", ("open the washer", "open the washing machine", "open the drum",
                         "洗濯機を開けて", "洗濯機をあけて", "ドラムを開けて", "扉を開けて",
                         "開けて", "あけて")),
        # Asking for the basket to be fetched is a reasonable thing to say and the robot cannot
        # do it -- the basket is fixed scenery. Having the verb means it can say so, instead of
        # falling through to "I do not know how to that".
        ("bring_basket", ("bring the basket", "fetch the basket", "get the basket",
                          "籠をもってきて", "かごをもってきて", "カゴをもってきて",
                          "籠を持ってきて", "かごを持ってきて", "カゴを持ってきて")),
        ("take_out", ("take out", "take it out", "get the towel", "take the towel",
                      "unload", "pull it out", "取り出して", "出して", "取って")),
        ("to_basket", ("in the basket", "into the basket", "to the basket", "in the hamper",
                       "かごに", "カゴに", "かごへ", "カゴへ", "洗濯かご")),
        ("to_counter", ("on the counter", "onto the counter", "to the counter",
                        "on the washstand", "on the vanity",
                        "洗面台に", "洗面台へ", "カウンターに", "台の上に", "台に置いて")),
        ("fold", ("fold", "folding", "畳んで", "たたんで", "折りたたんで", "折って")),
        ("describe", ("what do you see", "describe", "what can you see",
                      "何が見える", "何が見えますか", "見えるもの")),
        ("where", ("where are you", "your position", "どこにいる", "現在地", "どこですか")),
        ("home", ("go back to where you started", "back to the start",
                  "元の位置に戻って", "最初の位置に戻って", "戻って")),
    ),
    objects={
        "blue": ("blue towel", "the blue one", "青いタオル", "青いほう", "ブルー"),
        "pink": ("pink towel", "the pink one", "ピンクのタオル", "ピンク"),
        "towel": ("towel", "laundry", "washing", "タオル", "洗濯物", "洗濯もの"),
    },
    intransitive=frozenset({"open_washer", "to_basket", "describe", "where", "home"}),
    help_text=(
        "I can open the washing machine, take the laundry out, put it in the basket or on "
        "the counter, and fold it. Try: 「タオルを洗濯機から出して畳んで」"
    ),
    prompt="""You control a small humanoid robot in a home laundry room. There is a front-\
loading washing machine with a towel inside, a laundry basket on a stand, and a washstand \
counter with room to lay laundry out flat.

Available actions:
  open_washer        open the washing machine door
  take_out <towel>   take a towel out of the drum and hold it
  to_basket          put whatever is being held into the laundry basket
  to_counter <towel> put a towel down on the washstand counter
  fold <towel>       fold a towel that is lying on the counter
  describe           say what is currently in view
  where              report where in the room the robot is
  home               go back to where the robot started
  report <text>      say something to the user

The towels can be named "blue" or "pink". Leave the argument out if the user did not say.

Rules:
- The washing machine has to be opened before anything can be taken out of it.
- A towel has to be ON THE COUNTER before it can be folded -- folding needs a flat surface.
  So "take the towel out and fold it" is: open_washer -> take_out -> to_counter -> fold.
- Only use the action names listed above.
- Break a multi-part instruction into one step per action, in the order the user said them.

The user said: "{message}"

Reply with ONLY a JSON array of steps, no other text. Examples:

"open the washing machine"
[{{"action": "open_washer"}}]

"take the towel out and put it in the basket"
[{{"action": "open_washer"}}, {{"action": "take_out"}}, {{"action": "to_basket"}}]

"take the blue towel out of the washer and fold it"
[{{"action": "open_washer"}}, {{"action": "take_out", "argument": "blue"}}, \
{{"action": "to_counter", "argument": "blue"}}, {{"action": "fold", "argument": "blue"}}]

If the request is just conversation, reply with:
[{{"action": "report", "argument": "<your reply>"}}]""",
)


# ======================================================================================
# The outdoor patrol domain
# ======================================================================================

PATROL = Domain(
    name="patrol",
    verbs=(
        ("patrol", ("patrol", "walk the route", "go round the building", "do a lap",
                    "walk around the building", "circuit", "巡回", "見回り",
                    "一周", "周回", "ビルの周りを")),
        ("climb", ("climb", "go up the steps", "up the stairs", "go up to the entrance",
                   "階段", "段を上", "上って", "のぼって")),
        ("goto", ("go to", "walk to", "head to", "move to", "waypoint", "corner",
                  "行って", "移動して", "向かって", "地点")),
        ("look_around", ("look around", "scan", "have a look",
                         "見回して", "周りを見て", "あたりを見て")),
        ("describe", ("what do you see", "describe", "what can you see",
                      "何が見える", "何が見えますか")),
        ("where", ("where are you", "your position", "どこにいる", "現在地", "どこですか")),
        ("home", ("go back to where you started", "back to the start", "come back",
                  "元の位置に戻って", "最初の位置に戻って", "戻って")),
    ),
    intransitive=frozenset({"patrol", "climb", "look_around", "describe", "where", "home"}),
    help_text=(
        "I can patrol around the building, go to a numbered corner, climb the steps to the "
        "entrance, and tell you what I can see. Try: 「ビルの周りを1周して」"
    ),
    prompt="""You control a four-legged patrol robot outdoors, at a building on a small site. \
There is a lawn, trees, hedges, bollards, and a flight of three steps up to the building \
entrance. Four numbered waypoints mark the corners of the patrol route around the building.

Available actions:
  patrol [n]      walk the whole route round the building, n laps (default 1)
  goto <n>        go to numbered waypoint n (1 to 4)
  climb           climb the steps to the building entrance
  look_around     turn on the spot and report what is visible
  describe        say what is currently in view
  where           report which side of the building the robot is on
  home            go back to where the robot started
  report <text>   say something to the user

Rules:
- Only use the action names listed above. There is no action for opening anything: this robot
  has no arms.
- "go round the building" or "do a lap" is `patrol`, not four `goto` steps.
- Break a multi-part instruction into one step per action, in the order the user said them.

The user said: "{message}"

Reply with ONLY a JSON array of steps, no other text. Examples:

"patrol around the building"
[{{"action": "patrol"}}]

"go to corner 2, then climb the steps"
[{{"action": "goto", "argument": "2"}}, {{"action": "climb"}}]

"walk the route twice and tell me what you saw"
[{{"action": "patrol", "argument": "2"}}, {{"action": "describe"}}]

If the request is just conversation, reply with:
[{{"action": "report", "argument": "<your reply>"}}]""",
)


# ======================================================================================
# The lunar domain
# ======================================================================================

LUNAR = Domain(
    name="lunar",
    verbs=(
        ("survey", ("survey", "look around", "scan the area", "have a look round",
                    "見回して", "周りを見て", "観測", "探索")),
        ("attitude", ("how steep", "are you tilted", "attitude", "your tilt",
                      "傾き", "傾いて", "姿勢")),
        ("home", ("go back to the lander", "return to base", "come back", "back to the lander",
                  "着陸船に戻って", "基地に戻って", "戻って")),
        ("goto", ("go to", "drive to", "head to", "head for", "make for", "approach",
                  "行って", "向かって", "移動して", "まで行って")),
        ("describe", ("what do you see", "describe", "what can you see",
                      "何が見える", "何が見えますか")),
        ("where", ("where are you", "your position", "どこにいる", "現在地", "どこですか")),
    ),
    objects={
        "lander": ("lander", "base", "着陸船", "着陸機", "ランダー", "基地"),
        "ice": ("ice", "water", "deposit", "氷", "水", "氷床"),
        "beacon": ("beacon", "marker", "mast", "ビーコン", "目印", "標識"),
        "crater": ("crater", "rim", "クレーター", "クレータ", "縁", "ふち"),
    },
    intransitive=frozenset({"survey", "attitude", "describe", "where", "home"}),
    help_text=(
        "I can drive to the lander, the ice deposit, the beacon or the crater rim, survey the "
        "area, and report how steeply I am tilted. Try: 「クレーターの縁まで行って」"
    ),
    prompt="""You control a six-wheeled rover on the lunar south pole. The surface is cratered \
regolith. There is a lander (the rover's base), an ice deposit, a survey beacon, and the rim \
of a large crater. The Sun is low, so much of the surface is in deep shadow.

Available actions:
  goto <target>   drive to a target: lander, ice, beacon, or crater
  survey          turn a full circle and report what is visible and where the shadows are
  describe        say what is currently in view
  where           report where the rover is relative to the named places
  attitude        report how steeply the rover is pitched and rolled
  home            drive back to the lander
  report <text>   say something to the user

Rules:
- Only use the action names and target names listed above.
- Driving is slow and the terrain can stop the rover, so do not chain more than a few drives.
- Break a multi-part instruction into one step per action, in the order the user said them.

The user said: "{message}"

Reply with ONLY a JSON array of steps, no other text. Examples:

"drive to the beacon"
[{{"action": "goto", "argument": "beacon"}}]

"go to the ice and tell me what you see"
[{{"action": "goto", "argument": "ice"}}, {{"action": "describe"}}]

"have a look round, then come back to the lander"
[{{"action": "survey"}}, {{"action": "home"}}]

If the request is just conversation, reply with:
[{{"action": "report", "argument": "<your reply>"}}]""",
)


DOMAINS: dict[str, Domain] = {"home": HOME, "patrol": PATROL, "lunar": LUNAR}
