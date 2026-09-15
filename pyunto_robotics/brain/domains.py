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



# What every domain prompt has to end with, and none of them did.
#
# The model was answering correctly -- asked to move somewhere sunny it replied
# `fetch_power` -- and `parse_plan` threw the answer away, because it looks for a JSON array
# and nothing had asked the model for one. Worse, the prompts never included the user's
# message at all: the model was being asked to plan for a sentence it had not been shown, and
# only got the right answer by guessing from the domain description. Every LLM plan silently
# fell back to the keyword rules, which is why --llm appeared to do nothing.
#
# Kept in one place so a fifth domain cannot forget it.
_OUTPUT_FORMAT = """

The user said: "{message}"

Reply with ONLY a JSON array of steps, no other text. Examples:

  [{{"action": "<action>"}}]
  [{{"action": "<action>", "argument": "<target>"}}]
  [{{"action": "<first>"}}, {{"action": "<second>"}}]

If the request is just conversation, reply with:
  [{{"action": "report", "argument": "<your reply>"}}]
"""

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

    def replan(
        self,
        original: str,
        failed_step: str,
        failure: str,
        remaining: list[str],
        view: str = "",
    ) -> list[Step] | None:
        """Rework the rest of an errand after a step failed. None if it cannot help.

        This is the last tier of recovery, after looking around and wandering. Those two live
        in the skills, where the target is known and the problem is "I cannot see it". This one
        handles the other case: the robot knows exactly where things are, and the PLAN has been
        overtaken by events.

        The case it was written for: carrying the basket to the washer puts the basket exactly
        where the robot needed to stand to reach into the drum. Nothing is broken and nothing
        is lost -- the plan was simply written before the basket moved, and the fix is to do
        the remaining steps in a different order or from a different place. A planner that only
        ever sees the original sentence cannot work that out; one that is told what the robot
        can see and what just went wrong usually can.
        """
        try:
            self._load()
            from mlx_vlm import generate  # noqa: PLC0415
            from mlx_vlm.prompt_utils import apply_chat_template  # noqa: PLC0415

            question = _REPLAN_PROMPT.format(
                actions="\n".join(
                    f"  {action}" for action, _ in self.domain.verbs
                ),
                original=original,
                failed_step=failed_step,
                failure=failure,
                remaining=", ".join(remaining) or "(nothing)",
                view=view or "(nothing in particular)",
            )
            prompt = apply_chat_template(self._tokenizer, self._config, question, num_images=0)
            reply = generate(
                self._model, self._tokenizer, prompt, [],
                max_tokens=self.max_tokens, verbose=False,
            )
            text = reply if isinstance(reply, str) else getattr(reply, "text", str(reply))
            steps = parse_plan(text, allowed=tuple(a for a, _ in self.domain.verbs))
            return steps or None
        except Exception as e:  # noqa: BLE001 - recovery must never itself be fatal
            log.warning("could not replan (%s)", e)
            return None


_REPLAN_PROMPT = """A household robot was carrying out an errand and one step failed.

Available actions:
{actions}

The plan it was following:
  {original}

The step that failed:
  {failed_step}

What the robot reported:
  {failure}

Steps it had not reached yet:
  {remaining}

What the robot can see from where it is standing:
  {view}

Work out what it should do NOW to finish the errand. The failure is usually not fatal -- more
often something has moved, or the robot is standing in the wrong place, and doing the same
steps in a different order or after repositioning will work.

Reply with ONLY a JSON array of the remaining steps, no other text. For example:
[{{"action": "open_washer"}}, {{"action": "take_out"}}]

If the errand genuinely cannot be finished, reply with an empty array: []"""


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
                          "carry the basket", "move the basket", "basket to the washer",
                          "籠をもってきて", "かごをもってきて", "カゴをもってきて",
                          "籠を持ってきて", "かごを持ってきて", "カゴを持ってきて",
                          "洗濯かごを", "洗濯カゴを", "かごを運んで", "カゴを運んで",
                          "籠を運んで", "かごを持って", "カゴを持って")),
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
    intransitive=frozenset(
        {"open_washer", "close_washer", "to_basket", "describe", "where", "home"}
    ),
    help_text=(
        "I can open the washing machine, take the laundry out, put it in the basket or on "
        "the counter, and fold it. Try: 「タオルを洗濯機から出して畳んで」"
    ),
    prompt="""You control a small humanoid robot in a home laundry room. There is a front-\
loading washing machine with a towel inside, a laundry basket on a stand, and a washstand \
counter with room to lay laundry out flat.

Available actions:
  open_washer        open the washing machine door
  close_washer       push the washing machine door shut again
  take_out <towel>   take a towel out of the drum and hold it
  to_basket          put whatever is being held into the laundry basket
  to_counter <towel> put a towel down on the washstand counter
  fold <towel>       fold a towel that is lying on the counter
  bring_basket <where>  pick the laundry basket up and carry it somewhere
                     (washer, counter -- defaults to the washer)
  describe           say what is currently in view
  where              report where in the room the robot is
  home               go back to where the robot started
  report <text>      say something to the user

The towels can be named "blue" or "pink". Leave the argument out if the user did not say.

Rules:
- The washing machine has to be opened before anything can be taken out of it.
- A towel has to be ON THE COUNTER before it can be folded -- folding needs a flat surface.
  So "take the towel out and fold it" is: open_washer -> take_out -> to_counter -> fold.
- But ONLY fetch a towel when the user actually asks for it to be fetched. If they just say
  "fold the towel" and say nothing about the washing machine, the towel is already out and
  the whole plan is: fold. Do not add open_washer or take_out to a bare folding request --
  the robot would walk to the washer and rummage in an empty drum while the towel sits on
  the counter in front of it. Saying WHERE the towel is ("the towel on the counter",
  「カウンターの上のタオル」) is the same bare request: it is already on the counter, so the
  plan is still just: fold. Only the washing machine being named means fetching.
- The robot has ONE pair of hands and can hold one towel at a time. Put a towel down before
  picking anything else up.
- close_washer needs BOTH HANDS, so it cannot be done while carrying laundry. If the user asks
  to shut the door after taking the washing out, put the laundry down first and close the door
  after: take_out -> to_basket -> close_washer, or take_out -> to_counter -> close_washer.
- The counter IS the table laundry is folded on. "carry it to the folding table" is to_counter.
- The basket CAN be carried now. "bring the basket to the washer" is bring_basket with
  argument "washer"; it is a real errand, not a refusal.
- bring_basket needs BOTH HANDS, like close_washer, so it cannot be done while holding a
  towel. Fetch the basket BEFORE taking the laundry out, which is also the sensible order.
- Only use the action names listed above.
- Break a multi-part instruction into one step per action, in the order the user said them.
  Japanese chains actions with the て-form and 「、」 and no conjunction -- 「開けて、出して、
  畳んで」 is three actions, not one.

The user said: "{message}"

Reply with ONLY a JSON array of steps, no other text. Examples:

"open the washing machine"
[{{"action": "open_washer"}}]

"take the towel out and put it in the basket"
[{{"action": "open_washer"}}, {{"action": "take_out"}}, {{"action": "to_basket"}}]

"take the blue towel out of the washer and fold it"
[{{"action": "open_washer"}}, {{"action": "take_out", "argument": "blue"}}, \
{{"action": "to_counter", "argument": "blue"}}, {{"action": "fold", "argument": "blue"}}]

"タオルを畳んで"
  (nothing was said about the washing machine, so the towel is already out: just fold it)
[{{"action": "fold"}}]

"カウンターの上のタオルを畳んでください"
  (the towel is on the counter already; naming the counter is not a request to fetch it)
[{{"action": "fold"}}]

"洗濯カゴを洗濯機の前に持ってきて、洗濯機のドアを開けて、洗濯物をカゴに入れて、ドアを閉めて"
  (fetch the basket FIRST -- carrying it needs both hands, as does shutting the door)
[{{"action": "bring_basket", "argument": "washer"}}, {{"action": "open_washer"}}, \
{{"action": "take_out"}}, {{"action": "to_basket"}}, {{"action": "close_washer"}}]

"洗濯機を開けて、洗濯物を取り出して、洗濯機のドアを閉めて、畳むテーブルまで運んで"
  (the door is shut AFTER the laundry is put down, because closing needs both hands)
[{{"action": "open_washer"}}, {{"action": "take_out"}}, {{"action": "to_counter"}}, \
{{"action": "close_washer"}}]

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


SOLAR = Domain(
    name="solar",
    verbs=(
        # The whole errand in one instruction. This has to come first: "go and get power"
        # contains "go", and matching the bare verb would send the robot nowhere in particular
        # with no intention of charging when it arrived.
        # Order is everything here, because patterns are substrings and the words overlap.
        # "power the lights" contains "power"; "how much charge" contains "charge". So the
        # specific intents are listed before the general ones, and the general ones are
        # matched on the bare noun -- which is how people actually write the request
        # ("go and fetch SOME power" contains none of "fetch power" or "get power").
        ("power_lights", ("turn on the lights", "power the lights", "light the house",
                          "switch on the lights", "the lights", "照明をつけて", "電気をつけて",
                          "家の明かりをつけて", "ライトをつけて")),
        # Asking ABOUT the battery, before any instruction to go and fill it.
        #
        # 「あとどのくらいで充電が満タンになりますか？」 is a question, and it matched
        # fetch_power on the strength of 充電 -- so the robot answered a question by driving
        # off on an errand. Questions are recognised by what they ask about (charge level,
        # how long, how full) rather than by the word 充電, which appears in both.
        ("battery", ("how much charge", "how much power", "battery", "state of charge",
                     "how full", "how long until", "how long to", "when will it be full",
                     "バッテリー", "残量", "充電量", "どれくらい貯まった",
                     "満タン", "何%", "何パーセント", "充電レベル", "どのくらいで",
                     "あとどのくらい", "あとどれくらい", "充電は足りて")),
        # Matched on the SUN, not on the verb.
        #
        # These patterns only caught 探して ("search"), so 「日が当たるところに移動して」 --
        # which is how the instruction actually gets written -- fell through to `goto` on the
        # strength of 移動して, and the robot went hunting for a landmark called
        # "日が当たるところに移動して". The subject of the sentence is what matters here: any
        # instruction naming sunlight and a place means find the sun, whichever verb it uses.
        # "Go to the sun AND charge" is the whole errand, so it has to be tested before
        # either half. Both halves appear in it -- the sun words and the power words -- and
        # whichever is listed first would otherwise win and do only that half: measured
        # 「日の当たるところに移動して、充電して」 matching find_sun and stopping in the park
        # without charging.
        ("fetch_power", (
            "電力を取", "電力を取得", "充電してきて", "発電してきて", "電気を取って",
            "エネルギーを取", "取ってきて",
            # 「…に移動して、充電して」 -- the errand written as two clauses. It ends in
            # 充電して, not 充電してきて, so the patterns above missed it and the sun half
            # won: the robot walked to the park and stopped there without charging.
            "移動して、充電", "移動して充電", "行って、充電", "行って充電",
            "fetch power", "get power", "collect power", "fetch energy", "get energy",
            "collect energy", "charge up and come back",
        )),
        # Matched on the SUN, not on the verb.
        #
        # These patterns only caught 探して ("search"), so 「日が当たるところに移動して」 --
        # which is how the instruction actually gets written -- fell through to `goto` on the
        # strength of 移動して, and the robot went hunting for a landmark called
        # "日が当たるところに移動して". The subject of the sentence is what matters: an
        # instruction naming sunlight and a place means find the sun, whichever verb it uses.
        ("find_sun", ("find the sun", "find sunlight", "find somewhere sunny", "look for sun",
                      "sunny spot", "in the sun",
                      "日光", "日向", "日なた", "陽の当たる", "日の当たる", "日が当たる",
                      "陽が当たる", "太陽")),
        ("charge", ("charge here", "start charging", "collect here",
                    "ここで充電", "ここで発電")),
        # The bare nouns, last. Anything still mentioning power or charge after the specific
        # phrasings above have had their turn means the whole errand. Named `fetch_power`
        # again rather than a separate verb, because it IS the same action -- an earlier
        # attempt at a distinct name left a verb the skills had no handler for.
        ("fetch_power", ("power", "energy", "charge", "電力", "充電", "発電", "電気",
                         "エネルギー")),
        ("home", ("go home", "return home", "come back", "back to the carport",
                  "家に戻って", "帰って", "駐車場に戻って", "戻って")),
        ("goto", ("go to", "drive to", "head to", "head for", "make for", "approach",
                  "行って", "向かって", "移動して", "まで行って")),
        ("describe", ("what do you see", "describe", "what can you see",
                      "何が見える", "何が見えますか")),
        ("where", ("where are you", "your position", "どこにいる", "現在地", "どこですか")),
    ),
    objects={
        "park": ("park", "open ground", "the green", "公園", "広場", "空き地"),
        "street": ("street", "road", "道", "道路", "通り"),
        "home": ("home", "house", "carport", "家", "自宅", "駐車場", "カーポート"),
    },
    intransitive=frozenset({
        "fetch_power", "find_sun", "charge", "power_lights", "battery",
        "describe", "where", "home",
    }),
    help_text=(
        "I can go and find sunlight, charge there, come home and put the power into the house "
        "lights. Try: 「日光が当たる場所まで移動して、電力を取得してきて」"
    ),
    prompt="""You control a small four-wheeled robot with a solar panel on its back. It lives \
in a carport at a house. The carport roof shades it, so it cannot charge at home. The street \
outside is shaded by tall hedges. To the east there is an open park where the low afternoon \
sun still reaches the ground.

The robot's battery powers the house lights when it gets home.

Available actions:
  fetch_power     the whole errand: leave home, find sunlight, charge, come back, light the house
  find_sun        search for somewhere the sun actually reaches
  charge          stay put and collect power where the robot is now
  power_lights    put the stored charge into the house lights
  battery         report the state of charge
  goto <target>   drive to a target: park, street, or home
  home            drive back to the carport
  describe        say what is currently in view
  where           report where the robot is
  report <text>   say something to the user

Rules:
- Only use the action names and target names listed above.
- Prefer `fetch_power` when the user asks for power or energy without saying how.
- Charging in shade collects almost nothing, so find sunlight before charging.
""" + _OUTPUT_FORMAT,
)


MARS = Domain(
    name="mars",
    verbs=(
        ("survey", ("survey", "look around", "scan the area", "have a look round",
                    "見回して", "周りを見て", "観測して", "探索して")),
        ("attitude", ("how steep", "are you tilted", "attitude", "your tilt",
                      "傾き", "傾いて", "姿勢")),
        ("home", ("go back to the lander", "return to base", "back to base", "come back",
                  "着陸機に戻って", "着陸船に戻って", "基地に戻って", "戻って")),
        ("goto", ("go to", "drive to", "head to", "head for", "make for", "approach",
                  "行って", "向かって", "移動して", "まで行って")),
        ("describe", ("what do you see", "describe", "what can you see",
                      "何が見える", "何が見えますか")),
        ("where", ("where are you", "your position", "どこにいる", "現在地", "どこですか")),
    ),
    objects={
        "lander": ("lander", "base", "着陸機", "着陸船", "ランダー", "基地"),
        "cache": ("cache", "sample", "sample cache", "サンプル", "採取装置", "キャッシュ"),
        "beacon": ("beacon", "marker", "ビーコン", "目印", "標識"),
    },
    intransitive=frozenset({"survey", "attitude", "describe", "where", "home"}),
    help_text=(
        "I can drive to the sample cache, the beacon or the lander, survey what is visible, "
        "and report how steeply I am tilted. Try: 「サンプルまで行って」"
    ),
    prompt="""You control a six-wheeled rover on Mars, in an old outflow channel. There is a \
lander (the rover's base) up on the bank, a sample cache out on the channel floor beyond some \
dunes, and a beacon on the far bank. The ground is rocky and uneven.

Available actions:
  goto <target>   drive to a target: lander, cache, or beacon
  survey          turn a full circle and report what hardware is visible
  describe        say what is currently in view
  where           report how far the rover is from the lander
  attitude        report how steeply the rover is tilted
  home            drive back to the lander
  report <text>   say something to the user

Rules:
- Only use the action names and target names listed above.
- Prefer `goto` when the user names a place to drive to.
""" + _OUTPUT_FORMAT,
)


ORCHARD = Domain(
    name="orchard",
    verbs=(
        # The whole errand first: "go and fetch the apples" contains "go", and matching the
        # bare verb would walk the robot out and leave it standing among the trees.
        ("fetch", ("fetch", "bring the crates", "bring the apples", "bring in", "collect the",
                   "取ってきて", "運んできて", "持ってきて", "回収して")),
        # "What are you carrying?" before "carry it in": 「何を運んでいる」 contains 「運んで」,
        # so asking the question matched the instruction and the robot walked off to the shed
        # instead of answering.
        ("carrying", ("what are you carrying", "are you carrying", "carrying",
                      "何を運んでいる", "何を積んでいる", "積んでいる", "積載")),
        ("deliver", ("take them to the shed", "carry them in", "deliver", "take it in",
                     "小屋に運んで", "倉庫に運んで", "運んで", "納めて")),
        ("collect", ("load", "pick them up", "load up", "積んで", "積み込んで")),
        ("describe", ("what do you see", "describe", "what can you see",
                      "何が見える", "何が見えますか")),
        ("where", ("where are you", "your position", "どこにいる", "現在地", "どこですか")),
        ("goto", ("go to", "walk to", "head to", "head for", "approach",
                  "行って", "向かって", "移動して", "まで行って")),
    ),
    objects={
        "crates": ("crates", "crate", "apples", "fruit", "コンテナ", "かご", "リンゴ",
                   "りんご", "収穫物", "箱"),
        "shed": ("shed", "packing shed", "barn", "小屋", "倉庫", "作業場", "選果場"),
    },
    intransitive=frozenset({"fetch", "deliver", "collect", "carrying", "describe", "where"}),
    help_text=(
        "I can walk out to the crates, load them, and carry them to the packing shed. "
        "Try: 「リンゴのコンテナを取ってきて」"
    ),
    prompt="""You control a four-legged robot in an apple orchard. There are two rows of trees \
with a working lane between them. Crates of picked apples are stacked at the far end of the \
lane; the packing shed is at the near end, where the robot starts.

Available actions:
  fetch           the whole errand: walk to the crates, load them, carry them to the shed
  goto <target>   walk to a target: crates or shed
  collect         load the crates, if the robot is beside them
  deliver         carry what it has back to the shed
  carrying        report what the robot is carrying
  describe        say what is currently in view
  where           report how far the robot is from the shed
  report <text>   say something to the user

Rules:
- Only use the action names and target names listed above.
- Prefer `fetch` when the user asks for the fruit to be brought in without saying how.
""" + _OUTPUT_FORMAT,
)


HOTEL = Domain(
    name="hotel",
    verbs=(
        # "Clean both floors" before the bare "clean this corridor", and both before "floor",
        # because every one of them contains the word the next one matches on.
        ("clean", ("clean both", "clean the whole", "clean everything", "both floors",
                   "全部掃除", "両方掃除", "全フロア", "全部きれいに",
                   # 「両方のフロアを掃除して」 is the natural phrasing and matched none of the
                   # above -- it fell through to clean_floor and did one corridor. Match on
                   # the words that mean "both" rather than on whole sentences.
                   "両方", "両フロア", "二階とも", "全階")),
        ("clean_floor", ("clean this", "clean the corridor", "clean here", "clean",
                         "掃除して", "清掃して", "きれいにして")),
        ("board", ("get in the lift", "board the lift", "into the lift", "take the lift",
                   "エレベータに乗って", "エレベーターに乗って", "乗って")),
        ("ride", ("go up", "go down", "next floor", "other floor", "ride",
                  "上の階", "下の階", "階を移動", "上がって", "降りて")),
        ("floor", ("which floor", "what floor", "何階", "階数")),
        ("describe", ("what do you see", "describe", "what can you see",
                      "何が見える", "何が見えますか")),
        ("where", ("where are you", "your position", "どこにいる", "現在地", "どこですか")),
        ("goto", ("go to", "walk to", "head to", "行って", "向かって", "移動して")),
    ),
    objects={
        "lift": ("lift", "elevator", "エレベータ", "エレベーター", "昇降機"),
    },
    intransitive=frozenset({
        "clean", "clean_floor", "board", "ride", "floor", "describe", "where",
    }),
    help_text=(
        "I can clean a corridor, take the lift, and clean the corridor on the other floor. "
        "Try: 「両方のフロアを掃除して」"
    ),
    prompt="""You control a cleaning robot in a small hotel. There are two corridors, one on \
each of two floors, and a lift at the west end that connects them.

Available actions:
  clean           the whole job: clean this corridor, ride the lift, clean the other one
  clean_floor     clean the corridor the robot is standing in
  board           walk to the lift and get in
  ride            take the lift to the other floor
  floor           report which floor the robot is on
  describe        say where the robot is and what it has cleaned
  report <text>   say something to the user

Rules:
- Only use the action names listed above.
- Prefer `clean` when the user asks for the hotel or both floors to be cleaned.
""" + _OUTPUT_FORMAT,
)


HOUSE = Domain(
    name="house",
    verbs=(
        # Order matters throughout: these phrases overlap heavily. "Is the door locked?" is a
        # question and 「鍵をかけて」 is an instruction, and both contain the word for lock; the
        # question has to be tested first or asking becomes doing, which for a front door is
        # the worse way round to get wrong.
        ("lock_status", ("is the door locked", "is it locked", "did i lock", "door locked",
                         "鍵はかかってる", "施錠されてる", "鍵かかってる", "戸締まりした")),
        ("status", ("everything alright", "how is the house", "house status", "all okay",
                    "家の様子", "全部教えて", "状況", "ぜんぶ")),
        ("temperature", ("how warm", "how cold", "what is the temperature", "temperature",
                         "how hot", "何度", "気温", "室温", "温度は")),
        ("set_temperature", ("set the aircon", "set it to", "set to", "degrees",
                             "度にして", "度に設定")),
        ("warmer", ("warmer", "warm it up", "turn up the heat", "too cold",
                    "暖かく", "あたたかく", "温度を上げて", "寒い")),
        ("cooler", ("cooler", "cool it down", "turn it down", "too hot", "too warm",
                    "涼しく", "すずしく", "温度を下げて", "暑い")),
        ("aircon_off", ("turn the aircon off", "aircon off", "turn off the air",
                        "エアコンを消して", "エアコンオフ", "冷房を止めて", "暖房を止めて")),
        ("aircon_on", ("turn the aircon on", "aircon on", "turn on the air", "air conditioning",
                       "エアコンをつけて", "エアコンオン", "冷房", "暖房", "エアコン")),
        ("lights_off", ("lights off", "turn the lights off", "turn off the light",
                        "電気を消して", "明かりを消して", "照明を消して")),
        ("lights_on", ("lights on", "turn the lights on", "turn on the light",
                       "電気をつけて", "明かりをつけて", "照明をつけて")),
        ("unlock", ("unlock", "open the door", "鍵を開けて", "解錠", "開錠")),
        ("lock", ("lock up", "lock the door", "lock", "鍵をかけて", "施錠", "戸締まり")),
    ),
    objects={
        "living room": ("living room", "lounge", "リビング", "居間"),
        "bedroom": ("bedroom", "寝室", "ベッドルーム"),
        "kitchen": ("kitchen", "キッチン", "台所"),
    },
    intransitive=frozenset({
        "lock", "unlock", "lock_status", "status", "temperature", "warmer", "cooler",
        "aircon_on", "aircon_off", "lights_on", "lights_off",
    }),
    help_text=(
        "I look after the house: the air conditioning, the lights, the front door lock and "
        "the thermometers. Try: 「リビングのエアコンを24度にして」"
    ),
    prompt="""You control the devices in a house: air conditioning, lights and a thermometer \
in each of three rooms (living room, bedroom, kitchen), and the lock on the front door.

Available actions:
  temperature       report how warm a room is
  set_temperature   set the air conditioning to a given temperature
  warmer / cooler   nudge the air conditioning up or down
  aircon_on / aircon_off
  lights_on / lights_off
  lock / unlock     the front door
  lock_status       report whether the front door is locked
  status            report the whole house at once
  report <text>     say something to the user

Rules:
- Only use the action names listed above.
- Put the room in `where` when the user names one.
- Asking whether the door is locked is `lock_status`, not `lock`.
""" + _OUTPUT_FORMAT,
)


WATCH = Domain(
    name="watch",
    verbs=(
        # "How is she?" is the question this exists for, so it is matched first and widely.
        ("check", ("how is she", "how is he", "how is mum", "how is dad", "is she alright",
                   "is he alright", "where is she", "what is she doing",
                   "様子", "元気", "どうしてる", "どこにいる", "無事")),
        ("today", ("what happened today", "today so far", "how was today", "the day",
                   "今日は", "今日の様子", "一日", "きょうは")),
        ("watch", ("watch for", "keep an eye", "watch the next", "見ていて", "様子を見て",
                   "見守って")),
        # After `watch`, so 「何時間見ていて」 still asks to watch rather than for the time.
        # Note there is no bare 「何時間」 here for the same reason.
        ("time", ("what time is it", "what's the time", "the time now", "time now",
                  "何時", "なんじ", "時刻", "いま何時", "今の時間")),
        ("lock_status", ("is the door locked", "did she lock", "did he lock",
                         "鍵はかかってる", "施錠されてる", "戸締まり")),
        ("status", ("everything alright", "how is the house", "家の様子", "全部教えて")),
        # Before `temperature`: 「湿度」 contains 「度」, and temperature's 「何度」 would
        # otherwise swallow 「湿度は何度？」.
        ("humidity", ("humidity", "how humid", "how damp", "damp",
                      "湿度", "しつど", "乾燥", "じめじめ")),
        ("temperature", ("how warm", "how cold", "what is the temperature", "temperature",
                         "何度", "気温", "室温", "温度は")),
        # The doorphone: the one camera here, and it faces the street. Before `visitors` so
        # 「ドアフォンの映像」 gets the picture rather than the list.
        ("doorphone", ("doorphone", "door camera", "porch camera", "picture of the door",
                       "ドアフォン", "インターフォンの映像", "玄関の映像", "玄関のカメラ")),
        ("visitors", ("who came", "who called", "any visitors", "visitors", "the door today",
                      "来客", "訪問", "誰か来た", "誰が来た", "インターフォン", "チャイム")),
        ("set_temperature", ("set the aircon", "set it to", "度にして", "度に設定")),
        ("warmer", ("warmer", "warm it up", "暖かく", "温度を上げて", "寒い")),
        ("cooler", ("cooler", "cool it down", "涼しく", "温度を下げて", "暑い")),
        ("aircon_off", ("turn the aircon off", "aircon off", "エアコンを消して", "冷房を止めて")),
        ("aircon_on", ("turn the aircon on", "aircon on", "エアコンをつけて", "冷房", "エアコン")),
        ("lights_off", ("lights off", "turn the lights off", "電気を消して", "照明を消して")),
        ("lights_on", ("lights on", "turn the lights on", "電気をつけて", "照明をつけて")),
        ("unlock", ("unlock", "鍵を開けて", "解錠")),
        ("lock", ("lock up", "lock the door", "鍵をかけて", "施錠")),
    ),
    objects={
        "living room": ("living room", "リビング", "居間"),
        "bedroom": ("bedroom", "寝室"),
        "kitchen": ("kitchen", "キッチン", "台所"),
    },
    intransitive=frozenset({
        "check", "today", "watch", "time", "lock", "unlock", "lock_status", "status",
        "temperature", "humidity", "visitors", "doorphone",
        "warmer", "cooler", "aircon_on", "aircon_off",
        "lights_on", "lights_off",
    }),
    help_text=(
        "I watch the flat and tell you how she is. Ask 「様子はどう？」, 「今日は何してた？」 "
        "or 「今何時？」, and I can work the air conditioning, lights and lock too."
    ),
    prompt="""You watch a flat where an older person lives alone, through motion sensors in \
each room and a bed sensor. You also control the air conditioning, the lights and the front \
door lock.

You are reporting to a family member who lives elsewhere. Answer their questions, and speak \
plainly: say what the sensors saw, not what it might mean medically.

Available actions:
  check             where she is now and whether she is up
  today             what has happened today, with times
  watch <hours>     let time pass and report anything worth saying
  time              what time it is in the flat, and how fast the day is running
  temperature       how warm a room is
  humidity          how damp a room is
  visitors          who has been to the front door today, and whether she answered
  doorphone         the porch camera's view of the most recent caller
  set_temperature / warmer / cooler
  aircon_on / aircon_off
  lights_on / lights_off
  lock / unlock / lock_status
  status            the whole flat at once
  report <text>     say something to the user

Rules:
- Only use the action names listed above.
- Prefer `check` when asked how she is or where she is.
- Use `time` for "what time is it"; use `watch` for "watch her for N hours".
- There are no cameras inside the flat, by design. The only camera is the doorphone, which
  faces the street: `doorphone` shows who rang the bell, never the person being watched.
  If asked for a picture of a room or of her, say that and offer `check` instead.
""",
)


PET = Domain(
    name="pet",
    verbs=(
        # "Where is she?" is the question this exists for, so it is matched first and widely.
        ("find", ("where is", "find her", "find him", "find the cat", "look for",
                  "どこ", "探して", "さがして", "見つけて", "どこにいる")),
        # Before `look`: 「見回って」 contains 「見」, and look's verbs would swallow it.
        ("patrol", ("patrol", "look around", "check everywhere", "the whole flat",
                    "見回", "一周", "全部見て", "ぐるっと")),
        ("photo", ("photo", "picture", "send a picture", "show me",
                   "写真", "しゃしん", "撮って", "見せて")),
        # Camera moves. Before `go`, because 「右に向けて」 is a camera instruction and
        # 「右に行って」 is a driving one, and they differ by one character.
        ("tilt", ("look up", "look down", "tilt", "上に向けて", "下に向けて", "上を", "下を")),
        ("pan", ("pan", "turn the camera", "左に向けて", "右に向けて", "カメラを左", "カメラを右")),
        ("home", ("go home", "go back", "dock", "戻って", "帰って", "ドックに")),
        ("go", ("go to", "drive to", "move to", "へ行って", "に行って", "まで行って")),
        ("look", ("look at", "point at", "を見て", "の方を")),
        ("check", ("is she there", "can you see her", "写ってる", "見えてる", "いる？")),
    ),
    objects={
        "sofa": ("sofa", "couch", "ソファ"),
        "sill": ("windowsill", "window", "sill", "窓", "日なた"),
        "tree": ("cat tree", "tower", "キャットタワー", "タワー"),
        "shelf": ("bookshelf", "shelf", "本棚", "棚"),
        "bowls": ("bowl", "food", "water", "ごはん", "餌", "水"),
    },
    intransitive=frozenset({"find", "patrol", "photo", "home", "check", "pan", "tilt"}),
    help_text=(
        "留守番中の猫を見に行きます。「どこにいる？」「見回って」「写真を送って」、"
        "カメラだけ動かすなら「上に向けて」「左に向けて」。"
    ),
    prompt="""You are a small camera robot in a flat, looking after a cat while the owner is \
out. You can drive to places, aim your camera by panning and tilting, and photograph what you \
see.

You are reporting to the cat's owner, who is not at home and cannot check for themselves. So \
never guess: if you did not see the cat, say you did not find her rather than saying where \
she probably is.

Available actions:
  find              go looking for the cat and say where she is
  patrol            visit every one of her usual places and report each
  photo             send what the camera sees right now
  check             is she in view at this moment, without moving
  go <place>        drive to a place and aim at it
  look <place>      aim at a place without driving to it
  pan <degrees>     turn the camera left or right
  tilt <degrees>    aim the camera up or down
  home              return to the dock
  report <text>     say something to the owner

Her usual places are: sofa (underneath), sill (the sunny windowsill), tree (the cat tree), \
shelf (on top of the bookshelf), bowls (food and water).

Rules:
- Only use the action names listed above.
- Prefer `find` when asked where the cat is.
- `pan` and `tilt` move only the camera; `go` moves the robot.
""",
)


DOMAINS: dict[str, Domain] = {
    "home": HOME, "patrol": PATROL, "lunar": LUNAR, "solar": SOLAR, "mars": MARS,
    "orchard": ORCHARD, "hotel": HOTEL, "house": HOUSE, "watch": WATCH, "pet": PET,
}
