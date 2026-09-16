"""Turning a sentence into a plan, for each robot.

The general planner in brain/planner.py is written around one robot in one building: its verbs
are `open`/`leave`/`home`, its nouns are doors and whiteboards, and its prompt describes a
corridor with three rooms. None of that transfers to a rover on Mars or a camera looking for a
cat.

So each domain gets a Domain: its own verbs, its own words for things, and its own prompt. The
machinery underneath is shared -- the same rule matcher, the same JSON extraction, the same
fallback from model to rules -- because none of that is domain-specific. What changes is the
vocabulary, and vocabulary is data.

A Domain is deliberately small: a list of verbs with the phrasings people use, a list of
objects, and a prompt. Adding a fourth robot means adding one of these, not another planner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

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
    "hello", "hi", "hey",
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
    # "look around" has to beat "look at", and "clean this floor" has to beat "clean".
    verbs: tuple[tuple[str, tuple[str, ...]], ...]
    # Object words, mapped onto the name a skill expects.
    objects: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Verbs that take no argument, so a bare match is a complete step.
    intransitive: frozenset[str] = frozenset()
    # What to say when nothing matched.
    help_text: str = "I did not understand that."
    # The system prompt handed to the language model. `{message}` is substituted.
    #
    # Never read this directly -- use `full_prompt()`, which appends the output format. Two
    # domains were written without it and neither worked: the model was asked to plan for a
    # sentence it had never been shown, and to reply in a format nobody had described, so it
    # answered in prose and every plan silently fell through to the keyword rules. The
    # comment on _OUTPUT_FORMAT said it was kept in one place "so a fifth domain cannot
    # forget it"; the ninth and tenth did, because remembering was still the author's job.
    prompt: str = ""

    def full_prompt(self, message: str) -> str:
        """The prompt actually sent: this domain's, plus the output format, filled in.

        Appends _OUTPUT_FORMAT unless the domain already carries it, so the older domains
        that embed it by hand are unchanged.
        """
        body = self.prompt if "{message}" in self.prompt else self.prompt + _OUTPUT_FORMAT
        return body.format(message=message)

    def with_commands(self, path: str | Path) -> Domain:
        """This domain, with its command list replaced by one from a file.

        Command mode exists for sites that want a closed vocabulary, and a closed vocabulary
        is site-specific: a customer's equipment has its own words, an operator has their own
        six phrases, and neither belongs in this package. So the list is data a customer can
        supply rather than a table they would have to fork the SDK to change.

        The file is JSON, and says only what to match:

            {
              "verbs": {
                "find":  ["where is", "locate", "FIND-TGT"],
                "photo": ["photo", "image", "CAM-SNAP"]
              },
              "objects": {
                "sill": ["windowsill", "POS-03"]
              }
            }

        Actions not named here keep the built-in phrasings, so a customer overrides the two
        verbs their equipment words differently and inherits the rest. Actions that do not
        exist on the robot are rejected rather than silently ignored -- a command that can
        never fire is a fault in the file, and finding it at startup beats finding it when
        somebody types the word.
        """
        import json

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {action for action, _ in self.verbs}
        unknown = set(data.get("verbs", {})) - known
        if unknown:
            raise ValueError(
                f"{path}: no such action on this robot: {', '.join(sorted(unknown))}. "
                f"This robot does: {', '.join(sorted(known))}"
            )
        overrides = {a: tuple(words) for a, words in data.get("verbs", {}).items()}
        verbs = tuple((a, overrides.get(a, words)) for a, words in self.verbs)
        objects = dict(self.objects)
        objects.update({k: tuple(v) for k, v in data.get("objects", {}).items()})
        return replace(self, verbs=verbs, objects=objects)

    def verb(self, text: str) -> str | None:
        # Case-insensitive on both sides. Callers lower the message, but a site's command
        # list is written the way the equipment documents it -- "FIND-TGT", not "find-tgt" --
        # and a code that never matches because of its case is the most annoying kind of
        # silent failure.
        lowered = text.lower()
        for action, patterns in self.verbs:
            if any(pattern.lower() in lowered for pattern in patterns):
                return action
        return None

    def object_in(self, text: str) -> str | None:
        """Longest match wins, so "cat tree" beats "tree"."""
        lowered = text.lower()
        best: tuple[int, str] | None = None
        for name, words in self.objects.items():
            for word in words:
                if word.lower() in lowered and (best is None or len(word) > best[0]):
                    best = (len(word), name)
        return best[1] if best else None


class DomainRulePlanner:
    """Plans by matching a domain's verbs and objects. No model, no latency, no surprises.

    One step per message, exactly like the general RulePlanner -- and with the same caveat: a
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

    Shares the general LLMPlanner's design and its reasons: load through mlx_vlm rather than
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
                self.domain.full_prompt(message),
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

        The shape of case it is for: the robot parks where it needs to stand, and the thing it
        came for is now behind it -- or a door it meant to use is the one it is holding open.
        Nothing is broken and nothing is lost; the plan was simply written before the world
        moved, and the fix is to do the remaining steps in a different order or from a
        different place. A planner that only ever sees the original sentence cannot work that
        out; one that is told what the robot can see and what just went wrong usually can.
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


_REPLAN_PROMPT = """A robot was carrying out an errand and one step failed.

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

Reply with ONLY a JSON array of the remaining steps, no other text, using ONLY the actions
listed above. For example:
[{{"action": "goto", "argument": "shed"}}, {{"action": "deliver"}}]

If the errand genuinely cannot be finished, reply with an empty array: []"""


SOLAR = Domain(
    name="solar",
    verbs=(
        ("power_lights", (
            "turn on the lights", "power the lights", "light the house",
            "switch on the lights", "the lights"
        )),
        ("battery", (
            "how much charge", "how much power", "battery", "state of charge", "how full",
            "how long until", "how long to", "when will it be full"
        )),
        ("fetch_power", (
            "fetch power", "get power", "collect power", "fetch energy", "get energy",
            "collect energy", "charge up and come back"
        )),
        ("find_sun", (
            "find the sun", "find sunlight", "find somewhere sunny", "look for sun",
            "sunny spot", "in the sun"
        )),
        ("charge", ("charge here", "start charging", "collect here")),
        ("fetch_power", ("power", "energy", "charge")),
        ("home", ("go home", "return home", "come back", "back to the carport")),
        ("goto", ("go to", "drive to", "head to", "head for", "make for", "approach")),
        ("describe", ("what do you see", "describe", "what can you see")),
        ("where", ("where are you", "your position")),
    ),
    objects={
        "park": ("park", "open ground", "the green"),
        "street": ("street", "road"),
        "home": ("home", "house", "carport"),
    },
    intransitive=frozenset({
        "fetch_power", "find_sun", "charge", "power_lights", "battery",
        "describe", "where", "home",
    }),
    help_text=(
        "I can go and find sunlight, charge there, come home and put the power into the house "
        'lights. Try: "go and find some sunlight, and bring back power"'
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
        ("survey", ("survey", "look around", "scan the area", "have a look round")),
        ("attitude", ("how steep", "are you tilted", "attitude", "your tilt")),
        ("home", ("go back to the lander", "return to base", "back to base", "come back")),
        ("goto", ("go to", "drive to", "head to", "head for", "make for", "approach")),
        ("describe", ("what do you see", "describe", "what can you see")),
        ("where", ("where are you", "your position")),
    ),
    objects={
        "lander": ("lander", "base"),
        "cache": ("cache", "sample", "sample cache"),
        "beacon": ("beacon", "marker"),
    },
    intransitive=frozenset({"survey", "attitude", "describe", "where", "home"}),
    help_text=(
        "I can drive to the sample cache, the beacon or the lander, survey what is visible, "
        'and report how steeply I am tilted. Try: "drive to the sample"'
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
        ("fetch", ("fetch", "bring the crates", "bring the apples", "bring in", "collect the")),
        ("carrying", ("what are you carrying", "are you carrying", "carrying")),
        ("deliver", ("take them to the shed", "carry them in", "deliver", "take it in")),
        ("collect", ("load", "pick them up", "load up")),
        ("describe", ("what do you see", "describe", "what can you see")),
        ("where", ("where are you", "your position")),
        ("goto", ("go to", "walk to", "head to", "head for", "approach")),
    ),
    objects={
        "crates": ("crates", "crate", "apples", "fruit"),
        "shed": ("shed", "packing shed", "barn"),
    },
    intransitive=frozenset({"fetch", "deliver", "collect", "carrying", "describe", "where"}),
    help_text=(
        "I can walk out to the crates, load them, and carry them to the packing shed. "
        'Try: "fetch the crate of apples"'
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
        ("clean", ("clean both", "clean the whole", "clean everything", "both floors")),
        ("clean_floor", ("clean this", "clean the corridor", "clean here", "clean")),
        ("board", ("get in the lift", "board the lift", "into the lift", "take the lift")),
        ("ride", ("go up", "go down", "next floor", "other floor", "ride")),
        ("floor", ("which floor", "what floor")),
        ("describe", ("what do you see", "describe", "what can you see")),
        ("where", ("where are you", "your position")),
        ("goto", ("go to", "walk to", "head to")),
    ),
    objects={
        "lift": ("lift", "elevator"),
    },
    intransitive=frozenset({
        "clean", "clean_floor", "board", "ride", "floor", "describe", "where",
    }),
    help_text=(
        "I can clean a corridor, take the lift, and clean the corridor on the other floor. "
        'Try: "clean the rooms on both floors"'
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


WATCH = Domain(
    name="watch",
    verbs=(
        ("check", (
            "how is she", "how is he", "how is mum", "how is dad", "is she alright",
            "is he alright", "where is she", "what is she doing"
        )),
        ("today", ("what happened today", "today so far", "how was today", "the day")),
        ("watch", ("watch for", "keep an eye", "watch the next")),
        ("time", ("what time is it", "what's the time", "the time now", "time now")),
        ("lock_status", ("is the door locked", "did she lock", "did he lock")),
        ("status", ("everything alright", "how is the house")),
        ("humidity", ("humidity", "how humid", "how damp", "damp")),
        ("temperature", ("how warm", "how cold", "what is the temperature", "temperature")),
        ("doorphone", ("doorphone", "door camera", "porch camera", "picture of the door")),
        ("visitors", ("who came", "who called", "any visitors", "visitors", "the door today")),
        ("set_temperature", ("set the aircon", "set it to")),
        ("warmer", ("warmer", "warm it up")),
        ("cooler", ("cooler", "cool it down")),
        ("aircon_off", ("turn the aircon off", "aircon off")),
        ("aircon_on", ("turn the aircon on", "aircon on")),
        ("lights_off", ("lights off", "turn the lights off")),
        ("lights_on", ("lights on", "turn the lights on")),
        ("unlock", ("unlock")),
        ("lock", ("lock up", "lock the door")),
    ),
    objects={
        "living room": ("living room",),
        "bedroom": ("bedroom",),
        "kitchen": ("kitchen",),
    },
    intransitive=frozenset({
        "check", "today", "watch", "time", "lock", "unlock", "lock_status", "status",
        "temperature", "humidity", "visitors", "doorphone",
        "warmer", "cooler", "aircon_on", "aircon_off",
        "lights_on", "lights_off",
    }),
    help_text=(
        'I watch the flat and tell you how she is. Ask "how is she?", "what has she done '
        'today?" or "what time is it?", and I can work the air conditioning, lights and '
        "lock too."
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
        ("find", ("where is", "find her", "find him", "find the cat", "look for")),
        ("patrol", ("patrol", "look around", "check everywhere", "the whole flat")),
        ("photo", ("photo", "picture", "send a picture", "show me")),
        ("tilt", ("look up", "look down", "tilt")),
        ("pan", ("pan", "turn the camera")),
        ("home", ("go home", "go back", "dock")),
        ("go", ("go to", "drive to", "move to")),
        ("look", ("look at", "point at")),
        ("check", ("is she there", "can you see her")),
    ),
    objects={
        "sofa": ("sofa", "couch"),
        "sill": ("windowsill", "window", "sill"),
        "tree": ("cat tree", "tower"),
        "shelf": ("bookshelf", "shelf"),
        "bowls": ("bowl", "food", "water"),
    },
    intransitive=frozenset({"find", "patrol", "photo", "home", "check", "pan", "tilt"}),
    help_text=(
        'I go and look for the cat while you are out. Try "where is she?", "look around" '
        'or "send me a photo". To move only the camera: "look up", "turn left".'
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
    "solar": SOLAR, "watch": WATCH, "pet": PET,
    "mars": MARS, "orchard": ORCHARD, "hotel": HOTEL,
}
