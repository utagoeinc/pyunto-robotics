"""Every domain must actually ask the model the question.

Two domains shipped without the output format appended to their prompt. The model was being
asked to plan for a sentence it had never been shown, in a format nobody had described, so it
answered in prose and `parse_plan` threw the answer away -- and the planner fell back to the
keyword rules without a word. `--llm` appeared to do nothing, twice, for the same reason.

Nothing catches that at runtime: the fallback is by design, and a plan still comes back. So it
is pinned here instead.
"""

from __future__ import annotations

import pytest

from pyunto_robotics.brain.domains import DOMAINS


@pytest.mark.parametrize("key", sorted(DOMAINS))
def test_the_prompt_carries_the_user_message(key: str):
    """A prompt that never shows the model the sentence can only be guessing."""
    prompt = DOMAINS[key].full_prompt("SENTINEL PHRASE")
    assert "SENTINEL PHRASE" in prompt


@pytest.mark.parametrize("key", sorted(DOMAINS))
def test_the_prompt_asks_for_json(key: str):
    """`parse_plan` reads a JSON array. A prompt that does not ask for one is discarded."""
    prompt = DOMAINS[key].full_prompt("anything")
    assert "JSON array" in prompt


@pytest.mark.parametrize("key", sorted(DOMAINS))
def test_every_domain_is_english_only(key: str):
    """The keyword tables are English. A stray Japanese pattern can never match now."""
    import re

    japanese = re.compile(r"[぀-ヿ一-鿿]")
    domain = DOMAINS[key]
    for action, patterns in domain.verbs:
        for pattern in patterns:
            assert not japanese.search(pattern), f"{key}.{action}: {pattern!r}"
    for name, words in domain.objects.items():
        for word in words:
            assert not japanese.search(word), f"{key}.{name}: {word!r}"


def test_every_example_in_the_readme_gallery_reaches_an_action():
    """A README that promises a phrase the robot ignores is worse than one that promises none.

    Checked against the command lists, which is the WEAKER reader -- the model is the default
    and handles all of these, but it needs a 5.5 GB download that CI will not have. So this
    asserts that most of the gallery works even without the model, and names the ones that
    genuinely need it rather than pretending they do not exist.

    If that set grows, the gallery is drifting towards phrasings only the model can follow,
    which is worth knowing: command-mode users read the same README.
    """
    # Examples that only the language model resolves. Deliberately listed rather than removed:
    # they are the ones that show what reading a sentence buys you.
    NEEDS_THE_MODEL = {
        ("pet", "have you seen her anywhere?"),
        ("pet", "point the camera upwards a bit"),
        ("mars", "head over to that rock"),
    }
    import pathlib
    import re

    from pyunto_robotics.brain.domains import DomainRulePlanner

    text = pathlib.Path("README.md").read_text(encoding="utf-8")
    start = text.index("## The robots")
    section = text[start:text.index("## Writing in your own words", start)]

    robot, checked, unroutable = None, 0, []
    for line in section.splitlines():
        heading = re.match(r"### `(\w+)`", line)
        if heading:
            robot = heading.group(1)
            assert robot in DOMAINS, f"README names a robot that does not exist: {robot}"
            continue
        quoted = re.match(r'> \*"(.+)"\*', line)
        if quoted and robot:
            checked += 1
            if not DomainRulePlanner(DOMAINS[robot]).plan(quoted.group(1)).steps:
                unroutable.append((robot, quoted.group(1)))

    assert checked >= 15, f"only found {checked} examples; has the gallery moved?"
    surprises = set(unroutable) - NEEDS_THE_MODEL
    assert not surprises, f"examples that reach no action at all: {sorted(surprises)}"
    assert len(unroutable) <= len(NEEDS_THE_MODEL) + 2, (
        f"{len(unroutable)} of {checked} gallery examples need the model; "
        "the README is drifting away from command-mode readers"
    )


def test_the_gallery_pictures_are_committed():
    """The README shows a picture per robot. A broken image is the first thing a reader sees.

    Checks git, not the filesystem. The first version of this test asked whether the files
    existed locally, which they did -- while `*.png` in .gitignore quietly kept every one of
    them out of the repository, so `git add` staged nothing and the published README showed
    six broken images. A documentation asset is only real once it is committed.
    """
    import pathlib
    import re
    import subprocess

    text = pathlib.Path("README.md").read_text(encoding="utf-8")
    shown = re.findall(r"!\[[^\]]*\]\((docs/images/[^)]+)\)", text)
    assert shown, "the README gallery shows no images at all"

    tracked = set(
        subprocess.run(
            ["git", "ls-files", "docs/images"],
            capture_output=True, text=True, check=True,
        ).stdout.split()
    )
    missing = [p for p in shown if p not in tracked]
    assert not missing, f"README shows images that are not committed: {missing}"
