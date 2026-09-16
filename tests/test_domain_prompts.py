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
