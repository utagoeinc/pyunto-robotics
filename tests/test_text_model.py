"""Reading instructions off Apple silicon: a model server or the Claude API.

The planner only needs `generate(prompt, max_tokens) -> str`. These check the two HTTP
implementations send what their APIs expect and read back the text, and that a model which
cannot start leaves the robot in command mode instead of refusing to open.
"""
from __future__ import annotations

import pytest

from pyunto_robotics import cli, registry
from pyunto_robotics.brain import text_model
from pyunto_robotics.brain.domains import DomainLLMPlanner


class Resp:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_openai_compatible_sends_a_chat_request_and_reads_the_reply(monkeypatch):
    sent = {}

    def post(url, headers, json, timeout):
        sent.update(url=url, body=json)
        return Resp({"choices": [{"message": {"content": '[{"action": "clean"}]'}}]})

    monkeypatch.setattr(text_model.requests, "post", post)
    m = text_model.OpenAICompatibleModel("http://localhost:11434/v1/", "gemma4:e2b")
    assert m.generate("hi", 50) == '[{"action": "clean"}]'
    assert sent["url"] == "http://localhost:11434/v1/chat/completions"
    assert sent["body"]["model"] == "gemma4:e2b"
    assert sent["body"]["temperature"] == 0


def test_the_only_model_on_a_server_is_used_but_several_must_be_chosen(monkeypatch):
    monkeypatch.setattr(text_model.requests, "get",
                        lambda url, headers, timeout: Resp({"data": [{"id": "gemma4:e2b"}]}))
    assert text_model.OpenAICompatibleModel("http://x/v1").model == "gemma4:e2b"

    monkeypatch.setattr(text_model.requests, "get",
                        lambda url, headers, timeout: Resp({"data": [{"id": "a"}, {"id": "b"}]}))
    with pytest.raises(RuntimeError, match="--llm-model"):
        text_model.OpenAICompatibleModel("http://x/v1")


def test_claude_model_needs_a_key_and_reads_text_blocks(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        text_model.ClaudeModel()

    monkeypatch.setattr(text_model.requests, "post", lambda url, headers, json, timeout: Resp(
        {"content": [{"type": "text", "text": '[{"action": "find"}]'}]}))
    assert text_model.ClaudeModel(api_key="k").generate("hi", 10) == '[{"action": "find"}]'


def test_the_planner_uses_whatever_model_it_is_given():
    class Fixed:
        description = "fixed"

        def generate(self, prompt, max_tokens):
            return '```json\n[{"action": "clean"}]\n```'

    plan = DomainLLMPlanner(registry.get("hotel").domain, model=Fixed()).plan("sort the rooms")
    assert [s.action for s in plan.steps] == ["clean"]


def test_an_unreachable_server_leaves_the_robot_in_command_mode(monkeypatch, capsys):
    def down(*a, **k):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(text_model.requests, "get", down)
    assert cli._language_model(False, "http", "http://localhost:11434/v1", "gemma4:e2b") is None
    out = capsys.readouterr().out
    assert "connection refused" in out and "matching commands" in out


def test_off_apple_silicon_the_way_to_read_sentences_is_named(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "platform", "linux")
    assert cli._understanding(False) is False
    out = capsys.readouterr().out
    assert "--llm http" in out and "--llm claude-api" in out
