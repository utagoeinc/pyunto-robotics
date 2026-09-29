"""Where the robot's language model runs.

The planner needs one thing from a model: a prompt in, text out. Everything about *which*
model and *where* lives here, so the planner does not care whether it is talking to Gemma on
this Mac, to a server on the local network, or to the Claude API.

  MLXModel               Gemma 4 on Apple silicon, in this process (the original, and the
                         default there). Nothing leaves the machine.
  OpenAICompatibleModel  Any server that speaks the OpenAI chat API: Ollama, llama.cpp's
                         llama-server, LM Studio, vLLM. This is how Windows and Linux read
                         sentences -- all four run there, on CPU or GPU. Local by default.
  ClaudeModel            The Claude API. No local model at all, but the instruction is sent
                         to Anthropic to be understood, and the robot says so at start-up.

Why not bundle a cross-platform runtime (llama-cpp-python) instead: it compiles on install,
and a failed compile on Windows is the first thing a new user would see. Every serious local
runtime already offers this HTTP API, and people who run models locally usually have one.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

import requests

log = logging.getLogger(__name__)

DEFAULT_MLX_MODEL = "lmstudio-community/gemma-4-E2B-it-MLX-8bit"
DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5"


class TextModel(Protocol):
    """A prompt in, the model's text out. Raise on failure; the planner falls back to rules."""

    #: One line for the start-up banner: what runs, and where the instruction goes.
    description: str

    def generate(self, prompt: str, max_tokens: int) -> str: ...


class MLXModel:
    """Gemma 4 through mlx_vlm (it is multimodal, so mlx_lm cannot load it)."""

    def __init__(self, model_id: str = DEFAULT_MLX_MODEL):
        self.model_id = model_id
        self.description = f"{model_id} on this Mac (nothing is sent anywhere)"
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
                "mlx-vlm is not installed. Install the extra: pip install 'pyunto-robotics[llm]'"
            ) from e
        log.info("loading planner model %s (first run downloads weights)", self.model_id)
        self._model, self._tokenizer = load(self.model_id)
        self._config = load_config(self.model_id)

    def generate(self, prompt: str, max_tokens: int) -> str:
        self._load()
        from mlx_vlm import generate  # noqa: PLC0415
        from mlx_vlm.prompt_utils import apply_chat_template  # noqa: PLC0415

        templated = apply_chat_template(self._tokenizer, self._config, prompt, num_images=0)
        reply = generate(self._model, self._tokenizer, templated, [],
                         max_tokens=max_tokens, verbose=False)
        return reply if isinstance(reply, str) else getattr(reply, "text", str(reply))


class OpenAICompatibleModel:
    """POST {base_url}/chat/completions. Ollama: http://localhost:11434/v1."""

    def __init__(self, base_url: str, model: str | None = None, api_key: str | None = None,
                 timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.model = model or self._first_model()
        self.description = f"{self.model} at {self.base_url}"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _first_model(self) -> str:
        """Use the server's model when there is no doubt which one is meant.

        Guessing between several would quietly load a 30 GB model on somebody's laptop, so
        more than one is an error that names them.
        """
        r = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=10)
        r.raise_for_status()
        ids = [m.get("id") for m in (r.json().get("data") or []) if m.get("id")]
        if len(ids) == 1:
            return ids[0]
        if not ids:
            raise RuntimeError(f"{self.base_url} lists no models. Load one, or pass --llm-model.")
        raise RuntimeError(
            f"{self.base_url} has {len(ids)} models; choose one with --llm-model "
            f"(for example: {', '.join(ids[:4])})"
        )

    def check(self) -> None:
        """Fail at start-up, in one line, rather than on the first instruction."""
        r = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=10)
        r.raise_for_status()

    def generate(self, prompt: str, max_tokens: int) -> str:
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                # A plan is not creative writing: the same sentence should give the same steps.
                "temperature": 0,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        choice = (r.json().get("choices") or [{}])[0]
        return ((choice.get("message") or {}).get("content") or "").strip()


class ClaudeModel:
    """The Claude Messages API, called directly (no SDK dependency for one request shape)."""

    def __init__(self, model: str = DEFAULT_CLAUDE_MODEL, api_key: str | None = None,
                 timeout: float = 60.0):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.model = model
        self.timeout = timeout
        self.description = f"{model} via the Claude API (instructions are sent to Anthropic)"

    def generate(self, prompt: str, max_tokens: int) -> str:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return "".join(
            block.get("text", "") for block in r.json().get("content", [])
            if block.get("type") == "text"
        ).strip()
