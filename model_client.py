"""
model_client.py — the model-independence layer.

The engine depends on the `ModelClient` interface, never on a concrete model or URL.
Swap models by changing env vars; run tests with no server via MockModel.

  ARS_FABULA_BACKEND  = "server" | "mock"   (default: server)
  ARS_FABULA_BASE_URL = "http://localhost:8111/v1"
  ARS_FABULA_MODEL    = "gemma-12b"  (later: "gemma-26b-a4b", "gemma-31b")
  ARS_FABULA_API_KEY  = "not-needed"
"""
from __future__ import annotations
import os
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable

import requests


# ── Config ──────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    backend: str = "server"
    base_url: str = "http://localhost:8111/v1"
    model: str = "gemma-12b"
    api_key: str = "not-needed"
    temperature: float = 0.8
    max_tokens: int = 1024
    timeout: int = 120

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(
            backend=os.getenv("ARS_FABULA_BACKEND", "server"),
            base_url=os.getenv("ARS_FABULA_BASE_URL", "http://localhost:8111/v1"),
            model=os.getenv("ARS_FABULA_MODEL", "gemma-12b"),
            api_key=os.getenv("ARS_FABULA_API_KEY", "not-needed"),
            temperature=float(os.getenv("ARS_FABULA_TEMPERATURE", "0.8")),
            max_tokens=int(os.getenv("ARS_FABULA_MAX_TOKENS", "1024")),
            timeout=int(os.getenv("ARS_FABULA_TIMEOUT", "120")),
        )


# ── Interface ───────────────────────────────────────────────────────────

class ModelClient(ABC):
    """Minimal surface the engine needs. Any model that speaks it is swappable."""

    @abstractmethod
    def chat(self, messages: list[dict], tools: Optional[list] = None, **kw) -> dict:
        """Return an assistant message dict: {"role","content","tool_calls"?}."""
        ...

    def generate(self, messages: list[dict], **kw) -> str:
        """Convenience: text-only completion."""
        return self.chat(messages, **kw).get("content") or ""

    def health(self) -> bool:
        """Check if the backend is reachable. Default False for abstract clients."""
        return False


# ── Real backend: any OpenAI-compatible llama-server ────────────────────

class LlamaServerClient(ModelClient):
    def __init__(self, config: Optional[ModelConfig] = None):
        self.cfg = config or ModelConfig.from_env()

    def chat(self, messages, tools=None, **kw) -> dict:
        payload = {
            "model": kw.get("model", self.cfg.model),
            "messages": messages,
            "temperature": kw.get("temperature", self.cfg.temperature),
            "max_tokens": kw.get("max_tokens", self.cfg.max_tokens),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = kw.get("tool_choice", "auto")
        resp = requests.post(
            f"{self.cfg.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.cfg.api_key}",
                     "Content-Type": "application/json"},
            json=payload, timeout=self.cfg.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]

    def health(self) -> bool:
        try:
            base = self.cfg.base_url.rsplit("/v1", 1)[0]
            return requests.get(f"{base}/health", timeout=5).status_code == 200
        except requests.RequestException:
            return False


# ── Mock backend: scripted, no server (for unit/CI tests) ───────────────

class MockModel(ModelClient):
    """Returns programmed responses and records what it was asked.

    Program with either:
      MockModel(responses=["text 1", {"role":"assistant","tool_calls":[...]}, ...])
      MockModel(handler=lambda messages, tools: "...")  # dynamic
    """
    def __init__(self, responses: Optional[list] = None,
                 handler: Optional[Callable[[list, Optional[list]], object]] = None):
        self._queue = list(responses or [])
        self._handler = handler
        self.calls: list[dict] = []          # every chat() invocation, for assertions
        self.default = "The scene holds its breath for a moment, waiting."

    def chat(self, messages, tools=None, **kw) -> dict:
        self.calls.append({"messages": messages, "tools": tools, "kw": kw})
        if self._handler:
            out = self._handler(messages, tools)
        elif self._queue:
            out = self._queue.pop(0)
        else:
            out = self.default
        if isinstance(out, str):
            return {"role": "assistant", "content": out}
        return out  # already a message dict (may carry tool_calls)

    # test helpers
    def last_prompt_text(self) -> str:
        """Concatenated content of the most recent call's messages."""
        if not self.calls:
            return ""
        return "\n".join(str(m.get("content", "")) for m in self.calls[-1]["messages"])

    def queue(self, *responses):
        self._queue.extend(responses)


# ── Factory ─────────────────────────────────────────────────────────────

def get_model(config: Optional[ModelConfig] = None) -> ModelClient:
    cfg = config or ModelConfig.from_env()
    if cfg.backend == "mock":
        return MockModel()
    return LlamaServerClient(cfg)
