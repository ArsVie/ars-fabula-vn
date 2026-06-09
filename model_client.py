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
import time
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


# ── VRAM management — swap the LLM out so the image model can load ──────
#
# The LLM and the diffusion stack don't fit in VRAM together (e.g. Gemma
# 12B IQ4_XS ~6GB on an 8GB card). llama.cpp's llama-server has NO unload
# API, so we STOP the server process before a ComfyUI bake and RELAUNCH it
# afterward, polling /health before resuming scene turns.
#
# Default behavior auto-detects a WSL→Windows llama-server.exe and manages
# it with taskkill.exe / cmd.exe start. Everything is env-overridable:
#   ARS_FABULA_LLM_MANAGE      = auto (default) | win | none | ollama
#   ARS_FABULA_LLM_STOP_CMD    = "<shell cmd>"   # custom stop
#   ARS_FABULA_LLM_START_CMD   = "<shell cmd>"   # custom start (detached)
#   ARS_FABULA_LLM_UNLOAD_URL  = "http://.../unload"
#   ARS_FABULA_LLAMA_EXE_WIN   = C:\...\llama-server.exe
#   ARS_FABULA_LLAMA_MODEL_WIN = C:\...\model.gguf
#   ARS_FABULA_LLAMA_ARGS      = "<llama-server flags>"

_DEFAULT_LLAMA_EXE_WIN = r"C:\Users\vruizes\llama-cpp-setup\llama-server.exe"
_DEFAULT_LLAMA_MODEL_WIN = r"C:\Users\vruizes\Downloads\gemma-4-12b-it-IQ4_XS.gguf"
# 'none' reasoning profile from the launcher .bat — fastest, cleanest tool
# output for the VN (thinking tokens just add latency to bracket emission).
_DEFAULT_LLAMA_ARGS = ("-ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 16384 "
                       "--reasoning off -np 1 --cache-ram 0 "
                       "--host 0.0.0.0 --port 8111")


def _is_wsl() -> bool:
    if os.name != "posix":
        return False
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _win_to_wsl(win_path: str) -> str:
    """C:\\Users\\x → /mnt/c/Users/x (for existence checks from WSL)."""
    p = win_path.replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        return f"/mnt/{p[0].lower()}{p[2:]}"
    return p


def _llm_manage_mode(cfg: ModelConfig) -> str:
    mode = os.getenv("ARS_FABULA_LLM_MANAGE", "auto").lower()
    if mode != "auto":
        return mode
    # auto: a custom stop hook wins; else manage a WSL→Windows llama-server
    if os.getenv("ARS_FABULA_LLM_STOP_CMD"):
        return "cmd"
    exe = os.getenv("ARS_FABULA_LLAMA_EXE_WIN", _DEFAULT_LLAMA_EXE_WIN)
    if _is_wsl() and os.path.exists(_win_to_wsl(exe)):
        return "win"
    if "11434" in cfg.base_url:
        return "ollama"
    return "none"


def _win_health(cfg: ModelConfig) -> bool:
    try:
        base = cfg.base_url.rsplit("/v1", 1)[0]
        return requests.get(f"{base}/health", timeout=3).status_code == 200
    except requests.RequestException:
        return False


def release_vram(config: Optional[ModelConfig] = None) -> bool:
    """Stop/evict the LLM so ComfyUI can load. Returns True if it acted."""
    cfg = config or ModelConfig.from_env()
    if cfg.backend == "mock":
        return False
    mode = _llm_manage_mode(cfg)
    if mode == "none":
        return False
    import subprocess

    if mode == "cmd" or os.getenv("ARS_FABULA_LLM_STOP_CMD"):
        cmd = os.getenv("ARS_FABULA_LLM_STOP_CMD")
        if cmd:
            try:
                subprocess.run(cmd, shell=True, timeout=30, check=False)
                print(f"[vram] LLM stopped via STOP_CMD")
                return True
            except Exception as e:
                print(f"[vram] STOP_CMD failed: {e}")

    if mode == "win":
        # Kill the Windows llama-server.exe from WSL via interop
        for killer in ("taskkill.exe", "/mnt/c/Windows/System32/taskkill.exe"):
            try:
                subprocess.run([killer, "/IM", "llama-server.exe", "/F"],
                               timeout=20, check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("[vram] llama-server.exe stopped (taskkill)")
                # give the driver a moment to release VRAM
                time.sleep(2)
                return True
            except Exception:
                continue
        print("[vram] taskkill unavailable — could not stop llama-server")
        return False

    url = os.getenv("ARS_FABULA_LLM_UNLOAD_URL")
    if url:
        try:
            requests.post(url, json={}, timeout=15)
            print(f"[vram] LLM unload URL hit: {url}")
            return True
        except Exception as e:
            print(f"[vram] unload URL failed: {e}")

    if mode == "ollama":
        root = cfg.base_url.rsplit("/v1", 1)[0].rstrip("/")
        try:
            if requests.post(f"{root}/api/generate",
                             json={"model": cfg.model, "keep_alive": 0,
                                   "prompt": "", "stream": False}, timeout=15).ok:
                print(f"[vram] LLM unloaded via ollama keep_alive=0")
                return True
        except Exception:
            pass
    return False


def ensure_llm(config: Optional[ModelConfig] = None, timeout: int = 120) -> bool:
    """Make sure the LLM server is up and answering /health, relaunching it
    if we manage it. Returns True when healthy, False on give-up."""
    cfg = config or ModelConfig.from_env()
    if cfg.backend == "mock":
        return False
    if _win_health(cfg):
        return True

    mode = _llm_manage_mode(cfg)
    import subprocess

    started = False
    start_cmd = os.getenv("ARS_FABULA_LLM_START_CMD")
    if start_cmd:
        try:
            subprocess.Popen(start_cmd, shell=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            started = True
            print("[vram] LLM starting via START_CMD…")
        except Exception as e:
            print(f"[vram] START_CMD failed: {e}")
    elif mode == "win":
        exe = os.getenv("ARS_FABULA_LLAMA_EXE_WIN", _DEFAULT_LLAMA_EXE_WIN)
        model = os.getenv("ARS_FABULA_LLAMA_MODEL_WIN", _DEFAULT_LLAMA_MODEL_WIN)
        args = os.getenv("ARS_FABULA_LLAMA_ARGS", _DEFAULT_LLAMA_ARGS)
        # Execute the Windows .exe DIRECTLY via its WSL path — no cmd.exe,
        # no `start` (whose title arg gets mangled across the WSL boundary).
        # The exe is a Windows program, so the model path stays a Windows
        # path; Popen returns immediately and the process keeps running.
        import shlex
        exe_wsl = _win_to_wsl(exe)
        cmd = [exe_wsl, "-m", model] + shlex.split(args)
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL,
                             start_new_session=True)
            started = True
            print(f"[vram] llama-server.exe relaunching (detached): {exe_wsl}")
        except Exception as e:
            print(f"[vram] direct launch failed ({e}); trying the headless .bat")
            # Fallback: run the headless launcher .bat via cmd.exe /c
            bat = os.getenv("ARS_FABULA_LLAMA_BAT")
            if bat:
                for shell in ("cmd.exe", "/mnt/c/Windows/System32/cmd.exe"):
                    try:
                        subprocess.Popen([shell, "/c", bat],
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL,
                                         start_new_session=True)
                        started = True
                        print("[vram] llama-server relaunching via .bat…")
                        break
                    except Exception:
                        continue

    if not started:
        # Nothing we could launch — wait only a short grace period in case
        # the user is bringing it up by hand, then give up to mock.
        print("[vram] LLM not running and no launcher configured — "
              "start llama-server (or set ARS_FABULA_LLM_START_CMD). "
              "Using mock until it's up.")
        timeout = 6

    # Poll /health until the model finishes loading
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _win_health(cfg):
            print("[vram] LLM is healthy.")
            return True
        time.sleep(2)
    return _win_health(cfg)
