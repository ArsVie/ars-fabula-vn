"""
model_client.py — the model-independence layer.

The engine depends on the `ModelClient` interface, never on a concrete model or URL.
Swap models by changing env vars; run tests with no server via MockModel.

  ARS_FABULA_BACKEND  = "server" | "transformers" | "mock"   (default: server)
  ARS_FABULA_BASE_URL = "http://localhost:8110/v1"
  ARS_FABULA_MODEL    = "gemma-12b"  (later: "gemma-26b-a4b", "gemma-31b")
                        for the transformers backend: a HF repo id,
                        e.g. "google/gemma-3-4b-it"
  ARS_FABULA_API_KEY  = "not-needed"

The "transformers" backend runs the model in-process — built for Hugging
Face Spaces / ZeroGPU, where external servers (llama-server, ComfyUI)
cannot hold a GPU. GPU time is only granted inside @spaces.GPU calls.
"""
from __future__ import annotations
import os
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable, Iterator

import requests


# ── Config ──────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    backend: str = "server"
    base_url: str = "http://localhost:8110/v1"
    model: str = "gemma-4-26b-a4b"
    api_key: str = "not-needed"
    temperature: float = 0.8
    max_tokens: int = 1024
    timeout: int = 120
    # Anti-repetition sampling. llama.cpp's OpenAI endpoint honors both.
    # frequency penalizes tokens by how often they already appeared (n-gram
    # recycling); presence penalizes any reuse at all (topic recycling).
    frequency_penalty: float = 0.4
    presence_penalty: float = 0.3

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(
            backend=os.getenv("ARS_FABULA_BACKEND", "server"),
            base_url=os.getenv("ARS_FABULA_BASE_URL", "http://localhost:8110/v1"),
            model=os.getenv("ARS_FABULA_MODEL", "gemma-4-26b-a4b"),
            api_key=os.getenv("ARS_FABULA_API_KEY", "not-needed"),
            temperature=float(os.getenv("ARS_FABULA_TEMPERATURE", "0.8")),
            max_tokens=int(os.getenv("ARS_FABULA_MAX_TOKENS", "1024")),
            timeout=int(os.getenv("ARS_FABULA_TIMEOUT", "120")),
            frequency_penalty=float(os.getenv("ARS_FABULA_FREQUENCY_PENALTY", "0.4")),
            presence_penalty=float(os.getenv("ARS_FABULA_PRESENCE_PENALTY", "0.3")),
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

    def stream(self, messages: list[dict], tools: Optional[list] = None,
               **kw) -> Iterator[str]:
        """Yield text deltas as they arrive.

        Default (non-streaming) fallback: produce the whole completion as a
        single chunk. Real streaming backends override this. Callers buffer
        the deltas to line-break boundaries, so a one-shot multi-line chunk
        is split into complete lines just the same.
        """
        yield self.chat(messages, tools, **kw).get("content") or ""

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
            "frequency_penalty": kw.get("frequency_penalty",
                                        self.cfg.frequency_penalty),
            "presence_penalty": kw.get("presence_penalty",
                                       self.cfg.presence_penalty),
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

    def stream(self, messages, tools=None, **kw) -> Iterator[str]:
        """Stream text deltas via OpenAI-style SSE (stream: true)."""
        payload = {
            "model": kw.get("model", self.cfg.model),
            "messages": messages,
            "temperature": kw.get("temperature", self.cfg.temperature),
            "max_tokens": kw.get("max_tokens", self.cfg.max_tokens),
            "frequency_penalty": kw.get("frequency_penalty",
                                        self.cfg.frequency_penalty),
            "presence_penalty": kw.get("presence_penalty",
                                       self.cfg.presence_penalty),
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = kw.get("tool_choice", "auto")
        with requests.post(
            f"{self.cfg.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.cfg.api_key}",
                     "Content-Type": "application/json"},
            json=payload, timeout=self.cfg.timeout, stream=True,
        ) as resp:
            resp.raise_for_status()
            # SSE responses rarely declare a charset, and requests then
            # defaults text/* to ISO-8859-1 (RFC 2616) — which mangles
            # multi-byte UTF-8 (— ’ …) into mojibake ("â€”"). The payload
            # is JSON, which is always UTF-8; decode it as such.
            resp.encoding = "utf-8"
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue  # keep-alive / frame separator
                if not raw.startswith("data:"):
                    continue
                data = raw[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                piece = delta.get("content")
                if piece:
                    yield piece

    def health(self) -> bool:
        try:
            base = self.cfg.base_url.rsplit("/v1", 1)[0]
            return requests.get(f"{base}/health", timeout=5).status_code == 200
        except requests.RequestException:
            return False

    def server_model(self) -> str:
        """The model id the SERVER actually has loaded (GET /v1/models).

        llama-server's `model` payload field is cosmetic — it echoes whatever
        the client sends — so cfg.model can lie about the real base (e.g. a
        stale "gemma-12b" default while a 26B-A4B gguf is loaded). This asks the
        server what it loaded so trace provenance is correct. Cached after the
        first call; falls back to cfg.model if the endpoint is unreachable.
        """
        cached = getattr(self, "_server_model", None)
        if cached is not None:
            return cached
        try:
            payload = requests.get(f"{self.cfg.base_url}/models", timeout=5).json()
            # OpenAI shape: {"data": [{"id": ...}]}. The atomic-fork/ollama
            # shape: {"models": [{"name"/"model": ...}]}. Accept either.
            entries = payload.get("data") or payload.get("models") or []
            first = entries[0] if entries else {}
            model_id = first.get("id") or first.get("name") or first.get("model")
            if model_id:
                # Strip any directory prefix so the id is the gguf name/alias.
                # Cache ONLY a successful lookup — a failure (e.g. the server
                # still loading on the first turn) must not pin the fallback
                # label for the rest of the process.
                self._server_model = os.path.basename(str(model_id))
                return self._server_model
        except (requests.RequestException, ValueError, KeyError,
                IndexError, AttributeError):
            pass
        return self.cfg.model


# ── Transformers backend: in-process model (HF Spaces / ZeroGPU) ────────
#
# On ZeroGPU the model must be loaded and placed on "cuda" at startup
# (a CUDA emulation layer makes that legal outside @spaces.GPU); the real
# GPU only exists inside the @spaces.GPU-decorated generate call. The
# decorator is a no-op off-Spaces, so the same code path runs anywhere
# torch + transformers are installed.

# HF repo ids tried in order when ARS_FABULA_MODEL isn't itself a repo id.
# First choice is Gemma 4 12B in bf16 (~24GB — fits both the Space's 50GB
# ephemeral disk and the 48GB ZeroGPU slice; the VN was originally tuned
# against gemma-12b locally). The 26B-A4B MoE the dev box runs is NOT
# viable on ZeroGPU: its transformers-format repos are unquantized bf16
# (~52GB > the 50GB disk) and no transformers-loadable 4-bit repo exists —
# the 26B path is the Docker/T4 Space (llama.cpp GGUF), not this one.
# Gemma repos are license-gated: the Space needs an HF_TOKEN secret from
# an account that accepted the license, else we fall through to an
# ungated small model rather than dropping the whole demo to mock.
_HF_MODEL_CANDIDATES = [
    ("google/gemma-4-12b-it", ""),
    ("google/gemma-3-4b-it", ""),
    ("Qwen/Qwen3-4B-Instruct-2507", ""),
]

_HF_STATE: dict = {"model": None, "tokenizer": None, "id": None}

try:
    import spaces as _spaces
    _gpu_decorator = _spaces.GPU(duration=120)
except ImportError:
    _gpu_decorator = lambda f: f


def _hf_load(model_id: Optional[str] = None):
    """Load the chat model once, module-wide. Returns the loaded repo id
    or None if nothing could be loaded (no torch, gated repo, etc.)."""
    if _HF_STATE["model"] is not None:
        return _HF_STATE["id"]
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"[hf] transformers backend unavailable ({e})")
        return None

    if model_id and "/" in model_id:
        candidates = [(model_id, os.getenv("ARS_FABULA_QUANT", "4bit"))]
    else:
        candidates = list(_HF_MODEL_CANDIDATES)
    for mid, quant in candidates:
        try:
            print(f"[hf] loading {mid}" + (f" ({quant})" if quant else "") + "…")
            tok = AutoTokenizer.from_pretrained(mid)
            kwargs: dict = {"dtype": torch.bfloat16}
            if quant == "4bit" and torch.cuda.is_available():
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
                kwargs["device_map"] = "cuda"
            model = AutoModelForCausalLM.from_pretrained(mid, **kwargs)
            if "device_map" not in kwargs and torch.cuda.is_available():
                model = model.to("cuda")
            model.eval()
            _HF_STATE.update(model=model, tokenizer=tok, id=mid)
            print(f"[hf] {mid} ready on {model.device}")
            return mid
        except Exception as e:
            print(f"[hf] could not load {mid}: {type(e).__name__}: {e}")
    return None


@_gpu_decorator
def _hf_generate(messages: list[dict], max_new_tokens: int,
                 temperature: float) -> str:
    import torch
    tok, model = _HF_STATE["tokenizer"], _HF_STATE["model"]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-3),
            top_p=0.95,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[-1]:],
                      skip_special_tokens=True)


def _hf_generate_worker(model, gen_kwargs):
    """Run model.generate in a background thread (feeds a TextIteratorStreamer).
    Kept module-level so the @spaces.GPU generator below can spawn it."""
    import torch
    with torch.inference_mode():
        model.generate(**gen_kwargs)


@_gpu_decorator
def _hf_stream_generate(messages: list[dict], max_new_tokens: int,
                        temperature: float):
    """Yield text deltas token-by-token. The whole iteration runs inside ONE
    @spaces.GPU allocation (ZeroGPU holds the GPU for a generator's lifetime),
    so generate() in the worker thread and our decoding overlap — the engine's
    line-buffered stream parser turns the deltas into live beats. The previous
    behaviour (base ModelClient.stream → one giant chunk at the end) made every
    ZeroGPU turn feel like a long stall; this restores local-style streaming."""
    import torch  # noqa: F401  (ensures torch is importable before generate)
    from threading import Thread
    from transformers import TextIteratorStreamer
    tok, model = _HF_STATE["tokenizer"], _HF_STATE["model"]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True,
                                    skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-3),
        top_p=0.95,
        streamer=streamer,
    )
    thread = Thread(target=_hf_generate_worker, args=(model, gen_kwargs))
    thread.start()
    for delta in streamer:
        if delta:
            yield delta
    thread.join()


class TransformersClient(ModelClient):
    """In-process HF transformers chat model (the ZeroGPU path)."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.cfg = config or ModelConfig.from_env()
        self._loaded_id = _hf_load(self.cfg.model)

    def chat(self, messages, tools=None, **kw) -> dict:
        if not self.health():
            raise RuntimeError("transformers backend has no loaded model")
        text = _hf_generate(
            messages,
            max_new_tokens=kw.get("max_tokens", self.cfg.max_tokens),
            temperature=kw.get("temperature", self.cfg.temperature),
        )
        return {"role": "assistant", "content": text}

    def stream(self, messages, tools=None, **kw) -> Iterator[str]:
        """Token-by-token streaming (overrides the base one-chunk fallback).
        Used by the engine's run_turn_stream for every interactive turn."""
        if not self.health():
            raise RuntimeError("transformers backend has no loaded model")
        yield from _hf_stream_generate(
            messages,
            max_new_tokens=kw.get("max_tokens", self.cfg.max_tokens),
            temperature=kw.get("temperature", self.cfg.temperature),
        )

    def health(self) -> bool:
        return _HF_STATE["model"] is not None


def preload_transformers() -> bool:
    """Warm the in-process model at app startup (ZeroGPU wants the CUDA
    placement done at module/startup time, not lazily mid-request)."""
    cfg = ModelConfig.from_env()
    if cfg.backend != "transformers":
        return False
    return _hf_load(cfg.model) is not None


# ── Mock backend: scripted, no server (for unit/CI tests) ───────────────

class MockModel(ModelClient):
    """Returns programmed responses and records what it was asked.

    Program with either:
      MockModel(responses=["text 1", {"role":"assistant","tool_calls":[...]}, ...])
      MockModel(handler=lambda messages, tools: "...")  # dynamic
    """
    # Scripted output is not real model data — the engine skips trace logging
    # for mock models so tests and mock fallbacks don't pollute the dataset.
    is_mock = True

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
    if cfg.backend == "transformers":
        return TransformersClient(cfg)
    return LlamaServerClient(cfg)


# ── VRAM management — swap the LLM out so the image model can load ──────
#
# The LLM and the diffusion stack don't fit in VRAM together (e.g. Gemma
# 12B IQ4_XS ~6GB on an 8GB card). llama.cpp's llama-server has NO unload
# API, so we STOP the server process before a ComfyUI bake and RELAUNCH it
# afterward, polling /health before resuming scene turns.
#
# Default behavior auto-detects a WSL-native atomic-fork llama-server (Gemma
# 4 26B-A4B MoE + MTP) or falls back to a WSL→Windows llama-server.exe.
# Everything is env-overridable:
#   ARS_FABULA_LLM_MANAGE      = auto (default) | wsl | win | none | ollama
#   ARS_FABULA_LLM_STOP_CMD    = "<shell cmd>"   # custom stop
#   ARS_FABULA_LLM_START_CMD   = "<shell cmd>"   # custom start (detached)
#   ARS_FABULA_LLM_UNLOAD_URL  = "http://.../unload"
#   ARS_FABULA_LLAMA_EXE_WIN   = C:\\...\\llama-server.exe    # Windows mode
#   ARS_FABULA_LLAMA_MODEL_WIN = C:\\...\\model.gguf
#   ARS_FABULA_LLAMA_ARGS      = "<llama-server flags>"       # shared
#   ARS_FABULA_LLAMA_BIN       = /path/to/llama-server  (WSL) # WSL mode
#   ARS_FABULA_LLAMA_MODEL     = /path/to/model.gguf   (WSL)
#   ARS_FABULA_LLAMA_DRAFTER   = /path/to/mtp-drafter.gguf    # MTP
#   ARS_FABULA_LLAMA_LD_PATH   = /path/to/lib/dir             # LD_LIBRARY_PATH

_DEFAULT_LLAMA_EXE_WIN = r"C:\Users\vruizes\llama-cpp-setup\llama-server.exe"
_DEFAULT_LLAMA_MODEL_WIN = r"C:\Users\vruizes\Downloads\gemma-4-12b-it-IQ4_XS.gguf"
# 'none' reasoning profile from the launcher .bat — fastest, cleanest tool
# output for the VN (thinking tokens just add latency to bracket emission).
_DEFAULT_LLAMA_ARGS = ("-ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 16384 "
                       "--reasoning off -np 1 --cache-ram 0 "
                       "--host 0.0.0.0 --port 8110")

# ── WSL-native defaults (atomic-llama-cpp-turboquant, Gemma 4 26B-A4B) ──
_DEFAULT_LLAMA_BIN_WSL = "/mnt/c/Users/vruizes/atomic-llama-cpp-turboquant/" \
                         "build/bin/llama-server"
_DEFAULT_LLAMA_MODEL_WSL = "/mnt/c/Users/vruizes/Downloads/" \
                           "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
_DEFAULT_LLAMA_DRAFTER_WSL = "/mnt/c/Users/vruizes/Downloads/" \
                              "gemma-4-26B-A4B-it-assistant.Q4_K_M.gguf"
_DEFAULT_LLAMA_LD_PATH_WSL = "/mnt/c/Users/vruizes/" \
                              "atomic-llama-cpp-turboquant/build/bin"
# 32K context, CPU MoE offload (GPU can't fit 26B MoE experts), MTP drafter,
# no reasoning tokens (they add latency to bracket emission).
_DEFAULT_LLAMA_ARGS_WSL = ("--spec-type mtp --draft-block-size 3 "
                           "--draft-max 8 --draft-min 0 "
                           "-ngl 99 -ngld 99 -fa on -ctk q8_0 -ctv q8_0 "
                           "-c 32000 --cpu-moe --reasoning off "
                           "--host 0.0.0.0 --port 8110")


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
    # auto: a custom stop hook wins; then try WSL-native atomic fork,
    # then Windows .exe via interop, then ollama.
    if os.getenv("ARS_FABULA_LLM_STOP_CMD"):
        return "cmd"
    # Check for WSL-native atomic-fork binary first
    bin_path = os.getenv("ARS_FABULA_LLAMA_BIN", _DEFAULT_LLAMA_BIN_WSL)
    if _is_wsl() and os.path.exists(bin_path):
        return "wsl"
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
    # mock has nothing to stop; transformers is in-process — on ZeroGPU the
    # GPU is granted per-call and released automatically, so the whole
    # stop/relaunch swap dance does not apply.
    if cfg.backend in ("mock", "transformers"):
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

    if mode == "wsl":
        # Kill the WSL-native atomic-fork llama-server (Linux process)
        try:
            subprocess.run(["pkill", "-f", "build/bin/llama-server"],
                           timeout=15, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("[vram] WSL llama-server stopped (pkill build/bin/llama-server)")
            time.sleep(2)
            return True
        except Exception as e:
            print(f"[vram] pkill failed: {e}")
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


def ensure_llm(config: Optional[ModelConfig] = None, timeout: int = 300) -> bool:
    """Make sure the LLM server is up and answering /health, relaunching it
    if we manage it. Returns True when healthy, False on give-up."""
    cfg = config or ModelConfig.from_env()
    if cfg.backend == "mock":
        return False
    if cfg.backend == "transformers":
        # In-process model: there is no server to relaunch or poll.
        return _HF_STATE["model"] is not None or preload_transformers()
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

    elif mode == "wsl":
        # Launch the WSL-native atomic-fork llama-server (Gemma 4 26B-A4B
        # MoE + MTP drafter). The binary is a Linux ELF with CUDA deps, so
        # LD_LIBRARY_PATH must point at its build/bin for libggml*.so etc.
        bin_path = os.getenv("ARS_FABULA_LLAMA_BIN", _DEFAULT_LLAMA_BIN_WSL)
        model_path = os.getenv("ARS_FABULA_LLAMA_MODEL",
                               _DEFAULT_LLAMA_MODEL_WSL)
        drafter_path = os.getenv("ARS_FABULA_LLAMA_DRAFTER",
                                 _DEFAULT_LLAMA_DRAFTER_WSL)
        ld_path = os.getenv("ARS_FABULA_LLAMA_LD_PATH",
                            _DEFAULT_LLAMA_LD_PATH_WSL)
        args_str = os.getenv("ARS_FABULA_LLAMA_ARGS",
                             _DEFAULT_LLAMA_ARGS_WSL)
        import shlex
        cmd = [bin_path, "-m", model_path,
               "--mtp-head", drafter_path] + shlex.split(args_str)
        launch_env = os.environ.copy()
        launch_env["LD_LIBRARY_PATH"] = (
            f"{ld_path}:{launch_env.get('LD_LIBRARY_PATH', '')}"
        )
        try:
            subprocess.Popen(cmd, env=launch_env,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL,
                             start_new_session=True)
            started = True
            print(f"[vram] atomic-fork llama-server starting (WSL-native): "
                  f"{bin_path}")
        except Exception as e:
            print(f"[vram] WSL launch failed: {e}")

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
