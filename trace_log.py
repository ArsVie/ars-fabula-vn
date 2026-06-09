"""
trace_log.py — capture model prompt→response traces for later fine-tuning.

Every real narration turn is appended as one JSON line to a dated file under
the trace directory. Each record is already in the standard SFT chat shape
(a `messages` list ending in the assistant's raw response), with extra
`context` fields alongside for curating/filtering the dataset later:

    {"ts": "...", "model": "...", "context": {...},
     "messages": [{"role":"system",...},{"role":"user",...},
                  {"role":"assistant","content": "<raw model output>"}]}

To build a training set, read the lines and keep `record["messages"]`
(drop the trailing assistant turn at inference time).

Controls:
  ARS_FABULA_TRACE      = "1"|"on" (default) to log, "0"|"off"|"false" to disable
  ARS_FABULA_TRACE_DIR  = output directory (default: "logs/traces")

Mock/scripted responses are NOT logged — the engine skips logging when the
model reports `is_mock` (see run_turn), so tests and mock fallbacks don't
pollute the dataset.
"""
from __future__ import annotations
import os
import json
from datetime import datetime, date
from pathlib import Path
from typing import Optional


_DISABLED = {"0", "off", "false", "no", ""}


def traces_enabled() -> bool:
    return os.getenv("ARS_FABULA_TRACE", "1").strip().lower() not in _DISABLED


def _trace_dir() -> Path:
    return Path(os.getenv("ARS_FABULA_TRACE_DIR", "logs/traces"))


def log_turn(messages: list[dict], response: str, *,
             model: str = "", context: Optional[dict] = None) -> None:
    """Append one prompt→response trace as a JSON line.

    Best-effort: any IO/serialization error is swallowed so a logging
    problem can never break a turn. `messages` is the prompt sent to the
    model (system + user); the assistant's raw `response` is appended so the
    stored `messages` list is directly usable for supervised fine-tuning.
    """
    if not traces_enabled():
        return
    try:
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "context": context or {},
            "messages": list(messages) + [{"role": "assistant", "content": response}],
        }
        out_dir = _trace_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"turns-{date.today().isoformat()}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        # Never let trace logging interfere with the turn loop.
        pass
