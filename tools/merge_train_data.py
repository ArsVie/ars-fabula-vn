"""
merge_train_data.py — build data/sft/train.jsonl for the Modal run.

Merges:
  - data/sft/transformed.jsonl  (DeepSeek-transformed real-VN turns —
                                 the primary SFT targets)
  - data/sft/clean.jsonl        (our own validated traces — format
                                 exemplars; small, kept by default)

Shuffles deterministically and writes {"messages": ...} records.

Usage:
    python tools/merge_train_data.py
    python tools/merge_train_data.py --no-traces   # transformed only
"""
from __future__ import annotations
import argparse
import json
import random
import re
from pathlib import Path


def scrub(messages: list[dict]) -> list[dict]:
    """Strip 'Narration:' labels (a transform-model artifact present in
    early records) — engine narration is bare prose."""
    out = []
    for m in messages:
        c = m.get("content", "")
        if m.get("role") in ("user", "assistant"):
            c = re.sub(r"(?m)^\s*Narration:\s*", "", c)
        out.append({**m, "content": c})
    return out


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transformed", default="data/sft/transformed.jsonl")
    ap.add_argument("--traces", default="data/sft/clean.jsonl")
    ap.add_argument("--out", default="data/sft/train.jsonl")
    ap.add_argument("--no-traces", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    transformed = load(Path(args.transformed))
    traces = [] if args.no_traces else load(Path(args.traces))
    records = ([{"messages": scrub(r["messages"]),
                 "source": r.get("meta", {}).get("source", "transform")}
                for r in transformed]
               + [{"messages": r["messages"], "source": "own-trace"}
                  for r in traces])
    random.Random(args.seed).shuffle(records)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(transformed)} transformed + {len(traces)} traces "
          f"-> {len(records)} records in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
