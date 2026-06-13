"""
build_sft_dataset.py — turn logs/traces/*.jsonl into a validated SFT dataset.

Each trace record is already SFT-shaped (see trace_log.py):
    {"ts", "model", "context": {...},
     "messages": [system, user, {"role":"assistant","content": raw_output}]}

This script filters and validates them:
  - drops mock records and exact-duplicate prompts
  - drops records from stale/banned model labels (--drop-model, repeatable;
    default drops "gemma-12b" — pre-provenance-fix records whose true base
    model is unknown)
  - gates every assistant completion through vn_validate.validate_turn
  - writes passing records to <out>/clean.jsonl (messages only + provenance),
    rejects to <out>/rejects.jsonl with reasons, and prints stats

NOTE (data strategy): records exported here are our OWN model's turns. They
are format exemplars / DPO candidates — primary SFT targets come from the
DeepSeek-transformed real-VN-transcript pipeline (see next_session.md).

Usage:
    python tools/build_sft_dataset.py
    python tools/build_sft_dataset.py --in-dir logs/traces --out-dir data/sft \
        --drop-model gemma-12b --min-quality 0
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vn_validate import validate_turn  # noqa: E402


def _prompt_hash(messages: list[dict]) -> str:
    """Stable hash of the prompt (all messages except the assistant turn)."""
    prompt = [m for m in messages if m.get("role") != "assistant"]
    blob = json.dumps(prompt, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _iter_records(in_dir: Path):
    for path in sorted(in_dir.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield path.name, line_no, json.loads(line)
                except json.JSONDecodeError:
                    yield path.name, line_no, None


def build(in_dir: Path, out_dir: Path, drop_models: list[str],
          min_quality: float = 0.0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_path = out_dir / "clean.jsonl"
    reject_path = out_dir / "rejects.jsonl"

    seen_prompts: set[str] = set()
    reject_reasons: Counter = Counter()
    overlaps: list[float] = []
    n_total = n_clean = n_reject = 0

    with open(clean_path, "w", encoding="utf-8") as clean_f, \
         open(reject_path, "w", encoding="utf-8") as reject_f:
        for fname, line_no, rec in _iter_records(in_dir):
            n_total += 1

            def reject(reasons: list[str], record=None):
                nonlocal n_reject
                n_reject += 1
                for r in reasons:
                    reject_reasons[r.split(":")[0]] += 1
                reject_f.write(json.dumps(
                    {"source": f"{fname}:{line_no}", "reasons": reasons,
                     "record": record}, ensure_ascii=False) + "\n")

            if rec is None:
                reject(["unparseable JSON line"])
                continue
            messages = rec.get("messages") or []
            model = str(rec.get("model", ""))
            if len(messages) < 2 or messages[-1].get("role") != "assistant":
                reject(["record not SFT-shaped (no trailing assistant turn)"], rec)
                continue
            if "mock" in model.lower():
                reject(["mock model record"], rec)
                continue
            if any(d and d in model for d in drop_models):
                reject([f"dropped model label: {model!r}"], rec)
                continue

            phash = _prompt_hash(messages)
            if phash in seen_prompts:
                reject(["duplicate prompt"], rec)
                continue
            seen_prompts.add(phash)

            system_text = next((m.get("content", "") for m in messages
                                if m.get("role") == "system"), "")
            user_text = next((m.get("content", "") for m in messages
                              if m.get("role") == "user"), "")
            completion = messages[-1].get("content", "") or ""

            result = validate_turn(system_text, completion, history=user_text)
            overlap = result.stats.get("ngram_overlap")
            if overlap is not None:
                overlaps.append(overlap)
            if not result.ok:
                reject(result.reasons, rec)
                continue
            if overlap is not None and min_quality > 0 and overlap > 1 - min_quality:
                reject([f"repetition: overlap {overlap:.2f} above threshold"], rec)
                continue

            n_clean += 1
            clean_f.write(json.dumps(
                {"messages": messages, "model": model, "ts": rec.get("ts", ""),
                 "stats": result.stats}, ensure_ascii=False) + "\n")

    stats = {
        "total": n_total, "clean": n_clean, "rejected": n_reject,
        "top_reject_reasons": reject_reasons.most_common(10),
        "overlap_mean": (sum(overlaps) / len(overlaps)) if overlaps else None,
        "overlap_max": max(overlaps) if overlaps else None,
    }
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--in-dir", default="logs/traces",
                    help="directory of trace .jsonl files")
    ap.add_argument("--out-dir", default="data/sft",
                    help="output directory for clean.jsonl / rejects.jsonl")
    ap.add_argument("--drop-model", action="append", default=None,
                    metavar="SUBSTR",
                    help="drop records whose model label contains SUBSTR "
                         "(repeatable; default: gemma-12b)")
    ap.add_argument("--min-quality", type=float, default=0.0,
                    help="0..1 — reject records whose prompt-vs-completion "
                         "n-gram overlap exceeds (1 - MIN_QUALITY). 0 = off")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.is_dir():
        print(f"error: --in-dir {in_dir} is not a directory", file=sys.stderr)
        return 2
    drop_models = args.drop_model if args.drop_model is not None else ["gemma-12b"]

    stats = build(in_dir, Path(args.out_dir), drop_models, args.min_quality)

    print(f"records:  {stats['total']}")
    print(f"clean:    {stats['clean']}  -> {args.out_dir}/clean.jsonl")
    print(f"rejected: {stats['rejected']}  -> {args.out_dir}/rejects.jsonl")
    if stats["top_reject_reasons"]:
        print("top reject reasons:")
        for reason, count in stats["top_reject_reasons"]:
            print(f"  {count:5d}  {reason}")
    if stats["overlap_mean"] is not None:
        print(f"prompt-overlap (repetition signal): "
              f"mean {stats['overlap_mean']:.3f}, max {stats['overlap_max']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
