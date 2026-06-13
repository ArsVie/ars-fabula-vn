"""
eval_generations.py — score holdout generations with vn_validate.

Input: holdout_generations.jsonl from a Modal run
    {"messages": [system, user], "reference": "...", "generated": "..."}
(pull with: modal volume get ars-fabula-train /runs/<run>/holdout_generations.jsonl .)

Reports, per the roadmap's eval section: format validity rate, mean beat
length, choice presence, and reasons for failures — comparable across
base vs fine-tuned runs.

Usage:
    python tools/eval_generations.py runs/<run>/holdout_generations.jsonl
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vn_validate import validate_turn  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    n = ok = 0
    beat_lens, reasons_all = [], []
    has_choices = 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        system = next(m["content"] for m in rec["messages"]
                      if m["role"] == "system")
        user = next(m["content"] for m in rec["messages"]
                    if m["role"] == "user")
        result = validate_turn(system, rec["generated"], history=user)
        n += 1
        if result.ok:
            ok += 1
        else:
            reasons_all.extend(result.reasons)
        if result.stats.get("has_choices"):
            has_choices += 1
        mb = result.stats.get("max_beat_sentences")
        if mb is not None:
            beat_lens.append(mb)
    print(f"samples:            {n}")
    print(f"format validity:    {ok}/{n} ({ok / max(n, 1):.0%})")
    print(f"has offer_choices:  {has_choices}/{n}")
    if beat_lens:
        print(f"max beat sentences: mean {sum(beat_lens) / len(beat_lens):.2f}, "
              f"worst {max(beat_lens)}")
    if reasons_all:
        print("failure reasons:")
        for r in reasons_all[:15]:
            print(f"  - {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
