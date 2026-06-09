"""
sampler_sweep.py — Same A/B experiment as sampler_ab.py, but over EVERY
sampler ComfyUI has installed EXCEPT the ones already tested
(er_sde, flow_euler, euler). One character, fixed seeds, only the sampler
varies. Each sampler's outputs get a zero-padded index PREFIX (s01_, s02_…)
so they're easy to identify and sort in the folder and the report.

    .venv/bin/python3 tools/sampler_sweep.py                     # yuki, 3 seeds
    .venv/bin/python3 tools/sampler_sweep.py --char sakura --tries 3
    .venv/bin/python3 tools/sampler_sweep.py --exclude er_sde flow_euler euler dpmpp_2m

VRAM: stops the LLM first (frees the GPU for ComfyUI). Restart it after.

Output: static/sprites/_sampler_test/<char>_sweep/sNN_<sampler>_seed<seed>.png
        static/sprites/_sampler_test/<char>_sweep/REPORT.md
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cast_pipeline import (
    PRESET_CHARACTERS, SPRITE_NEGATIVE, SCHEDULER, GENERATED_DIR,
    _comfy_health, _comfy_submit, _comfy_wait, _comfy_fetch_image, COMFY_HOST,
)
# Reuse the exact fixed-setup workflow + sampler-list query from the A/B tool
from sampler_ab import _valid_samplers, _base_workflow, STEPS, CFG, WIDTH, HEIGHT, TEST_DIR

# Samplers covered by sampler_ab.py — excluded here by default.
DEFAULT_EXCLUDE = ["er_sde", "flow_euler", "euler"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep every other sampler on one character's base gen")
    ap.add_argument("--char", default="yuki", help="preset character key (default: yuki)")
    ap.add_argument("--tries", type=int, default=3, help="seeds per sampler (default: 3)")
    ap.add_argument("--exclude", nargs="+", default=DEFAULT_EXCLUDE,
                    help=f"samplers to skip (default: {DEFAULT_EXCLUDE})")
    args = ap.parse_args()

    preset = PRESET_CHARACTERS.get(args.char)
    if not preset:
        print(f"Unknown character '{args.char}'. Options: {list(PRESET_CHARACTERS)}")
        return 1

    if not _comfy_health():
        print(f"ComfyUI not reachable at {COMFY_HOST}. Start it first.")
        return 1

    valid = _valid_samplers()
    if not valid:
        print("Couldn't read ComfyUI's sampler list — needed to enumerate the sweep. Aborting.")
        return 1

    excluded = set(args.exclude)
    samplers = sorted(s for s in valid if s not in excluded)
    if not samplers:
        print("No samplers left after exclusions.")
        return 1

    # Free the GPU for ComfyUI (stop the LLM).
    try:
        from model_client import release_vram
        release_vram()
    except Exception as e:
        print(f"[sweep] release_vram failed ({e}); continuing")

    base_seed = preset["seed"]
    seeds = [base_seed + i for i in range(args.tries)]
    out_dir = os.path.join(TEST_DIR, f"{args.char}_sweep")
    os.makedirs(out_dir, exist_ok=True)
    prompt, negative = preset["identity_prompt"], SPRITE_NEGATIVE

    print(f"\n[sweep] {args.char}: {len(samplers)} sampler(s) × {len(seeds)} seed(s)")
    print(f"[sweep] excluded: {sorted(excluded)}")
    print(f"[sweep] sweeping: {samplers}\n")

    # Index prefix per sampler → easy identification + stable sort order
    labels = {s: f"s{i + 1:02d}" for i, s in enumerate(samplers)}

    grid = {}  # (sampler, seed) -> filename or None
    for sampler in samplers:
        pfx = labels[sampler]
        for seed in seeds:
            tag = f"{pfx}_{sampler}_seed{seed}"
            print(f"  [{pfx}] {sampler} seed {seed}…", end=" ", flush=True)
            dest = os.path.join(out_dir, f"{tag}.png")
            try:
                pid = _comfy_submit(_base_workflow(prompt, negative, seed, sampler),
                                    f"sweep_{args.char}_{tag}")
                result = _comfy_wait(pid)
                ok = bool(result and _comfy_fetch_image(result, dest))
            except Exception as e:
                print(f"FAILED ({e})")
                grid[(sampler, seed)] = None
                continue
            print("ok" if ok else "no output")
            grid[(sampler, seed)] = f"{tag}.png" if ok else None

    # Markdown contact sheet: rows = sampler (prefixed), cols = seed
    lines = [f"# Sampler sweep — {args.char}", "",
             f"Fixed: scheduler `{SCHEDULER}`, steps {STEPS}, cfg {CFG}, "
             f"{WIDTH}×{HEIGHT}. Only the sampler varies. "
             f"Excluded (see sampler_ab.py): {sorted(excluded)}.", "",
             "| # | sampler | " + " | ".join(f"seed {s}" for s in seeds) + " |",
             "|---|" + "---|" * (len(seeds) + 1)]
    for sampler in samplers:
        cells = []
        for s in seeds:
            f = grid.get((sampler, s))
            cells.append(f"![{f}]({f})" if f else "✗")
        lines.append(f"| {labels[sampler]} | **{sampler}** | " + " | ".join(cells) + " |")
    with open(os.path.join(out_dir, "REPORT.md"), "w") as fp:
        fp.write("\n".join(lines) + "\n")

    done = sum(1 for v in grid.values() if v)
    print(f"\n[sweep] Done — {done}/{len(grid)} renders. Images + REPORT.md in: {out_dir}")
    print("[sweep] LLM was stopped — restart it (or run app.py) before playing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
