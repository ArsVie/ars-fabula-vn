"""
sampler_ab.py — A/B test ComfyUI samplers on ONE character's base gen.

Everything is held fixed (prompt, negative, seeds, steps, cfg, scheduler,
resolution) EXCEPT the sampler, so any difference you see is the sampler's.
Runs N seeds × each sampler and writes a markdown contact sheet.

    .venv/bin/python3 tools/sampler_ab.py                       # yuki, 3 seeds
    .venv/bin/python3 tools/sampler_ab.py --char sakura --tries 3
    .venv/bin/python3 tools/sampler_ab.py --samplers er_sde flow_euler euler

Sampler resolution: the requested list is checked against ComfyUI's actual
KSampler sampler_name options. If `flow_euler` isn't registered, it falls
back to plain `euler` (deduped). Unknown samplers are skipped with a note.

VRAM: stops the LLM first (frees the GPU for ComfyUI). Restart your LLM
after, or just run app.py which manages it.

Output: static/sprites/_sampler_test/<char>/<sampler>_seed<seed>.png
        static/sprites/_sampler_test/<char>/REPORT.md
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cast_pipeline import (
    PRESET_CHARACTERS, SPRITE_NEGATIVE, SCHEDULER, GENERATED_DIR,
    UNET_NAME, CLIP_NAME, VAE_NAME,
    _comfy_health, _comfy_submit, _comfy_wait, _comfy_fetch_image, _session,
    COMFY_HOST,
)

TEST_DIR = os.path.join(os.path.dirname(GENERATED_DIR), "_sampler_test")
STEPS, CFG, WIDTH, HEIGHT = 30, 4.0, 512, 768


def _valid_samplers() -> set:
    """ComfyUI's actual KSampler sampler_name options."""
    try:
        obj = _session().get(f"{COMFY_HOST}/object_info/KSampler", timeout=15).json()
        opts = obj["KSampler"]["input"]["required"]["sampler_name"][0]
        return set(opts) if isinstance(opts, list) else set()
    except Exception as e:
        print(f"[ab] couldn't fetch sampler list ({e}); assuming all requested are valid")
        return set()


def _resolve(requested, valid):
    """Map requested samplers onto installed ones (flow_euler→euler fallback),
    deduped, order-preserving. Returns [(label, sampler_name)]."""
    out, seen = [], set()
    for name in requested:
        if valid and name not in valid:
            if name == "flow_euler" and "euler" in (valid or {"euler"}):
                resolved, label = "euler", "flow_euler→euler (fallback)"
                print(f"[ab] '{name}' not installed — falling back to 'euler'")
            else:
                print(f"[ab] '{name}' not installed and no fallback — skipping")
                continue
        else:
            resolved, label = name, name
        if resolved in seen:
            print(f"[ab] '{resolved}' already queued — skipping duplicate ({label})")
            continue
        seen.add(resolved)
        out.append((label, resolved))
    return out


def _base_workflow(prompt, negative, seed, sampler):
    return {
        "1": {"inputs": {"filename_prefix": "ab", "images": ["8", 0]}, "class_type": "SaveImage"},
        "8": {"inputs": {"samples": ["19", 0], "vae": ["15", 0]}, "class_type": "VAEDecode"},
        "11": {"inputs": {"text": prompt, "clip": ["45", 0]}, "class_type": "CLIPTextEncode"},
        "12": {"inputs": {"text": negative, "clip": ["45", 0]}, "class_type": "CLIPTextEncode"},
        "15": {"inputs": {"vae_name": VAE_NAME}, "class_type": "VAELoader"},
        "19": {"inputs": {
            "seed": seed, "steps": STEPS, "cfg": CFG,
            "sampler_name": sampler, "scheduler": SCHEDULER, "denoise": 1.0,
            "model": ["44", 0], "positive": ["11", 0], "negative": ["12", 0],
            "latent_image": ["28", 0],
        }, "class_type": "KSampler"},
        "28": {"inputs": {"width": WIDTH, "height": HEIGHT, "batch_size": 1}, "class_type": "EmptyLatentImage"},
        "44": {"inputs": {"unet_name": UNET_NAME, "weight_dtype": "default"}, "class_type": "UNETLoader"},
        "45": {"inputs": {"clip_name": CLIP_NAME, "type": "stable_diffusion", "device": "default"}, "class_type": "CLIPLoader"},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="A/B test samplers on one character's base gen")
    ap.add_argument("--char", default="yuki", help="preset character key (default: yuki)")
    ap.add_argument("--tries", type=int, default=3, help="seeds per sampler (default: 3)")
    ap.add_argument("--samplers", nargs="+",
                    default=["er_sde", "flow_euler", "euler"],
                    help="samplers to compare")
    args = ap.parse_args()

    preset = PRESET_CHARACTERS.get(args.char)
    if not preset:
        print(f"Unknown character '{args.char}'. Options: {list(PRESET_CHARACTERS)}")
        return 1

    if not _comfy_health():
        print(f"ComfyUI not reachable at {COMFY_HOST}. Start it first.")
        return 1

    # Free the GPU for ComfyUI (stop the LLM).
    try:
        from model_client import release_vram
        release_vram()
    except Exception as e:
        print(f"[ab] release_vram failed ({e}); continuing")

    samplers = _resolve(args.samplers, _valid_samplers())
    if not samplers:
        print("No usable samplers after resolution. Aborting.")
        return 1

    base_seed = preset["seed"]
    seeds = [base_seed + i for i in range(args.tries)]
    out_dir = os.path.join(TEST_DIR, args.char)
    os.makedirs(out_dir, exist_ok=True)
    prompt, negative = preset["identity_prompt"], SPRITE_NEGATIVE

    print(f"\n[ab] {args.char}: {len(samplers)} sampler(s) × {len(seeds)} seed(s), "
          f"scheduler={SCHEDULER}, steps={STEPS}, cfg={CFG}\n")

    grid = {}  # (label, seed) -> filename or None
    for label, sampler in samplers:
        for seed in seeds:
            tag = f"{sampler}_seed{seed}"
            print(f"  [{label}] seed {seed}…", end=" ", flush=True)
            dest = os.path.join(out_dir, f"{tag}.png")
            try:
                pid = _comfy_submit(_base_workflow(prompt, negative, seed, sampler),
                                    f"ab_{args.char}_{tag}")
                result = _comfy_wait(pid)
                ok = bool(result and _comfy_fetch_image(result, dest))
            except Exception as e:
                print(f"FAILED ({e})")
                grid[(label, seed)] = None
                continue
            print("ok" if ok else "no output")
            grid[(label, seed)] = f"{tag}.png" if ok else None

    # Markdown contact sheet: rows = sampler, cols = seed
    lines = [f"# Sampler A/B — {args.char}", "",
             f"Fixed: scheduler `{SCHEDULER}`, steps {STEPS}, cfg {CFG}, "
             f"{WIDTH}×{HEIGHT}. Only the sampler varies.", "",
             "| sampler | " + " | ".join(f"seed {s}" for s in seeds) + " |",
             "|" + "---|" * (len(seeds) + 1)]
    for label, _ in samplers:
        cells = []
        for s in seeds:
            f = grid.get((label, s))
            cells.append(f"![{f}]({f})" if f else "✗")
        lines.append(f"| **{label}** | " + " | ".join(cells) + " |")
    report = os.path.join(out_dir, "REPORT.md")
    with open(report, "w") as fp:
        fp.write("\n".join(lines) + "\n")

    print(f"\n[ab] Done. Images + REPORT.md in: {out_dir}")
    print("[ab] LLM was stopped — restart it (or run app.py) before playing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
