"""
rembg_cast.py — Run ONLY the background-removal (+ auto-crop) step over
already-generated cast sprites.

Use this when sprites were baked before rembg was installed (white
backgrounds), or after installing rembg, to fix the matte without
re-running the whole ComfyUI pipeline.

    .venv/bin/python3 tools/rembg_cast.py                 # all generated chars
    .venv/bin/python3 tools/rembg_cast.py yuki sakura     # only these
    .venv/bin/python3 tools/rembg_cast.py --no-crop       # matte only, skip crop
    .venv/bin/python3 tools/rembg_cast.py --force         # re-matte even if already transparent

Operates in-place on static/sprites/generated/<key>/<key>_<expr>.png.
Requires rembg + onnxruntime in this venv:
    .venv/bin/pip install rembg onnxruntime
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vn_contracts import STANDARD_EXPRESSIONS
from cast_pipeline import (GENERATED_DIR, _remove_bg, _has_transparency,
                           _autocrop_sprite_set)


def _char_dirs(only):
    if not os.path.isdir(GENERATED_DIR):
        return []
    keys = sorted(d for d in os.listdir(GENERATED_DIR)
                  if os.path.isdir(os.path.join(GENERATED_DIR, d)))
    if only:
        wanted = {k.lower() for k in only}
        keys = [k for k in keys if k.lower() in wanted]
    return keys


def main() -> int:
    ap = argparse.ArgumentParser(description="Background removal over generated cast sprites")
    ap.add_argument("chars", nargs="*", help="Character keys (default: all)")
    ap.add_argument("--no-crop", action="store_true", help="Skip the union-box auto-crop")
    ap.add_argument("--force", action="store_true",
                    help="Re-matte sprites that already have transparency")
    args = ap.parse_args()

    keys = _char_dirs(args.chars)
    if not keys:
        print(f"No generated character folders found under {GENERATED_DIR}")
        return 1

    total = matted = skipped = failed = 0
    for key in keys:
        char_dir = os.path.join(GENERATED_DIR, key)
        print(f"\n[{key}]")
        touched = False
        for expr in STANDARD_EXPRESSIONS:
            path = os.path.join(char_dir, f"{key}_{expr}.png")
            if not os.path.exists(path):
                continue
            total += 1
            if not args.force and _has_transparency(path):
                print(f"  {expr}: already transparent — skip")
                skipped += 1
                continue
            _remove_bg(path)              # in-place; prints its own warnings
            if _has_transparency(path):
                print(f"  {expr}: matted ✓")
                matted += 1
                touched = True
            else:
                print(f"  {expr}: NO transparency after rembg ✗")
                failed += 1

        if touched and not args.no_crop:
            _autocrop_sprite_set(char_dir, key)   # union-box crop keeps the set aligned

    print(f"\nDone — {matted} matted, {skipped} already-ok, {failed} failed, "
          f"of {total} sprites across {len(keys)} character(s).")
    if failed:
        print("Some sprites still have no transparency — see the rembg warning "
              "above (install: .venv/bin/pip install rembg onnxruntime).")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
