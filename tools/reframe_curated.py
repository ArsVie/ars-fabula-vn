"""Re-apply the face-normalized ZOOM framing to an already-baked (curated) cast.

The curated sprites under static/sprites/curated/<key>/ are FRAMED composites
(the old bake: figure scaled, feet on the canvas bottom edge) — not raw mattes —
so this script first RECOVERS the old bake's scale, then re-frames to the
current spec in cast_pipeline:

  • every face = FACE_TARGET_FRAC of the SCENE height (all equal)
  • feet sit FEET_DROP_FACES face-heights BELOW the canvas bottom (leg crop)
  • canvas WIDTH grows to fit the figure (no side clipping); heights uniform

Scale recovery: `<key>_base.png` is the raw RGB render on a near-white
background; thresholding it gives the figure bbox in base coordinates, and the
framed sprite's alpha bbox gives the same figure in sprite coordinates — the
height ratio is the old scale, which maps the face-gate mask's face height
into sprite pixels.

Originals are backed up to <char_dir>/_preframe/ on first run and ALWAYS
re-read from there, so the script is idempotent — tweak the env knobs
(ARS_FABULA_FACE_TARGET / ARS_FABULA_FEET_DROP) and re-run freely.

Usage:
    python tools/reframe_curated.py                 # all chars under curated/
    python tools/reframe_curated.py --char yuki     # one character
    python tools/reframe_curated.py --dir static/sprites/curated
    python tools/reframe_curated.py --restore       # put _preframe originals back
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cast_pipeline import (FACE_TARGET_FRAC, FEET_DROP_FACES, UI_SPRITE_FRAC,
                           MIN_FIGURE_CANVAS_FRAC, STANDARD_EXPRESSIONS)

# Background threshold for the base render ("white background" style tag):
# a pixel is background if every channel exceeds this.
BG_THRESHOLD = 235


def _figure_bbox_on_white(img: Image.Image) -> tuple[int, int, int, int] | None:
    """Bbox of the non-background figure in an RGB-on-white base render."""
    rgb = img.convert("RGB")
    # point() per band, then AND: background = all three channels bright
    masks = [band.point(lambda v: 255 if v > BG_THRESHOLD else 0)
             for band in rgb.split()]
    from PIL import ImageChops
    bg = ImageChops.multiply(ImageChops.multiply(masks[0], masks[1]), masks[2])
    fig = bg.point(lambda v: 0 if v else 255)   # invert: figure pixels white
    return fig.getbbox()


def reframe_char(char_dir: str, key: str) -> bool:
    mask_path = os.path.join(char_dir, "_facegate_mask.png")
    base_path = os.path.join(char_dir, f"{key}_base.png")
    if not os.path.exists(mask_path) or not os.path.exists(base_path):
        print(f"  [{key}] SKIP — needs _facegate_mask.png + {key}_base.png")
        return False

    expr_files = [f"{key}_{e}.png" for e in STANDARD_EXPRESSIONS]
    expr_files = [f for f in expr_files if os.path.exists(os.path.join(char_dir, f))]
    if not expr_files:
        print(f"  [{key}] SKIP — no expression sprites")
        return False

    # ── Backup originals once; always re-frame FROM the backup (idempotent) ──
    pre_dir = os.path.join(char_dir, "_preframe")
    os.makedirs(pre_dir, exist_ok=True)
    for f in expr_files:
        dst = os.path.join(pre_dir, f)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(char_dir, f), dst)

    # ── Measure: face (mask, base coords) + figure (base render, base coords) ──
    with Image.open(mask_path) as m:
        fb = m.convert("L").getbbox()
    if not fb:
        print(f"  [{key}] SKIP — empty face mask")
        return False
    face_h_base = fb[3] - fb[1]

    with Image.open(base_path) as b:
        base_fig = _figure_bbox_on_white(b)
        bW, bH = b.size
    if not base_fig or (base_fig[2] - base_fig[0]) >= bW - 2:
        print(f"  [{key}] SKIP — couldn't isolate the figure in {key}_base.png "
              f"(background not near-white?)")
        return False

    # ── Union alpha bbox of the ORIGINAL framed sprites (sprite coords) ──
    union = None
    sW = sH = 0
    for f in expr_files:
        with Image.open(os.path.join(pre_dir, f)) as img:
            rgba = img.convert("RGBA")
            sW, sH = rgba.size
            bb = rgba.getchannel("A").getbbox()
        if bb:
            union = bb if union is None else (
                min(union[0], bb[0]), min(union[1], bb[1]),
                max(union[2], bb[2]), max(union[3], bb[3]))
    if union is None:
        print(f"  [{key}] SKIP — sprites have no alpha content")
        return False

    # ── Recover the old bake's scale (base px → sprite px) ──
    s_h = (union[3] - union[1]) / (base_fig[3] - base_fig[1])
    s_w = (union[2] - union[0]) / (base_fig[2] - base_fig[0])
    if abs(s_h - s_w) / max(s_h, s_w) > 0.05:
        print(f"  [{key}] WARNING: scale recovery disagrees "
              f"(h={s_h:.3f} vs w={s_w:.3f}) — sprite may already be re-framed "
              f"or side-clipped; using height")
    face_h_sprite = face_h_base * s_h

    # ── New transform (same spec as cast_pipeline._frame_upper_body) ──
    H0 = sH                                            # keep cast-uniform height
    face_px = H0 * FACE_TARGET_FRAC / UI_SPRITE_FRAC   # target face on canvas
    rescale = face_px / face_h_sprite
    # Min-figure-height floor (anti-"midget"): boost short figures up to the
    # MIN_FIGURE_CANVAS_FRAC crown floor — identical to cast_pipeline._frame_upper_body.
    fig_h_sprite = union[3] - union[1]
    crown_frac = (fig_h_sprite * rescale - FEET_DROP_FACES * face_px) / H0
    if crown_frac < MIN_FIGURE_CANVAS_FRAC and fig_h_sprite > 1:
        rescale = (MIN_FIGURE_CANVAS_FRAC * H0 + FEET_DROP_FACES * face_px) / fig_h_sprite
        print(f"  [{key}] short figure ({crown_frac:.0%} crown) boosted to "
              f"the {MIN_FIGURE_CANVAS_FRAC:.0%} min-height floor")
    pad = 12
    fig_w = (union[2] - union[0]) * rescale
    W0 = max(sW, int(fig_w) + 1 + 2 * pad)             # widen to fit, never clip
    cx = (union[0] + union[2]) / 2.0
    off_x = int(round(W0 / 2.0 - cx * rescale))
    off_y = int(round(H0 + FEET_DROP_FACES * face_px - union[3] * rescale))
    nw, nh = max(1, int(round(sW * rescale))), max(1, int(round(sH * rescale)))

    for f in expr_files:
        with Image.open(os.path.join(pre_dir, f)) as img:
            resized = img.convert("RGBA").resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGBA", (W0, H0), (0, 0, 0, 0))
        canvas.paste(resized, (off_x, off_y), resized)
        canvas.save(os.path.join(char_dir, f))

    crown_scene = (H0 - (off_y + union[1] * rescale)) / H0 * UI_SPRITE_FRAC
    print(f"  [{key}] reframed {len(expr_files)} sprites: face {FACE_TARGET_FRAC:.0%} "
          f"of scene, feet at -{FEET_DROP_FACES:g} faces, crown at {crown_scene:.0%} "
          f"of scene, canvas {W0}x{H0} (old scale {s_h:.3f}, zoom ×{rescale:.2f})")
    if crown_scene > UI_SPRITE_FRAC:
        print(f"  [{key}] WARNING: crown overflows the canvas top — lower "
              f"ARS_FABULA_FACE_TARGET or ARS_FABULA_FEET_DROP and re-run")
    return True


def restore_char(char_dir: str, key: str) -> bool:
    pre_dir = os.path.join(char_dir, "_preframe")
    if not os.path.isdir(pre_dir):
        print(f"  [{key}] nothing to restore")
        return False
    n = 0
    for f in os.listdir(pre_dir):
        if f.endswith(".png"):
            shutil.copy2(os.path.join(pre_dir, f), os.path.join(char_dir, f))
            n += 1
    print(f"  [{key}] restored {n} original sprites from _preframe/")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=os.path.join("static", "sprites", "curated"),
                    help="cast directory (default: static/sprites/curated)")
    ap.add_argument("--char", action="append", default=None,
                    help="character key (repeatable; default: all)")
    ap.add_argument("--restore", action="store_true",
                    help="restore the _preframe originals instead of reframing")
    args = ap.parse_args()

    root = args.dir
    if not os.path.isdir(root):
        sys.exit(f"not a directory: {root}")
    keys = args.char or sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))

    print(f"{'Restoring' if args.restore else 'Reframing'} cast in {root} "
          f"(face={FACE_TARGET_FRAC:.0%} of scene, feet at -{FEET_DROP_FACES:g} faces)")
    for key in keys:
        char_dir = os.path.join(root, key)
        if not os.path.isdir(char_dir):
            print(f"  [{key}] SKIP — no such folder")
            continue
        (restore_char if args.restore else reframe_char)(char_dir, key)


if __name__ == "__main__":
    main()
