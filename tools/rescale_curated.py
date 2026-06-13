"""Rescale curated sprites so faces are 22% of the output canvas height.

Old framing: face = H * FACE_TARGET_FRAC = 896 * 0.135 = 120.96 px
New framing: face = H0 * 0.22            = 1254 * 0.22 = 275.88 px

Constant scale factor: 275.88 / 120.96 ≈ 2.281

Instead of center-crop (which clips the face), anchors the crop on the
character's alpha-bbox crown + horizontal center so the face region is
preserved — only trims excess body below.

Usage: python tools/rescale_curated.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image
from cast_pipeline import CURATED_DIR, STANDARD_EXPRESSIONS

H = 896
FACE_TARGET_FRAC = 0.135
SPRITE_CANVAS_MULT = 1.4
H0 = int(H * SPRITE_CANVAS_MULT)

FACTOR = (H0 * 0.22) / (H * FACE_TARGET_FRAC)


def rescale_curated():
    if not os.path.isdir(CURATED_DIR):
        print(f"  [rescale] No curated dir at {CURATED_DIR}")
        return

    chars = sorted(d.name for d in os.scandir(CURATED_DIR) if d.is_dir())
    if not chars:
        print(f"  [rescale] No character dirs found in {CURATED_DIR}")
        return

    print(f"  [rescale] Scale factor: {FACTOR:.4f}×")
    print()

    for char in chars:
        char_dir = os.path.join(CURATED_DIR, char)
        expr_paths = [
            os.path.join(char_dir, f"{char}_{e}.png")
            for e in STANDARD_EXPRESSIONS
        ]
        expr_paths = [p for p in expr_paths if os.path.exists(p)]
        if not expr_paths:
            continue

        # ---- Union alpha bbox across all expressions ----
        union = None
        cw = ch = 0
        for p in expr_paths:
            with Image.open(p) as img:
                rgba = img.convert("RGBA")
                cw, ch = rgba.size
                bbox = rgba.getchannel("A").getbbox()
            if bbox:
                union = (
                    min(union[0], bbox[0]) if union else bbox[0],
                    min(union[1], bbox[1]) if union else bbox[1],
                    max(union[2], bbox[2]) if union else bbox[2],
                    max(union[3], bbox[3]) if union else bbox[3],
                )
        if union is None:
            print(f"  [rescale] {char}: no alpha bbox — skipping")
            continue

        crown_y = union[1]  # top of head
        cx = (union[0] + union[2]) / 2.0  # figure center-x

        nw = max(1, int(round(cw * FACTOR)))
        nh = max(1, int(round(ch * FACTOR)))

        # Crop anchored on crown+center so the face stays intact
        crop_left = int(round(cx * (FACTOR - 1)))
        crop_top = int(round(crown_y * (FACTOR - 1)))

        # Clamp so crop stays inside scaled image
        crop_left = max(0, min(crop_left, nw - cw))
        crop_top = max(0, min(crop_top, nh - ch))

        for p in expr_paths:
            with Image.open(p) as img:
                scaled = img.convert("RGBA").resize((nw, nh), Image.LANCZOS)
                cropped = scaled.crop((crop_left, crop_top,
                                       crop_left + cw, crop_top + ch))
                cropped.save(p)

        print(f"  [rescale] {char}: {len(expr_paths)} sprites "
              f"{cw}×{ch} → {nw}×{nh} → crop at ({crop_left},{crop_top})")

    print()
    print(f"  [rescale] Done — {len(chars)} characters")


if __name__ == "__main__":
    rescale_curated()
