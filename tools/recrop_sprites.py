"""
recrop_sprites.py — Apply the union-box auto-crop to ALREADY-generated
sprites (so you don't need a full ComfyUI re-bake to fix the framing).

    python tools/recrop_sprites.py

Idempotent: re-running on already-cropped sets just trims ~nothing.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cast_pipeline import GENERATED_DIR, _autocrop_sprite_set

if __name__ == "__main__":
    if not os.path.isdir(GENERATED_DIR):
        print(f"Nothing to crop — {GENERATED_DIR} doesn't exist yet.")
        sys.exit(0)
    done = 0
    for key in sorted(os.listdir(GENERATED_DIR)):
        char_dir = os.path.join(GENERATED_DIR, key)
        if os.path.isdir(char_dir):
            _autocrop_sprite_set(char_dir, key)
            done += 1
    print(f"\nDone — processed {done} character folder(s).")
