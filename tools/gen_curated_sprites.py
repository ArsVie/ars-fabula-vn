"""Generate all character sprites via ComfyUI, then copy to curated dir."""
import os, shutil, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cast_pipeline import generate_cast, GENERATED_DIR, CURATED_DIR

print("[gen] Generating all 4 characters...", flush=True)
cast = generate_cast({"characters": ["yuki", "haruki", "sakura", "ryo"]})
print(f"[gen] Generated {len(cast)} characters", flush=True)

for m in cast:
    key = m.character.name.lower()
    curated_char_dir = os.path.join(CURATED_DIR, key)
    os.makedirs(curated_char_dir, exist_ok=True)
    for expr, path in m.expression_map.items():
        if os.path.exists(path):
            dst = os.path.join(curated_char_dir, f"{key}_{expr}.png")
            shutil.copy2(path, dst)
            sz = os.path.getsize(dst) // 1024
            print(f"  {key}_{expr}: {sz}KB -> {dst}", flush=True)

total = sum(len(files) for _, _, files in os.walk(CURATED_DIR) if files)
print(f"\n[gen] Done — {total} sprites in curated dir", flush=True)
