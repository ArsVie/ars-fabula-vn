"""
comfy_doctor.py — Diagnose the headless ComfyUI connection layer by layer.

Run this BEFORE the cast pipeline whenever generation misbehaves:

    python tools/comfy_doctor.py
    ARS_FABULA_COMFY=http://172.x.x.x:8188 python tools/comfy_doctor.py

Checks, in order (each one isolates a different failure mode):
  1. Proxy environment variables (the usual cause of instant
     [Errno 32] Broken pipe on corporate machines)
  2. GET  /system_stats   — server reachable at all
  3. GET  /object_info    — required node classes installed
                            + model FILENAMES match what's installed
  4. POST /upload/image   — POST path + multipart works
  5. POST /prompt + /view — full round trip with a GPU-free workflow
                            (LoadImage → SaveImage)
"""
from __future__ import annotations
import io
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cast_pipeline import (COMFY_HOST, _session, UNET_NAME, CLIP_NAME,
                           VAE_NAME, SAM_NAME, ANIME_FACE_MODEL)

OK, FAIL, WARN = "  [ok]  ", "  [FAIL]", "  [warn]"

REQUIRED_NODES = [
    "LoadImage", "SaveImage", "KSampler", "EmptyLatentImage",
    "CLIPTextEncode", "VAEDecode", "VAELoader", "UNETLoader", "CLIPLoader",
    "UltralyticsDetectorProvider", "SAMLoader", "FaceDetailer",
    "BboxDetectorSEGS", "SegsToCombinedMask", "MaskToImage",
]


def _choices(obj_info: dict, node: str, field: str) -> list:
    """Extract a loader widget's option list from /object_info."""
    try:
        inp = obj_info[node]["input"]
        spec = (inp.get("required", {}).get(field)
                or inp.get("optional", {}).get(field))
        return spec[0] if spec and isinstance(spec[0], list) else []
    except Exception:
        return []


def clear_queue() -> None:
    """Interrupt the running job and clear all pending ones."""
    s = _session()
    try:
        s.post(f"{COMFY_HOST}/interrupt", timeout=10)
        s.post(f"{COMFY_HOST}/queue", json={"clear": True}, timeout=10)
        q = s.get(f"{COMFY_HOST}/queue", timeout=10).json()
        print(f"Queue cleared — running: {len(q.get('queue_running', []))}, "
              f"pending: {len(q.get('queue_pending', []))}")
    except Exception as e:
        print(f"Clear failed: {e}")


def main() -> int:
    if "--clear-queue" in sys.argv:
        clear_queue()
        return 0

    failures = 0
    print(f"\nComfyUI doctor — target: {COMFY_HOST}\n")

    # 1 — proxy environment
    proxies = {k: v for k, v in os.environ.items()
               if k.lower() in ("http_proxy", "https_proxy", "all_proxy") and v}
    if proxies:
        print(f"{WARN} proxy env vars set: {', '.join(proxies)} — the pipeline "
              "bypasses them (trust_env=False), but other tools may not")
    else:
        print(f"{OK} no proxy env vars")

    s = _session()

    # 2 — reachability
    try:
        r = s.get(f"{COMFY_HOST}/system_stats", timeout=5)
        r.raise_for_status()
        devs = [d.get("name", "?") for d in r.json().get("devices", [])]
        print(f"{OK} GET /system_stats — devices: {', '.join(devs) or 'unknown'}")
    except Exception as e:
        print(f"{FAIL} GET /system_stats: {e}")
        print("\n  Server unreachable. If ComfyUI runs on WINDOWS and this "
              "script runs in WSL, 127.0.0.1 is not the same loopback —\n"
              "  start ComfyUI with `--listen 0.0.0.0` and run with\n"
              "  ARS_FABULA_COMFY=http://$(ip route show default | awk '{print $3}'):8188\n")
        return 1

    # 3 — nodes + model files
    try:
        obj = s.get(f"{COMFY_HOST}/object_info", timeout=30).json()
        missing = [n for n in REQUIRED_NODES if n not in obj]
        if missing:
            failures += 1
            print(f"{FAIL} missing node classes: {', '.join(missing)} "
                  "(install Impact-Pack / Impact-Subpack via ComfyUI Manager)")
        else:
            print(f"{OK} all {len(REQUIRED_NODES)} required node classes present")

        for node, field, wanted in [
            ("UNETLoader", "unet_name", UNET_NAME),
            ("CLIPLoader", "clip_name", CLIP_NAME),
            ("VAELoader", "vae_name", VAE_NAME),
            ("SAMLoader", "model_name", SAM_NAME),
            ("UltralyticsDetectorProvider", "model_name", ANIME_FACE_MODEL),
        ]:
            opts = _choices(obj, node, field)
            if wanted in opts:
                print(f"{OK} {node}: '{wanted}' installed")
            else:
                failures += 1
                near = [o for o in opts if isinstance(o, str)
                        and wanted.split('/')[-1][:6].lower() in o.lower()]
                hint = f" — did you mean: {near[:3]}" if near else f" — available: {opts[:5]}"
                print(f"{FAIL} {node}: '{wanted}' NOT installed{hint}")
    except Exception as e:
        failures += 1
        print(f"{FAIL} GET /object_info: {e}")

    # 4 — POST path (upload a small, definitely-valid PNG).
    # Must be a real PNG: ComfyUI's LoadImage falls back to its VIDEO
    # decoder when PIL can't open an image, which produces a baffling
    # av.error.ExternalError instead of "bad image".
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (128, 64, 200)).save(buf, format="PNG")
    buf.seek(0)
    try:
        r = s.post(f"{COMFY_HOST}/upload/image",
                   files={"image": ("_doctor.png", buf, "image/png")},
                   data={"overwrite": "true"}, timeout=30)
        r.raise_for_status()
        print(f"{OK} POST /upload/image — multipart POST path works")
    except Exception as e:
        failures += 1
        print(f"{FAIL} POST /upload/image: {e}")
        print("       GET works but POST dies → almost certainly a proxy or "
              "middlebox; check `env | grep -i proxy` and your VPN client")
        return 1

    # 5 — queue state (stuck/backlogged queues make every new job hang)
    try:
        q = s.get(f"{COMFY_HOST}/queue", timeout=10).json()
        running = q.get("queue_running", []) or []
        pending = q.get("queue_pending", []) or []
        if running or pending:
            print(f"{WARN} queue: {len(running)} running, {len(pending)} pending "
                  "— earlier jobs are ahead of (or blocking) new ones.")
            print("        Clear with: python tools/comfy_doctor.py --clear-queue")
        else:
            print(f"{OK} queue empty")
    except Exception as e:
        print(f"{WARN} GET /queue: {e}")
        running = pending = []

    # 6 — full round trip, no GPU needed
    try:
        wf = {
            "1": {"inputs": {"image": "_doctor.png"}, "class_type": "LoadImage"},
            "2": {"inputs": {"filename_prefix": "_doctor_out", "images": ["1", 0]},
                  "class_type": "SaveImage"},
        }
        r = s.post(f"{COMFY_HOST}/prompt",
                   json={"prompt": wf, "client_id": "doctor"}, timeout=30)
        if not r.ok:
            failures += 1
            print(f"{FAIL} POST /prompt: HTTP {r.status_code} — {r.text[:500]}")
        else:
            pid = r.json()["prompt_id"]
            img = None
            for _ in range(30):
                time.sleep(1)
                hist = s.get(f"{COMFY_HOST}/history/{pid}", timeout=10).json()
                entry = hist.get(pid) or {}
                for nout in (entry.get("outputs") or {}).values():
                    if nout.get("images"):
                        img = nout["images"][0]
                        break
                if img:
                    break
            if not img:
                failures += 1
                # Diagnose WHY: still queued, or executed and failed?
                hist = s.get(f"{COMFY_HOST}/history/{pid}", timeout=10).json()
                entry = hist.get(pid) or {}
                q = s.get(f"{COMFY_HOST}/queue", timeout=10).json()
                queued_ids = [item[1] for item in
                              (q.get("queue_running", []) + q.get("queue_pending", []))
                              if isinstance(item, (list, tuple)) and len(item) > 1]
                if pid in queued_ids:
                    print(f"{FAIL} test job is STILL IN THE QUEUE after 30s — the "
                          "queue is blocked by an earlier job.")
                    print("        Run: python tools/comfy_doctor.py --clear-queue")
                    print("        Also check the ComfyUI terminal — the blocking "
                          "job's log (e.g. an OOM during model load) will be there.")
                elif entry:
                    status = entry.get("status") or {}
                    print(f"{FAIL} test job executed but produced no output — "
                          f"status: {status.get('status_str')}")
                    for msg in status.get("messages", []):
                        print(f"        {msg}")
                else:
                    print(f"{FAIL} test job vanished: not in history, not in queue "
                          "(server restarted mid-test?)")
            else:
                v = s.get(f"{COMFY_HOST}/view",
                          params={"filename": img["filename"],
                                  "subfolder": img.get("subfolder", ""),
                                  "type": img.get("type", "output")},
                          timeout=15)
                if v.ok and v.content[:4] == b"\x89PNG":
                    print(f"{OK} POST /prompt → execute → GET /view — full round trip works")
                else:
                    failures += 1
                    print(f"{FAIL} GET /view: HTTP {v.status_code}")
    except Exception as e:
        failures += 1
        print(f"{FAIL} round trip: {e}")

    print(f"\n{'All checks passed — run the cast pipeline.' if not failures else f'{failures} check(s) failed — fix the first failure above, then re-run.'}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
