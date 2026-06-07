"""
cast_pipeline.py — Generate or load character casts for the VN.

Pipeline (per the corrected anime pipeline doc / Wave 0):
    ComfyUI anime base gen
      → ANIME face detect gate (yolov8x6_animeface — NOT the generic
        face_yolov8m/n, which under-detects anime faces; retry on miss)
      → ExpressionEditor warp (ComfyUI-AdvancedLivePortrait) for the
        expression set — warping preserves identity, unlike inpainting
      → rembg isnet-anime matte → alpha sprites

NOTE: AdvancedLivePortrait/InsightFace carries a NON-COMMERCIAL license
flag. Fine for this PoC; revisit before any commercial use.

Two modes:
1. GENERATE (default): requires ComfyUI at localhost:8188 with the Anima
   model, Impact-Pack (UltralyticsDetectorProvider) and
   ComfyUI-AdvancedLivePortrait (ExpressionEditor) custom nodes.
2. CURATED: loads pre-made sprites from static/sprites/curated/<name>/.

Same return type either way — list[CastMember] (frozen Wave 0.5 contract:
generate_cast(prefs) -> list[CastMember]) — so nothing downstream changes.
Use vn_contracts.cast_to_dict() to build SceneState's name->member dict.

Usage:
    from cast_pipeline import generate_cast, load_curated_cast
    from vn_contracts import cast_to_dict

    cast = generate_cast({"mood": "romance", "preset": "childhood_friends"})
    scene_cast = cast_to_dict(cast)
"""
from __future__ import annotations
import json
import os
import sys
import time
import urllib.request
import urllib.error
import shutil
from typing import Optional

from vn_contracts import Character, CastMember, STANDARD_EXPRESSIONS

# ── Constants ─────────────────────────────────────────────────────────
COMFY_HOST = "http://127.0.0.1:8188"
COMFY_OUTPUT = os.path.expanduser("~/comfy/ComfyUI/output")
PROJECT_DIR = os.path.dirname(__file__)
CURATED_DIR = os.path.join(PROJECT_DIR, "static", "sprites", "curated")
GENERATED_DIR = os.path.join(PROJECT_DIR, "static", "sprites", "generated")

QUALITY = "masterpiece, best quality, score_7, safe"
NEGATIVE = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia, bad anatomy, ugly, deformed"

# Anime-specific face detector (Impact-Pack UltralyticsDetectorProvider model).
# The plan is explicit: do NOT use the generic face_yolov8m/n on anime art.
ANIME_FACE_MODEL = "bbox/yolov8x6_animeface.pt"

# How many times to regenerate the base (seed bump) if no anime face is found
FACE_GATE_RETRIES = 2

# ── Preset character templates ────────────────────────────────────────

PRESET_CHARACTERS: dict[str, dict] = {
    "yuki": {
        "identity_prompt": (
            f"{QUALITY}. 1girl, yuki tanaka, long silver hair, straight hair, "
            "red eyes, fair skin, flat chest, high school uniform, white shirt, "
            "blue skirt, red ribbon, portrait, upper body, looking at viewer, neutral face"
        ),
        "personality": "Quiet and reserved but deeply caring. Speaks in short sentences. "
                       "Notices small details about others. Opens up slowly. "
                       "Has a dry sense of humor that surprises people.",
        "voice_id": "af_bella",
        "role": "classmate",
        "seed": 42,
    },
    "haruki": {
        "identity_prompt": (
            f"{QUALITY}. 1boy, haruki tanaka, short messy black hair, "
            "brown eyes, fair skin, high school uniform, white shirt, "
            "loose tie, portrait, upper body, looking at viewer, neutral face, tsundere"
        ),
        "personality": "Outgoing and energetic with a mischievous streak. "
                       "Teases people he likes. Loyal to friends. "
                       "Acts confident but gets flustered easily.",
        "voice_id": "am_michael",   # Kokoro male voice (am_*; af_* are female)
        "role": "classmate",
        "seed": 43,
    },
    "sakura": {
        "identity_prompt": (
            f"{QUALITY}. 1girl, sakura haruno, long pink hair, side ponytail, "
            "green eyes, fair skin, high school uniform, white shirt, "
            "pink bow, portrait, upper body, looking at viewer, smile, genki"
        ),
        "personality": "Cheerful and energetic genki girl. Always smiling. "
                       "Best friend energy. Gives great hugs. "
                       "Perceptive despite her bubbly exterior.",
        "voice_id": "af_sky",
        "role": "best friend",
        "seed": 44,
    },
    "ryo": {
        "identity_prompt": (
            f"{QUALITY}. 1boy, ryo akira, short blue hair, sharp eyes, "
            "tall, cool expression, high school uniform slightly disheveled, "
            "portrait, upper body, looking at viewer, calm face, cool"
        ),
        "personality": "Cool and aloof on the surface. Secretly a huge romantic. "
                       "Gives surprisingly good advice. "
                       "Has a hidden soft spot for cute things.",
        "voice_id": "am_adam",      # Kokoro male voice
        "role": "senpai",
        "seed": 45,
    },
}

# Kept for prompt-based pipelines and the contact sheet (describes the
# *intent* of each expression; the warp pipeline uses EXPRESSION_PARAMS).
EXPRESSION_TAGS = {
    "neutral": "neutral expression, looking at viewer, relaxed",
    "smile": "smile, open mouth, happy, looking at viewer, blush",
    "laugh": "laughing, open mouth, eyes closed, happy, joyful",
    "sad": "sad, downturned eyebrows, tears, wet eyes, looking down",
    "angry": "angry, furrowed eyebrows, glare, gritting teeth, frown",
    "surprised": "surprised, wide eyes, open mouth, shocked, eyebrows raised",
    "embarrassed": "embarrassed, heavy blush, averted eyes, sweatdrop, looking away",
}

# ExpressionEditor (AdvancedLivePortrait) parameter sets per expression.
# These WARP the base face — identity is preserved by construction.
# All unset params default to 0. Tune by eye via the contact sheet.
EXPRESSION_PARAMS: dict[str, dict] = {
    "neutral":     {},
    "smile":       {"smile": 1.0, "eyebrow": 5.0, "blink": 2.0},
    "laugh":       {"smile": 1.3, "aaa": 30.0, "blink": 18.0, "eyebrow": 6.0},
    "sad":         {"smile": -0.3, "eyebrow": -10.0, "blink": 6.0,
                    "rotate_pitch": 6.0, "pupil_y": -4.0},
    "angry":       {"smile": -0.2, "eyebrow": -14.0, "blink": 3.0, "eee": 8.0},
    "surprised":   {"blink": -12.0, "eyebrow": 12.0, "aaa": 25.0},
    "embarrassed": {"smile": 0.5, "blink": 7.0, "eyebrow": -3.0,
                    "rotate_yaw": -9.0, "pupil_x": 7.0},
}

# Defaults for ExpressionEditor widgets not covered by EXPRESSION_PARAMS
_EXPRESSION_EDITOR_DEFAULTS = {
    "rotate_pitch": 0.0, "rotate_yaw": 0.0, "rotate_roll": 0.0,
    "blink": 0.0, "eyebrow": 0.0, "wink": 0.0,
    "pupil_x": 0.0, "pupil_y": 0.0,
    "aaa": 0.0, "eee": 0.0, "woo": 0.0, "smile": 0.0,
    "src_ratio": 1.0, "sample_ratio": 1.0,
    "sample_parts": "OnlyExpression", "crop_factor": 1.7,
}


# ═══════════════════════════════════════════════════════════════════════
# ComfyUI helpers
# ═══════════════════════════════════════════════════════════════════════

def _comfy_upload(path: str, name: str) -> dict:
    """Upload an image to ComfyUI's input folder."""
    with open(path, "rb") as f:
        data = f.read()
    boundary = "----BoundaryMA4"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{COMFY_HOST}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _comfy_submit(workflow: dict, client_id: str) -> str:
    """Submit a workflow to ComfyUI and return the prompt_id."""
    payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
    req = urllib.request.Request(
        f"{COMFY_HOST}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["prompt_id"]


def _comfy_wait(prompt_id: str, timeout: int = 180) -> Optional[str]:
    """Wait for a ComfyUI prompt to complete. Returns output filename or None."""
    for i in range(timeout // 2):
        time.sleep(2)
        try:
            with urllib.request.urlopen(f"{COMFY_HOST}/history/{prompt_id}") as hr:
                hist = json.loads(hr.read())
        except Exception:
            continue
        if prompt_id in hist and hist[prompt_id].get("status", {}).get("completed"):
            outputs = hist[prompt_id].get("outputs", {})
            for nid, nout in outputs.items():
                for img in nout.get("images", []):
                    return img["filename"]
            return None
    return None


def _comfy_health() -> bool:
    """Check if ComfyUI is reachable."""
    try:
        req = urllib.request.Request(f"{COMFY_HOST}/system_stats")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════
# Base generation (txt2img)
# ═══════════════════════════════════════════════════════════════════════

def _generate_base(character_key: str, preset: dict, out_dir: str,
                   seed: Optional[int] = None) -> Optional[str]:
    """Generate base character image via ComfyUI txt2img.

    Returns path to the generated base PNG, or None on failure.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{character_key}_base.png")
    seed = preset["seed"] if seed is None else seed

    workflow = {
        "1": {"inputs": {"filename_prefix": f"base_{character_key}", "images": ["8", 0]}, "class_type": "SaveImage"},
        "8": {"inputs": {"samples": ["19", 0], "vae": ["15", 0]}, "class_type": "VAEDecode"},
        "11": {"inputs": {"text": preset["identity_prompt"], "clip": ["45", 0]}, "class_type": "CLIPTextEncode"},
        "12": {"inputs": {"text": NEGATIVE, "clip": ["45", 0]}, "class_type": "CLIPTextEncode"},
        "15": {"inputs": {"vae_name": "qwen_image_vae.safetensors"}, "class_type": "VAELoader"},
        "19": {"inputs": {
            "seed": seed, "steps": 30, "cfg": 4.0,
            "sampler_name": "er_sde", "scheduler": "simple", "denoise": 1.0,
            "model": ["44", 0], "positive": ["11", 0], "negative": ["12", 0],
            "latent_image": ["28", 0],
        }, "class_type": "KSampler"},
        "28": {"inputs": {"width": 512, "height": 768, "batch_size": 1}, "class_type": "EmptyLatentImage"},
        "44": {"inputs": {"unet_name": "anime-base-v1.0.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
        "45": {"inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "stable_diffusion", "device": "default"}, "class_type": "CLIPLoader"},
    }

    pid = _comfy_submit(workflow, f"base_{character_key}")
    result = _comfy_wait(pid)
    if result:
        src = os.path.join(COMFY_OUTPUT, result)
        if os.path.exists(src):
            shutil.copy2(src, out_path)
            print(f"  [cast] Generated base for {character_key} (seed={seed}): {out_path}")
            return out_path
    return None


# ═══════════════════════════════════════════════════════════════════════
# Anime face detection gate
# ═══════════════════════════════════════════════════════════════════════

def _detect_anime_face(base_path: str, character_key: str) -> Optional[bool]:
    """Check the base image contains a detectable anime face.

    Runs an Impact-Pack detector workflow with the ANIME face model and
    inspects the combined detection mask: non-empty mask == face found.

    Returns:
        True  — anime face detected
        False — no face detected (caller should retry base gen)
        None  — gate unavailable (node/model missing); caller proceeds
    """
    _comfy_upload(base_path, f"{character_key}_base.png")

    workflow = {
        "10": {"inputs": {"image": f"{character_key}_base.png"}, "class_type": "LoadImage"},
        "70": {"inputs": {"model_name": ANIME_FACE_MODEL}, "class_type": "UltralyticsDetectorProvider"},
        "71": {"inputs": {
            "bbox_detector": ["70", 0], "image": ["10", 0],
            "threshold": 0.3, "dilation": 0, "crop_factor": 1.0,
            "drop_size": 10, "labels": "all",
        }, "class_type": "BboxDetectorSEGS"},
        "72": {"inputs": {"segs": ["71", 0]}, "class_type": "SegsToCombinedMask"},
        "73": {"inputs": {"mask": ["72", 0]}, "class_type": "MaskToImage"},
        "74": {"inputs": {"filename_prefix": f"facegate_{character_key}", "images": ["73", 0]}, "class_type": "SaveImage"},
    }

    try:
        pid = _comfy_submit(workflow, f"facegate_{character_key}")
    except Exception as e:
        print(f"  [cast] Face gate unavailable ({e}); proceeding without gate")
        return None

    result = _comfy_wait(pid, timeout=60)
    if not result:
        print(f"  [cast] Face gate produced no output; proceeding without gate")
        return None

    mask_path = os.path.join(COMFY_OUTPUT, result)
    try:
        from PIL import Image
        with Image.open(mask_path) as img:
            extrema = img.convert("L").getextrema()
        return extrema[1] > 0   # any non-black pixel => a face was detected
    except Exception as e:
        print(f"  [cast] Face gate mask check failed ({e}); proceeding without gate")
        return None


def _generate_base_with_face_gate(character_key: str, preset: dict, out_dir: str) -> Optional[str]:
    """Base gen + anime-face-detect gate, retrying with bumped seeds."""
    for attempt in range(FACE_GATE_RETRIES + 1):
        seed = preset["seed"] + attempt
        base_path = _generate_base(character_key, preset, out_dir, seed=seed)
        if not base_path:
            return None
        detected = _detect_anime_face(base_path, character_key)
        if detected is not False:   # True, or gate unavailable
            if detected is True:
                print(f"  [cast] Anime face confirmed for {character_key}")
            return base_path
        print(f"  [cast] No anime face in {character_key} base "
              f"(attempt {attempt + 1}/{FACE_GATE_RETRIES + 1}); regenerating...")
    print(f"  [cast] Face gate failed for {character_key} after retries; using last base")
    return base_path


# ═══════════════════════════════════════════════════════════════════════
# Expression generation via ExpressionEditor warp (AdvancedLivePortrait)
# ═══════════════════════════════════════════════════════════════════════

def _generate_expression(base_path: str, character_key: str, expr_name: str) -> Optional[str]:
    """Generate a single expression by WARPING the base face with the
    ExpressionEditor node (ComfyUI-AdvancedLivePortrait).

    This is the Wave-0-validated approach: warping moves facial landmarks
    on the SAME image, so identity holds by construction (vs. inpainting,
    which re-draws the face and can drift).

    Returns path to the expression PNG, or None on failure.
    """
    out_dir = os.path.join(GENERATED_DIR, character_key)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{character_key}_{expr_name}.png")

    # neutral == the base itself, just copied through
    if expr_name == "neutral":
        shutil.copy2(base_path, out_path)
        return out_path

    _comfy_upload(base_path, f"{character_key}_base.png")

    params = dict(_EXPRESSION_EDITOR_DEFAULTS)
    params.update(EXPRESSION_PARAMS.get(expr_name, {}))

    workflow = {
        "10": {"inputs": {"image": f"{character_key}_base.png"}, "class_type": "LoadImage"},
        "80": {"inputs": {
            "src_image": ["10", 0],
            **params,
        }, "class_type": "ExpressionEditor"},
        "1": {"inputs": {"filename_prefix": f"{character_key}_{expr_name}", "images": ["80", 0]}, "class_type": "SaveImage"},
    }

    pid = _comfy_submit(workflow, f"{character_key}_{expr_name}")
    result = _comfy_wait(pid, timeout=200)
    if result:
        src = os.path.join(COMFY_OUTPUT, result)
        if os.path.exists(src):
            shutil.copy2(src, out_path)
            return out_path
    return None


# ═══════════════════════════════════════════════════════════════════════
# Background removal (rembg isnet-anime)
# ═══════════════════════════════════════════════════════════════════════

_REMBG_SESSION = None


def _get_rembg_session():
    """Lazily create (and cache) the isnet-anime rembg session."""
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        from rembg import new_session
        _REMBG_SESSION = new_session("isnet-anime")
    return _REMBG_SESSION


def _remove_bg(input_path: str) -> Optional[str]:
    """Remove background using rembg with the isnet-anime matte.

    Overwrites the input in-place with an RGBA PNG. Uses the rembg 2.x
    session API (`new_session("isnet-anime")` — the old `model_name=`
    kwarg no longer exists and would silently skip matting).
    """
    out_path = input_path
    try:
        from rembg import remove
        with open(input_path, "rb") as f:
            img_data = f.read()
        try:
            result = remove(img_data, session=_get_rembg_session())
        except Exception as e:
            # Last-resort fallback: default model (u2net) — better than no alpha
            print(f"  [cast] isnet-anime session failed ({e}); using default rembg model")
            result = remove(img_data)
        with open(out_path, "wb") as f:
            f.write(result)
        return out_path
    except ImportError:
        print("  [cast] rembg not available, skipping background removal")
        return input_path
    except Exception as e:
        print(f"  [cast] rembg error: {e}")
        return input_path


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

def generate_cast(preferences: Optional[dict] = None) -> list[CastMember]:
    """Generate a full cast via ComfyUI (or fall back to curated).

    This is the main entry point for the casting beat. It:
    1. Decides which 3-4 characters to generate based on preferences
    2. Generates base images (anime-face-gated) + warped expression sets
    3. Removes backgrounds for alpha compositing
    4. Returns locked CastMember objects

    Args:
        preferences: Optional dict with keys like "mood", "preferences",
            "preset", "characters".

    Returns:
        list[CastMember] with expression_map filled (frozen contract).
    """
    # Decide character keys
    if preferences and "preset" in preferences:
        preset_name = preferences["preset"]
        char_keys = _get_preset_chars(preset_name)
    elif preferences and "characters" in preferences:
        char_keys = preferences["characters"]
    else:
        char_keys = ["yuki", "haruki", "sakura"]  # Default 3-character cast

    # Try ComfyUI generation
    comfy_ok = _comfy_health()
    if comfy_ok:
        print("[cast] ComfyUI reachable — generating cast...")
        return _generate_via_comfy(char_keys)

    # Fall back to curated
    print("[cast] ComfyUI not available — using curated cast.")
    return load_curated_cast(char_keys)


def load_curated_cast(char_keys: Optional[list[str]] = None) -> list[CastMember]:
    """Load pre-made character sprites from the curated directory.

    Each character has a folder under static/sprites/curated/<name>/
    with PNG files named <name>_<expression>.png

    Args:
        char_keys: Optional list of character keys to load. Defaults to all.

    Returns:
        list[CastMember] with loaded sprite paths (frozen contract).
    """
    if char_keys is None:
        char_keys = list(PRESET_CHARACTERS.keys())

    cast: list[CastMember] = []
    for key in char_keys:
        preset = PRESET_CHARACTERS.get(key)
        if not preset:
            continue

        member = _build_member(key, preset)

        # Check curated folder
        curated_char_dir = os.path.join(CURATED_DIR, key)
        if os.path.isdir(curated_char_dir):
            for expr in STANDARD_EXPRESSIONS:
                expr_path = os.path.join(curated_char_dir, f"{key}_{expr}.png")
                if os.path.exists(expr_path):
                    member.expression_map[expr] = expr_path

        # If missing expressions, use generated directory
        generated_char_dir = os.path.join(GENERATED_DIR, key)
        if os.path.isdir(generated_char_dir):
            for expr in STANDARD_EXPRESSIONS:
                if expr not in member.expression_map:
                    expr_path = os.path.join(generated_char_dir, f"{key}_{expr}.png")
                    if os.path.exists(expr_path):
                        member.expression_map[expr] = expr_path

        # If still missing, create placeholder sprites
        for expr in STANDARD_EXPRESSIONS:
            if expr not in member.expression_map:
                placeholder = _make_placeholder_sprite(key, expr, member)
                if placeholder:
                    member.expression_map[expr] = placeholder

        cast.append(member)

    return cast


def _build_member(key: str, preset: dict) -> CastMember:
    character = Character(
        name=key.capitalize(),
        identity_prompt=preset["identity_prompt"],
        seed=preset["seed"],
        personality=preset["personality"],
        voice_id=preset["voice_id"],
        role=preset["role"],
    )
    return CastMember(character=character)


def _generate_via_comfy(char_keys: list[str]) -> list[CastMember]:
    """Generate a cast using ComfyUI. Full pipeline per character."""
    cast: list[CastMember] = []

    for key in char_keys:
        preset = PRESET_CHARACTERS.get(key)
        if not preset:
            continue

        char_dir = os.path.join(GENERATED_DIR, key)
        os.makedirs(char_dir, exist_ok=True)

        # Step 1: Generate base (anime-face-gated)
        print(f"  [cast] Generating {key} (seed={preset['seed']})...")
        base_path = _generate_base_with_face_gate(key, preset, char_dir)
        if not base_path:
            print(f"  [cast] Base gen failed for {key}, falling back to curated")
            cast.extend(load_curated_cast([key]))
            continue

        # Step 2: Warp expressions via ExpressionEditor
        print(f"  [cast] Warping expressions for {key}...")
        for expr_name in STANDARD_EXPRESSIONS:
            expr_path = _generate_expression(base_path, key, expr_name)
            status = "OK" if expr_path else "FAILED"
            print(f"    {expr_name}: {status}")

        # Step 3: Remove backgrounds (isnet-anime matte)
        print(f"  [cast] Removing backgrounds for {key}...")
        for expr_name in STANDARD_EXPRESSIONS:
            src = os.path.join(char_dir, f"{key}_{expr_name}.png")
            if os.path.exists(src):
                _remove_bg(src)

        # Step 4: Build CastMember
        member = _build_member(key, preset)
        for expr_name in STANDARD_EXPRESSIONS:
            path = os.path.join(char_dir, f"{key}_{expr_name}.png")
            if os.path.exists(path):
                member.expression_map[expr_name] = path

        # Any expression that failed gets a placeholder so the sprite-set
        # integrity invariant (every emotion present) still holds.
        for expr_name in STANDARD_EXPRESSIONS:
            if expr_name not in member.expression_map:
                placeholder = _make_placeholder_sprite(key, expr_name, member)
                if placeholder:
                    member.expression_map[expr_name] = placeholder

        cast.append(member)

    return cast


def _get_preset_chars(preset_name: str) -> list[str]:
    """Get character keys for a named cast preset."""
    presets = {
        "childhood_friends": ["yuki", "haruki", "sakura"],
        "love_triangle": ["yuki", "haruki", "ryo"],
        "girls_night": ["yuki", "sakura"],
        "full_cast": ["yuki", "haruki", "sakura", "ryo"],
        "default": ["yuki", "haruki", "sakura"],
    }
    return presets.get(preset_name, presets["default"])


def _make_placeholder_sprite(char_key: str, expr: str, member: CastMember) -> Optional[str]:
    """Create a colored placeholder sprite for missing expressions."""
    char_dir = os.path.join(GENERATED_DIR, char_key)
    os.makedirs(char_dir, exist_ok=True)
    path = os.path.join(char_dir, f"{char_key}_{expr}.png")

    try:
        from PIL import Image, ImageDraw
        # Base color per character
        colors = {
            "yuki": (200, 210, 230),
            "haruki": (180, 190, 200),
            "sakura": (230, 200, 210),
            "ryo": (190, 200, 220),
        }
        r, g, b = colors.get(char_key, (200, 200, 200))

        # Slight variation per expression
        expr_shifts = {
            "neutral": (0, 0, 0),
            "smile": (10, 10, 0),
            "laugh": (20, 15, 0),
            "sad": (0, 0, -20),
            "angry": (20, 0, 0),
            "surprised": (0, 10, 20),
            "embarrassed": (20, 0, 10),
        }
        dr, dg, db = expr_shifts.get(expr, (0, 0, 0))

        img = Image.new("RGBA", (member.sprite_width, member.sprite_height),
                        (min(255, r + dr), min(255, g + dg), min(255, b + db), 255))
        draw = ImageDraw.Draw(img)
        draw.text((member.sprite_width // 2, member.sprite_height // 2),
                  f"{char_key}\n{expr}", fill=(255, 255, 255, 255), anchor="mm")
        img.save(path)
        return path
    except ImportError:
        return None
