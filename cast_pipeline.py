"""
cast_pipeline.py — Generate or load character casts for the VN.

Pipeline (mirrors the VALIDATED ComfyUI workflow
`tools/workflows/anima_expression_sheet.json` — tested by hand on this
machine's ComfyUI install):
    Anima base gen (UNETLoader anima-base-v1.0 + Qwen CLIP/VAE)
      → face detect gate (face_yolov8n via Impact-Pack; retry on miss)
      → FaceDetailer per expression — only the face region is
        re-generated with the expression prompt (denoise 0.55; the sheet's
        note suggests 0.4–0.7, lower = subtler), so body/clothes/background
        stay identical and identity holds
      → rembg isnet-anime matte → alpha sprites

Two modes:
1. GENERATE (default): requires ComfyUI at localhost:8188 with the Anima
   model and Impact-Pack (UltralyticsDetectorProvider, FaceDetailer, SAM).
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
import re
from typing import Optional

from vn_contracts import Character, CastMember, STANDARD_EXPRESSIONS

# ── Constants ─────────────────────────────────────────────────────────
# Override with ARS_FABULA_COMFY if ComfyUI runs elsewhere (e.g. on the
# Windows side of WSL: start it with `--listen 0.0.0.0` and point this at
# the Windows host IP).
COMFY_HOST = os.getenv("ARS_FABULA_COMFY", "http://127.0.0.1:8188").rstrip("/")
COMFY_OUTPUT = os.path.expanduser("~/comfy/ComfyUI/output")  # legacy; results now fetched via /view

# Transport for ComfyUI: "http" talks to a running ComfyUI server (Docker/T4
# Space + local dev — unchanged). "embed" drives ComfyUI's PromptExecutor
# in-process inside an @spaces.GPU call (ZeroGPU Gradio Space, where a
# subprocess server can't get a GPU). Same workflow dicts either way — only
# the four _comfy_* helpers below branch on this. See comfy_embed.py.
COMFY_MODE = os.getenv("ARS_FABULA_COMFY_MODE", "http").lower()
PROJECT_DIR = os.path.dirname(__file__)
CURATED_DIR = os.path.join(PROJECT_DIR, "static", "sprites", "curated")
GENERATED_DIR = os.path.join(PROJECT_DIR, "static", "sprites", "generated")

# Anima-tuned (per the June 2026 ADT/LDG synthesis): pony/score_ quality
# tags DEGRADE Anima output; `absurdres` is the only quality tag that helps.
# `safe` is the SFW content rating. (Override via env if you re-tune.)
QUALITY = os.getenv("ARS_FABULA_QUALITY", "absurdres")
NEGATIVE = os.getenv("ARS_FABULA_NEGATIVE",
                     "worst quality, low quality, blurry, jpeg artifacts, "
                     "sepia, bad anatomy, ugly, deformed")

# Anima sampler/scheduler. dpm_2_ancestral won the A/B sweep on this
# install (tools/sampler_sweep.py — best subjective quality). Env-overridable
# to retest others without code edits, e.g.:
#   ARS_FABULA_SAMPLER=flow_euler python app.py
SAMPLER = os.getenv("ARS_FABULA_SAMPLER", "dpm_2_ancestral")
SCHEDULER = os.getenv("ARS_FABULA_SCHEDULER", "simple")

# Fixed generation seed — every base + expression starts here, so the whole
# cast is reproducible run-to-run. Env-overridable. The face-gate / expression
# retries still bump it locally (+attempt) to self-heal a bad draw.
SEED = int(os.getenv("ARS_FABULA_SEED", "42"))

# Shared style block — prepended to EVERY prompt (cast + backgrounds) so the
# whole game shares one look. Generic style descriptors by default; add
# `@artistname` danbooru artist tags here (or via env) to narrow further.
STYLE_TAGS = os.getenv(
    "ARS_FABULA_STYLE",
    "official art, light novel illustration style, clean lineart, "
    "soft cel shading, pastel color palette, soft lighting, "
    "shiny detailed eyes, smooth shading",
)

# Full-body base canvas (taller than the old 512x768). We generate the whole
# body for CONSISTENT framing, then crop the upper portion — full-body
# composition lands the head at a consistent place/scale across characters,
# which upper-body framing did not.
BASE_WIDTH, BASE_HEIGHT = 512, 896

# Fast generation: bake the base at a LOWER RESOLUTION (same 4:7 aspect) for a
# much quicker cast. Also uses lighter FaceDetailer parameters (fewer steps,
# smaller guide/max size, euler sampler) so both the base gen and expression
# pass are faster. Toggled per run from the UI via set_fast_generation(); fast
# dims are env-overridable.
FAST_GENERATION = False
FAST_BASE_WIDTH = int(os.getenv("ARS_FABULA_FAST_WIDTH", "384"))
FAST_BASE_HEIGHT = int(os.getenv("ARS_FABULA_FAST_HEIGHT", "672"))

# Expression FaceDetailer params are now UNIFIED across all modes (fast and
# standard). The only mode difference is base canvas resolution — see
# _base_dims(). These lighter params (euler, 15 steps, guide_size=256) come
# from the user's validated expression sheet; the old heavier config
# (dpm_2_ancestral, 20 steps, guide_size=384) added cost with no quality gain.


def set_fast_generation(enabled: bool) -> None:
    """Enable/disable the lower-resolution fast bake for subsequent runs."""
    global FAST_GENERATION
    FAST_GENERATION = bool(enabled)


def _base_dims() -> tuple[int, int]:
    """Base canvas size for the current mode (fast = lower resolution)."""
    if FAST_GENERATION:
        return FAST_BASE_WIDTH, FAST_BASE_HEIGHT
    return BASE_WIDTH, BASE_HEIGHT


def _char_dir(key: str) -> str:
    """Per-character generated-sprite folder for the current mode. Fast (low-
    res) bakes use a separate '<key>_fast' folder so they never collide with —
    or get silently reused as — the full-resolution set, and vice versa."""
    return os.path.join(GENERATED_DIR, f"{key}_fast" if FAST_GENERATION else key)


# Fraction of the (matted) image height kept for the on-screen sprite,
# measured DOWN from the head crown. ~0.62 = head + torso (waist-up).
UPPER_BODY_FRAC = float(os.getenv("ARS_FABULA_SPRITE_FRAC", "0.62"))

# Face-anchored ZOOMED framing — equal faces, cropped legs, real height gaps.
# Using the face-gate mask, every sprite is scaled so the detected FACE is the
# SAME fraction of the SCENE height (FACE_TARGET_FRAC), then composited so the
# FEET sit FEET_DROP_FACES face-heights BELOW the canvas bottom (the lower legs
# crop off-screen — the zoom). Equal faces + a shared feet baseline mean a
# character drawn taller (smaller face ⇒ more face-heights of body) still has
# their head reach higher in the frame, so implied heights survive the zoom.
# All sprites share one pixel size, so the UI's uniform height-lock scales them
# identically. Falls back to the crude alpha-crown crop when no mask exists.
#
# GEOMETRY LIMIT: the visible span is (figure-height-in-faces − FEET_DROP_FACES)
# × FACE_TARGET_FRAC of the scene. The current cast measures 7.6–8.3 faces tall.
# A bigger face fraction = a closer (more "bust-framed") shot; the crown stays
# on-screen only while  (tallest_faces − FEET_DROP_FACES) × FACE_CANVAS_FRAC ≤ 1
# where FACE_CANVAS_FRAC = FACE_TARGET_FRAC / UI_SPRITE_FRAC (face ÷ CANVAS).
#
# Tuned for FACE_CANVAS_FRAC ≈ 0.20 (faces a fifth of the canvas height — bigger
# than the old 0.13/0.86≈0.15, which read as small, distant figures whose short
# legs made big-headed art look stumpy/"midget", but NOT a tight bust: at 0.20
# the sprites stay narrow enough that three fit side-by-side under the UI's
# max-width:34% cap without overlapping. Feet drop −3.3 faces (an upper-body
# crop) so the tallest (8.3-face) crown still clears the top:
#   FACE_TARGET_FRAC 0.172 (÷0.86 ⇒ 0.20 of canvas);  (8.3−3.3)×0.20 = 1.0 ≤ 1.
FACE_TARGET_FRAC = float(os.getenv("ARS_FABULA_FACE_TARGET", "0.172"))  # face height ÷ SCENE height
FEET_DROP_FACES = float(os.getenv("ARS_FABULA_FEET_DROP", "3.3"))      # feet this many face-heights below the bottom
UI_SPRITE_FRAC = 0.86  # .vn-sprite { height: 86% } in ui/styles.py — canvas→scene factor
SPRITE_CANVAS_MULT = float(os.getenv("ARS_FABULA_SPRITE_CANVAS", "1.4"))  # output canvas size ÷ base size (headroom for tall figures)
# Minimum visible figure height (crown→canvas-bottom) as a fraction of the
# canvas. Faces are normalised equal, so a SHORT character (few face-heights)
# would otherwise sit low and small in frame — the "midget" read. When a
# figure's crown would fall below this floor we scale that character up (its
# face exceeds the equal-face target) so every cast member fills the frame to at
# least this height. Set ≥ the tallest natural crown to disable the boost.
MIN_FIGURE_CANVAS_FRAC = float(os.getenv("ARS_FABULA_MIN_FIGURE", "0.82"))  # crown height ÷ CANVAS height

# ── Sprite pose block (researched danbooru framing/pose tags) ─────────
# Locked onto EVERY identity prompt — but ONLY the essentials: framing,
# eye contact, and a mattable background. Body attitude (slight 3/4
# angle, hand on hip, arms crossed…) is character design and belongs in
# the appearance tags; VN sprites are rarely dead-frontal.
#   upper body        — waist-up framing (the VN sprite standard;
#                       cowboy_shot = mid-thigh if taller sprites wanted)
#   standing, looking at viewer — sprite stance + player eye contact
#                       (eye contact holds even at a slight body angle)
#   solo              — exactly one character
#   simple/white background — clean rembg matte
# Hard angle failures (profile, from behind/above/below) are handled by
# SPRITE_NEGATIVE instead of over-constraining the positive prompt.
# NOTE: NO weighted tags — (tag:weight) emphasis is inconsistent on Anima,
# so everything here is plain comma-separated tags.
SPRITE_POSE_TAGS = ("solo, full body, standing, full length, "
                    "looking at viewer, calm expression, "
                    "simple background, white background")

# Framing failure modes, negated for BASE generation only (the FaceDetailer
# pass keeps the plain NEGATIVE — these would fight it). We now WANT full
# body, so the old full-body/legs/cowboy negatives are gone; we still kill
# wrong camera angles, multi-character, and Anima's "coomer bias"
# (exaggerated proportions, ADT #106) so the school cast stays normal.
SPRITE_NEGATIVE = (f"{NEGATIVE}, close-up, portrait, profile, from behind, "
                   "from above, from below, dutch angle, multiple views, "
                   "multiple girls, multiple boys, 2girls, 2boys, "
                   "text, watermark, signature, out of frame")


def _spaces_not_underscores(tags: str) -> str:
    """Underscores break Anima's tag parser — normalize to spaces.
    (The LLM designer can emit canonical danbooru tags like
    'looking_at_viewer'; convert them.)"""
    return tags.replace("_", " ")


def build_identity_prompt(appearance_tags: str) -> str:
    """Quality + style + the character's APPEARANCE tags + the locked pose.

    appearance_tags should describe only the character (1girl/1boy, hair,
    eyes, build, clothing, accessories, aura) — pose/framing/background
    come from SPRITE_POSE_TAGS and must not be duplicated or contradicted.
    """
    tags = _spaces_not_underscores(appearance_tags.strip().rstrip(","))
    return f"{QUALITY}, {STYLE_TAGS}. {tags}, {SPRITE_POSE_TAGS}"


def _face_anchor(identity_or_tags: str) -> str:
    """Pull the must-preserve face traits (eye + hair colour) out of a
    character's tags, so the FaceDetailer keeps them across expressions.

    Returns e.g. "red eyes, silver hair" (either part may be missing)."""
    parts = []
    m = re.search(r'([a-z]+(?:\s[a-z]+)?)\s+eyes', identity_or_tags.lower())
    if m:
        parts.append(f"{m.group(1).strip()} eyes")
    m = re.search(r'([a-z]+(?:\s[a-z]+)?)\s+hair', identity_or_tags.lower())
    if m:
        parts.append(f"{m.group(1).strip()} hair")
    return ", ".join(parts)

# Face detector (Impact-Pack UltralyticsDetectorProvider model) — taken
# verbatim from the validated expression sheet, which confirmed it detects
# Anima-style faces on this install. If detection ever misses, try an
# anime-specific model (e.g. yolov8x6_animeface.pt) here.
ANIME_FACE_MODEL = "bbox/face_yolov8n.pt"

# How many times to regenerate the base (seed bump) if no anime face is found
FACE_GATE_RETRIES = 2

# ── Preset character templates ────────────────────────────────────────

PRESET_CHARACTERS: dict[str, dict] = {
    "yuki": {
        "identity_prompt": build_identity_prompt(
            "1girl, yuki tanaka, long silver hair, straight hair, "
            "red eyes, fair skin, flat chest, high school uniform, white shirt, "
            "blue skirt, red ribbon, gentle expression, quiet reserved aura, "
            "soft lighting"
        ),
        "personality": "Quiet and reserved but a deeply caring girl. Speaks in short sentences. "
                       "Notices small details about others. Opens up slowly. "
                       "Has a dry sense of humor that surprises people.",
        "voice_id": "af_bella",
        "role": "classmate",
        "seed": SEED,
    },
    "marie": {
        "identity_prompt": build_identity_prompt(
            "1girl, marie, long blonde hair, drill hair, blue eyes, "
            "fair skin, elegant, high school uniform, white shirt, "
            "blue skirt, red ribbon, refined posture, graceful aura, "
            "proud expression, ojou-sama"
        ),
        "personality": "A refined, well-bred ojou-sama from old money. Proud and "
                       "elegant, with a signature 'ohohoho' laugh behind a raised "
                       "hand. Competitive and a touch haughty, but genuinely caring "
                       "underneath — she'd never admit how much she enjoys company.",
        "voice_id": "af_sarah",
        "role": "ojou-sama",
        "seed": SEED,
    },
    "sakura": {
        "identity_prompt": build_identity_prompt(
            "1girl, long pink hair, side ponytail, "
            "green eyes, fair skin, freckles, high school uniform, white shirt, "
            "pink bow, cheerful genki energy, bubbly"
        ),
        "personality": "Cheerful and energetic genki girl. Always smiling. "
                       "Best friend energy. Gives great hugs. "
                       "Perceptive despite her bubbly exterior.",
        "voice_id": "af_sky",
        "role": "best friend",
        "seed": SEED,
    },
    "ryo": {
        "identity_prompt": build_identity_prompt(
            "1boy, ryo akira, short blue hair, sharp eyes, "
            "tall build, high school uniform, blazer, "
            "cool aura, sharp gaze, composed"
        ),
        "personality": "Cool and aloof on the surface. Secretly a huge romantic. "
                       "Gives surprisingly good advice. "
                       "Has a hidden soft spot for cute things.",
        "voice_id": "am_adam",      # Kokoro male voice
        "role": "senpai",
        "seed": SEED,
    },
}

# Expression prompts fed to the FaceDetailer. Each is engineered to read
# DISTINCTLY from the others along three axes — eyes, mouth, brows — plus a
# signature effect, so the 7 don't blur together:
#   eyes:  open / soft / >_< squint / wide+shrunk / averted
#   mouth: closed / slight / open grin / frown / wavy
#   extra: blush / tears / anger vein / sweatdrops
# NOTE on closed eyes: the face detector runs on the open-eyed BASE image,
# THEN inpaints the expression — so closed/squint eyes in the prompt do NOT
# break detection (earlier worry was wrong). Plain tags only (no weights).
EXPRESSION_TAGS = {
    # "neutral" is the DEFAULT sprite (get_sprite falls back here) AND a callable
    # emotion — deadpan/stoic characters need a true flat affect. We avoid the bare
    # token "neutral expression" (Anima reads it as grumpy → angry/sad drift) and use
    # "expressionless" with relaxed brows + closed mouth so it stays flat, not cross.
    "neutral": "expressionless, calm, looking at viewer, relaxed, closed mouth, level eyebrows",
    # SMILE = Passive Contentment (Eyes wide and calm, subtle lip curve)
    "smile": "gentle smile, parted lips, pleasant expression, calm eyes, looking at viewer, serene",
    # LAUGH = Dynamic Joy (Eyes naturally squinting/crinkling from a wide, open-mouthed grin)
    "laugh": "joyful laughter, wide open mouth, laughing expression, squinting eyes, crinkling eyes, big happy grin, radiant smile, elevated cheeks",
    "sad": "sad expression, deep frown, downturned mouth, worried eyebrows, downcast eyes, casting eyes downward, distressed",
    "angry": "angry glare, slanting v-shaped eyebrows, furrowed brow, gritting teeth, tight jaw, intense eyes, looking at viewer",
    "surprised": "surprised expression, wide-eyed, shocked, shrunken pupils, slightly open gasping mouth, raised elevated eyebrows",
    "embarrassed": "embarrassed, heavy blush, (@_@), spiral eyes, averted eyes, flying sweatdrops, nervous",
}

# Per-expression NEGATIVES, appended to the plain NEGATIVE for the
# FaceDetailer pass. Each kills that expression's observed failure mode —
# drifting back toward the base's calm face (laugh rendering as a mild
# closed-mouth smile, sad as neutral-pleasant, surprised as calm) — without
# constraining the rest of the set.
EXPRESSION_NEGATIVE_TAGS = {
    "laugh": "closed mouth, expressionless, neutral expression, frown, slight smile",
    "sad": "smile, grin, laughing, happy, cheerful, expressionless",
    "surprised": "smile, expressionless, calm, half-closed eyes, closed eyes, closed mouth",
    "angry": "smile, happy, expressionless, calm",
}

# FaceDetailer strength — the sheet's note: 0.4–0.7, lower = subtler.
EXPRESSION_DENOISE = 0.6
# The big-delta expressions need a stronger inpaint to actually commit
# (at 0.6 they often kept too much of the base's calm face — the source of
# inconsistent laugh/sad/surprised sets); 0.7 is the top of the sheet's
# recommended range, identity still held by the anchor + face-only mask.
EXPRESSION_DENOISE_OVERRIDES = {
    "laugh": 0.7,
    "sad": 0.68,
    "surprised": 0.7,
}

# Model files (must match the names installed in ComfyUI — taken verbatim
# from the validated workflow).
UNET_NAME = "anima-base-v1.0-Q5_K_M.gguf"
CLIP_NAME = "qwen_3_06b_base.safetensors"
VAE_NAME = "qwen_image_vae.safetensors"
SAM_NAME = "sam_vit_b_01ec64.pth"


# ═══════════════════════════════════════════════════════════════════════
# Character design — concept phrase -> appearance tags (LLM or heuristic)
# ═══════════════════════════════════════════════════════════════════════

FEMALE_VOICES = ["af_bella", "af_sky", "af_nicole", "af_sarah"]
MALE_VOICES = ["am_michael", "am_adam"]

# Keyword fallbacks for when no LLM is reachable — coarse but coherent.
_CONCEPT_DESIGNS = [
    (("childhood friend",), "girl",
     "1girl, brown hair, medium hair, ahoge, warm brown eyes, fair skin, "
     "school uniform, white shirt, red ribbon, hands behind back, "
     "cheerful aura, approachable", "childhood friend"),
    (("genki", "energetic", "cheerful"), "girl",
     "1girl, orange hair, short hair, bright eyes, fang, school uniform, "
     "loose socks, hand on hip, energetic aura, sporty", "genki classmate"),
    (("shy", "bookworm", "quiet", "library"), "girl",
     "1girl, dark blue hair, long hair, blunt bangs, glasses, purple eyes, "
     "cardigan, school uniform, holding book, head tilt, timid aura",
     "shy bookworm"),
    (("cool", "aloof", "senpai", "stoic"), "boy",
     "1boy, black hair, short hair, sharp grey eyes, tall, school uniform, "
     "blazer, hands in pockets, composed, cool aura", "cool senpai"),
    (("delinquent", "rebel", "punk"), "boy",
     "1boy, blonde hair, slicked back hair, sharp eyes, open collar, "
     "school uniform, untucked shirt, piercing, hands in pockets, "
     "rough aura", "delinquent"),
    (("class rep", "president", "strict"), "girl",
     "1girl, black hair, long hair, hime cut, serious eyes, school uniform, "
     "armband, own hands together, neat, dignified aura",
     "class representative"),
    (("gyaru", "gal",), "girl",
     "1girl, blonde hair, long wavy hair, tan skin, nail polish, "
     "school uniform, loose ribbon, accessories, hand on hip, "
     "confident aura", "gyaru"),
    (("athlete", "sporty", "club"), "boy",
     "1boy, brown hair, very short hair, tan skin, athletic build, "
     "school uniform, sports towel, arms crossed, energetic aura",
     "athlete"),
]

_DEFAULT_DESIGN = ("girl",
                   "1girl, {hair} hair, expressive eyes, fair skin, "
                   "school uniform, white shirt",
                   "classmate")
_HAIR_ROTATION = ["chestnut", "ash blonde", "teal", "auburn", "lavender", "black"]


def _stable_seed(*parts: str) -> int:
    import zlib
    return zlib.crc32("|".join(parts).encode("utf-8")) & 0x7FFFFFFF


def _heuristic_design(name: str, concept: str) -> dict:
    """No-LLM fallback: map concept keywords to a coherent design."""
    low = (concept or "").lower()
    if any(k in low for k in ("boy", "male", "guy", "him", "prince", "butler")):
        forced_gender = "boy"
    elif any(k in low for k in ("girl", "female", "her", "princess", "maid")):
        forced_gender = "girl"
    else:
        forced_gender = None

    for keywords, gender, tags, role in _CONCEPT_DESIGNS:
        if any(k in low for k in keywords):
            if forced_gender and forced_gender != gender:
                tags = tags.replace("1girl", "1boy") if forced_gender == "boy" \
                    else tags.replace("1boy", "1girl")
                gender = forced_gender
            return {"gender": gender, "tags": tags, "role": role,
                    "personality": concept or f"{name} — {role}."}

    gender, tags, role = _DEFAULT_DESIGN
    gender = forced_gender or gender
    hair = _HAIR_ROTATION[_stable_seed(name, concept) % len(_HAIR_ROTATION)]
    tags = tags.format(hair=hair)
    if gender == "boy":
        tags = tags.replace("1girl", "1boy")
    return {"gender": gender, "tags": tags, "role": role,
            "personality": concept or f"{name}, a {role}."}


def design_character(name: str, concept: str) -> dict:
    """Concept phrase -> {gender, tags, personality, role}.

    Uses the story LLM when reachable (CHARACTER_DESIGNER_PROMPT,
    strict-JSON reply); falls back to the keyword heuristics so the
    casting beat never blocks on a missing server.
    """
    try:
        from model_client import get_model, ModelConfig
        from vn_prompt import CHARACTER_DESIGNER_PROMPT
        cfg = ModelConfig.from_env()
        if cfg.backend != "mock":
            client = get_model(cfg)
            if client.health():
                raw = client.generate([
                    {"role": "system", "content": CHARACTER_DESIGNER_PROMPT},
                    {"role": "user", "content": f"Name: {name}\nConcept: {concept}"},
                ]) or ""
                start, end = raw.find("{"), raw.rfind("}")
                if start != -1 and end > start:
                    data = json.loads(raw[start:end + 1])
                    tags = str(data.get("tags", "")).strip()
                    if tags and ("1girl" in tags or "1boy" in tags):
                        return {
                            "gender": "boy" if "1boy" in tags else "girl",
                            "tags": tags,
                            "personality": str(data.get("personality", concept)).strip()
                                           or concept,
                            "role": str(data.get("role", "classmate")).strip()
                                    or "classmate",
                        }
                print(f"  [cast] LLM design for '{name}' unusable — using heuristics")
    except Exception as e:
        print(f"  [cast] LLM design failed for '{name}' ({e}) — using heuristics")
    return _heuristic_design(name, concept)


def build_custom_specs(cards: list[dict]) -> list[tuple[str, dict]]:
    """Turn UI character cards into (cache_key, preset) specs.

    Card fields: name (required), concept (regular mode), tags (advanced
    mode — overrides the designer; pose block still appended).
    The cache key embeds a hash of the final prompt, so editing a
    character's design automatically invalidates their sprite cache.
    """
    import hashlib
    specs: list[tuple[str, dict]] = []
    voice_idx = {"girl": 0, "boy": 0}

    for i, card in enumerate(cards):
        name = (card.get("name") or "").strip()
        if not name:
            continue
        concept = (card.get("concept") or "").strip()
        adv_tags = (card.get("tags") or "").strip()

        if adv_tags:   # advanced mode: tags are authoritative
            gender = "boy" if "1boy" in adv_tags else "girl"
            design = {"gender": gender, "tags": adv_tags,
                      "personality": concept or f"{name}.",
                      "role": concept.split(",")[0][:24] if concept else "custom"}
        else:
            design = design_character(name, concept)

        pool = MALE_VOICES if design["gender"] == "boy" else FEMALE_VOICES
        voice = pool[voice_idx[design["gender"]] % len(pool)]
        voice_idx[design["gender"]] += 1

        identity_prompt = build_identity_prompt(design["tags"])
        slug = "".join(ch for ch in name.lower() if ch.isalnum()) or f"char{i}"
        key = f"{slug}_{hashlib.md5(identity_prompt.encode()).hexdigest()[:8]}"

        specs.append((key, {
            "name": name,
            "identity_prompt": identity_prompt,
            "personality": design["personality"],
            "voice_id": voice,
            "role": design["role"],
            "seed": SEED,
        }))
    return specs


# ═══════════════════════════════════════════════════════════════════════
# ComfyUI helpers
# ═══════════════════════════════════════════════════════════════════════

_COMFY_SESSION = None


def _session():
    """Shared requests session for ComfyUI.

    trust_env=False is the important part: it makes requests IGNORE
    http_proxy/https_proxy environment variables (common on corporate
    machines), which otherwise hijack even 127.0.0.1 traffic and kill
    POSTs with [Errno 32] Broken pipe while GETs appear to work.
    """
    global _COMFY_SESSION
    if _COMFY_SESSION is None:
        import requests
        s = requests.Session()
        s.trust_env = False
        _COMFY_SESSION = s
    return _COMFY_SESSION


def _comfy_upload(path: str, name: str) -> dict:
    """Upload an image to ComfyUI's input folder.

    Uses `requests` with a proper multipart encoder — the previous
    hand-rolled multipart body could make aiohttp drop the connection
    mid-write (the "broken pipe"). `overwrite=true` matters: without it,
    re-uploading the same filename makes ComfyUI save a renamed copy and
    the workflow's LoadImage silently keeps reading the OLD image.
    """
    if COMFY_MODE == "embed":
        import comfy_embed
        dst = os.path.join(comfy_embed.input_dir(), name)
        shutil.copyfile(path, dst)   # overwrite, like overwrite=true above
        return {"name": name}
    with open(path, "rb") as f:
        resp = _session().post(
            f"{COMFY_HOST}/upload/image",
            files={"image": (name, f, "image/png")},
            data={"overwrite": "true"},
            timeout=60,
        )
    if not resp.ok:
        raise RuntimeError(
            f"ComfyUI upload of '{name}' failed: HTTP {resp.status_code} — {resp.text[:300]}")
    return resp.json()


def _comfy_submit(workflow: dict, client_id: str) -> str:
    """Submit a workflow to ComfyUI and return the prompt_id.

    On a 400, surfaces ComfyUI's validation error (bad node class, missing
    model file, wrong input name) instead of dying with a transport error.
    """
    resp = _session().post(
        f"{COMFY_HOST}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=60,
    )
    if not resp.ok:
        raise RuntimeError(
            f"ComfyUI rejected workflow '{client_id}': HTTP {resp.status_code} — {resp.text[:1000]}")
    return resp.json()["prompt_id"]


def _comfy_wait(prompt_id: str, timeout: int = 180) -> Optional[dict]:
    """Wait for a ComfyUI prompt to finish.

    Returns the first output image's info dict ({filename, subfolder, type})
    or None. Doesn't rely on the 'completed' status flag (absent in some
    ComfyUI versions) — the presence of outputs is the ground truth — and
    prints execution errors from the history so failures aren't silent.
    """
    for i in range(timeout // 2):
        time.sleep(2)
        try:
            hist = _session().get(f"{COMFY_HOST}/history/{prompt_id}", timeout=15).json()
        except Exception:
            continue
        if prompt_id not in hist:
            continue
        entry = hist[prompt_id]

        # Outputs present → done, regardless of what status says
        for nout in (entry.get("outputs") or {}).values():
            for img in nout.get("images", []):
                return img

        # Surface execution errors instead of timing out silently
        status = entry.get("status") or {}
        for msg in status.get("messages", []):
            if isinstance(msg, (list, tuple)) and msg and msg[0] == "execution_error":
                detail = msg[1] if len(msg) > 1 else {}
                print(f"  [cast] ComfyUI execution error in node "
                      f"{detail.get('node_type', '?')}: "
                      f"{detail.get('exception_message', detail)}")
                return None
        if status.get("status_str") == "error":
            print(f"  [cast] ComfyUI reported an error for prompt {prompt_id}: {status}")
            return None
        if status.get("completed"):
            return None   # finished with no image outputs
    print(f"  [cast] Timed out waiting for prompt {prompt_id}")
    return None


def _comfy_run(workflow: dict, client_id: str,
               timeout: int = 180) -> Optional[dict]:
    """Run a workflow and return its first-image info dict, via whichever
    transport COMFY_MODE selects. Unifies the http submit+wait pair and the
    in-process executor so the call sites are transport-agnostic."""
    if COMFY_MODE == "embed":
        import comfy_embed
        return comfy_embed.run_workflow(workflow, client_id)
    pid = _comfy_submit(workflow, client_id)
    return _comfy_wait(pid, timeout=timeout)


def _comfy_fetch_image(img_info: dict, dest_path: str) -> Optional[str]:
    """Download a generated image from ComfyUI via its /view endpoint.

    This replaces copying from a guessed output directory on disk — which
    breaks whenever ComfyUI lives somewhere else (or SaveImage used a
    subfolder). Works regardless of where/how ComfyUI is installed.
    """
    # Embed transport already wrote the file to ComfyUI's output dir on disk;
    # just copy it to the destination the caller expects (no HTTP /view).
    if COMFY_MODE == "embed":
        src = img_info.get("_path")
        if not src or not os.path.exists(src):
            print(f"  [cast] embed output missing for {img_info}")
            return None
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copyfile(src, dest_path)
        return dest_path
    try:
        resp = _session().get(
            f"{COMFY_HOST}/view",
            params={
                "filename": img_info.get("filename", ""),
                "subfolder": img_info.get("subfolder", ""),
                "type": img_info.get("type", "output"),
            },
            timeout=60,
        )
        if not resp.ok:
            print(f"  [cast] /view fetch failed: HTTP {resp.status_code} for {img_info}")
            return None
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(resp.content)
        return dest_path
    except Exception as e:
        print(f"  [cast] /view fetch failed for {img_info}: {e}")
        return None


def _comfy_health() -> bool:
    """Check if ComfyUI is reachable (proxy-bypassing, like all other calls)."""
    if COMFY_MODE == "embed":
        import comfy_embed
        return comfy_embed.health()
    try:
        return _session().get(f"{COMFY_HOST}/system_stats", timeout=3).ok
    except Exception:
        return False


def comfy_free_vram() -> bool:
    """Unload ComfyUI's models from VRAM via its /free endpoint.

    Called after a cast bake so the scene LLM has the GPU to itself.
    ComfyUI reloads its models lazily on the next /prompt, so this is safe
    to call between casting beats. No-op if ComfyUI isn't reachable.
    """
    # Embed mode: ZeroGPU grants/reclaims the GPU per @spaces.GPU call, so
    # there is no resident server holding VRAM to free between beats.
    if COMFY_MODE == "embed":
        return False
    try:
        r = _session().post(f"{COMFY_HOST}/free",
                            json={"unload_models": True, "free_memory": True},
                            timeout=20)
        if r.ok:
            print("[vram] ComfyUI models unloaded (/free)")
            return True
    except Exception as e:
        print(f"[vram] ComfyUI /free failed (ignored): {e}")
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
    base_w, base_h = _base_dims()

    workflow = {
        "1": {"inputs": {"filename_prefix": f"base_{character_key}", "images": ["8", 0]}, "class_type": "SaveImage"},
        "8": {"inputs": {"samples": ["19", 0], "vae": ["15", 0]}, "class_type": "VAEDecode"},
        "11": {"inputs": {"text": preset["identity_prompt"], "clip": ["45", 0]}, "class_type": "CLIPTextEncode"},
        "12": {"inputs": {"text": SPRITE_NEGATIVE, "clip": ["45", 0]}, "class_type": "CLIPTextEncode"},
        "15": {"inputs": {"vae_name": VAE_NAME}, "class_type": "VAELoader"},
        "19": {"inputs": {
            "seed": seed, "steps": 30, "cfg": 4.0,
            "sampler_name": SAMPLER, "scheduler": SCHEDULER, "denoise": 1.0,
            "model": ["44", 0], "positive": ["11", 0], "negative": ["12", 0],
            "latent_image": ["28", 0],
        }, "class_type": "KSampler"},
        "28": {"inputs": {"width": base_w, "height": base_h, "batch_size": 1}, "class_type": "EmptyLatentImage"},
        "44": {"inputs": {"unet_name": UNET_NAME}, "class_type": "UnetLoaderGGUF"},
        "45": {"inputs": {"clip_name": CLIP_NAME, "type": "stable_diffusion", "device": "default"}, "class_type": "CLIPLoader"},
    }

    result = _comfy_run(workflow, f"base_{character_key}")
    if result and _comfy_fetch_image(result, out_path):
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
    try:
        _comfy_upload(base_path, f"{character_key}_base.png")
    except Exception as e:
        print(f"  [cast] Face gate upload failed ({e}); proceeding without gate")
        return None

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
        result = _comfy_run(workflow, f"facegate_{character_key}", timeout=60)
    except Exception as e:
        print(f"  [cast] Face gate unavailable ({e}); proceeding without gate")
        return None

    if not result:
        print(f"  [cast] Face gate produced no output; proceeding without gate")
        return None

    mask_path = os.path.join(_char_dir(character_key), "_facegate_mask.png")
    if not _comfy_fetch_image(result, mask_path):
        print(f"  [cast] Face gate mask fetch failed; proceeding without gate")
        return None
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
# Expression generation via FaceDetailer (the validated expression sheet)
# ═══════════════════════════════════════════════════════════════════════

def _generate_expression(base_path: str, character_key: str, expr_name: str,
                         expr_tags: str, seed: int, anchor: str = "") -> Optional[str]:
    """Generate a single expression with the validated FaceDetailer sheet
    (tools/workflows/anima_expression_sheet.json, API form).

    Only the detected face region is re-generated with the expression
    prompt — body, clothes and background stay identical, which is what
    preserves identity across the set. `anchor` (eye/hair colour) is folded
    into the prompt so those traits don't drift across expressions.

    Returns path to the expression PNG, or None on failure.
    """
    out_dir = _char_dir(character_key)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{character_key}_{expr_name}.png")

    # neutral == the base itself, just copied through (base prompts are
    # already neutral-faced; copying beats re-generating for identity)
    if expr_name == "neutral":
        shutil.copy2(base_path, out_path)
        return out_path

    _comfy_upload(base_path, f"{character_key}_base.png")

    anchor_part = f"{anchor}, " if anchor else ""
    expr_prompt = f"{QUALITY}, {STYLE_TAGS}. {anchor_part}{expr_tags}"

    # Per-expression strength + negatives (see EXPRESSION_DENOISE_OVERRIDES /
    # EXPRESSION_NEGATIVE_TAGS): big-delta expressions inpaint harder and
    # negate their drift-back-to-calm failure mode.
    denoise = EXPRESSION_DENOISE_OVERRIDES.get(expr_name, EXPRESSION_DENOISE)
    expr_negative = NEGATIVE
    if expr_name in EXPRESSION_NEGATIVE_TAGS:
        expr_negative = f"{NEGATIVE}, {EXPRESSION_NEGATIVE_TAGS[expr_name]}"

    # FaceDetailer inputs — unified across all modes (see the module-level
    # comment above; only base resolution differs per mode).
    # Unified FaceDetailer params from the validated expression sheet.
    # Lighter than the old heavier config (guide_size 384→256, steps 20→15,
    # dpm_2_ancestral→euler) with no quality loss. bbox_threshold 0.5 catches
    # expressions reliably (was 0.3 with more dilation/crop to compensate).
    _fd_inputs = {
        "image": ["10", 0], "model": ["44", 0], "clip": ["45", 0], "vae": ["15", 0],
        "guide_size": 256, "guide_size_for": True, "max_size": 384,
        "seed": seed, "steps": 15, "cfg": 4.0,
        "sampler_name": "euler", "scheduler": SCHEDULER,
        "positive": ["11", 0], "negative": ["12", 0],
        "denoise": denoise, "feather": 15,
        "noise_mask": True, "force_inpaint": True,
        "bbox_threshold": 0.5, "bbox_dilation": 25, "bbox_crop_factor": 2.0,
        "sam_detection_hint": "center-1", "sam_dilation": 0,
        "sam_threshold": 0.93, "sam_bbox_expansion": 0,
        "sam_mask_hint_threshold": 0.7, "sam_mask_hint_use_negative": "False",
        "drop_size": 10, "wildcard": "", "cycle": 1,
        "noise_mask_feather": 20,
        "bbox_detector": ["70", 0], "sam_model_opt": ["73", 0],
    }

    workflow = {
        "10": {"inputs": {"image": f"{character_key}_base.png"}, "class_type": "LoadImage"},
        "11": {"inputs": {"text": expr_prompt, "clip": ["45", 0]}, "class_type": "CLIPTextEncode"},
        "12": {"inputs": {"text": expr_negative, "clip": ["45", 0]}, "class_type": "CLIPTextEncode"},
        "15": {"inputs": {"vae_name": VAE_NAME}, "class_type": "VAELoader"},
        "44": {"inputs": {"unet_name": UNET_NAME}, "class_type": "UnetLoaderGGUF"},
        "45": {"inputs": {"clip_name": CLIP_NAME, "type": "stable_diffusion", "device": "default"}, "class_type": "CLIPLoader"},
        "70": {"inputs": {"model_name": ANIME_FACE_MODEL}, "class_type": "UltralyticsDetectorProvider"},
        "73": {"inputs": {"model_name": SAM_NAME, "device_mode": "AUTO"}, "class_type": "SAMLoader"},
        "80": {"inputs": _fd_inputs, "class_type": "FaceDetailer"},
        "1": {"inputs": {"filename_prefix": f"{character_key}_{expr_name}", "images": ["80", 0]}, "class_type": "SaveImage"},
    }

    result = _comfy_run(workflow, f"{character_key}_{expr_name}_{seed}",
                        timeout=200)
    if result and _comfy_fetch_image(result, out_path):
        return out_path
    return None


# Per-expression retry count on failure (bumped seed). 0 = no retry.
EXPRESSION_RETRIES = 1


def _generate_expression_retry(base_path: str, character_key: str,
                               expr_name: str, expr_tags: str,
                               seed: int, anchor: str = "") -> Optional[str]:
    """Generate one expression, retrying with a bumped seed if it fails
    (transient face-detect miss / empty output). Self-heals the occasional
    'laugh didn't generate' without re-running the whole cast."""
    for attempt in range(EXPRESSION_RETRIES + 1):
        path = _generate_expression(base_path, character_key, expr_name,
                                    expr_tags, seed + attempt * 1000, anchor=anchor)
        if path:
            return path
        if attempt < EXPRESSION_RETRIES:
            print(f"    {expr_name}: empty/failed — retrying with bumped seed")
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


_REMBG_WARNED = False


def _warn_rembg_missing(detail: str) -> None:
    """One-time, prominent, actionable warning — silent failure here is why
    sprites came out with white backgrounds."""
    global _REMBG_WARNED
    if _REMBG_WARNED:
        return
    _REMBG_WARNED = True
    print("\n" + "!" * 64)
    print("  [cast] BACKGROUND REMOVAL DID NOT RUN — sprites will keep their")
    print(f"         white background. Reason: {detail}")
    print("         Install it INTO THIS venv (rembg + an onnxruntime):")
    print("           .venv/bin/pip install rembg onnxruntime")
    print("         (use onnxruntime-gpu instead for CUDA). The isnet-anime")
    print("         model at ~/.u2net/isnet-anime.pth is already present.")
    print("!" * 64 + "\n")


def _has_transparency(path: str) -> bool:
    try:
        from PIL import Image
        with Image.open(path) as img:
            if img.mode not in ("RGBA", "LA", "PA"):
                return False
            alpha = img.convert("RGBA").getchannel("A").getextrema()
        return alpha[0] < 250   # some pixel is meaningfully transparent
    except Exception:
        return False


def _remove_bg(input_path: str) -> Optional[str]:
    """Remove background using rembg with the isnet-anime matte.

    Overwrites the input in-place with an RGBA PNG. Uses the rembg 2.x
    session API (`new_session("isnet-anime")` — the old `model_name=`
    kwarg no longer exists and would silently skip matting). Verifies the
    output actually became transparent; warns loudly if not.
    """
    out_path = input_path
    try:
        from rembg import remove
    except ImportError:
        _warn_rembg_missing("`import rembg` failed (not installed in this venv)")
        return input_path

    try:
        with open(input_path, "rb") as f:
            img_data = f.read()
        try:
            result = remove(img_data, session=_get_rembg_session())
        except Exception as e:
            print(f"  [cast] isnet-anime session failed ({e}); trying default rembg model")
            result = remove(img_data)
        with open(out_path, "wb") as f:
            f.write(result)
        if not _has_transparency(out_path):
            _warn_rembg_missing("rembg ran but produced no transparency "
                                "(onnxruntime missing/broken, or model failed to load)")
        return out_path
    except Exception as e:
        _warn_rembg_missing(f"rembg error: {e}")
        return input_path


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

def _frame_upper_body(char_dir: str, key: str,
                      frac: Optional[float] = None, pad: int = 12) -> None:
    """Crop a character's full-body renders to a consistent waist-up sprite.

    Vertical: anchored on the head crown (union alpha-bbox TOP) and a FIXED
    height (frac × image height) down from it — so the head sits at the same
    place and the framing is identical across the whole cast.
    Horizontal: the character's union extent + padding (trims side
    whitespace; widths may differ per character, which is fine — the UI
    height-locks sprites).
    The SAME box is applied to all 7 expressions so they don't jump on swap.
    """
    try:
        from PIL import Image
    except ImportError:
        return
    frac = UPPER_BODY_FRAC if frac is None else frac

    paths = [os.path.join(char_dir, f"{key}_{e}.png") for e in STANDARD_EXPRESSIONS]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        return

    # Union alpha bbox across the set → stable crown + horizontal extent
    union = None
    H = W = 0
    for p in paths:
        with Image.open(p) as img:
            rgba = img.convert("RGBA")
            W, H = rgba.size
            bbox = rgba.getchannel("A").getbbox()
        if bbox:
            union = bbox if union is None else (
                min(union[0], bbox[0]), min(union[1], bbox[1]),
                max(union[2], bbox[2]), max(union[3], bbox[3]),
            )
    if union is None:
        return

    # Detected face height (base px), from the face-gate mask if present.
    face_h = None
    mask_path = os.path.join(char_dir, "_facegate_mask.png")
    if os.path.exists(mask_path):
        try:
            with Image.open(mask_path) as m:
                ml = m.convert("L")
                _, mH = ml.size
                fb = ml.getbbox()           # bbox of the white (detected-face) region
            if fb and mH:
                face_h = (fb[3] - fb[1]) * (H / mH)   # scale mask→sprite if res differs
        except Exception as e:
            print(f"  [cast] face mask unusable for {key} ({e}); crude crop")

    if face_h and face_h > 1:
        # FACE-NORMALIZED ZOOMED bake: every face = FACE_TARGET_FRAC of the
        # SCENE height (the UI shows the canvas at UI_SPRITE_FRAC of the
        # screen, so the canvas target is FACE_TARGET_FRAC / UI_SPRITE_FRAC),
        # and the FEET sit FEET_DROP_FACES face-heights BELOW the canvas
        # bottom (lower legs crop off — the zoom-in). Equal faces + shared
        # feet baseline ⇒ taller figures' heads still reach higher.
        H0 = int(H * SPRITE_CANVAS_MULT)
        face_px = H0 * FACE_TARGET_FRAC / UI_SPRITE_FRAC   # target face height on canvas
        scale = face_px / face_h
        # ── Min-figure-height floor (anti-"midget") ──────────────────────
        # Feet are pinned FEET_DROP_FACES below the bottom, so the crown lands
        # at  (figure_h × scale − FEET_DROP × face_px)  above the bottom. A
        # SHORT figure clears too little of the canvas at the equal-face scale
        # and reads as a small body with a big head. If its crown would fall
        # below MIN_FIGURE_CANVAS_FRAC of the canvas, scale that character UP
        # (face exceeds the equal target — a deliberate trade) so its crown
        # reaches the floor and it fills the frame like the taller cast.
        fig_h = union[3] - union[1]
        crown_frac = (fig_h * scale - FEET_DROP_FACES * face_px) / H0
        if crown_frac < MIN_FIGURE_CANVAS_FRAC and fig_h > 1:
            scale = (MIN_FIGURE_CANVAS_FRAC * H0 + FEET_DROP_FACES * face_px) / fig_h
            print(f"  [cast] {key}: short figure ({crown_frac:.0%} crown) boosted "
                  f"to the {MIN_FIGURE_CANVAS_FRAC:.0%} min-height floor")
        # Canvas WIDTH grows to fit the scaled figure (wide poses — drill hair,
        # full skirts — must never hit the vertical edges; a hard clip line
        # mid-screen looks broken). Heights stay uniform across the cast, which
        # is what keeps faces equal under the UI height-lock; widths may vary.
        fig_w = (union[2] - union[0]) * scale
        W0 = max(int(W * SPRITE_CANVAS_MULT), int(fig_w) + 1 + 2 * pad)
        cx = (union[0] + union[2]) / 2.0               # figure center-x (base px)
        off_x = int(round(W0 / 2.0 - cx * scale))      # center horizontally
        # union bottom (feet) lands FEET_DROP_FACES faces below the canvas edge
        off_y = int(round(H0 + FEET_DROP_FACES * face_px - union[3] * scale))
        sw, sh = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
        crown_clip = -(off_y + union[1] * scale)       # px of head above the canvas top
        for p in paths:
            with Image.open(p) as img:
                resized = img.convert("RGBA").resize((sw, sh), Image.LANCZOS)
            canvas = Image.new("RGBA", (W0, H0), (0, 0, 0, 0))
            canvas.paste(resized, (off_x, off_y), resized)   # mask = alpha; clips overflow
            canvas.save(p)
        print(f"  [cast] Framed {len(paths)} sprites for {key} "
              f"(face-normalized zoom, scale={scale:.2f}, face={face_h:.0f}px → "
              f"{FACE_TARGET_FRAC:.0%} of scene, feet at -{FEET_DROP_FACES:g} faces, {W0}x{H0})")
        if crown_clip > 1:
            print(f"  [cast] WARNING: {key}'s crown overflows the canvas top by "
                  f"{crown_clip:.0f}px — lower ARS_FABULA_FACE_TARGET or ARS_FABULA_FEET_DROP")
        return

    # ── Fallback: crude alpha-crown + fixed-fraction crop (no usable mask) ──
    crown_y = max(0, union[1] - pad)
    crop_h = int(H * frac)
    bottom = min(H, crown_y + crop_h)
    left = max(0, union[0] - pad)
    right = min(W, union[2] + pad)
    box = (left, crown_y, right, bottom)
    for p in paths:
        with Image.open(p) as img:
            img.convert("RGBA").crop(box).save(p)
    print(f"  [cast] Framed {len(paths)} sprites for {key} (alpha-crown crop) to {box}")


# Back-compat alias: tools/recrop_sprites.py calls _autocrop_sprite_set.
_autocrop_sprite_set = _frame_upper_body


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
    # Decide the cast specs: custom cards take precedence over presets.
    # VRAM choreography (LLM and diffusion can't co-reside):
    #   custom mode needs the design LLM first, so free ComfyUI to make room
    #   for it; preset mode designs nothing and skips that step.
    if preferences and preferences.get("custom_cast"):
        comfy_free_vram()                       # room for the design LLM
        try:                                    # bring the LLM up if a prior bake killed it
            from model_client import ensure_llm
            ensure_llm(timeout=300)
        except Exception as e:
            print(f"[cast] ensure_llm before design failed ({e}); designing with heuristics")
        specs = build_custom_specs(preferences["custom_cast"])   # LLM designs
        if not specs:
            specs = [(k, PRESET_CHARACTERS[k]) for k in ["yuki", "marie", "sakura"]]
    else:
        if preferences and "preset" in preferences:
            char_keys = _get_preset_chars(preferences["preset"])
        elif preferences and "characters" in preferences:
            char_keys = preferences["characters"]
        else:
            char_keys = ["yuki", "marie", "sakura"]  # Default 3-character cast
        specs = [(k, PRESET_CHARACTERS[k]) for k in char_keys
                 if k in PRESET_CHARACTERS]

    # Try ComfyUI generation
    comfy_ok = _comfy_health()
    if comfy_ok:
        # Only a real bake loads diffusion models — a fully-cached reuse
        # doesn't, so don't needlessly evict the LLM in that case.
        will_bake = bool(os.getenv("ARS_FABULA_REGEN")) or \
            not all(_generated_complete(k) for k, _ in specs)
        if will_bake:
            # Evict the LLM before ComfyUI loads its models — covers both the
            # design LLM just used (custom mode) and one left resident from a
            # prior scene (Reset → regenerate).
            from model_client import release_vram
            release_vram()
        print("[cast] ComfyUI reachable — generating cast...")
        return _generate_via_comfy(specs)

    # ComfyUI down — previously baked sprites beat placeholders
    if all(_generated_complete(k) for k, _ in specs):
        print("[cast] ComfyUI not available — using previously generated sprites.")
        return [_load_generated_member(k, p) for k, p in specs]

    # Fall back to curated (presets) / placeholders (custom)
    print("[cast] ComfyUI not available — using curated/placeholder cast.")
    out: list[CastMember] = []
    for k, p in specs:
        if k in PRESET_CHARACTERS:
            out.extend(load_curated_cast([k]))
        else:
            out.append(_placeholder_member(k, p))
    return out


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
        name=preset.get("name", key.capitalize()),
        identity_prompt=preset["identity_prompt"],
        seed=preset["seed"],
        personality=preset["personality"],
        voice_id=preset["voice_id"],
        role=preset["role"],
    )
    return CastMember(character=character)


def _placeholder_member(key: str, preset: dict) -> CastMember:
    """A custom character with placeholder sprites (no ComfyUI, no curated art)."""
    member = _build_member(key, preset)
    for expr in STANDARD_EXPRESSIONS:
        placeholder = _make_placeholder_sprite(key, expr, member)
        if placeholder:
            member.expression_map[expr] = placeholder
    return member


def _generate_via_comfy(specs: list[tuple[str, dict]]) -> list[CastMember]:
    """Generate a cast using ComfyUI. Full pipeline per character.

    Any HTTP/ComfyUI failure for one character (broken pipe, rejected
    workflow, timeout) falls back to curated sprites (presets) or
    placeholders (custom characters) instead of crashing the batch.
    """
    cast: list[CastMember] = []

    for key, preset in specs:
        try:
            cast.append(_generate_one_via_comfy(key, preset))
        except Exception as e:
            print(f"  [cast] ComfyUI pipeline failed for {key}: {e}")
            if key in PRESET_CHARACTERS:
                print(f"  [cast] Falling back to curated sprites for {key}")
                cast.extend(load_curated_cast([key]))
            else:
                print(f"  [cast] Falling back to placeholder sprites for {key}")
                cast.append(_placeholder_member(key, preset))

    return cast


def _generated_complete(key: str) -> bool:
    """True if a full generated sprite set already exists on disk
    (for the current mode — fast and full are cached separately)."""
    char_dir = _char_dir(key)
    return all(os.path.exists(os.path.join(char_dir, f"{key}_{e}.png"))
               for e in STANDARD_EXPRESSIONS)


def _load_generated_member(key: str, preset: dict) -> CastMember:
    """Build a CastMember from already-generated sprites on disk."""
    char_dir = _char_dir(key)
    member = _build_member(key, preset)
    for expr in STANDARD_EXPRESSIONS:
        member.expression_map[expr] = os.path.join(char_dir, f"{key}_{expr}.png")
    return member


def _generate_one_via_comfy(key: str, preset: dict, base_path: str = None) -> CastMember:
    """Full ComfyUI pipeline for one character. Raises on hard failure.

    Cached: if the character's full sprite set already exists under
    static/sprites/generated/<key>/, it is reused — the bake happens once,
    not on every casting beat. Force a re-bake by deleting that folder or
    running with ARS_FABULA_REGEN=1.

    When *base_path* is given, step 1 (base generation) is skipped — the
    caller already ran the face-gated base gen and now only needs
    expressions baked from it. The base is copied into the char dir if
    it isn't already there.
    """
    if _generated_complete(key) and not os.getenv("ARS_FABULA_REGEN"):
        print(f"  [cast] Reusing generated sprites for {key} "
              f"(delete {_char_dir(key)}/ or set ARS_FABULA_REGEN=1 to re-bake)")
        return _load_generated_member(key, preset)

    char_dir = _char_dir(key)
    os.makedirs(char_dir, exist_ok=True)

    # Step 1: Generate base (face-gated) — skipped if base_path provided and exists
    if base_path and os.path.exists(base_path):
        dest = os.path.join(char_dir, f"{key}_base.png")
        if os.path.abspath(base_path) != os.path.abspath(dest):
            shutil.copy2(base_path, dest)
        print(f"  [cast] Using supplied base for {key}: {base_path}")
    else:
        print(f"  [cast] Generating {key} (seed={preset['seed']})...")
        base_path = _generate_base_with_face_gate(key, preset, char_dir)
        if not base_path:
            raise RuntimeError("base generation produced no image")

    # Step 2: Generate expressions via the FaceDetailer sheet.
    # The eye/hair anchor keeps those traits from drifting across the set.
    anchor = _face_anchor(preset.get("identity_prompt", ""))
    print(f"  [cast] Generating expressions for {key} (anchor: {anchor or 'none'})...")
    for expr_name in STANDARD_EXPRESSIONS:
        tags = EXPRESSION_TAGS[expr_name]
        expr_path = _generate_expression_retry(base_path, key, expr_name,
                                               tags, preset["seed"], anchor=anchor)
        status = "OK" if expr_path else "FAILED"
        print(f"    {expr_name}: {status}")

    # Step 3: Remove backgrounds (isnet-anime matte)
    print(f"  [cast] Removing backgrounds for {key}...")
    for expr_name in STANDARD_EXPRESSIONS:
        src = os.path.join(char_dir, f"{key}_{expr_name}.png")
        if os.path.exists(src):
            _remove_bg(src)

    # Step 3.5: Crop full-body renders to a consistent waist-up sprite,
    # anchored on the head crown so the whole cast lines up.
    _frame_upper_body(char_dir, key)

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

    return member


def _get_preset_chars(preset_name: str) -> list[str]:
    """Get character keys for a named cast preset."""
    presets = {
        "childhood_friends": ["yuki", "marie", "sakura"],
        "love_triangle": ["yuki", "marie", "ryo"],
        "girls_night": ["yuki", "marie", "sakura"],  # was yuki+sakura; Marie joins the girls
        "full_cast": ["yuki", "marie", "sakura", "ryo"],
        "default": ["yuki", "marie", "sakura"],
    }
    return presets.get(preset_name, presets["default"])


def _make_placeholder_sprite(char_key: str, expr: str, member: CastMember) -> Optional[str]:
    """Create a colored placeholder sprite for missing expressions."""
    char_dir = _char_dir(char_key)
    os.makedirs(char_dir, exist_ok=True)
    path = os.path.join(char_dir, f"{char_key}_{expr}.png")

    try:
        from PIL import Image, ImageDraw
        # Base color per character
        colors = {
            "yuki": (200, 210, 230),
            "marie": (235, 225, 180),
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
