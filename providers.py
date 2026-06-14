"""
providers.py — Swappable Background and Voice providers.

BackgroundProvider: preset library + live gen hook.
VoiceProvider: Chatterbox (emotion) + Kokoro (fast) selectable.

Both are interfaces with concrete implementations.
"""
from __future__ import annotations
import os
import subprocess
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

# ── Asset paths ──────────────────────────────────────────────────────
ASSET_DIR = os.path.join(os.path.dirname(__file__), "static")
BG_DIR = os.path.join(ASSET_DIR, "bg")


# ═══════════════════════════════════════════════════════════════════════
# Background Provider
# ═══════════════════════════════════════════════════════════════════════

# Preset background library — maps keys to image paths or generation prompts
PRESET_BACKGROUNDS: dict[str, dict] = {
    "classroom": {
        "path": "bg_classroom.png",
        "label": "High School Classroom",
        "prompt": "sunlit Japanese high school classroom, desks in rows, blackboard, afternoon light through windows, anime style",
    },
    "hallway": {
        "path": "bg_hallway.png",
        "label": "School Hallway",
        "prompt": "Japanese school hallway with lockers, shoe cubbies, afternoon sunlight, empty corridor, anime style",
    },
    "rooftop": {
        "path": "bg_rooftop.png",
        "label": "School Rooftop",
        "prompt": "school rooftop at sunset, city skyline in distance, fence, water tower, sky gradient, anime style",
    },
    "courtyard": {
        "path": "bg_courtyard.png",
        "label": "Courtyard",
        "prompt": "school courtyard with cherry blossom trees in full bloom, benches, students walking, spring, anime style",
    },
    "cafe": {
        "path": "bg_cafe.png",
        "label": "Café",
        "prompt": "cozy cafe interior, warm lighting, wooden tables, plants, window seat, afternoon, anime style",
    },
    "park": {
        "path": "bg_park.png",
        "label": "City Park",
        "prompt": "city park with green grass, trees, pond, benches, jogging path, blue sky, anime style",
    },
    "library": {
        "path": "bg_library.png",
        "label": "School Library",
        "prompt": "quiet school library, tall bookshelves, reading tables, lamp light, large windows, anime style",
    },
    "home_room": {
        "path": "bg_home_room.png",
        "label": "Character's Room",
        "prompt": "cozy Japanese bedroom, low bed, desk with laptop, posters on wall, evening light, anime style",
    },
    "gate": {
        "path": "bg_gate.png",
        "label": "School Gate",
        "prompt": "school gate at the end of the day, cherry blossom petals falling, sunset, students leaving, anime style",
    },
    "sunset_hill": {
        "path": "bg_sunset_hill.png",
        "label": "Sunset Hill",
        "prompt": "hill overlooking the city at sunset, grass, wind, two silhouettes, romantic atmosphere, anime style",
    },
}


# Background → music-track mapping. Many backgrounds deliberately share one
# track (a track key maps to static/music/<track>.mp3). A manifest — not a
# per-bg-key filename convention — lets scenes share moods for free and gives
# runtime-generated novel backgrounds a sensible fallback ("calm"). Add tracks
# by dropping <track>.mp3 into static/music/; a missing file just plays nothing.
BG_MUSIC: dict[str, str] = {
    "classroom":   "school_day",
    "hallway":     "school_day",
    "library":     "school_day",
    "courtyard":   "school_day",
    "gate":        "sunset",
    "rooftop":     "sunset",
    "sunset_hill": "sunset",
    "cafe":        "cozy",
    "home_room":   "cozy",
    "park":        "calm",
}
DEFAULT_MUSIC = "calm"


def bg_music_track(key: str) -> str:
    """Track key for a background key (falls back to the default mood)."""
    return BG_MUSIC.get(key, DEFAULT_MUSIC)


@dataclass
class BackgroundInfo:
    """Information about a background image."""
    key: str
    label: str
    path: str            # absolute path to the image file
    is_placeholder: bool = False
    music: str = DEFAULT_MUSIC   # BGM track key → static/music/<music>.mp3


class BackgroundProvider(ABC):
    """Interface for providing background images."""

    @abstractmethod
    def get_background(self, key: str) -> BackgroundInfo:
        """Get a background image by key. Returns info with path."""
        ...

    @abstractmethod
    def list_available(self) -> list[str]:
        """List all available background keys."""
        ...

    def get_prompt_for_gen(self, key: str) -> str:
        """Get the generation prompt for a key (for live gen hook)."""
        info = PRESET_BACKGROUNDS.get(key)
        return info["prompt"] if info else f"anime style background, {key}"


class PresetBackgroundProvider(BackgroundProvider):
    """Uses pre-rendered background images from static/bg/."""

    def get_background(self, key: str) -> BackgroundInfo:
        info = PRESET_BACKGROUNDS.get(key)
        if not info:
            # Fallback to first available
            key = list(PRESET_BACKGROUNDS.keys())[0]
            info = PRESET_BACKGROUNDS[key]

        path = os.path.join(BG_DIR, info["path"])
        is_placeholder = not os.path.exists(path)
        if is_placeholder:
            path = os.path.join(BG_DIR, "_placeholder.png")
            # Use a default placeholder if we have one
            if not os.path.exists(path):
                path = self._make_placeholder(key)

        return BackgroundInfo(
            key=key,
            label=info["label"],
            path=path,
            is_placeholder=is_placeholder,
            music=bg_music_track(key),
        )

    def list_available(self) -> list[str]:
        return list(PRESET_BACKGROUNDS.keys())

    def _make_placeholder(self, key: str) -> str:
        """Create a colored placeholder background."""
        colors = {
            "classroom": (200, 190, 160),
            "hallway": (180, 175, 155),
            "rooftop": (200, 180, 140),
            "courtyard": (160, 190, 140),
            "cafe": (180, 150, 120),
            "park": (140, 180, 130),
            "library": (160, 150, 130),
            "home_room": (170, 140, 130),
            "gate": (180, 170, 140),
            "sunset_hill": (200, 150, 100),
        }
        try:
            from PIL import Image, ImageDraw, ImageFont
            r, g, b = colors.get(key, (150, 150, 150))
            img = Image.new("RGB", (1024, 768), (r, g, b))
            draw = ImageDraw.Draw(img)
            draw.text((512, 384), key, fill=(255, 255, 255), anchor="mm")
            os.makedirs(BG_DIR, exist_ok=True)
            path = os.path.join(BG_DIR, "_placeholder.png")
            img.save(path)
            return path
        except ImportError:
            # Fallback — write a minimal PNG manually
            path = os.path.join(BG_DIR, "_placeholder.png")
            if not os.path.exists(path):
                self._write_minimal_png(path, r, g, b)
            return path

    def _write_minimal_png(self, path: str, r: int, g: int, b: int):
        """Write a minimal valid 1x1 PNG as absolute last resort."""
        import struct, zlib
        width, height = 1024, 768
        raw = b""
        for y in range(height):
            raw += b"\x00" + bytes([r, g, b]) * width
        compressed = zlib.compress(raw)
        def chunk(ctype, data):
            c = ctype + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(png)


class ComfyUIBackgroundProvider(PresetBackgroundProvider):
    """Extends PresetBackgroundProvider with live gen for novel keys.

    Live gen happens ONLY for keys outside the preset library — and per the
    plan's hard rule it should be invoked from beat boundaries (casting /
    pre-staging), never mid-scene-turn.
    """

    def __init__(self, comfy_host: str = "http://127.0.0.1:8188"):
        super().__init__()
        self.comfy_host = comfy_host
        self._generated_paths: dict[str, str] = {}

    def get_background(self, key: str) -> BackgroundInfo:
        if key in PRESET_BACKGROUNDS:
            return super().get_background(key)

        # Novel key — generate once, then serve from disk
        if key not in self._generated_paths:
            path = self._generate_background(key)
            if path:
                self._generated_paths[key] = path

        if key in self._generated_paths and os.path.exists(self._generated_paths[key]):
            return BackgroundInfo(
                key=key,
                label=key.replace("_", " ").title(),
                path=self._generated_paths[key],
                is_placeholder=False,
                music=bg_music_track(key),
            )

        # Generation failed — fall back to preset/placeholder behavior
        return super().get_background(key)

    def _generate_background(self, key: str) -> Optional[str]:
        """Generate a background via ComfyUI txt2img, WAIT for completion,
        and copy the result into static/bg/ so it can actually be served.

        Returns the local path of the generated image, or None on failure.
        """
        prompt_text = self.get_prompt_for_gen(key)
        try:
            # Reuse the cast pipeline's ComfyUI helpers and model names
            # (UNETLoader + Qwen CLIP/VAE — same trio as the validated
            # expression sheet; anima is a UNET-only file, so
            # CheckpointLoaderSimple would fail to load it).
            from cast_pipeline import (_comfy_run, _comfy_fetch_image,
                                       UNET_NAME, CLIP_NAME, VAE_NAME,
                                       SAMPLER, SCHEDULER, STYLE_TAGS)

            workflow = {
                "3": {"class_type": "KSampler", "inputs": {"seed": abs(hash(key)) % (2**31), "steps": 30, "cfg": 4.0, "sampler_name": SAMPLER, "scheduler": SCHEDULER, "denoise": 1.0, "model": ["44", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
                "44": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": UNET_NAME}},
                "45": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP_NAME, "type": "stable_diffusion", "device": "default"}},
                "15": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
                "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 768, "batch_size": 1}},
                "6": {"class_type": "CLIPTextEncode", "inputs": {"text": f"absurdres, {STYLE_TAGS}, {prompt_text}", "clip": ["45", 0]}},
                "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "worst quality, low quality, blurry", "clip": ["45", 0]}},
                "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["15", 0]}},
                "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"bg_{key}", "images": ["8", 0]}},
            }

            result = _comfy_run(workflow, f"bg_{key}", timeout=120)
            if not result:
                print(f"[bg] Generation failed/timed out for '{key}'")
                return None

            dst = os.path.join(BG_DIR, f"bg_{key}.png")
            if not _comfy_fetch_image(result, dst):
                return None
            print(f"[bg] Generated background '{key}' -> {dst}")
            return dst
        except Exception as e:
            print(f"[bg] Failed to generate background '{key}': {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════
# Voice Provider
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class VoiceLine:
    """A voice line ready for playback."""
    text: str
    character: str
    emotion: str           # happy, sad, angry, surprised, neutral, embarrassed
    wav_path: Optional[str] = None
    duration_ms: int = 0


class VoiceProvider(ABC):
    """Interface for generating voice lines."""

    @abstractmethod
    def speak(self, text: str, character: str, emotion: str = "neutral",
              voice_id: Optional[str] = None) -> VoiceLine:
        """Generate a voiced line. Returns a VoiceLine with path to WAV.

        Args:
            text: The line to voice.
            character: Character name (for filenames/metadata).
            emotion: Emotion tag (happy, sad, angry, surprised, neutral, embarrassed).
            voice_id: The character's locked voice (Character.voice_id).
                Providers that support per-voice synthesis MUST honor it.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class KokoroVoiceProvider(VoiceProvider):
    """Fast TTS via Kokoro (local model on port 11436 by default)."""

    def __init__(self, base_url: str = "http://localhost:11436"):
        self.base_url = base_url
        self._name = "kokoro"

    @property
    def name(self) -> str:
        return self._name

    def speak(self, text: str, character: str, emotion: str = "neutral",
              voice_id: Optional[str] = None) -> VoiceLine:
        try:
            import urllib.request
            import json as _json
            payload = _json.dumps({"text": text, "voice": voice_id or "af_bella"}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/v1/audio/speech",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            wav_dir = os.path.join(ASSET_DIR, "voice")
            os.makedirs(wav_dir, exist_ok=True)
            wav_path = os.path.join(wav_dir, f"{character}_{hash(text) % 10000}.wav")
            with open(wav_path, "wb") as f:
                f.write(data)
            return VoiceLine(text=text, character=character, emotion=emotion,
                             wav_path=wav_path, duration_ms=len(data) // 32)
        except Exception as e:
            print(f"[voice] Kokoro failed: {e}")
            return VoiceLine(text=text, character=character, emotion=emotion)


class ChatterboxVoiceProvider(VoiceProvider):
    """Emotion-rich voice via Chatterbox-Turbo."""

    def __init__(self, base_url: str = "http://localhost:11437"):
        self.base_url = base_url
        self._name = "chatterbox"

    @property
    def name(self) -> str:
        return self._name

    def speak(self, text: str, character: str, emotion: str = "neutral",
              voice_id: Optional[str] = None) -> VoiceLine:
        try:
            import urllib.request
            import json as _json
            payload = _json.dumps({
                "text": text,
                "speaker": voice_id or character,
                "emotion": emotion,
            }).encode()
            req = urllib.request.Request(
                f"{self.base_url}/tts",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            wav_dir = os.path.join(ASSET_DIR, "voice")
            os.makedirs(wav_dir, exist_ok=True)
            wav_path = os.path.join(wav_dir, f"{character}_{emotion}_{hash(text) % 10000}.wav")
            with open(wav_path, "wb") as f:
                f.write(data)
            return VoiceLine(text=text, character=character, emotion=emotion,
                             wav_path=wav_path, duration_ms=len(data) // 32)
        except Exception as e:
            print(f"[voice] Chatterbox failed: {e}")
            return VoiceLine(text=text, character=character, emotion=emotion)


class MockVoiceProvider(VoiceProvider):
    """No actual TTS — returns a voice line with just metadata (for testing)."""

    def __init__(self):
        self._name = "mock"

    @property
    def name(self) -> str:
        return self._name

    def speak(self, text: str, character: str, emotion: str = "neutral",
              voice_id: Optional[str] = None) -> VoiceLine:
        return VoiceLine(
            text=text,
            character=character,
            emotion=emotion,
            duration_ms=int(len(text) * 60),  # ~60ms per character
        )


def get_voice_provider(backend: str = "mock") -> VoiceProvider:
    """Factory: returns the appropriate VoiceProvider based on backend name.

    Args:
        backend: "mock", "kokoro", "chatterbox"

    Returns:
        VoiceProvider instance.
    """
    if backend == "kokoro":
        return KokoroVoiceProvider()
    elif backend == "chatterbox":
        return ChatterboxVoiceProvider()
    return MockVoiceProvider()


def get_background_provider(backend: str = "preset") -> BackgroundProvider:
    """Factory: returns the appropriate BackgroundProvider.

    Args:
        backend: "preset", "comfyui"

    Returns:
        BackgroundProvider instance.
    """
    if backend == "comfyui":
        return ComfyUIBackgroundProvider()
    return PresetBackgroundProvider()
