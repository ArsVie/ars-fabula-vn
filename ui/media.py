"""Image embedding helpers — data: URIs so sprites/backgrounds always render
without a Gradio file-route dependency.
"""
from __future__ import annotations
import base64
import io
import os
from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=256)
def _data_uri_cached(path: str, mtime: float) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def data_uri(path: Optional[str]) -> str:
    """Return a data: URI for an image path, or '' if unavailable."""
    if not path or not os.path.exists(path):
        return ""
    try:
        return _data_uri_cached(path, os.path.getmtime(path))
    except OSError:
        return ""


@lru_cache(maxsize=64)
def _thumb_uri_cached(path: str, mtime: float, height: int) -> str:
    try:
        from PIL import Image
        with Image.open(path) as img:
            img = img.convert("RGBA")
            w = max(1, int(img.width * height / img.height))
            img = img.resize((w, height), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        # Fall back to the full image (bigger payload, still works)
        return _data_uri_cached(path, mtime)


def thumb_data_uri(path: Optional[str], height: int = 160) -> str:
    """Downscaled data: URI — keeps the cast-panel payload small."""
    if not path or not os.path.exists(path):
        return ""
    try:
        return _thumb_uri_cached(path, os.path.getmtime(path), height)
    except OSError:
        return ""


# Background music lives in static/music/<track>.mp3. Unlike sprites/backgrounds
# (small, inlined as data: URIs), audio files are large, so they're served by
# Gradio's file route instead — allowed_paths=[STATIC_DIR] is set in app.py.
_MUSIC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "music")


def music_url(track: Optional[str]) -> str:
    """Return the Gradio file-route URL for a BGM track key, or '' if the file
    isn't present (so the client simply plays no music rather than erroring)."""
    if not track:
        return ""
    path = os.path.join(_MUSIC_DIR, f"{track}.mp3")
    if not os.path.exists(path):
        return ""
    return "/gradio_api/file=" + os.path.abspath(path).replace("\\", "/")
