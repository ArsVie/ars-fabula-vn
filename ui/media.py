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
