"""
test_comfy_transport.py — the ComfyUI transport seam (http vs embed).

These are pure unit tests: no ComfyUI, no torch, no GPU, no server. They lock
in that
  • the default transport stays "http" (Docker/T4 + local dev unchanged), and
  • "embed" routes every _comfy_* helper to the in-process executor module
    (comfy_embed) instead of HTTP.

Run: pytest tests/test_comfy_transport.py -v
"""
import os
import sys
import importlib
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytestmark = pytest.mark.unit


@pytest.fixture
def cp_http(monkeypatch):
    """cast_pipeline reloaded with the default (http) transport."""
    monkeypatch.delenv("ARS_FABULA_COMFY_MODE", raising=False)
    import cast_pipeline
    return importlib.reload(cast_pipeline)


@pytest.fixture
def cp_embed(monkeypatch):
    """cast_pipeline reloaded in embed transport."""
    monkeypatch.setenv("ARS_FABULA_COMFY_MODE", "embed")
    import cast_pipeline
    cp = importlib.reload(cast_pipeline)
    yield cp
    # leave the module in its default state for other test files
    monkeypatch.delenv("ARS_FABULA_COMFY_MODE", raising=False)
    importlib.reload(cp)


def test_default_mode_is_http(cp_http):
    assert cp_http.COMFY_MODE == "http"


def test_embed_mode_set(cp_embed):
    assert cp_embed.COMFY_MODE == "embed"


def test_embed_health_routes_to_comfy_embed(cp_embed, monkeypatch):
    import comfy_embed
    monkeypatch.setattr(comfy_embed, "health", lambda: True)
    assert cp_embed._comfy_health() is True
    monkeypatch.setattr(comfy_embed, "health", lambda: False)
    assert cp_embed._comfy_health() is False


def test_embed_free_vram_is_noop(cp_embed):
    # ZeroGPU reclaims the GPU per call — nothing resident to free.
    assert cp_embed.comfy_free_vram() is False


def test_embed_run_routes_to_executor(cp_embed, monkeypatch):
    import comfy_embed
    seen = {}

    def fake_run(workflow, client_id):
        seen["wf"], seen["cid"] = workflow, client_id
        return {"filename": "y.png", "subfolder": "", "type": "output",
                "_path": "/tmp/y.png"}

    monkeypatch.setattr(comfy_embed, "run_workflow", fake_run)
    out = cp_embed._comfy_run({"1": {"class_type": "SaveImage"}}, "base_x")
    assert seen["cid"] == "base_x"
    assert out["filename"] == "y.png"


def test_embed_upload_copies_into_input_dir(cp_embed, tmp_path):
    src = tmp_path / "base.png"
    src.write_bytes(b"PNG")
    cp_embed._comfy_upload(str(src), "char_base.png")
    import comfy_embed
    dst = os.path.join(comfy_embed.input_dir(), "char_base.png")
    assert os.path.exists(dst)


def test_embed_fetch_copies_from_path(cp_embed, tmp_path):
    src = tmp_path / "out.png"
    src.write_bytes(b"DATA")
    dst = tmp_path / "fetched" / "x.png"
    res = cp_embed._comfy_fetch_image({"_path": str(src)}, str(dst))
    assert res == str(dst)
    assert dst.read_bytes() == b"DATA"


def test_embed_fetch_missing_path_returns_none(cp_embed, tmp_path):
    dst = tmp_path / "x.png"
    assert cp_embed._comfy_fetch_image({"_path": "/no/such/file"}, str(dst)) is None


# ── comfy_embed module-level helpers (no torch needed) ──────────────────

def test_comfy_embed_health_false_without_models(monkeypatch, tmp_path):
    import comfy_embed
    # Point at an empty dir → no nodes.py, no models → health is False.
    monkeypatch.setattr(comfy_embed, "COMFY_DIR", str(tmp_path))
    assert comfy_embed.health() is False


def test_output_node_ids_picks_saveimage():
    import comfy_embed
    wf = {
        "1": {"class_type": "SaveImage", "inputs": {"filename_prefix": "p"}},
        "2": {"class_type": "KSampler", "inputs": {}},
    }
    assert comfy_embed._output_node_ids(wf) == ["1"]
    assert comfy_embed._save_prefixes(wf) == ["p"]
