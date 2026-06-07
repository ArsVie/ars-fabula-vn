"""
conftest.py — Shared fixtures, markers, and test helpers for the VN.

Run: pytest -v
     pytest -m "not live" -v   (deterministic tests only)
"""
import os
import sys
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_client import MockModel, ModelConfig


# ── Markers ─────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "unit: deterministic, no server")
    config.addinivalue_line("markers", "mock: full engine via MockModel, no server")
    config.addinivalue_line("markers", "live: requires a running model server")


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def live_model():
    """Check if a real model server is available for live/eval tests."""
    cfg = ModelConfig.from_env()
    from model_client import LlamaServerClient
    client = LlamaServerClient(cfg)
    if not client.health():
        pytest.skip(f"No model server at {cfg.base_url}")
    return client


@pytest.fixture
def mock_model():
    return MockModel()


@pytest.fixture
def cast_list():
    """The cast as the pipeline returns it (frozen contract: list[CastMember])."""
    from cast_pipeline import load_curated_cast
    return load_curated_cast()


@pytest.fixture
def cast(cast_list):
    """The cast as SceneState consumes it (name -> CastMember dict)."""
    from vn_contracts import cast_to_dict
    return cast_to_dict(cast_list)


@pytest.fixture
def scene(cast):
    """Create a fresh initial scene with the curated cast."""
    from vn_contracts import SceneState
    return SceneState(
        cast=cast,
        present=[],
        background="classroom",
        mood="peaceful",
        turn=0,
        chapter="Prologue",
    )


@pytest.fixture
def engine(mock_model, scene):
    """Create a VN engine with MockModel."""
    from vn_engine import VNEngine
    return VNEngine(model=mock_model, cast=scene.cast, scene=scene)


# ── Helper: Expression set integrity check ──────────────────────────────

def assert_cast_has_expressions(cast):
    """Assert every cast member has the standard expression set."""
    from vn_contracts import STANDARD_EXPRESSIONS
    for name, member in cast.items():
        for expr in STANDARD_EXPRESSIONS:
            assert expr in member.expression_map, (
                f"{name} missing expression: {expr}"
            )
        # All paths should exist
        for expr, path in member.expression_map.items():
            assert os.path.exists(path), (
                f"{name}/{expr} sprite not found: {path}"
            )
