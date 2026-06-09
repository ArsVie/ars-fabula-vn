"""
test_trace_log.py — per-turn prompt→response trace capture for fine-tuning.

Guards two properties that protect the dataset:
  - real-model turns are logged in SFT-ready chat shape, with context;
  - mock/scripted turns are NOT logged (no test/mock-fallback pollution);
  - ARS_FABULA_TRACE=0 disables logging entirely.
"""
import glob
import json

import pytest

from model_client import MockModel
from vn_engine import VNEngine
from vn_contracts import SceneState


class _FakeReal(MockModel):
    """MockModel's scripted behavior, but presents as a real backend."""
    is_mock = False

    class cfg:
        model = "test-model"


def _scene(cast):
    return SceneState(cast=cast, present=[next(iter(cast))],
                      background="classroom", mood="peaceful",
                      turn=0, chapter="Prologue")


def _records(trace_dir):
    recs = []
    for fp in glob.glob(str(trace_dir / "*.jsonl")):
        with open(fp, encoding="utf-8") as f:
            recs.extend(json.loads(line) for line in f if line.strip())
    return recs


@pytest.mark.mock
def test_real_model_turn_is_logged(cast, tmp_path, monkeypatch):
    monkeypatch.setenv("ARS_FABULA_TRACE", "1")
    monkeypatch.setenv("ARS_FABULA_TRACE_DIR", str(tmp_path))

    speaker = next(iter(cast))
    model = _FakeReal(responses=[f'[{speaker}]: "Hello there."'])
    VNEngine(model=model, cast=cast, scene=_scene(cast)).run_turn("I wave.")

    recs = _records(tmp_path)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["model"] == "test-model"
    assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]
    assert "Hello there." in rec["messages"][-1]["content"]
    assert rec["context"]["player_input"] == "I wave."


@pytest.mark.mock
def test_mock_model_is_not_logged(cast, tmp_path, monkeypatch):
    monkeypatch.setenv("ARS_FABULA_TRACE", "1")
    monkeypatch.setenv("ARS_FABULA_TRACE_DIR", str(tmp_path))

    VNEngine(model=MockModel(responses=["hi"]), cast=cast,
             scene=_scene(cast)).run_turn("hello")

    assert _records(tmp_path) == []


@pytest.mark.mock
def test_trace_disabled_by_env(cast, tmp_path, monkeypatch):
    monkeypatch.setenv("ARS_FABULA_TRACE", "0")
    monkeypatch.setenv("ARS_FABULA_TRACE_DIR", str(tmp_path))

    model = _FakeReal(responses=["narration only"])
    VNEngine(model=model, cast=cast, scene=_scene(cast)).run_turn("I look around.")

    assert _records(tmp_path) == []
