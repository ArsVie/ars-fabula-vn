"""
test_engine_mock.py — Full VN engine loop via MockModel, no server needed.
Exercises the entire chain: choice → engine → mock model → tool parsing → scene mutation.
Run: pytest -m mock -v
"""
import json
import pytest

pytestmark = pytest.mark.mock


# ═══════════════════════════════════════════════════════════════════════
# VN Engine: Full turn loop
# ═══════════════════════════════════════════════════════════════════════

class TestVNTurnLoop:
    def test_first_turn_returns_narration(self, engine):
        """A basic turn should return some narration text."""
        narration, scene = engine.run_turn("I open my eyes to a new day.")
        assert narration
        assert len(narration) > 0
        assert scene.turn >= 1

    def test_tool_calls_parsed_in_narration(self, engine):
        """The mock model generates tool calls; they should be parsed."""
        narration, scene = engine.run_turn("I want to talk to Yuki")
        # Scene should have mutated from tool calls
        # The mock handler generates advance_scene calls, so background should be set
        assert scene.background is not None
        assert scene.mood is not None

    def test_cast_lock_cleanup(self, engine):
        """Non-cast character dialogue should be removed from narration."""
        # Create a response with a non-cast character
        from model_client import MockModel
        from vn_engine import VNEngine

        def bad_handler(messages, tools):
            return (
                '[TOOL: advance_scene description="A scene" background="classroom" mood="peaceful"]\n\n'
                '[Yuki]: "Hello!"\n'
                '[FakeChar]: "I should not exist"\n'
                '[Haruki]: "Hi Yuki!"\n'
            )

        mock = MockModel(handler=bad_handler)
        eng = VNEngine(model=mock, cast=engine.cast, scene=engine.scene)
        narration, _ = eng.run_turn("A normal choice")

        # FakeChar dialogue should be cleaned
        assert "FakeChar" not in narration or "CAST-LOCK" in narration

    def test_multi_turn_preserves_state(self, engine):
        """Multiple turns should preserve and build scene state."""
        # First turn
        n1, s1 = engine.run_turn("I look around the classroom")
        t1 = s1.turn

        # Second turn — state should persist
        n2, s2 = engine.run_turn("I talk to Yuki")
        assert s2.turn > t1

    def test_expression_changes_in_narration(self, engine):
        """Expression changes triggered by mock handler should reflect."""
        from model_client import MockModel
        from vn_engine import VNEngine

        def expr_handler(messages, tools):
            return (
                '[TOOL: set_expression character="Yuki" expression="smile"]\n\n'
                '[TOOL: advance_scene description="The sun shines." background="classroom" mood="cheerful"]\n\n'
                '[Yuki]: "Good morning!"'
            )

        mock = MockModel(handler=expr_handler)
        eng = VNEngine(model=mock, cast=engine.cast, scene=engine.scene)
        narration, scene = eng.run_turn("I say good morning")

        assert "Good morning" in narration
        assert scene.mood == "cheerful"


# ═══════════════════════════════════════════════════════════════════════
# Choices: Flow from model → engine → UI
# ═══════════════════════════════════════════════════════════════════════

class TestChoiceFlow:
    def test_choices_set_by_model(self, engine):
        """The model should be able to set choices via offer_choices tool."""
        from model_client import MockModel
        from vn_engine import VNEngine

        def choice_handler(messages, tools):
            return (
                '[TOOL: advance_scene description="A choice approaches" background="classroom" mood="peaceful"]\n\n'
                '[TOOL: offer_choices choices=\'[{"id":"a","text":"Talk to Yuki"},{"id":"b","text":"Talk to Haruki"}]\']'
            )

        mock = MockModel(handler=choice_handler)
        eng = VNEngine(model=mock, cast=engine.cast, scene=engine.scene)
        narration, scene = eng.run_turn("What should I do?")

        assert len(scene.choices) == 2
        assert scene.choices[0].text == "Talk to Yuki"
        assert scene.choices[1].text == "Talk to Haruki"

    def test_apply_choice_effects(self, engine):
        """After a choice is selected, effects should be applied."""
        from vn_contracts import Choice

        # Set up some choices first
        engine.scene.choices = [
            Choice(id="a", text="Go to rooftop", next_background="rooftop", next_mood="romantic"),
            Choice(id="b", text="Stay here"),
        ]

        # Apply choice A
        engine.apply_choice_effects(engine.scene.choices[0])
        assert engine.scene.background == "rooftop"
        assert engine.scene.mood == "romantic"

    def test_choice_affection_triggers(self, engine):
        """Choosing a romance option should trigger affection flags."""
        from vn_contracts import Choice

        choice = Choice(
            id="a", text="Confess",
            affection_flag=("Yuki", "confession"),
        )
        engine.apply_choice_effects(choice)
        assert engine.scene.has_affection("Yuki", "confession")


# ═══════════════════════════════════════════════════════════════════════
# Save/Load: Scene serialization
# ═══════════════════════════════════════════════════════════════════════

from model_client import MockModel


class _StreamMock(MockModel):
    """MockModel that emits its scripted text as many small deltas — optionally
    splitting mid-token — to exercise the engine's line-buffered reassembly."""

    def __init__(self, text: str, chunk: int = 5):
        super().__init__(responses=[text])
        self._text = text
        self._chunk = chunk

    def stream(self, messages, tools=None, **kw):
        self.calls.append({"messages": messages, "tools": tools, "kw": kw})
        t = self._text
        for i in range(0, len(t), self._chunk):
            yield t[i:i + self._chunk]


def _fresh_engine(cast, model):
    from vn_engine import VNEngine
    from vn_contracts import SceneState
    scene = SceneState(cast=cast, present=[], background="classroom",
                       mood="peaceful", turn=0, chapter="Prologue")
    return VNEngine(model=model, cast=cast, scene=scene)


class TestStreaming:
    SCRIPT = (
        '[TOOL: set_expression character="Yuki" expression="smile"]\n'
        '[TOOL: advance_scene description="The sun shines." '
        'background="classroom" mood="cheerful"]\n\n'
        '[Yuki]: "Good morning!"\n'
        'The classroom is quiet.\n'
    )

    def test_stream_final_matches_blocking(self, cast):
        """Streaming in tiny deltas must agree with the one-shot path on the
        final narration and scene mutations, and must yield progressively."""
        stream_eng = _fresh_engine(cast, _StreamMock(self.SCRIPT, chunk=5))
        block_eng = _fresh_engine(cast, MockModel(responses=[self.SCRIPT]))

        yields = list(stream_eng.run_turn_stream("hi"))
        stream_narr, stream_scene = yields[-1]
        block_narr, block_scene = block_eng.run_turn("hi")

        assert stream_narr == block_narr
        assert stream_scene.mood == block_scene.mood == "cheerful"
        assert stream_scene.expressions.get("Yuki") == "smile"
        assert len(yields) >= 2  # at least one partial + the final

    def test_multiline_tool_call_not_split(self, cast):
        """A pretty-printed multi-line tool call, streamed in 4-char chunks,
        must be reassembled into one unit — not split across newlines."""
        script = (
            "[TOOL: offer_choices choices='[\n"
            '  {"id":"a","text":"Talk to Yuki"},\n'
            '  {"id":"b","text":"Talk to Haruki"}\n'
            "]']\n"
        )
        eng = _fresh_engine(cast, _StreamMock(script, chunk=4))
        *_, (_narr, scene) = eng.run_turn_stream("choose")
        assert len(scene.choices) == 2
        assert scene.choices[0].text == "Talk to Yuki"
        assert scene.choices[1].text == "Talk to Haruki"

    def test_dialogue_recorded_via_stream(self, cast):
        """Dialogue lines are recorded into scene state during streaming."""
        eng = _fresh_engine(cast, _StreamMock(self.SCRIPT, chunk=3))
        *_, (narr, scene) = eng.run_turn_stream("hi")
        assert "Good morning" in narr
        assert any(d["speaker"] == "Yuki" and "Good morning" in d["text"]
                   for d in scene.dialogue)


class TestSaveLoad:
    def test_scene_state_roundtrip(self, scene):
        """Scene state should serialize and deserialize correctly."""
        import pickle

        # Modify scene
        scene.set_affection("Yuki", "first_meeting")
        scene.turn = 5
        scene.background = "rooftop"
        scene.mood = "romantic"

        # Roundtrip through pickle
        data = pickle.dumps(scene)
        restored = pickle.loads(data)

        assert restored.turn == 5
        assert restored.background == "rooftop"
        assert restored.mood == "romantic"
        assert restored.has_affection("Yuki", "first_meeting")

    def test_cast_persistence_across_saves(self, scene):
        """Cast should survive serialization."""
        import pickle
        data = pickle.dumps(scene)
        restored = pickle.loads(data)
        assert restored.cast.keys() == scene.cast.keys()
        for name in scene.cast:
            assert name in restored.cast
