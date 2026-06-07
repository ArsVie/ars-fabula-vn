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
