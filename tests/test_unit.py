"""
test_unit.py — Deterministic unit tests for the VN contracts and engine.
No server needed. Run: pytest -m unit -v
"""
import os
import pytest

pytestmark = pytest.mark.unit


# ═══════════════════════════════════════════════════════════════════════
# Contracts: SceneState
# ═══════════════════════════════════════════════════════════════════════

class TestSceneState:
    def test_initial_state(self, scene):
        assert scene.turn == 0
        assert scene.background == "classroom"
        assert scene.mood == "peaceful"
        assert scene.chapter == "Prologue"
        assert len(scene.cast) >= 2

    def test_affection_flags(self, scene):
        scene.set_affection("Yuki", "first_meeting")
        flags = scene.active_affection("Yuki")
        assert "first_meeting" in flags

        scene.set_affection("Haruki", "confided_in")
        assert "confided_in" in scene.active_affection("Haruki")

        # Check Yuki's unaffected
        assert "confided_in" not in scene.active_affection("Yuki")

    def test_has_affection(self, scene):
        scene.set_affection("Yuki", "gave_gift")
        assert scene.has_affection("Yuki", "gave_gift")
        assert not scene.has_affection("Yuki", "nonexistent")
        assert not scene.has_affection("Haruki", "gave_gift")

    def test_present_characters(self, scene):
        assert len(scene.present) == 0  # present starts empty; characters enter via tool
        # Adding a character to present
        scene.present.append("Yuki")
        assert "Yuki" in scene.present


# ═══════════════════════════════════════════════════════════════════════
# Contracts: CastMember
# ═══════════════════════════════════════════════════════════════════════

class TestCastMember:
    def test_expression_map_completeness(self, cast):
        from vn_contracts import STANDARD_EXPRESSIONS
        for name, member in cast.items():
            for expr in STANDARD_EXPRESSIONS:
                assert expr in member.expression_map, (
                    f"{name} missing expression: {expr}"
                )

    def test_get_sprite_fallback(self, cast):
        for member in cast.values():
            # Getting a known expression should work
            neutral_path = member.get_sprite("neutral")
            assert neutral_path is not None

            # Getting an unknown expression should fall back to neutral
            fallback = member.get_sprite("nonexistent_expression")
            assert fallback is not None

    def test_character_immutability(self, cast):
        for name, member in cast.items():
            char = member.character
            assert isinstance(char.name, str) and char.name
            assert isinstance(char.role, str) and char.role
            assert isinstance(char.personality, str) and char.personality


# ═══════════════════════════════════════════════════════════════════════
# Engine: Tool parsing and routing
# ═══════════════════════════════════════════════════════════════════════

class TestToolParsing:
    def test_parse_advance_scene(self):
        from vn_engine import parse_tool_calls
        text = '[TOOL: advance_scene description="A sunny day" background="rooftop" mood="cheerful"]'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["tool"] == "advance_scene"
        assert calls[0]["description"] == "A sunny day"
        assert calls[0]["background"] == "rooftop"
        assert calls[0]["mood"] == "cheerful"

    def test_parse_set_expression(self):
        from vn_engine import parse_tool_calls
        text = '[TOOL: set_expression character="Yuki" expression="smile"]'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["tool"] == "set_expression"
        assert calls[0]["character"] == "Yuki"
        assert calls[0]["expression"] == "smile"

    def test_parse_offer_choices(self):
        from vn_engine import parse_tool_calls
        text = '[TOOL: offer_choices choices=\'[{"id":"a","text":"Go left"}]\']'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["tool"] == "offer_choices"

    def test_parse_save_checkpoint(self):
        from vn_engine import parse_tool_calls
        text = '[TOOL: save_checkpoint]'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["tool"] == "save_checkpoint"

    def test_parse_multiple_tools(self):
        from vn_engine import parse_tool_calls
        text = (
            'Some narration...\n\n'
            '[TOOL: set_expression character="Yuki" expression="surprised"]\n\n'
            'More text...\n\n'
            '[TOOL: advance_scene description="Afternoon" background="courtyard" mood="peaceful"]\n\n'
            'And then...\n\n'
            '[TOOL: offer_choices choices=\'[{"id":"a","text":"Talk"}]\']'
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 3
        assert calls[0]["tool"] == "set_expression"
        assert calls[1]["tool"] == "advance_scene"
        assert calls[2]["tool"] == "offer_choices"

    def test_no_tools_returns_empty(self):
        from vn_engine import parse_tool_calls
        text = "Just some regular narration without any tool calls."
        calls = parse_tool_calls(text)
        assert len(calls) == 0

    def test_parse_choices_with_apostrophes(self):
        """Regression: apostrophes in choice text ("I've") must not
        truncate the single-quoted JSON argument (broke story start)."""
        import json as _json
        from vn_engine import parse_tool_calls
        text = (
            "[TOOL: offer_choices choices='["
            '{"id":"yes","text":"\\"Sure, sounds good!\\"","consequence":"go"},'
            '{"id":"no","text":"\\"Maybe tomorrow? I\'ve got stuff to do.\\"","consequence":"decline"},'
            '{"id":"tease","text":"\\"Look who\'s getting serious~\\"","consequence":"tease"}'
            "]']"
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        choices = _json.loads(calls[0]["choices"])
        assert len(choices) == 3
        assert "I've" in choices[1]["text"]
        assert "who's" in choices[2]["text"]

    def test_parse_apostrophe_in_double_quoted_arg(self):
        from vn_engine import parse_tool_calls
        text = '[TOOL: advance_scene description="It\'s a beautiful day [truly]" background="park" mood="peaceful"]'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["description"] == "It's a beautiful day [truly]"
        assert calls[0]["background"] == "park"

    def test_offer_choices_with_apostrophes_routes_ok(self, scene):
        """End-to-end: the exact failure — offer_choices must succeed."""
        from vn_engine import parse_tool_calls, _handle_offer_choices
        text = (
            "[TOOL: offer_choices choices='["
            '{"id":"a","text":"\\"I\'ve been waiting.\\""},'
            '{"id":"b","text":"Don\'t say anything"}'
            "]']"
        )
        calls = parse_tool_calls(text)
        result = _handle_offer_choices(calls[0], scene)
        assert result.success, result.message
        assert len(scene.choices) == 2
        assert scene.choices[1].text == "Don't say anything"

    def test_parse_apostrophe_in_plain_single_quoted_arg(self):
        """Regression: a bare apostrophe inside a plain single-quoted value
        (character='O'Brien') must not truncate the value. A quote closes
        the value only when followed by the next key or end-of-args."""
        from vn_engine import parse_tool_calls
        text = "[TOOL: set_expression character='O'Brien' expression='smile']"
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["character"] == "O'Brien"
        assert calls[0]["expression"] == "smile"

    def test_parse_apostrophe_in_trailing_single_quoted_arg(self):
        """The apostrophe-bearing value is the last argument (no key after)."""
        from vn_engine import parse_tool_calls
        text = "[TOOL: advance_scene background='park' mood='it's complicated']"
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["background"] == "park"
        assert calls[0]["mood"] == "it's complicated"

    def test_parse_bare_unquoted_json_value(self):
        """Robustness: a weak model may omit the single-quote wrapper around a
        JSON value (choices=[{...}] instead of choices='[{...}]'). The bare
        array must still be captured whole, not dropped."""
        from vn_engine import parse_tool_calls
        text = '[TOOL: offer_choices choices=[{"id":"a","text":"Go"}, {"id":"b","text":"Stay"}]]'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["tool"] == "offer_choices"
        assert calls[0].get("choices", "").startswith("[")
        assert '"id"' in calls[0]["choices"]

    def test_offer_choices_lenient_loose_json(self, scene):
        """Robustness regression (real-model output): bare array, unquoted keys
        and numeric ids — offer_choices must still parse, and ids coerce to str
        so they match the player's click."""
        from vn_engine import parse_tool_calls, _handle_offer_choices
        text = ('[TOOL: offer_choices choices=[{id: 1, text: "Cafe.", consequence: "x"}, '
                '{id: 2, text: "Park.", consequence: "y"}]]')
        calls = parse_tool_calls(text)
        result = _handle_offer_choices(calls[0], scene)
        assert result.success, result.message
        assert len(scene.choices) == 2
        assert scene.choices[0].id == "1"   # coerced int -> str
        assert scene.choices[1].text == "Park."

    def test_loads_lenient_preserves_colon_in_values(self):
        """The bare-key repair must not corrupt colons inside string values."""
        from vn_engine import _loads_lenient
        data = _loads_lenient('[{id: 1, text: "meet at 3:30 pm: rooftop"}]')
        assert data[0]["text"] == "meet at 3:30 pm: rooftop"


# ═══════════════════════════════════════════════════════════════════════
# Engine: Cast-lock enforcement
# ═══════════════════════════════════════════════════════════════════════

class TestCastLock:
    def test_check_valid_character(self, cast):
        from vn_engine import _check_cast_lock
        for name in cast:
            assert _check_cast_lock(name, cast) == name

    def test_check_case_insensitive(self, cast):
        from vn_engine import _check_cast_lock
        for name in cast:
            assert _check_cast_lock(name.lower(), cast) == name
            assert _check_cast_lock(name.upper(), cast) == name

    def test_check_nonexistent_character(self, cast):
        from vn_engine import _check_cast_lock
        assert _check_cast_lock("Totoro", cast) is None
        assert _check_cast_lock("RandomNPC", cast) is None

    def test_clean_noncast_dialogue(self, cast):
        from vn_engine import _clean_noncast_dialogue
        text = (
            '[Yuki]: "Hello!"\n'
            '[Totoro]: "I should not be here"\n'
            '[Sakura]: "Hey!"\n'
        )
        cleaned = _clean_noncast_dialogue(text, cast)
        assert "[Yuki]:" in cleaned
        assert "[Sakura]:" in cleaned
        assert "[Totoro]:" not in cleaned
        # Totoro's line should be replaced
        assert "someone speaks" in cleaned.lower()

    def test_repair_cast_violation_flags(self, cast):
        from vn_engine import _repair_cast_violation
        text = '[Yuki]: "Hi"\n[RandomNPC]: "Hello"'
        result = _repair_cast_violation(text, cast)
        assert "CAST-LOCK VIOLATION" in result
        assert "RandomNPC" in result


# ═══════════════════════════════════════════════════════════════════════
# Engine: Scene mutation via tool routing
# ═══════════════════════════════════════════════════════════════════════

class TestToolRouting:
    def test_advance_scene_mutation(self, scene):
        from vn_engine import _handle_advance_scene
        tool = {"tool": "advance_scene", "description": "Sunset scene",
                "background": "rooftop", "mood": "romantic"}
        result = _handle_advance_scene(tool, scene)
        assert result.success
        assert scene.background == "rooftop"
        assert scene.mood == "romantic"
        assert "Sunset" in scene.description

    def test_set_expression_valid(self, scene):
        from vn_engine import _handle_set_expression
        tool = {"tool": "set_expression", "character": "Yuki", "expression": "smile"}
        result = _handle_set_expression(tool, scene, scene.cast)
        assert result.success

    def test_set_expression_invalid_character(self, scene):
        from vn_engine import _handle_set_expression
        tool = {"tool": "set_expression", "character": "Nobody", "expression": "smile"}
        result = _handle_set_expression(tool, scene, scene.cast)
        assert not result.success
        assert "Cast-lock violation" in result.message

    def test_offer_choices_sets_state(self, scene):
        from vn_engine import _handle_offer_choices
        choices_json = '[{"id":"a","text":"Talk to Yuki"},{"id":"b","text":"Leave"}]'
        tool = {"tool": "offer_choices", "choices": choices_json}
        result = _handle_offer_choices(tool, scene)
        assert result.success
        assert len(scene.choices) == 2
        assert scene.choices[0].id == "a"
        assert scene.choices[0].text == "Talk to Yuki"

    def test_apply_choice_effects(self, scene):
        from vn_contracts import Choice
        choice = Choice(
            id="a", text="Confess",
            next_background="rooftop",
            next_mood="romantic",
            affection_flag=("Yuki", "confession"),
        )
        from vn_engine import VNEngine
        engine = VNEngine(model=None, cast=scene.cast, scene=scene)
        engine.apply_choice_effects(choice)
        assert scene.background == "rooftop"
        assert scene.mood == "romantic"
        assert scene.has_affection("Yuki", "confession")


# ═══════════════════════════════════════════════════════════════════════
# Cast Pipeline: Expression set integrity
# ═══════════════════════════════════════════════════════════════════════

class TestCastPipeline:
    def test_curated_cast_has_all_expressions(self, cast):
        """Every curated cast member must have all standard expressions."""
        from vn_contracts import STANDARD_EXPRESSIONS
        for name, member in cast.items():
            for expr in STANDARD_EXPRESSIONS:
                assert expr in member.expression_map, (
                    f"{name} missing expression: {expr}"
                )

    def test_curated_cast_sprite_paths_exist(self, cast):
        """All expression sprites must exist on disk (even if placeholder)."""
        for name, member in cast.items():
            for expr, path in member.expression_map.items():
                assert os.path.exists(path), (
                    f"{name}/{expr} sprite not found at: {path}"
                )

    def test_cast_character_identity(self, cast):
        """Each cast member must have a valid Character."""
        for name, member in cast.items():
            assert member.character is not None
            assert member.character.name == name
            assert member.character.seed > 0

    def test_standard_expressions_constant(self):
        """STANDARD_EXPRESSIONS must match across the codebase."""
        from vn_contracts import STANDARD_EXPRESSIONS as SE
        from vn_engine import _handle_set_expression
        # All should be present
        assert "neutral" in SE
        assert "smile" in SE
        assert "sad" in SE
        assert "angry" in SE
        assert "surprised" in SE
        assert len(SE) >= 6

    def test_pipeline_returns_list(self, cast_list):
        """Frozen contract: the cast pipeline returns list[CastMember]."""
        from vn_contracts import CastMember
        assert isinstance(cast_list, list)
        assert all(isinstance(m, CastMember) for m in cast_list)

    def test_cast_to_dict_roundtrip(self, cast_list):
        from vn_contracts import cast_to_dict
        d = cast_to_dict(cast_list)
        assert set(d.keys()) == {m.character.name for m in cast_list}
        for name, member in d.items():
            assert member.character.name == name


# ═══════════════════════════════════════════════════════════════════════
# Engine: set_background tool + expression state
# ═══════════════════════════════════════════════════════════════════════

class TestSetBackgroundTool:
    def test_parse_set_background(self):
        from vn_engine import parse_tool_calls
        calls = parse_tool_calls('[TOOL: set_background background="rooftop"]')
        assert len(calls) == 1
        assert calls[0]["tool"] == "set_background"
        assert calls[0]["background"] == "rooftop"

    def test_set_background_mutation(self, scene):
        from vn_engine import _handle_set_background
        result = _handle_set_background(
            {"tool": "set_background", "background": "sunset_hill"}, scene)
        assert result.success
        assert scene.background == "sunset_hill"

    def test_set_background_requires_key(self, scene):
        from vn_engine import _handle_set_background
        before = scene.background
        result = _handle_set_background({"tool": "set_background"}, scene)
        assert not result.success
        assert scene.background == before

    def test_set_background_routed(self, scene):
        from vn_engine import _route_tool
        result = _route_tool({"tool": "set_background", "background": "cafe"},
                             scene, scene.cast)
        assert result.success
        assert scene.background == "cafe"


class TestExpressionState:
    def test_set_expression_updates_scene_state(self, scene):
        from vn_engine import _handle_set_expression
        result = _handle_set_expression(
            {"tool": "set_expression", "character": "Yuki", "expression": "smile"},
            scene, scene.cast)
        assert result.success
        assert scene.expressions.get("Yuki") == "smile"

    def test_invalid_expression_leaves_state(self, scene):
        from vn_engine import _handle_set_expression
        result = _handle_set_expression(
            {"tool": "set_expression", "character": "Yuki", "expression": "zalgo"},
            scene, scene.cast)
        assert not result.success
        assert "Yuki" not in scene.expressions

    def test_choice_expression_changes_applied(self, scene):
        from vn_contracts import Choice
        from vn_engine import VNEngine
        engine = VNEngine(model=None, cast=scene.cast, scene=scene)
        choice = Choice(id="a", text="Smile back",
                        expression_changes={"Yuki": "smile", "Nobody": "sad"})
        engine.apply_choice_effects(choice)
        assert scene.expressions.get("Yuki") == "smile"
        assert "Nobody" not in scene.expressions

    def test_state_banner_shows_expressions(self, scene):
        from vn_prompt import build_state_banner
        scene.expressions["Yuki"] = "embarrassed"
        banner = build_state_banner(scene)
        assert "Yuki: embarrassed" in banner


# ═══════════════════════════════════════════════════════════════════════
# Engine: enter_character / exit_character tools
# ═══════════════════════════════════════════════════════════════════════

class TestEnterExitCharacter:
    def test_enter_character_adds_to_present(self, scene):
        from vn_engine import _handle_enter_character
        assert "Yuki" not in scene.present
        result = _handle_enter_character(
            {"tool": "enter_character", "character": "Yuki"}, scene, scene.cast)
        assert result.success
        assert "Yuki" in scene.present
        assert result.data == {"character": "Yuki", "present": "enter"}

    def test_enter_character_idempotent(self, scene):
        from vn_engine import _handle_enter_character
        scene.present.append("Yuki")
        result = _handle_enter_character(
            {"tool": "enter_character", "character": "Yuki"}, scene, scene.cast)
        assert result.success
        assert scene.present.count("Yuki") == 1  # no duplicate

    def test_enter_character_cast_lock_violation(self, scene):
        from vn_engine import _handle_enter_character
        result = _handle_enter_character(
            {"tool": "enter_character", "character": "Nobody"}, scene, scene.cast)
        assert not result.success
        assert "Cast-lock violation" in result.message

    def test_exit_character_removes_from_present(self, scene):
        from vn_engine import _handle_exit_character
        scene.present.append("Yuki")
        result = _handle_exit_character(
            {"tool": "exit_character", "character": "Yuki"}, scene, scene.cast)
        assert result.success
        assert "Yuki" not in scene.present
        assert result.data == {"character": "Yuki", "present": "exit"}

    def test_exit_character_not_present_no_error(self, scene):
        from vn_engine import _handle_exit_character
        assert "Yuki" not in scene.present
        result = _handle_exit_character(
            {"tool": "exit_character", "character": "Yuki"}, scene, scene.cast)
        assert result.success  # no-op, not an error

    def test_exit_character_cast_lock_violation(self, scene):
        from vn_engine import _handle_exit_character
        result = _handle_exit_character(
            {"tool": "exit_character", "character": "Nobody"}, scene, scene.cast)
        assert not result.success
        assert "Cast-lock violation" in result.message

    def test_enter_character_routed(self, scene):
        from vn_engine import _route_tool
        result = _route_tool(
            {"tool": "enter_character", "character": "Yuki"}, scene, scene.cast)
        assert result.success
        assert "Yuki" in scene.present

    def test_exit_character_routed(self, scene):
        from vn_engine import _route_tool
        scene.present.append("Yuki")
        result = _route_tool(
            {"tool": "exit_character", "character": "Yuki"}, scene, scene.cast)
        assert result.success
        assert "Yuki" not in scene.present

    def test_parse_enter_character(self):
        from vn_engine import parse_tool_calls
        calls = parse_tool_calls('[TOOL: enter_character character="Yuki"]')
        assert len(calls) == 1
        assert calls[0]["tool"] == "enter_character"
        assert calls[0]["character"] == "Yuki"

    def test_parse_exit_character(self):
        from vn_engine import parse_tool_calls
        calls = parse_tool_calls('[TOOL: exit_character character="Haruki"]')
        assert len(calls) == 1
        assert calls[0]["tool"] == "exit_character"
        assert calls[0]["character"] == "Haruki"


# ═══════════════════════════════════════════════════════════════════════
# Save/Load: JSON slots via the engine (SaveData contract)
# ═══════════════════════════════════════════════════════════════════════

class TestSaveLoadJson:
    def test_engine_save_load_roundtrip(self, scene, tmp_path):
        from vn_engine import VNEngine
        from vn_contracts import Choice

        engine = VNEngine(model=None, cast=scene.cast, scene=scene)
        scene.turn = 7
        scene.background = "sunset_hill"
        scene.mood = "romantic"
        scene.description = "The hill at dusk."
        scene.set_affection("Yuki", "confession")
        scene.expressions["Yuki"] = "embarrassed"
        scene.choices = [Choice(id="a", text="Stay", next_mood="cozy",
                                affection_flag=("Yuki", "stayed"))]

        path = engine.save(slot=1, save_dir=str(tmp_path))
        assert os.path.exists(path)

        # Wreck the live scene, then restore
        scene.turn = 0
        scene.background = "classroom"
        scene.mood = "peaceful"
        scene.affection_flags = []
        scene.expressions = {}
        scene.choices = []

        restored = engine.load(slot=1, save_dir=str(tmp_path))
        assert restored.turn == 7
        assert restored.background == "sunset_hill"
        assert restored.mood == "romantic"
        assert restored.description == "The hill at dusk."
        assert restored.has_affection("Yuki", "confession")
        assert restored.expressions.get("Yuki") == "embarrassed"
        assert len(restored.choices) == 1
        assert restored.choices[0].affection_flag == ("Yuki", "stayed")

    def test_load_missing_slot_raises(self, scene, tmp_path):
        from vn_engine import VNEngine
        engine = VNEngine(model=None, cast=scene.cast, scene=scene)
        with pytest.raises(FileNotFoundError):
            engine.load(slot=9, save_dir=str(tmp_path))

    def test_savedata_dict_roundtrip(self, scene):
        from vn_contracts import SaveData
        scene.set_affection("Haruki", "teased")
        data = SaveData.from_scene(scene)
        clone = SaveData.from_dict(data.to_dict())
        assert clone.cast_names == data.cast_names
        assert clone.affection_snapshot == data.affection_snapshot


# ═══════════════════════════════════════════════════════════════════════
# Voice: selective voicing of the emotional beat
# ═══════════════════════════════════════════════════════════════════════

class TestSelectiveVoicing:
    def _engine_with_voice(self, scene, handler):
        from model_client import MockModel
        from vn_engine import VNEngine
        from providers import MockVoiceProvider
        return VNEngine(model=MockModel(handler=handler), cast=scene.cast,
                        scene=scene, voice=MockVoiceProvider())

    def test_voices_line_after_expression_change(self, scene):
        def handler(messages, tools):
            return ('[TOOL: set_expression character="Yuki" expression="embarrassed"]\n\n'
                    '[Yuki]: "I... I made you a bento."\n')
        engine = self._engine_with_voice(scene, handler)
        engine.run_turn("I open the box")
        assert engine.last_voice is not None
        assert engine.last_voice.character == "Yuki"
        assert engine.last_voice.emotion == "embarrassed"
        assert "bento" in engine.last_voice.text

    def test_no_voice_without_expression_change(self, scene):
        def handler(messages, tools):
            return '[Yuki]: "Just a normal line."\n'
        engine = self._engine_with_voice(scene, handler)
        engine.run_turn("hello")
        assert engine.last_voice is None

    def test_dialogue_recorded_in_scene(self, scene):
        def handler(messages, tools):
            return ('[TOOL: set_expression character="Yuki" expression="smile"]\n\n'
                    '[Yuki]: "Good morning!"\n')
        engine = self._engine_with_voice(scene, handler)
        engine.run_turn("good morning")
        assert any(d["speaker"] == "Yuki" and d["text"] == "Good morning!"
                   for d in scene.dialogue)
