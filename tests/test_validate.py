"""
test_validate.py — vn_validate (dataset gate / eval metric) + the
history-trim prompt helpers from vn_prompt.

All unit-marked: deterministic, no server.
"""
import pytest

from vn_validate import validate_turn, cast_names_from_system, ngram_overlap
from vn_prompt import (build_recent_history, build_full_prompt,
                       VN_SYSTEM_PROMPT, MOOD_TAGS, BACKGROUND_KEYS)

CAST = {"Yuki": None, "Marie": None, "Sakura": None}

CLEAN_TURN = """[TOOL: advance_scene description="The rooftop at lunch." background="rooftop" mood="cheerful"]
[TOOL: enter_character character="Yuki"]

The door swings open behind you.

[Yuki]: "There you are! I've been looking everywhere."

[TOOL: set_expression character="Yuki" expression="smile"]
[TOOL: offer_choices choices='[{"id": "a", "text": "Wave her over", "consequence": "friendly"}, {"id": "b", "text": "Pretend not to hear", "consequence": "cold"}]']
"""


@pytest.mark.unit
class TestValidateTurn:
    def test_clean_turn_passes(self):
        r = validate_turn(CAST, CLEAN_TURN)
        assert r.ok, r.reasons
        assert r.stats["tool_calls"] == 4
        assert r.stats["has_choices"] is True
        assert r.stats["choice_count"] == 2
        assert r.stats["speakers"] == ["Yuki"]

    def test_cast_violation_dialogue(self):
        turn = CLEAN_TURN + '\n[Hiro]: "Who am I?"\n'
        r = validate_turn(CAST, turn)
        assert not r.ok
        assert any("cast-lock" in x and "Hiro" in x for x in r.reasons)

    def test_cast_violation_tool_character(self):
        turn = '[TOOL: enter_character character="Hiro"]\nNarration.\n'
        r = validate_turn(CAST, turn)
        assert not r.ok
        assert any("enter_character" in x for x in r.reasons)

    def test_bad_background(self):
        turn = ('[TOOL: advance_scene description="x" background="space_station" '
                'mood="cheerful"]\nNarration continues.\n')
        r = validate_turn(CAST, turn)
        assert not r.ok
        assert any("space_station" in x for x in r.reasons)

    def test_bad_mood_and_expression(self):
        turn = ('[TOOL: advance_scene description="x" background="cafe" mood="grimdark"]\n'
                '[TOOL: set_expression character="Yuki" expression="ecstatic"]\n'
                'Narration.\n')
        r = validate_turn(CAST, turn)
        assert not r.ok
        assert any("grimdark" in x for x in r.reasons)
        assert any("ecstatic" in x for x in r.reasons)

    def test_malformed_tool_call(self):
        turn = 'Some narration.\n[TOOL: advance_scene description="never closes\n'
        r = validate_turn(CAST, turn)
        assert not r.ok
        assert any("malformed" in x for x in r.reasons)

    def test_bad_choices_json(self):
        turn = 'Narration.\n[TOOL: offer_choices choices=\'[{"id": }]\']\n'
        r = validate_turn(CAST, turn)
        assert not r.ok
        assert any("offer_choices" in x for x in r.reasons)

    def test_duplicate_choice_ids(self):
        turn = ('Narration.\n[TOOL: offer_choices choices=\'[{"id":"a","text":"x"},'
                '{"id":"a","text":"y"}]\']\n')
        r = validate_turn(CAST, turn)
        assert not r.ok
        assert any("duplicate" in x for x in r.reasons)

    def test_no_choices_but_narration_ok(self):
        r = validate_turn(CAST, "The bell rings. Yuki looks up at the sky.")
        assert r.ok
        assert r.stats["has_choices"] is False

    def test_empty_completion_rejected(self):
        r = validate_turn(CAST, "   ")
        assert not r.ok

    def test_mojibake_rejected(self):
        # UTF-8 streamed but decoded as Latin-1 (pre-fix SSE turns): an
        # em-dash's bytes E2 80 94 become the chars U+00E2 U+0080 U+0094.
        mangled_dash = chr(0xE2) + chr(0x80) + chr(0x94)
        turn = f'[Sakura]: "Wha{mangled_dash}?! Just... the heat!"\n'
        r = validate_turn(CAST, turn)
        assert not r.ok
        assert any("mojibake" in x for x in r.reasons)
        # cp1252-rendered variant of the same bytes ("a-circumflex, euro"):
        r2 = validate_turn(CAST, "Wha" + chr(0xE2) + chr(0x20AC) + chr(0x201D) + "?!")
        assert not r2.ok

    def test_clean_unicode_not_flagged(self):
        r = validate_turn(CAST, '[Yuki]: "Wha—?! It’s café time… ☆"')
        assert r.ok, r.reasons

    def test_unknown_tool(self):
        r = validate_turn(CAST, '[TOOL: summon_dragon name="Fafnir"]\nNarration.\n')
        assert not r.ok
        assert any("unknown tool" in x for x in r.reasons)

    def test_overlap_recorded_with_history(self):
        history = "The golden sunset spills through the large windows of the classroom."
        turn = ("The golden sunset spills through the large windows once more.\n"
                '[Yuki]: "Hi."\n')
        r = validate_turn(CAST, turn, history=history)
        assert r.ok  # style metric, not fatal
        assert r.stats["ngram_overlap"] is not None
        assert r.stats["ngram_overlap"] > 0

    def test_cast_from_system_prompt(self):
        system = VN_SYSTEM_PROMPT.format(
            cast_descriptions="- Yuki (childhood friend): warm.\n"
                              "- Marie (class rep): precise.",
            player_name="Player")
        assert cast_names_from_system(system) == ["Yuki", "Marie"]
        r = validate_turn(system, '[Yuki]: "Hello!"')
        assert r.ok, r.reasons


@pytest.mark.unit
class TestNgramOverlap:
    def test_disjoint_is_zero(self):
        a = "one two three four five six"
        b = "seven eight nine ten eleven twelve"
        assert ngram_overlap(a, b) == 0.0

    def test_identical_is_one(self):
        a = "the quick brown fox jumps over the lazy dog"
        assert ngram_overlap(a, a) == 1.0

    def test_too_short_is_none(self):
        assert ngram_overlap("hi", "there") is None


@pytest.mark.unit
class TestRecentHistory:
    LOG = ("Opening narration block.\n\n"
           "▸ Player: go to the roof\n\n"
           "Turn one narration, on the roof.\n\n"
           "▸ Player: talk to Yuki\n\n"
           "Turn two narration, Yuki turns around.")

    def test_empty_log(self):
        assert build_recent_history("") == ""
        assert build_recent_history("   \n  ") == ""

    def test_keeps_tail_turns(self):
        out = build_recent_history(self.LOG, max_turns=1)
        assert "talk to Yuki" in out
        assert "Turn two narration" in out
        assert "go to the roof" not in out

    def test_char_cap(self):
        long_log = "\n\n".join(f"Narration block {i} " + "x" * 200
                               for i in range(20))
        out = build_recent_history(long_log, max_turns=10, max_chars=500)
        assert 0 < len(out) <= 500
        assert out.endswith("Narration block 19 " + "x" * 200)

    def test_prompt_injection(self):
        # The RECENT STORY block lands above the [STATE] banner when present,
        # and is absent otherwise.
        class FakeScene:
            chapter, turn, player_name = "Prologue", 3, "Player"
            background, mood = "rooftop", "cheerful"
            present, expressions, cast, choices = [], {}, {}, []

            def active_affection(self, name):
                return []

        _, user = build_full_prompt(FakeScene(), {}, "talk to Yuki",
                                    history="Earlier story text.")
        assert "RECENT STORY" in user
        assert user.index("Earlier story text.") < user.index("[STATE]")
        _, user2 = build_full_prompt(FakeScene(), {}, "talk to Yuki")
        assert "RECENT STORY" not in user2


@pytest.mark.unit
class TestPromptEnums:
    def test_enums_injected_into_prompt(self):
        assert "__MOOD_TAGS__" not in VN_SYSTEM_PROMPT
        assert "__BACKGROUND_KEYS__" not in VN_SYSTEM_PROMPT
        for mood in MOOD_TAGS:
            assert mood in VN_SYSTEM_PROMPT
        for bg in BACKGROUND_KEYS:
            assert f"`{bg}`" in VN_SYSTEM_PROMPT
