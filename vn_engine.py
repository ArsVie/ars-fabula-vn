"""
vn_engine.py — The VN tool loop.

Wraps an LLM ModelClient and the locked cast. Processes player choices
through the model, enforces cast-lock, and returns scene mutations.

Architecture (same pattern as the DM):
  1. Player makes a choice
  2. Engine builds prompt with [STATE] banner + choice text
  3. Sends to model
  4. Parses [TOOL: ...] calls from response
  5. Mutates SceneState accordingly
  6. Returns (narration, updated_scene)
"""
from __future__ import annotations
import ast
import json
import re
from typing import Optional, Iterator

import os

from vn_contracts import (SceneState, CastMember, Choice, ActionResult,
                          AffectionFlag, SaveData,
                          member_to_dict, member_from_dict, cast_fingerprint)
from vn_prompt import build_full_prompt, build_state_banner
from model_client import ModelClient
from trace_log import log_turn

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saves")
CAST_DIR = os.path.join(SAVE_DIR, "cast")    # cast cards, keyed by fingerprint
NUM_SLOTS = 5


# ── Cast cards + slot metadata (module-level: usable without an engine) ──

def save_cast_card(cast: dict[str, CastMember]) -> str:
    """Persist the cast as a card under saves/cast/<hash>.json (idempotent —
    the fingerprint covers identity, so an existing card is reused).
    Returns the fingerprint."""
    fp = cast_fingerprint(cast)
    os.makedirs(CAST_DIR, exist_ok=True)
    path = os.path.join(CAST_DIR, f"{fp}.json")
    if not os.path.exists(path):
        card = {"hash": fp,
                "members": [member_to_dict(m) for m in cast.values()]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(card, f, ensure_ascii=False, indent=2)
    return fp


def load_cast_card(cast_hash: str) -> Optional[dict[str, CastMember]]:
    """Rebuild a cast dict from its card. None if the card doesn't exist."""
    path = os.path.join(CAST_DIR, f"{cast_hash}.json")
    if not (cast_hash and os.path.exists(path)):
        return None
    with open(path, "r", encoding="utf-8") as f:
        card = json.load(f)
    members = [member_from_dict(d) for d in card.get("members", [])]
    return {m.character.name: m for m in members} or None


def read_save(slot: int, save_dir: Optional[str] = None) -> Optional[SaveData]:
    """Read a slot's SaveData without an engine. None if the slot is empty
    or unreadable."""
    path = os.path.join(save_dir or SAVE_DIR, f"slot_{slot}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return SaveData.from_dict(json.load(f))
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def list_saves(save_dir: Optional[str] = None) -> list[dict]:
    """Slot metadata for the save/load UIs — one entry per slot, in order.
    Empty/unreadable slots get {"slot": i, "empty": True}."""
    out = []
    for i in range(NUM_SLOTS):
        data = read_save(i, save_dir)
        if data is None:
            out.append({"slot": i, "empty": True})
            continue
        out.append({
            "slot": i, "empty": False,
            "name": data.save_name or "",
            "chapter": data.chapter, "turn": data.turn,
            "cast_names": list(data.cast_names),
            "saved_at": data.saved_at or "",
            "cast_hash": data.cast_hash or "",
        })
    return out

# Expression -> voice emotion mapping (selective voicing of the emotional beat)
_EXPRESSION_EMOTION = {
    "neutral": "neutral", "smile": "happy", "laugh": "happy",
    "sad": "sad", "angry": "angry", "surprised": "surprised",
    "embarrassed": "embarrassed",
}


# ── Regex for parsing [TOOL: ...] calls ──────────────────────────────
_TOOL_CALL_RE = re.compile(r'\[TOOL:')


def _scan_balanced(text: str, start: int) -> int:
    """Scan text from an opening bracket at `start` to its balanced close.

    Honors JSON double-quoted strings (brackets inside them don't count)
    and backslash escapes. Returns the index AFTER the closing bracket,
    or -1 if unbalanced.
    """
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
        elif c == '\\':
            esc = True
        elif in_str:
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in '[{':
            depth += 1
        elif c in ']}':
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def _split_complete_lines(buffer: str) -> tuple[list[str], str]:
    """Cut `buffer` into complete lines on newlines that are NOT inside an
    open bracket span or a double-quoted string.

    This keeps a multi-line `[TOOL: ... choices=[{...}]]` call (or any bracketed
    marker the model pretty-prints across lines) intact as one unit, so the
    line-buffered streaming parser never sees a half a tool call. Brackets and
    backslash-escaped double quotes are tracked exactly as in parse_tool_calls;
    single quotes are NOT string delimiters (apostrophes in prose are fine).

    Returns (complete_lines, remainder). `complete_lines` have no trailing '\\n'.
    """
    depth = in_str = esc = 0
    lines: list[str] = []
    start = 0
    for i, c in enumerate(buffer):
        if esc:
            esc = 0
        elif c == '\\':
            esc = 1
        elif in_str:
            if c == '"':
                in_str = 0
        elif c == '"':
            in_str = 1
        elif c in '[{':
            depth += 1
        elif c in ']}':
            if depth > 0:
                depth -= 1
        elif c == '\n' and depth == 0:
            lines.append(buffer[start:i])
            start = i + 1
    return lines, buffer[start:]


def _parse_tool_args(args_str: str) -> dict:
    """Parse `key=value` pairs from a tool-call argument string.

    Values may be:
      - 'single-quoted'  — if the value starts with [ or { (JSON), it is
        scanned bracket-aware so apostrophes inside JSON strings (e.g.
        "I've") do NOT terminate the value; otherwise scans to the next '.
      - "double-quoted"  — scans to the closing quote, honoring \\" escapes
        (apostrophes inside are fine).
      - bare             — integer or word.
    """
    result: dict = {}
    i, n = 0, len(args_str)
    key_re = re.compile(r'(\w+)\s*=\s*')

    while i < n:
        while i < n and args_str[i].isspace():
            i += 1
        m = key_re.match(args_str, i)
        if not m:
            break
        key = m.group(1)
        i = m.end()
        if i >= n:
            result[key] = ""
            break

        ch = args_str[i]
        if ch == "'":
            i += 1
            if i < n and args_str[i] in '[{':
                end = _scan_balanced(args_str, i)
                if end == -1:
                    result[key] = args_str[i:]
                    break
                result[key] = args_str[i:end]
                i = end
                if i < n and args_str[i] == "'":
                    i += 1
            else:
                # Plain single-quoted value. A bare apostrophe inside the
                # value (e.g. character='O'Brien') must NOT end it. Treat a
                # quote as the closing delimiter only when it is followed by
                # end-of-args or by the next key (whitespace* + word + '=').
                # Otherwise it is a literal apostrophe; keep scanning.
                j = i
                while True:
                    j = args_str.find("'", j)
                    if j == -1:
                        result[key] = args_str[i:]
                        i = n
                        break
                    if re.match(r"\s*(\w+\s*=|$)", args_str[j + 1:]):
                        result[key] = args_str[i:j]
                        i = j + 1
                        break
                    j += 1  # apostrophe inside the value
        elif ch == '"':
            i += 1
            buf = []
            esc = False
            while i < n:
                c = args_str[i]
                if esc:
                    buf.append(c)
                    esc = False
                elif c == '\\':
                    esc = True
                elif c == '"':
                    break
                else:
                    buf.append(c)
                i += 1
            result[key] = "".join(buf)
            i += 1  # past the closing quote
        elif ch in '[{':
            # Bare (unquoted) JSON value, e.g. choices=[{...}]. Some models
            # omit the single-quote wrapper the prompt asks for; scan the
            # balanced blob just as we do for the quoted-JSON case.
            end = _scan_balanced(args_str, i)
            if end == -1:
                result[key] = args_str[i:]
                i = n
            else:
                result[key] = args_str[i:end]
                i = end
        else:
            m2 = re.match(r'(\d+)|(\w+)', args_str[i:])
            if not m2:
                i += 1
                continue
            if m2.group(1) is not None:
                result[key] = int(m2.group(1))
            else:
                result[key] = m2.group(2)
            i += m2.end()

    return result


def parse_tool_calls(response: str) -> list[dict]:
    """Extract ALL tool calls from the model's response.

    Supports [TOOL: name key="value" key='[json]' key=123 key=word] format.
    Robust to apostrophes in dialogue/JSON text ("I've", "don't") — the
    end-of-call scanner tracks only double-quoted strings (with escapes)
    and bracket depth, never single quotes.

    Returns a list of dicts with keys:
      - "tool": the tool name
      - any parsed keyword arguments
    """
    calls = []
    for m in _TOOL_CALL_RE.finditer(response):
        start = m.end()
        # Find the matching closing bracket: depth starts at 1 ([TOOL:).
        # Double-quoted strings are skipped (escape-aware); single quotes
        # are NOT treated as string delimiters here — apostrophes in prose
        # would corrupt the state.
        depth = 1
        in_str = False
        esc = False
        end_pos = -1

        for i in range(start, len(response)):
            ch = response[i]
            if esc:
                esc = False
                continue
            if ch == '\\':
                esc = True
                continue
            if in_str:
                if ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    end_pos = i
                    break

        if end_pos == -1:
            continue  # Malformed tool call, skip

        # Extract the content between [TOOL: and the matching ]
        inner = response[m.end():end_pos].strip()

        # First token is the tool name
        space_idx = inner.find(' ')
        if space_idx == -1:
            tool_name = inner
            args_str = ""
        else:
            tool_name = inner[:space_idx]
            args_str = inner[space_idx + 1:].strip()

        result = {"tool": tool_name}
        if args_str:
            result.update(_parse_tool_args(args_str))

        calls.append(result)

    return calls


# ── Cast-lock enforcement ────────────────────────────────────────────

def _check_cast_lock(character_name: str, cast: dict[str, CastMember]) -> Optional[str]:
    """Validate a character name against the locked cast.

    Returns the canonical name (properly cased) if found,
    or None if the character doesn't exist in the cast.
    """
    # Direct match
    if character_name in cast:
        return character_name

    # Case-insensitive match
    for name in cast:
        if name.lower() == character_name.lower():
            return name

    return None


def _find_cast_violations(response: str, cast: dict[str, CastMember]) -> list[str]:
    """Find non-cast character names used as dialogue speakers.

    We look for the [character_name]: pattern in dialogue lines.
    Returns the list of violating names (deduped, in order of appearance).
    """
    dialogue_refs = re.findall(r'\[(\w+)\]:\s*"', response)
    violations = []
    for ref in dialogue_refs:
        if not _check_cast_lock(ref, cast) and ref not in violations:
            violations.append(ref)
    return violations


def _repair_cast_violation(response: str, cast: dict[str, CastMember]) -> str:
    """Check a model response for non-cast character names and flag them.

    Returns the response with violations flagged via [CAST-LOCK VIOLATION].
    Does NOT stop execution — just flags for the engine to handle.
    """
    violations = _find_cast_violations(response, cast)
    if not violations:
        return response

    # Append violation warning
    violation_msg = (
        "\n\n[CAST-LOCK VIOLATION: The following character names are not in the "
        f"locked cast and will be ignored: {', '.join(violations)}. "
        "Only use characters from the cast list.]"
    )
    return response + violation_msg


def _clean_noncast_dialogue(response: str, cast: dict[str, CastMember]) -> str:
    """Remove dialogue lines from non-cast characters.

    If a line is tagged with a character not in the cast, strip that line.
    This is the enforcement mechanism — the model CANNOT speak through
    a character that doesn't have a sprite.
    """
    lines = response.split("\n")
    cleaned = []
    for line in lines:
        m = re.match(r'^\[(\w+)\]:', line)
        if m:
            char_name = m.group(1)
            if not _check_cast_lock(char_name, cast):
                # Replace with a narration mention instead
                cleaned.append(f"*(someone speaks)*")
                continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ═══════════════════════════════════════════════════════════════════════
# Tool handlers
# ═══════════════════════════════════════════════════════════════════════

def _handle_advance_scene(tool_data: dict, scene: SceneState) -> ActionResult:
    """Handle advance_scene tool call.

    Mutates scene: background, mood, description, present characters.
    """
    description = tool_data.get("description", "")
    background = tool_data.get("background", scene.background)
    mood = tool_data.get("mood", scene.mood)

    if description:
        scene.description = description
    scene.background = background
    scene.mood = mood

    return ActionResult(
        success=True,
        message=f"Scene advanced: {background} ({mood})",
        data={"background": background, "mood": mood},
    )


def _handle_set_background(tool_data: dict, scene: SceneState) -> ActionResult:
    """Handle set_background tool call.

    Changes only the background — narration/mood untouched.
    """
    background = tool_data.get("background", "")
    if not background:
        return ActionResult(success=False, message="set_background: no background key given")

    scene.background = background
    return ActionResult(
        success=True,
        message=f"Background set: {background}",
        data={"background": background},
    )


def _handle_set_expression(tool_data: dict, scene: SceneState, cast: dict[str, CastMember]) -> ActionResult:
    """Handle set_expression tool call.

    Validates that the character exists and the expression is valid,
    then records it as the character's current expression in scene state.
    """
    char_name = tool_data.get("character", "")
    expression = tool_data.get("expression", "")

    canonical = _check_cast_lock(char_name, cast)
    if not canonical:
        return ActionResult(
            success=False,
            message=f"Cast-lock violation: '{char_name}' is not in the cast. Ignored.",
        )

    if expression not in cast[canonical].expression_map:
        return ActionResult(
            success=False,
            message=f"Expression '{expression}' not available for {canonical}. Available: {list(cast[canonical].expression_map.keys())}",
        )

    if scene.expressions.get(canonical) == expression:
        # No-op: already showing this expression. Succeed without reporting a
        # visual change (data carries no "expression" key), so the streaming
        # loop doesn't count it as an expression change worth voicing.
        return ActionResult(
            success=True,
            message=f"Expression unchanged: {canonical} already {expression}",
        )

    scene.expressions[canonical] = expression
    return ActionResult(
        success=True,
        message=f"Expression set: {canonical} -> {expression}",
        data={"character": canonical, "expression": expression},
    )


def _handle_enter_character(tool_data: dict, scene: SceneState, cast: dict[str, CastMember]) -> ActionResult:
    """Handle enter_character tool call.

    Validates cast-lock, then appends to scene.present if not already present.
    """
    char_name = tool_data.get("character", "")

    canonical = _check_cast_lock(char_name, cast)
    if not canonical:
        return ActionResult(
            success=False,
            message=f"Cast-lock violation: '{char_name}' is not in the cast. Ignored.",
        )

    if canonical not in scene.present:
        scene.present.append(canonical)
    return ActionResult(
        success=True,
        message=f"{canonical} entered the scene",
        data={"character": canonical, "present": "enter"},
    )


def _handle_exit_character(tool_data: dict, scene: SceneState, cast: dict[str, CastMember]) -> ActionResult:
    """Handle exit_character tool call.

    Removes from scene.present. No-op if not present.
    """
    char_name = tool_data.get("character", "")

    canonical = _check_cast_lock(char_name, cast)
    if not canonical:
        return ActionResult(
            success=False,
            message=f"Cast-lock violation: '{char_name}' is not in the cast. Ignored.",
        )

    if canonical in scene.present:
        scene.present.remove(canonical)
    return ActionResult(
        success=True,
        message=f"{canonical} exited the scene",
        data={"character": canonical, "present": "exit"},
    )


def _loads_lenient(blob: str):
    """json.loads with a best-effort repair for common model deviations.

    Weaker models emit loosely-formatted JSON — most often unquoted object
    keys (`{id: 1, text: ...}` instead of `{"id": 1, "text": ...}`), or
    Python-repr style with single-quoted keys/values
    (`[{'id': 'study', 'text': "I've ..."}]`). Try a strict parse first;
    on failure, fall back to ast.literal_eval (which handles single quotes —
    apostrophes inside values included — natively), then quote bare keys and
    retry both. Raises json.JSONDecodeError if every repair fails.
    """
    try:
        return json.loads(blob)
    except json.JSONDecodeError as err:
        strict_err = err

    # Python-repr style (single quotes, True/False/None) parses as a literal.
    try:
        return ast.literal_eval(blob)
    except (ValueError, SyntaxError):
        pass

    # Quote bare object keys following '{' or ','. The lookbehind anchor
    # (a brace/comma + optional space) keeps us from touching colons that
    # live inside string values like "go to the cafe: now".
    repaired = re.sub(r'([{,]\s*)([A-Za-z_]\w*)(\s*:)', r'\1"\2"\3', blob)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    # Bare keys AND single-quoted values combined: the key repair above makes
    # it a valid Python literal even though it still isn't strict JSON.
    try:
        return ast.literal_eval(repaired)
    except (ValueError, SyntaxError):
        raise strict_err


def _handle_offer_choices(tool_data: dict, scene: SceneState) -> ActionResult:
    """Handle offer_choices tool call.

    Parses the choices JSON and sets them on the scene state. Tolerates the
    loose JSON weaker models tend to emit (unquoted keys, numeric ids).
    """
    choices_json = tool_data.get("choices", "")
    if not choices_json:
        return ActionResult(success=False, message="No choices provided")

    try:
        choices_data = _loads_lenient(choices_json)
    except json.JSONDecodeError:
        return ActionResult(success=False, message="Invalid choices JSON")

    if not isinstance(choices_data, list):
        return ActionResult(success=False, message="choices must be a JSON array")

    choices = []
    for cd in choices_data:
        if not isinstance(cd, dict):
            continue
        choice = Choice(
            # ids are matched against the player's click (a string), so coerce
            # numeric ids like {"id": 1} to "1" — otherwise 1 == "1" is False.
            id=str(cd.get("id", f"c_{len(choices)}")),
            text=cd.get("text", ""),
            consequence=cd.get("consequence", ""),
            next_background=cd.get("next_background"),
            next_mood=cd.get("next_mood"),
            expression_changes=cd.get("expression_changes", {}),
        )
        # Affection flag if present
        if "affection_flag" in cd:
            flag_data = cd["affection_flag"]
            if isinstance(flag_data, list) and len(flag_data) == 2:
                choice.affection_flag = (flag_data[0], flag_data[1])
        choices.append(choice)

    scene.choices = choices
    return ActionResult(
        success=True,
        message=f"Offered {len(choices)} choices",
        data={"choice_count": len(choices)},
    )


def _handle_save_checkpoint(tool_data: dict, scene: SceneState) -> ActionResult:
    """Handle save_checkpoint tool call.

    Marks the scene for auto-save (no-op in engine, UI handles actual serialization).
    """
    scene.save_slot = 0  # auto-save slot
    return ActionResult(success=True, message="Checkpoint saved")


# ── Tool router ──────────────────────────────────────────────────────

_TOOL_HANDLERS = {
    "advance_scene": _handle_advance_scene,
    "set_background": _handle_set_background,
    "set_expression": _handle_set_expression,
    "enter_character": _handle_enter_character,
    "exit_character": _handle_exit_character,
    "offer_choices": _handle_offer_choices,
    "save_checkpoint": _handle_save_checkpoint,
}


def _route_tool(tool_data: dict, scene: SceneState, cast: dict[str, CastMember]) -> ActionResult:
    """Route a tool call to the appropriate handler."""
    tool_name = tool_data.get("tool", "")
    handler = _TOOL_HANDLERS.get(tool_name)
    if not handler:
        return ActionResult(success=False, message=f"Unknown tool: {tool_name}")

    # Different handlers need different signatures
    cast_tools = ("set_expression", "enter_character", "exit_character")
    if tool_name in cast_tools:
        return handler(tool_data, scene, cast)
    return handler(tool_data, scene)


# ═══════════════════════════════════════════════════════════════════════
# VN Engine
# ═══════════════════════════════════════════════════════════════════════

class VNEngine:
    """The VN tool loop.

    Wraps a ModelClient and a locked cast. Processes player choices
    through the model, enforces cast-lock, and returns scene mutations.

    Usage:
        engine = VNEngine(model=my_client, cast=my_cast, scene=initial_scene)
        narration, updated_scene = engine.run_turn("I choose to talk to Yuki")
    """

    def __init__(self, model: ModelClient, cast: dict[str, CastMember],
                 scene: SceneState, voice=None):
        """
        Args:
            model: Any ModelClient (real or mock).
            cast: The locked cast (name -> CastMember).
            scene: Initial scene state.
            voice: Optional VoiceProvider for selective voicing of key
                   emotional lines. None disables voicing.
        """
        self.model = model
        self.cast = cast
        self.scene = scene
        self.voice = voice
        self.last_voice = None  # VoiceLine of the most recent voiced beat (or None)

    def run_turn(self, player_choice_text: str,
                 history: str = "") -> tuple[str, SceneState]:
        """Run one VN turn (blocking).

        Drains run_turn_stream and returns its final (narration, scene). All
        turn logic lives in the streaming generator; this is the one-shot
        convenience for callers (and tests) that don't need incremental output.

        Args:
            player_choice_text: The choice text the player selected.
            history: Trimmed session-log tail (vn_prompt.build_recent_history)
                so the model knows what it already narrated.

        Returns:
            (narration_text, updated_scene_state)
        """
        final = (None, self.scene)
        for final in self.run_turn_stream(player_choice_text, history=history):
            pass
        return final

    def run_turn_stream(self, player_choice_text: str, history: str = ""
                        ) -> Iterator[tuple[str, SceneState]]:
        """Run one VN turn, streaming line-buffered.

        Streams the model response and processes it ONE COMPLETE LINE AT A
        TIME (buffering deltas to '\\n' boundaries), so `[TOOL: ...]` calls and
        `[Name]: "..."` speaker prefixes are never split mid-token. Per line:
        route tool calls (mutating the scene live), record dialogue, and voice
        the first eligible emotional beat. Yields (narration_so_far, scene) as
        lines complete; the FINAL yield equals what run_turn returns — the
        cleaned narration plus cast-lock / engine notes.

        Args:
            player_choice_text: The choice text the player selected.

        Yields:
            (narration_so_far, updated_scene_state) — last yield is final.
        """
        # 0. The previous turn's choices are consumed by making one
        self.scene.choices = []

        # 1. Build prompt
        system, user = build_full_prompt(self.scene, self.cast, player_choice_text,
                                         player_name=self.scene.player_name,
                                         history=history)

        # Also inject the current scene description for context
        if self.scene.description:
            user = f"Current scene: {self.scene.description}\n\n{user}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # Per-turn streaming state
        tool_results: list[ActionResult] = []
        changed_exprs: set[str] = set()   # characters whose expression changed so far
        self.last_voice = None
        voiced = False

        def process_line(line: str):
            """Handle one complete line: route tool calls, record dialogue,
            voice the first eligible beat. Mutates scene + closure state."""
            nonlocal voiced
            # Route tool calls on this line (fires as the line arrives)
            for tc in parse_tool_calls(line):
                result = _route_tool(tc, self.scene, self.cast)
                tool_results.append(result)
                if result.success and "expression" in result.data:
                    changed_exprs.add(result.data["character"])
            # Record dialogue (cast-lock cleaning is line-based, so a single
            # cleaned line matches what the full-text pass would record).
            cleaned = _clean_noncast_dialogue(line, self.cast)
            for m in re.finditer(r'^\[(\w+)\]:\s*"([^"]*)"', cleaned, re.MULTILINE):
                speaker = _check_cast_lock(m.group(1), self.cast)
                if not speaker:
                    continue
                self.scene.dialogue.append({
                    "speaker": speaker,
                    "text": m.group(2),
                    "expression": self.scene.expressions.get(speaker, "neutral"),
                })
                # Selective voicing — the first dialogue line spoken by a
                # character whose expression changed earlier in this turn.
                if (self.voice is not None and not voiced
                        and speaker in changed_exprs and m.group(2).strip()):
                    expr = self.scene.expressions.get(speaker, "neutral")
                    emotion = _EXPRESSION_EMOTION.get(expr, "neutral")
                    try:
                        self.last_voice = self.voice.speak(
                            m.group(2), speaker, emotion=emotion,
                            voice_id=self.cast[speaker].character.voice_id,
                        )
                    except TypeError:
                        # Provider without voice_id kwarg (older interface)
                        self.last_voice = self.voice.speak(m.group(2), speaker, emotion=emotion)
                    except Exception as e:
                        print(f"[engine] voice failed: {e}")
                    voiced = True

        # 2. Stream the model, line-buffered (bracket/quote aware so a
        #    multi-line tool call is never split across the newline).
        response = ""
        pending = ""
        for delta in self.model.stream(messages):
            if not delta:
                continue
            response += delta
            pending += delta
            lines, pending = _split_complete_lines(pending)
            for line in lines:
                process_line(line)
            if lines:
                # Partial narration for incremental display (notes come last).
                yield _clean_noncast_dialogue(response, self.cast), self.scene
        # Flush the trailing partial line, if any.
        if pending:
            process_line(pending)

        # 3. Cast-lock check — over the full response, before cleaning.
        violations = _find_cast_violations(response, self.cast)

        # Trace the raw prompt→response for fine-tuning (real models only;
        # mock/scripted output is skipped). Best-effort, never raises.
        if not getattr(self.model, "is_mock", False):
            # Prefer the SERVER-reported model id (cfg.model is a cosmetic label
            # that can lie — e.g. a stale "gemma-12b" default). Falls back to
            # cfg.model when the client can't ask the server.
            model_id = getattr(getattr(self.model, "cfg", None), "model", "")
            resolve = getattr(self.model, "server_model", None)
            if callable(resolve):
                try:
                    model_id = resolve() or model_id
                except Exception:
                    pass
            log_turn(
                messages, response,
                model=model_id,
                context={
                    "turn": self.scene.turn,
                    "scene": self.scene.description,
                    "mood": self.scene.mood,
                    "present": list(self.scene.present),
                    "player_input": player_choice_text,
                    "cast_violations": violations,
                },
            )

        # 4. Final narration: cleaned response + cast-lock / engine notes.
        narration = _clean_noncast_dialogue(response, self.cast)
        if violations:
            narration += (
                "\n\n[Engine: CAST-LOCK removed non-cast speaker(s): "
                f"{', '.join(violations)}]"
            )
        engine_notes = [f"[Engine: {tr.message}]"
                        for tr in tool_results if not tr.success]
        if engine_notes:
            narration += "\n\n" + "\n".join(engine_notes)

        # 5. Increment turn and emit the final result.
        self.scene.turn += 1
        yield narration, self.scene

    def apply_choice_effects(self, choice: Choice):
        """Apply the effects of a player-selected choice.

        Mutates scene state: background, mood, affection flags, expressions.

        Args:
            choice: The Choice the player picked.
        """
        if choice.next_background:
            self.scene.background = choice.next_background
        if choice.next_mood:
            self.scene.mood = choice.next_mood
        if choice.affection_flag:
            char_name, flag = choice.affection_flag
            self.scene.set_affection(char_name, flag)
        for char_name, expr in (choice.expression_changes or {}).items():
            canonical = _check_cast_lock(char_name, self.cast)
            if canonical and expr in self.cast[canonical].expression_map:
                self.scene.expressions[canonical] = expr

    # ── Save / Load ──────────────────────────────────────────────────

    def save(self, slot: int = 0, save_dir: Optional[str] = None,
             save_name: str = "", log_text: str = "") -> str:
        """Serialize the current scene to a JSON save slot.

        Also writes the cast card (saves/cast/<hash>.json) and stamps the
        save with its fingerprint, a player-given name, and the local time —
        so the slot can be described in the UI and loaded from the main menu.
        Returns the path of the written save file.
        """
        save_dir = save_dir or SAVE_DIR
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"slot_{slot}.json")
        data = SaveData.from_scene(self.scene)
        data.save_name = (save_name or "").strip()[:48]
        from datetime import datetime
        data.saved_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        data.cast_hash = save_cast_card(self.scene.cast)
        data.log_text = log_text or ""
        self.scene.save_slot = slot
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    def load(self, slot: int = 0, save_dir: Optional[str] = None) -> SceneState:
        """Restore scene state from a JSON save slot (cast stays locked).

        Raises FileNotFoundError if the slot doesn't exist.
        """
        save_dir = save_dir or SAVE_DIR
        path = os.path.join(save_dir, f"slot_{slot}.json")
        with open(path, "r", encoding="utf-8") as f:
            data = SaveData.from_dict(json.load(f))
        data.apply_to(self.scene)
        self.scene.save_slot = slot
        return self.scene

    def reset_scene(self) -> SceneState:
        """Reset the scene to a fresh state with the same cast."""
        self.scene = SceneState(
            cast=self.cast,
            present=[],
            background="classroom",
            mood="peaceful",
            turn=0,
            chapter="Prologue",
        )
        return self.scene
