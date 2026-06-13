"""
vn_validate.py — single source of truth for "is this a well-formed VN turn".

Used two ways:
  1. Dataset filter — tools/build_sft_dataset.py gates every trace record
     (and later every DeepSeek-transformed example) through validate_turn.
  2. Eval metric — held-out prompts are scored on format validity,
     cast-lock violations, beat length, and cross-turn n-gram overlap.

All parsing reuses the engine's own parsers (vn_engine) and the canonical
enums (vn_prompt / vn_contracts) — nothing is reimplemented, so a record
that validates here is exactly a record the engine would accept at runtime.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Optional, Union

from vn_contracts import STANDARD_EXPRESSIONS
from vn_prompt import MOOD_TAGS, BACKGROUND_KEYS
from vn_engine import (parse_tool_calls, _check_cast_lock,
                       _find_cast_violations, _loads_lenient, _TOOL_HANDLERS)

# Tools whose `character` argument must satisfy cast-lock.
_CAST_TOOLS = {"set_expression", "enter_character", "exit_character"}

# Matches the cast block lines of the system prompt: "- Name (role): ..."
_CAST_LINE_RE = re.compile(r'^- (.+?) \([^)]*\):', re.MULTILINE)


@dataclass
class ValidationResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def cast_names_from_system(system_text: str) -> list[str]:
    """Extract the locked cast names from a rendered system prompt.

    The cast block renders as "- Name (role): personality" lines (see
    vn_prompt.build_cast_descriptions). Used by the dataset exporter, where
    only the trace's message text is available — not a live cast dict.
    """
    return _CAST_LINE_RE.findall(system_text or "")


def _as_cast(system_or_cast: Union[str, dict, list, set, None]) -> dict:
    """Normalize the cast argument to a name->member dict.

    Accepts a cast dict (name -> CastMember), an iterable of names, or a
    rendered system-prompt string (names are parsed out of it). Values may
    be None when only names are known — expression checks then fall back to
    STANDARD_EXPRESSIONS instead of the member's expression_map.
    """
    if isinstance(system_or_cast, dict):
        return system_or_cast
    if isinstance(system_or_cast, str):
        return {name: None for name in cast_names_from_system(system_or_cast)}
    if system_or_cast:
        return {name: None for name in system_or_cast}
    return {}


def ngram_overlap(text: str, history: str, n: int = 4) -> Optional[float]:
    """Fraction of `text`'s word n-grams that already appear in `history`.

    The repetition signal: high overlap means the turn re-narrates what the
    player already read. None when either side is too short to compare.
    """
    def grams(s: str) -> set[tuple[str, ...]]:
        words = re.findall(r"[\w']+", s.lower())
        return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}

    text_grams = grams(text)
    hist_grams = grams(history)
    if not text_grams or not hist_grams:
        return None
    return len(text_grams & hist_grams) / len(text_grams)


def _strip_tools(completion: str) -> str:
    """The narration-only text — balanced [TOOL: ...] spans removed (a tool
    call may span lines when the model pretty-prints its choices JSON)."""
    out: list[str] = []
    i, n = 0, len(completion)
    while i < n:
        start = completion.find("[TOOL:", i)
        if start == -1:
            out.append(completion[i:])
            break
        out.append(completion[i:start])
        # Scan to the balanced close, double-quote aware (same rules as
        # parse_tool_calls). Unbalanced → drop the rest of the line only.
        depth, in_str, esc = 1, False, False
        end = -1
        for j in range(start + len("[TOOL:"), n):
            c = completion[j]
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif in_str:
                if c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end == -1:
            nl = completion.find("\n", start)
            end = n if nl == -1 else nl
        i = end
    return "".join(out)


def validate_turn(system_or_cast, completion: str,
                  history: str = "") -> ValidationResult:
    """Validate one assistant turn against the VN protocol.

    Args:
        system_or_cast: the locked cast — a name->CastMember dict, an
            iterable of names, or the rendered system prompt (names parsed).
        completion: the assistant's raw turn text.
        history: optional prior story text; when given, the cross-turn
            n-gram overlap is recorded in stats (non-fatal).

    Returns:
        ValidationResult(ok, reasons, stats). `ok` is False only on protocol
        violations; style metrics are recorded in stats but never fail.
    """
    cast = _as_cast(system_or_cast)
    reasons: list[str] = []
    stats: dict = {}

    if not (completion or "").strip():
        return ValidationResult(False, ["empty completion"], stats)
    if not cast:
        reasons.append("no cast available to validate against")

    # Mojibake (UTF-8 read as Latin-1): "â€”" for —, "â€™" for ’, "Ã©" for é.
    # Streamed turns logged before the SSE encoding fix carry these; they
    # must never reach training data.
    if re.search('\u00e2\u20ac|[\u00c2\u00c3\u00e2][\u0080-\u00bf]', completion):
        reasons.append("mojibake: UTF-8/Latin-1 double-decoding artifacts")

    # ── Tool calls parse cleanly, and every [TOOL: opener was consumed ──
    calls = parse_tool_calls(completion)
    n_openers = completion.count("[TOOL:")
    if len(calls) < n_openers:
        reasons.append(f"malformed tool call: {n_openers - len(calls)} of "
                       f"{n_openers} [TOOL: openers failed to parse")
    stats["tool_calls"] = len(calls)

    choices_ok = False
    has_choices = False
    for call in calls:
        tool = call.get("tool", "")
        if tool not in _TOOL_HANDLERS:
            reasons.append(f"unknown tool: {tool!r}")
            continue

        if tool in _CAST_TOOLS:
            char = str(call.get("character", ""))
            if cast and not _check_cast_lock(char, cast):
                reasons.append(f"{tool}: character {char!r} not in cast")

        if tool == "set_expression":
            expr = str(call.get("expression", ""))
            char = _check_cast_lock(str(call.get("character", "")), cast) if cast else None
            member = cast.get(char) if char else None
            valid = (member.expression_map.keys()
                     if member is not None and getattr(member, "expression_map", None)
                     else STANDARD_EXPRESSIONS)
            if expr not in valid:
                reasons.append(f"set_expression: invalid expression {expr!r}")

        if tool in ("advance_scene", "set_background"):
            bg = str(call.get("background", "")) or None
            if bg and bg not in BACKGROUND_KEYS:
                reasons.append(f"{tool}: invalid background {bg!r}")
        if tool == "advance_scene":
            mood = str(call.get("mood", "")) or None
            if mood and mood not in MOOD_TAGS:
                reasons.append(f"advance_scene: invalid mood {mood!r}")

        if tool == "offer_choices":
            has_choices = True
            blob = call.get("choices", "")
            try:
                data = _loads_lenient(blob) if blob else None
            except json.JSONDecodeError:
                data = None
            if not isinstance(data, list) or not data:
                reasons.append("offer_choices: choices JSON unparseable or empty")
                continue
            ids = [str(c.get("id", "")) for c in data if isinstance(c, dict)]
            if len(ids) != len(data):
                reasons.append("offer_choices: non-object entries in choices")
            if len(set(ids)) != len(ids):
                reasons.append("offer_choices: duplicate choice ids")
            if any(not (c.get("text") or "").strip()
                   for c in data if isinstance(c, dict)):
                reasons.append("offer_choices: choice with empty text")
            if not [r for r in reasons if r.startswith("offer_choices:")]:
                choices_ok = True
            stats["choice_count"] = len(data)
    stats["has_choices"] = has_choices

    # ── Dialogue speakers satisfy cast-lock ──
    if cast:
        violations = _find_cast_violations(completion, cast)
        if violations:
            reasons.append(f"cast-lock: non-cast speakers {violations}")
        speakers = []
        for m in re.finditer(r'^\[(\w+)\]:\s*"', completion, re.MULTILINE):
            canon = _check_cast_lock(m.group(1), cast)
            if canon and canon not in speakers:
                speakers.append(canon)
        stats["speakers"] = speakers

    # ── Turn must end with valid choices OR continuing narration ──
    narration = _strip_tools(completion).strip()
    if has_choices and not choices_ok:
        pass  # already a reason above
    elif not has_choices and not narration:
        reasons.append("turn has neither narration nor choices")

    # ── Style metrics (recorded, never fatal) ──
    beats = [b.strip() for b in narration.split("\n")
             if b.strip() and not re.match(r'^\[\w+\]:', b.strip())]
    stats["beats"] = len(beats)
    stats["max_beat_sentences"] = max(
        (len(re.findall(r'[.!?]+(?:\s|$)', b)) for b in beats), default=0)
    if history:
        stats["ngram_overlap"] = ngram_overlap(narration, history)

    return ValidationResult(ok=not reasons, reasons=reasons, stats=stats)
