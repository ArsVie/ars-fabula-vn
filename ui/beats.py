"""Response → display beats: strip protocol markers, split a turn into paced
narration/dialogue beats, and render the screen for a given beat.
"""
from __future__ import annotations
import os
import re
from typing import Optional

from vn_contracts import CastMember, SceneState
from providers import BackgroundProvider

from .screens import _cast_panel_html_inner, _vn_screen_html


_MARKER_PREFIXES = ("TOOL", "Engine", "CAST-LOCK", "STATE", "ENDSTATE")


def clean_narration(text: str) -> str:
    """Strip protocol/engine markers from narration for player-facing display.

    The engine returns the raw model response as the transcript, which still
    contains the `[TOOL: ...]` calls (the protocol) plus any `[Engine: ...]` /
    `[CAST-LOCK ...]` notes. Those are machine markers, not prose — remove them
    before the text reaches the dialogue box or the Scene Log.

    Bracket-depth + quote aware, so the JSON inside an offer_choices call
    (which carries its own [ ] { } and quoted strings) is removed as one unit
    rather than truncated at the first ']'.
    """
    if not text:
        return text
    out = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == '[':
            m = re.match(r'\[\s*([A-Za-z-]+)', text[i:])
            if m and m.group(1) in _MARKER_PREFIXES:
                depth = in_str = esc = 0
                j = i
                while j < n:
                    c = text[j]
                    if esc:
                        esc = 0
                    elif c == '\\':
                        esc = 1
                    elif in_str:
                        if c == '"':
                            in_str = 0
                    elif c == '"':
                        in_str = 1
                    elif c == '[':
                        depth += 1
                    elif c == ']':
                        depth -= 1
                        if depth == 0:
                            j += 1
                            break
                    j += 1
                i = j
                continue
        out.append(text[i])
        i += 1
    # Collapse the blank lines left where markers were removed.
    return re.sub(r'\n[ \t]*(?:\n[ \t]*)+\n', '\n\n', "".join(out)).strip()


def _split_markers(text: str) -> list[tuple[str, str]]:
    """Walk a model response into ordered ('marker'|'text', chunk) segments.

    Markers are the [TOOL:]/[Engine:]/[CAST-LOCK:]… spans (bracket + quote
    aware so an offer_choices JSON blob is one span). Dialogue lines like
    `[Yuki]: "..."` are NOT markers — their prefix isn't in _MARKER_PREFIXES —
    so they stay in the text segments where the beat splitter can see them.
    """
    segs: list[tuple[str, str]] = []
    buf: list[str] = []
    i, n = 0, len(text)

    def flush():
        if buf:
            segs.append(("text", "".join(buf)))
            buf.clear()

    while i < n:
        if text[i] == '[':
            m = re.match(r'\[\s*([A-Za-z-]+)', text[i:])
            if m and m.group(1) in _MARKER_PREFIXES:
                depth = in_str = esc = 0
                j = i
                while j < n:
                    c = text[j]
                    if esc:
                        esc = 0
                    elif c == '\\':
                        esc = 1
                    elif in_str:
                        if c == '"':
                            in_str = 0
                    elif c == '"':
                        in_str = 1
                    elif c == '[':
                        depth += 1
                    elif c == ']':
                        depth -= 1
                        if depth == 0:
                            j += 1
                            break
                    j += 1
                flush()
                segs.append(("marker", text[i:j]))
                i = j
                continue
        buf.append(text[i])
        i += 1
    flush()
    return segs


# Speaker line: an optionally-bracketed name prefix ("Yuki:", "[Yuki]:") then
# the line. We do NOT trust the bracket/quote shape — the captured name is
# validated against the locked cast in _segment_beats, and only a real cast
# member makes it a speaker (and gets highlighted). This keeps the highlight
# consistent no matter how the model punctuated the line. Group 2 is the rest
# of the line (wrapping quotes stripped separately).
_SPEAKER_RE = re.compile(r'^\s*\[?\s*([^\[\]:]{1,40}?)\s*\]?\s*:\s*(.+)$')


def _strip_wrapping_quotes(text: str) -> str:
    """Drop one layer of matching wrapping quotes (straight or smart)."""
    t = text.strip()
    for a, b in (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")):
        if len(t) >= 2 and t[0] == a and t[-1] == b:
            return t[1:-1].strip()
    return t

# Narration pacing: cap each narration beat so the dialogue box never holds a
# wall of text. Whichever limit hits first ends the chunk; the rest becomes
# the next click-through beat (content is split, never truncated).
NARR_MAX_SENTENCES = 2
NARR_MAX_CHARS = 240
_SENTENCE_RE = re.compile(r'[^.!?…]*[.!?…]+["”\')\]]?|\S[^.!?…]*$')


def _chunk_narration(text: str) -> list[str]:
    """Split narration into <=NARR_MAX_SENTENCES, <=NARR_MAX_CHARS chunks."""
    text = text.strip()
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]
    if not sentences:
        return [text]
    chunks, cur = [], ""
    n = 0
    for s in sentences:
        candidate = (cur + " " + s).strip() if cur else s
        if cur and (n >= NARR_MAX_SENTENCES or len(candidate) > NARR_MAX_CHARS):
            chunks.append(cur)
            cur, n = s, 1
        else:
            cur, n = candidate, n + 1
    if cur:
        chunks.append(cur)
    return chunks


def _segment_beats(response: str, cast: dict[str, CastMember],
                   init_expr: dict[str, str], init_bg: str,
                   init_present: list[str] | None = None,
                   voice_line=None) -> list[dict]:
    """Split a turn's response into ordered display beats for paced reading.

    A beat is one narration paragraph or one dialogue line, tagged with the
    sprite expressions, background, and present characters in effect at that
    point — so a set_expression or enter_character that PRECEDES a line lands
    on that line's beat (the VN's emotional lever). Tool/engine markers are
    consumed to advance the running visual state, never shown. The single
    voiced line (if any) is attached to the beat whose dialogue it matches.

    Args:
        response: Raw model response text.
        cast: The locked cast.
        init_expr: Initial expression state before processing this response.
        init_bg: Initial background before processing this response.
        init_present: Initial present list before processing this response.
            If None, starts empty (characters must be called in).
        voice_line: Optional VoiceLine from the engine.
    """
    from vn_engine import parse_tool_calls, _check_cast_lock

    expr = dict(init_expr or {})
    bg = init_bg or "classroom"
    present = list(init_present) if init_present is not None else []
    beats: list[dict] = []
    narr: list[str] = []

    def flush_narr():
        text = " ".join(s for s in narr).strip()
        narr.clear()
        # One beat per <=2-sentence chunk so long narration paces out
        # instead of cluttering the box.
        for chunk in _chunk_narration(text):
            beats.append({"speaker": "", "text": chunk,
                          "expressions": dict(expr), "background": bg,
                          "present": list(present), "voice": None})

    for kind, chunk in _split_markers(response):
        if kind == "marker":
            for call in parse_tool_calls(chunk):
                tool = call.get("tool")
                if tool == "set_expression":
                    canon = _check_cast_lock(call.get("character", ""), cast)
                    e = call.get("expression", "")
                    if canon and canon in cast and e in cast[canon].expression_map:
                        expr[canon] = e
                elif tool in ("advance_scene", "set_background") and call.get("background"):
                    bg = call["background"]
                elif tool == "enter_character":
                    canon = _check_cast_lock(call.get("character", ""), cast)
                    if canon and canon not in present:
                        present.append(canon)
                elif tool == "exit_character":
                    canon = _check_cast_lock(call.get("character", ""), cast)
                    if canon and canon in present:
                        present.remove(canon)
            continue
        for line in chunk.split("\n"):
            s = line.strip()
            if not s:
                flush_narr()
                continue
            m = _SPEAKER_RE.match(s)
            canon = _check_cast_lock(m.group(1).strip(), cast) if m else None
            if canon:
                flush_narr()
                beats.append({"speaker": canon,
                              "text": _strip_wrapping_quotes(m.group(2)),
                              "expressions": dict(expr), "background": bg,
                              "present": list(present), "voice": None})
            else:
                narr.append(s)
        flush_narr()

    if voice_line and getattr(voice_line, "wav_path", None) and os.path.exists(voice_line.wav_path):
        target = (voice_line.text or "").strip()
        for b in beats:
            if b["speaker"] and b["text"].strip() == target:
                b["voice"] = voice_line.wav_path
                break

    return beats


def _render_beats_screen(beats: list[dict], idx: int, scene: SceneState,
                         bg_provider: BackgroundProvider,
                         prev_bg_path: str,
                         log_text: str = "") -> tuple[str, str, Optional[str]]:
    """Render the VN screen for beat `idx`. Choices appear only on the final
    beat; earlier beats show the ▼ continue hint. The HUD, scene-log panel and
    cast panel are baked into the canvas HTML. Returns
    (screen_html, new_prev_bg_path, voice_path)."""
    cast_html = _cast_panel_html_inner(scene.cast)

    if not beats:
        bg_info = bg_provider.get_background(scene.background)
        screen = _vn_screen_html(bg_info.path, scene.cast, scene.present,
                                 scene.expressions, choices=scene.choices,
                                 prev_bg_path=prev_bg_path,
                                 log_text=log_text, cast_html=cast_html)
        return screen, bg_info.path, None

    idx = max(0, min(idx, len(beats) - 1))
    beat = beats[idx]
    is_last = idx == len(beats) - 1
    bg_info = bg_provider.get_background(beat["background"])

    screen = _vn_screen_html(
        bg_path=bg_info.path,
        cast=scene.cast,
        present=beat.get("present", scene.present),
        expressions=beat["expressions"],
        dialogue_speaker=beat["speaker"],
        dialogue_text=beat["text"] if beat["speaker"] else "",
        narration=beat["text"] if not beat["speaker"] else "",
        choices=scene.choices if is_last else None,
        prev_bg_path=prev_bg_path,
        more=not is_last,
        log_text=log_text,
        cast_html=cast_html,
    )

    voice_path = beat.get("voice")
    if voice_path and not os.path.exists(voice_path):
        voice_path = None
    return screen, bg_info.path, voice_path
