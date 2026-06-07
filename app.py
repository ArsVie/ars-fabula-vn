"""
Wave 1C / 2 — Gradio VN UI.

Ars-Fabula Visual Novel interface with:
  - VN grammar: background layer → sprite layer → name/dialogue box → choice menu
  - Scene-turn transitions: double-buffered background crossfade + animated
    sprite/expression swap (the plan's polish lever — timing, not fidelity)
  - The CASTING BEAT as an experience: portraits resolve one by one while
    "assembling your cast…" plays (loading is a feature, not a stall)
  - Selective voiced lines (VoiceProvider) played on the emotional beat
  - Save/Load controls (JSON slots via the engine)
  - Per-session gr.State for SceneState persistence
  - Mock LLM fallback when the model server is unavailable

Images are embedded as base64 data URIs — no file-route/allowed_paths
dependency, so sprites and backgrounds always render.

Run with:
    python app.py                            # production (tries real model)
    ARS_FABULA_BACKEND=mock python app.py    # test mode (no server)
    ARS_FABULA_VOICE=kokoro python app.py    # voiced lines via Kokoro
"""
from __future__ import annotations
import base64
import json
import os
import re
import time
from functools import lru_cache
from typing import Optional

import gradio as gr

from vn_contracts import SceneState, CastMember, Choice, cast_to_dict
from vn_engine import VNEngine
from cast_pipeline import load_curated_cast, generate_cast
from providers import BackgroundProvider, get_voice_provider, get_background_provider
from model_client import get_model, ModelConfig, MockModel


# ── Constants ─────────────────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(PROJECT_DIR, "static")

# Default cast (loaded at startup) — pipeline returns list (frozen contract);
# SceneState wants the name->member dict.
DEFAULT_CAST = cast_to_dict(load_curated_cast())


def make_initial_scene(cast=None) -> SceneState:
    if cast is None:
        cast = DEFAULT_CAST
    return SceneState(
        cast=cast,
        present=[],  # characters enter via enter_character as they appear
        background="classroom",
        mood="peaceful",
        turn=0,
        chapter="Prologue",
    )


# ── Image embedding (always renders; no file-route dependency) ────────

@lru_cache(maxsize=256)
def _data_uri_cached(path: str, mtime: float) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def data_uri(path: Optional[str]) -> str:
    """Return a data: URI for an image path, or '' if unavailable."""
    if not path or not os.path.exists(path):
        return ""
    try:
        return _data_uri_cached(path, os.path.getmtime(path))
    except OSError:
        return ""


# ── Mock VN LLM ───────────────────────────────────────────────────────

def _make_mock_engine(cast: dict[str, CastMember]) -> MockModel:
    """Create a MockModel that generates plausible VN narration + tool calls."""
    char_names = list(cast.keys())

    def handler(messages, tools):
        user_msgs = [m for m in messages if m["role"] == "user"]
        choice_text = user_msgs[-1]["content"] if user_msgs else ""

        c1 = char_names[0] if len(char_names) > 0 else "Someone"
        c2 = char_names[1] if len(char_names) > 1 else "Someone else"

        choice_lower = choice_text.lower()

        if "talk" in choice_lower or c1.lower() in choice_lower:
            expressions = ["surprised", "smile", "embarrassed"]
            return (
                f"[TOOL: enter_character character=\"{c1}\"]\n\n"
                f"[TOOL: set_expression character=\"{c1}\" expression=\"{expressions[0]}\"]\n\n"
                f"[TOOL: advance_scene description=\"The afternoon sun filters through the classroom windows, casting warm rectangles of light across the wooden floor.\" background=\"classroom\" mood=\"cheerful\"]\n\n"
                f"[TOOL: set_expression character=\"{c1}\" expression=\"{expressions[1]}\"]\n\n"
                f"[{c1}]: \"Oh, hey! I didn't expect you to come over today. What's up?\"\n\n"
                f"You feel a warmth in your chest at their genuine smile. The classroom, usually so ordinary, seems to glow a little brighter when they're in it.\n\n"
                f"[TOOL: set_expression character=\"{c1}\" expression=\"{expressions[2]}\"]\n\n"
                f"[{c1}]: \"Actually… I've been meaning to talk to you too. There's something I wanted to say.\"\n\n"
                f"The air around you seems to still. A cherry blossom petal drifts past the open window.\n\n"
                f"[TOOL: offer_choices choices='[{{\"id\":\"ask\",\"text\":\"\\\"What is it?\\\"\",\"consequence\":\"lean in\"}},{{\"id\":\"deflect\",\"text\":\"\\\"Can it wait? I'm kind of busy...\\\"\",\"consequence\":\"avoid the conversation\"}},{{\"id\":\"tease\",\"text\":\"\\\"Look who's getting serious all of a sudden~\\\"\",\"consequence\":\"tease them playfully\"}}]']"
            )

        elif "look" in choice_lower or "alone" in choice_lower:
            return (
                f"[TOOL: enter_character character=\"{c1}\"]\n\n"
                f"[TOOL: set_expression character=\"{c1}\" expression=\"sad\"]\n\n"
                f"[TOOL: advance_scene description=\"The classroom feels emptier when it's just you. Through the window, you can see the courtyard where students are laughing, the sound muffled by glass.\" background=\"classroom\" mood=\"melancholic\"]\n\n"
                f"Your eyes drift to {c1}'s desk. A notebook left behind, pages fluttering in the breeze from the open window.\n\n"
                f"[TOOL: exit_character character=\"{c1}\"]\n\n"
                f"The door clicks shut behind them. You're alone with the afternoon light.\n\n"
                f"[TOOL: offer_choices choices='[{{\"id\":\"wait\",\"text\":\"Wait for them to come back\",\"consequence\":\"sit and wait\"}},{{\"id\":\"go_find\",\"text\":\"Go look for them\",\"consequence\":\"search the school\"}},{{\"id\":\"leave\",\"text\":\"Head home\",\"consequence\":\"leave for the day\"}}]']"
            )

        elif "confess" in choice_lower or "love" in choice_lower:
            return (
                f"[TOOL: enter_character character=\"{c1}\"]\n\n"
                f"[TOOL: set_expression character=\"{c1}\" expression=\"surprised\"]\n\n"
                f"[TOOL: advance_scene description=\"Time seems to stop. The world narrows to just the two of you — the distant sounds of the school fading into nothing.\" background=\"rooftop\" mood=\"romantic\"]\n\n"
                f"[TOOL: set_expression character=\"{c1}\" expression=\"embarrassed\"]\n\n"
                f"For a long, breathless moment, neither of you speaks. The wind carries petals between you, swirling in a lazy dance.\n\n"
                f"[TOOL: set_expression character=\"{c1}\" expression=\"smile\"]\n\n"
                f"[{c1}]: \"I… I didn't think you'd ever say that. I feel the same way.\"\n\n"
                f"The sunset paints the world in shades of gold and pink. Somewhere, a bell tolls the hour. But right now, it's just the two of you.\n\n"
                f"[TOOL: offer_choices choices='[{{\"id\":\"hold_hands\",\"text\":\"Take their hand\",\"consequence\":\"a gentle touch\",\"next_background\":\"sunset_hill\",\"next_mood\":\"romantic\",\"affection_flag\":[\"{c1}\",\"confession\"]}},{{\"id\":\"hug\",\"text\":\"Pull them into a hug\",\"consequence\":\"embrace\",\"next_background\":\"rooftop\",\"next_mood\":\"romantic\"}},{{\"id\":\"speak\",\"text\":\"\\\"I've wanted to say it for so long...\\\"\",\"consequence\":\"vulnerable truth\",\"next_background\":\"rooftop\",\"next_mood\":\"romantic\"}}]']"
            )

        else:
            return (
                f"[TOOL: enter_character character=\"{c1}\"]\n\n"
                f"[TOOL: advance_scene description=\"The bell rings, signaling the end of another school day. Students begin packing their bags, the air filled with chatter and the shuffle of papers.\" background=\"classroom\" mood=\"peaceful\"]\n\n"
                f"[TOOL: set_expression character=\"{c1}\" expression=\"smile\"]\n\n"
                f"[{c1}]: \"Hey, you ready to head out? I was thinking we could grab something from the café near the station.\"\n\n"
                f"They pack their bag with practiced ease, glancing at you expectantly. There's a warmth in their eyes that makes even an ordinary school day feel special.\n\n"
                f"[TOOL: offer_choices choices='[{{\"id\":\"yes_cafe\",\"text\":\"\\\"Sure, sounds good!\\\"\",\"consequence\":\"go to the cafe together\"}},{{\"id\":\"rain_check\",\"text\":\"\\\"Maybe tomorrow? I've got stuff to do.\\\"\",\"consequence\":\"politely decline\"}},{{\"id\":\"invite_others\",\"text\":\"\\\"Mind if {c2} comes too?\\\"\",\"consequence\":\"make it a group outing\"}}]']"
            )

    return MockModel(handler=handler)


# ── LLM / engine builders ─────────────────────────────────────────────

def _try_real_llm():
    """Try to connect to the real model server. Returns client or None."""
    cfg = ModelConfig.from_env()
    if cfg.backend == "mock":
        return None
    try:
        client = get_model(cfg)
        if client.health():
            print(f"[app] Connected to LLM server at {cfg.base_url}")
            return client
    except Exception as e:
        print(f"[app] LLM server unavailable: {e}")
    return None


def build_engine(cast: dict[str, CastMember], scene: SceneState) -> tuple[VNEngine, str]:
    """Build the VN engine: real model if reachable, mock otherwise.
    Voice provider comes from ARS_FABULA_VOICE (mock/kokoro/chatterbox)."""
    voice = get_voice_provider(os.getenv("ARS_FABULA_VOICE", "mock"))

    real = _try_real_llm()
    if real:
        return VNEngine(model=real, cast=cast, scene=scene, voice=voice), "real LLM"

    mock = _make_mock_engine(cast)
    print("[app] Using MOCK VN engine (no server needed)")
    return VNEngine(model=mock, cast=cast, scene=scene, voice=voice), "mock LLM"


# ═══════════════════════════════════════════════════════════════════════
# CSS — VN themed (incl. scene-turn crossfade + sprite swap animations)
# ═══════════════════════════════════════════════════════════════════════

CSS = """
/* ── Global ── */
.gradio-container {
    background: #1a1a2e !important;
    color: #e0d5c8 !important;
    font-family: 'Georgia', 'Palatino', 'Noto Serif JP', serif !important;
    max-width: 100% !important;
}

/* ── VN Screen ── */
.vn-screen {
    position: relative;
    width: 100%;
    height: 550px;
    background: #0a0a1a;
    border: 2px solid #2a2a4a;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 8px;
}

/* Double-buffered background layers: .old sits below, .new fades in over it */
.vn-bg {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    object-fit: cover;
}
.vn-bg.new { animation: vnBgFade 1.1s ease both; }
@keyframes vnBgFade { from { opacity: 0; } to { opacity: 1; } }

.vn-chars {
    position: absolute;
    bottom: 120px;
    left: 0;
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: flex-end;
    gap: 20px;
    pointer-events: none;
}
.vn-sprite {
    height: 350px;
    width: auto;
    animation: vnSpriteIn 0.45s ease both;
}
@keyframes vnSpriteIn {
    from { opacity: 0.35; transform: translateY(6px); }
    to   { opacity: 1;    transform: translateY(0); }
}

/* ── Dialogue Box ── */
.vn-dialogue {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    background: linear-gradient(0deg, rgba(10, 10, 30, 0.95) 0%, rgba(10, 10, 30, 0.85) 100%);
    padding: 16px 24px;
    border-top: 2px solid #4a4a6a;
    min-height: 110px;
}
.vn-name {
    font-weight: bold;
    color: #e8a87c;
    font-size: 1.1rem;
    margin-bottom: 4px;
}
.vn-text { color: #e0d5c8; font-size: 1rem; line-height: 1.6; margin: 0; }
.vn-text.narrator { font-style: italic; color: #b0a898; }

/* Typewriter cursor — blinks while text is being revealed */
.vn-text.typing::after {
    content: '▌';
    animation: vnBlink 0.7s step-end infinite;
    color: #e8a87c;
    margin-left: 1px;
}
@keyframes vnBlink { 50% { opacity: 0; } }

/* Click-to-advance affordance: the whole screen is the "next" button while
   there are more beats to read; the ▼ blinks in the dialogue corner. */
.vn-screen.advancing { cursor: pointer; }
.vn-continue {
    position: absolute;
    right: 16px; bottom: 10px;
    color: #e8a87c; font-size: 1rem;
    animation: vnPulse 1.2s ease infinite;
    pointer-events: none;
}

/* ── Choices ── */
.vn-choices {
    position: absolute;
    bottom: 130px; left: 0; right: 0;
    padding: 12px 24px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    pointer-events: auto;
}
.vn-choice-btn {
    background: rgba(40, 40, 80, 0.9) !important;
    border: 1px solid #4a4a6a !important;
    color: #e0d5c8 !important;
    font-family: 'Georgia', serif !important;
    font-size: 0.95rem !important;
    padding: 10px 16px !important;
    border-radius: 6px !important;
    text-align: left !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
.vn-choice-btn:hover {
    background: rgba(60, 60, 120, 0.9) !important;
    border-color: #e8a87c !important;
    color: #fff !important;
}

/* ── Casting beat ── */
.vn-casting {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 18px;
}
.vn-casting .casting-title {
    color: #e8a87c;
    font-size: 1.25rem;
    font-style: italic;
    animation: vnPulse 1.6s ease infinite;
}
@keyframes vnPulse { 0%,100% { opacity: 0.55; } 50% { opacity: 1; } }
.vn-casting .portraits { display: flex; gap: 24px; }
.vn-portrait {
    width: 110px; height: 160px;
    border-radius: 8px;
    border: 1px solid #4a4a6a;
    background: rgba(30,30,60,0.7);
    overflow: hidden;
    display: flex; align-items: center; justify-content: center;
}
.vn-portrait img {
    width: 100%; height: 100%; object-fit: cover; object-position: top;
    animation: vnBgFade 0.8s ease both;
}
.vn-portrait.pending { color: #555a7a; font-size: 2rem; }
.vn-portrait .pname {
    color: #e8a87c; font-size: 0.8rem; text-align: center;
}

/* ── Scene Info / Cast Panel ── */
.scene-info {
    color: #8888aa;
    font-size: 0.8rem;
    padding: 4px 8px;
    border-left: 2px solid #4a4a6a;
    margin: 4px 0;
}
.cast-card {
    background: linear-gradient(135deg, #1a1a3e 0%, #16213e 100%);
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
}
.cast-card .name { font-weight: bold; color: #e8a87c; font-size: 0.95rem; }
.cast-card .role { color: #8888aa; font-size: 0.8rem; font-style: italic; }
.cast-card .personality { color: #b0a898; font-size: 0.8rem; margin-top: 4px; line-height: 1.3; }

/* ── Buttons / headings ── */
button {
    border: 1px solid #3a3a5a !important;
    background: linear-gradient(180deg, #2a2a4a 0%, #1a1a3e 100%) !important;
    color: #e0d5c8 !important;
    font-family: 'Georgia', serif !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
}
button:hover { border-color: #e8a87c !important; color: #fff !important; }
h1, h2, h3 { font-family: 'Georgia', serif !important; color: #e8a87c !important; }
"""

# Installed via launch(js=...) — gr.HTML strips <script> tags, so a page
# load hook is the reliable way to wire the in-screen choice buttons.
#
# IMPORTANT (Gradio 6): the js string is run as a FUNCTION BODY
# (`new Function(...args, <js>)` then called), NOT eval'd as an expression.
# So it must be plain statements — an `() => {...}` arrow wrapper would just
# be constructed and discarded, and the listener would never attach.
ONLOAD_JS = """
// ── Typewriter: reveals text char-by-char, two-stage click ─────
const TYPING_SPEED = 35;  // ms per character
let _typewriterTimer = null;

function startTypewriter(screen) {
    // Clear any running timer
    if (_typewriterTimer) { clearInterval(_typewriterTimer); _typewriterTimer = null; }

    const textEl = screen.querySelector('.vn-text');
    if (!textEl) return;

    const fullText = textEl.getAttribute('data-full-text') || textEl.textContent || '';
    if (!fullText) return;
    textEl.setAttribute('data-full-text', fullText);

    // If reduced-motion, reveal instantly
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        textEl.textContent = fullText;
        textEl.classList.remove('typing');
        screen.setAttribute('data-typing', '0');
        return;
    }

    let idx = 0;
    textEl.textContent = '';
    textEl.classList.add('typing');
    screen.setAttribute('data-typing', '1');

    _typewriterTimer = setInterval(function() {
        idx++;
        if (idx >= fullText.length) {
            clearInterval(_typewriterTimer);
            _typewriterTimer = null;
            textEl.textContent = fullText;
            textEl.classList.remove('typing');
            screen.setAttribute('data-typing', '0');
        } else {
            textEl.textContent = fullText.slice(0, idx);
        }
    }, TYPING_SPEED);
}

function completeTypewriter(screen) {
    if (_typewriterTimer) { clearInterval(_typewriterTimer); _typewriterTimer = null; }
    const textEl = screen.querySelector('.vn-text');
    if (!textEl) return;
    const fullText = textEl.getAttribute('data-full-text') || textEl.textContent || '';
    textEl.textContent = fullText;
    textEl.classList.remove('typing');
    screen.setAttribute('data-typing', '0');
}

// Watch for VN screen changes and start typewriter
const _typewriterObserver = new MutationObserver(function() {
    const screen = document.querySelector('.vn-screen:not(.vn-casting)');
    if (screen && screen.getAttribute('data-typing') !== '0') {
        // Small delay so the DOM settles
        setTimeout(function() { startTypewriter(screen); }, 50);
    }
});
// Start observing once the body is ready
(function waitBody() {
    if (document.body) {
        _typewriterObserver.observe(document.body, { childList: true, subtree: true });
    } else {
        setTimeout(waitBody, 100);
    }
})();

// ── Click bridge ───────────────────────────────────────────────
const clickHidden = (sel) => {
    const el = document.querySelector(sel + ' button') || document.querySelector(sel);
    if (el) el.click();
};

document.addEventListener('click', function(e) {
    const btn = e.target.closest('.vn-choice-btn');
    if (btn) {
        e.preventDefault();
        const id = btn.getAttribute('data-choice-id');
        const text = btn.getAttribute('data-choice-text');
        if (!id) return;

        const setVal = (boxId, val) => {
            const ta = document.querySelector('#' + boxId + ' textarea, #' + boxId + ' input');
            if (!ta) return;
            const proto = ta.tagName === 'TEXTAREA'
                ? window.HTMLTextAreaElement.prototype
                : window.HTMLInputElement.prototype;
            const setter = Object.getOwnPropertyDescriptor(proto, 'value');
            if (setter && setter.set) { setter.set.call(ta, val); }
            else { ta.value = val; }
            ta.dispatchEvent(new Event('input', { bubbles: true }));
        };
        setVal('choice-id-box', id);
        setVal('choice-text-box', text);

        setTimeout(() => clickHidden('#submit-choice-btn'), 150);
        return;
    }

    // Two-stage click: if typing, complete text; else advance
    const screen = e.target.closest('.vn-screen');
    if (screen) {
        if (screen.getAttribute('data-typing') === '1') {
            e.preventDefault();
            completeTypewriter(screen);
            return;
        }
        // If no more beats and choices visible, don't advance (let choice buttons work)
        if (screen.getAttribute('data-more') === '1' && !e.target.closest('.vn-choices')) {
            e.preventDefault();
            clickHidden('#advance-btn');
        }
    }
});

// Space / Enter: complete typewriter or advance
document.addEventListener('keydown', function(e) {
    if (e.key !== ' ' && e.key !== 'Enter') return;
    const tag = (document.activeElement && document.activeElement.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;

    const screen = document.querySelector('.vn-screen');
    if (!screen) return;

    if (screen.getAttribute('data-typing') === '1') {
        e.preventDefault();
        completeTypewriter(screen);
        return;
    }

    if (screen.getAttribute('data-more') === '1') {
        e.preventDefault();
        clickHidden('#advance-btn');
    }
});
"""


# ═══════════════════════════════════════════════════════════════════════
# UI Builders
# ═══════════════════════════════════════════════════════════════════════

def _vn_screen_html(bg_path: str, cast: dict[str, CastMember],
                    present: list[str], expressions: dict[str, str],
                    dialogue_speaker: str = "", dialogue_text: str = "",
                    choices: list[Choice] | None = None,
                    narration: str = "",
                    prev_bg_path: str = "",
                    more: bool = False) -> str:
    """Build the VN screen HTML: crossfading bg layers, sprites, dialogue, choices.

    `more=True` marks the screen as having further beats to read this turn:
    it adds a ▼ continue hint and a `data-more="1"` flag the JS click/keyboard
    hook uses to advance the reader one beat at a time.
    """
    bg_layers = ""
    prev_uri = data_uri(prev_bg_path) if prev_bg_path and prev_bg_path != bg_path else ""
    curr_uri = data_uri(bg_path)
    if prev_uri:
        bg_layers += f'<img class="vn-bg old" src="{prev_uri}" alt="">'
    if curr_uri:
        bg_layers += f'<img class="vn-bg new" src="{curr_uri}" alt="">'

    # Sprites — keyed by expression so a swap re-triggers the entry animation
    sprite_html = ""
    for name in present:
        member = cast.get(name)
        if not member:
            continue
        expr = expressions.get(name, "neutral")
        sprite_uri = data_uri(member.get_sprite(expr))
        if sprite_uri:
            sprite_html += (f'<img class="vn-sprite" data-expr="{name}:{expr}" '
                            f'src="{sprite_uri}" alt="{name} ({expr})">')

    # Dialogue
    dialogue_html = ""
    if dialogue_text:
        if dialogue_speaker:
            dialogue_html = f'<div class="vn-name">{dialogue_speaker}</div>'
        dialogue_html += f'<p class="vn-text{" narrator" if not dialogue_speaker else ""}">{dialogue_text}</p>'
    elif narration:
        dialogue_html = f'<p class="vn-text narrator">{narration}</p>'
    if more:
        dialogue_html += '<div class="vn-continue">▼</div>'

    # Choices
    choices_html = ""
    if choices:
        parts = ['<div class="vn-choices">']
        for c in choices:
            safe_text = c.text.replace('"', '&quot;')
            parts.append(
                f'<button class="vn-choice-btn" '
                f'data-choice-id="{c.id}" data-choice-text="{safe_text}">'
                f'{c.text}</button>'
            )
        parts.append('</div>')
        choices_html = "\n".join(parts)

    return f"""
    <div class="vn-screen{' advancing' if more else ''}" data-more="{'1' if more else '0'}" data-typing="0">
        {bg_layers}
        <div class="vn-chars">{sprite_html}</div>
        {choices_html}
        <div class="vn-dialogue">{dialogue_html}</div>
    </div>"""


def _casting_screen_html(char_keys: list[str],
                         resolved: dict[str, CastMember],
                         message: str = "Assembling your cast…") -> str:
    """The casting-beat screen: portraits resolve one by one."""
    portraits = []
    for key in char_keys:
        name = key.capitalize()
        member = resolved.get(name)
        if member:
            uri = data_uri(member.get_sprite("neutral"))
            inner = (f'<img src="{uri}" alt="{name}">' if uri
                     else f'<div class="pname">{name}</div>')
            portraits.append(
                f'<div style="text-align:center;">'
                f'<div class="vn-portrait">{inner}</div>'
                f'<div class="pname" style="margin-top:4px;color:#e8a87c;'
                f'font-size:0.8rem;">{name}</div></div>'
            )
        else:
            portraits.append(
                f'<div style="text-align:center;">'
                f'<div class="vn-portrait pending">…</div>'
                f'<div class="pname" style="margin-top:4px;color:#555a7a;'
                f'font-size:0.8rem;">?</div></div>'
            )

    return f"""
    <div class="vn-screen">
        <div class="vn-casting">
            <div class="casting-title">🌸 {message}</div>
            <div class="portraits">{''.join(portraits)}</div>
        </div>
    </div>
    """


def _cast_panel_html(cast: dict[str, CastMember]) -> str:
    """Build HTML for the cast info panel."""
    if not cast:
        return '<div style="padding:8px;color:#8888aa;">No cast yet — start a story.</div>'
    parts = ['<div style="padding:8px;">']
    for name, member in cast.items():
        char = member.character
        parts.append(
            f'<div class="cast-card">'
            f'<div class="name">{char.name}</div>'
            f'<div class="role">{char.role} · ♪ {char.voice_id}</div>'
            f'<div class="personality">{char.personality}</div>'
            f'<div style="color:#666;font-size:0.75rem;margin-top:4px;">'
            f'Expressions: {", ".join(member.expression_map.keys())}'
            f'</div></div>'
        )
    parts.append('</div>')
    return "\n".join(parts)


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


_DIALOGUE_RE = re.compile(r'^\[(\w+)\]:\s*"(.*)"\s*$')


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
        if text:
            beats.append({"speaker": "", "text": text,
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
            m = _DIALOGUE_RE.match(s)
            if m:
                flush_narr()
                speaker = _check_cast_lock(m.group(1), cast) or m.group(1)
                beats.append({"speaker": speaker, "text": m.group(2),
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
                         prev_bg_path: str) -> tuple[str, str, Optional[str]]:
    """Render the VN screen for beat `idx`. Choices appear only on the final
    beat; earlier beats show the ▼ continue hint. Returns
    (screen_html, new_prev_bg_path, voice_path)."""
    if not beats:
        bg_info = bg_provider.get_background(scene.background)
        screen = _vn_screen_html(bg_info.path, scene.cast, scene.present,
                                 scene.expressions, choices=scene.choices,
                                 prev_bg_path=prev_bg_path)
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
    )

    voice_path = beat.get("voice")
    if voice_path and not os.path.exists(voice_path):
        voice_path = None
    return screen, bg_info.path, voice_path


# ═══════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════

def _make_callbacks(bg_provider: BackgroundProvider):
    # All turn callbacks return/yield 11-tuples:
    # (scene, engine, screen, cast, cast_panel, log, info, audio, prev_bg,
    #  beats, beat_idx)   ← the last two drive read-at-your-own-pace.

    def _empty(scene_state=None):
        scene = scene_state or make_initial_scene()
        return (scene, None,
                _vn_screen_html("", {}, [], {}),
                {}, _cast_panel_html({}),
                "", "No active scene", None, "", [], 0)

    def on_start(cast_preset: str):
        """The CASTING BEAT — a generator so portraits resolve one by one.

        Loading is the feature: yield the casting screen immediately,
        reveal each cast member as they're ready, then open the scene.
        """
        if cast_preset == "generated":
            char_keys = ["yuki", "haruki", "sakura"]
            loader = lambda k: None   # generated mode batches below
        else:
            char_keys = ["yuki", "haruki", "sakura"]
            loader = lambda k: load_curated_cast([k])[0]

        resolved: dict[str, CastMember] = {}

        # Beat 1 — the casting screen, empty portraits
        yield (None, None,
               _casting_screen_html(char_keys, resolved),
               {}, _cast_panel_html({}),
               "", "🎬 Casting…", None, "", [], 0)

        # Beat 2 — portraits resolve
        if cast_preset == "generated":
            # One long bake; ComfyUI does the whole batch (falls back to
            # curated internally if unreachable). Show the wait honestly.
            yield (None, None,
                   _casting_screen_html(char_keys, resolved,
                                        "Generating your cast… (this is the bake — "
                                        "everything after this is instant)"),
                   {}, _cast_panel_html({}), "", "🎬 Casting…", None, "", [], 0)
            members = generate_cast({"preset": "default"})
            for m in members:
                resolved[m.character.name] = m
                yield (None, None,
                       _casting_screen_html(char_keys, resolved),
                       {}, _cast_panel_html(resolved),
                       "", "🎬 Casting…", None, "", [], 0)
        else:
            for key in char_keys:
                member = loader(key)
                if member:
                    resolved[member.character.name] = member
                time.sleep(0.45)   # let the beat land — loading as a feature
                yield (None, None,
                       _casting_screen_html(char_keys, resolved),
                       {}, _cast_panel_html(resolved),
                       "", "🎬 Casting…", None, "", [], 0)

        cast = resolved
        if not cast:
            yield _empty()
            return

        # Beat 3 — lock the cast, build the engine, open the scene
        scene = SceneState(
            cast=cast, present=[],
            background="classroom", mood="peaceful",
            turn=0, chapter="Prologue",
        )
        engine, status = build_engine(cast, scene)
        pre_expr, pre_bg = dict(scene.expressions), scene.background
        pre_present = list(scene.present)
        raw, scene = engine.run_turn("")
        beats = _segment_beats(raw, scene.cast, pre_expr, pre_bg,
                               init_present=pre_present, voice_line=engine.last_voice)

        screen, prev_bg, voice_path = _render_beats_screen(beats, 0, scene, bg_provider, "")
        yield (scene, engine, screen, cast, _cast_panel_html(cast),
               clean_narration(raw),
               f"🌸 {scene.chapter} · {scene.background} ({scene.mood}) · {status}",
               voice_path, prev_bg, beats, 0)

    def on_choice(choice_id: str, choice_text: str,
                  scene: SceneState, engine, prev_bg: str):
        """A choice click → THE SCENE TURN."""
        if not engine or not scene or not choice_id:
            return _empty(scene)

        for choice in scene.choices:
            if choice.id == choice_id:
                engine.apply_choice_effects(choice)
                break

        # Snapshot the visual state going INTO the turn so the beat splitter
        # can replay expression/background changes in order.
        pre_expr, pre_bg = dict(scene.expressions), scene.background
        pre_present = list(scene.present)
        raw, scene = engine.run_turn(choice_text or choice_id)
        beats = _segment_beats(raw, scene.cast, pre_expr, pre_bg,
                               init_present=pre_present, voice_line=engine.last_voice)

        screen, new_prev_bg, voice_path = _render_beats_screen(
            beats, 0, scene, bg_provider, prev_bg)

        return (scene, engine, screen, scene.cast, _cast_panel_html(scene.cast),
                clean_narration(raw),
                f"Turn {scene.turn} · {scene.background} ({scene.mood})",
                voice_path, new_prev_bg, beats, 0)

    def on_save(slot: float, scene: SceneState, engine):
        if not engine:
            return "Nothing to save — start a story first."
        path = engine.save(int(slot))
        return f"💾 Saved to slot {int(slot)} ({os.path.basename(path)})"

    def on_load(slot: float, scene: SceneState, engine, prev_bg: str):
        if not engine:
            return (*_empty(scene), "Nothing to load into — start a story first.")
        try:
            scene = engine.load(int(slot))
        except FileNotFoundError:
            return (scene, engine, gr.skip(), scene.cast,
                    _cast_panel_html(scene.cast), gr.skip(),
                    f"Slot {int(slot)} is empty.", None, prev_bg, gr.skip(), gr.skip(),
                    f"⚠️ Slot {int(slot)} is empty.")

        # A restored save has no per-beat tool sequence — show it as a single
        # static beat (the saved description) with its choices already visible.
        beats = [{"speaker": "", "text": scene.description or "",
                  "expressions": dict(scene.expressions),
                  "background": scene.background,
                  "present": list(scene.present), "voice": None}]
        screen, new_prev_bg, _ = _render_beats_screen(beats, 0, scene, bg_provider, prev_bg)
        return (scene, engine, screen, scene.cast, _cast_panel_html(scene.cast),
                scene.description,
                f"📂 Loaded slot {int(slot)} · Turn {scene.turn} · {scene.background} ({scene.mood})",
                None, new_prev_bg, beats, 0,
                f"📂 Loaded slot {int(slot)}.")

    def on_advance(beats, idx, scene: SceneState, prev_bg: str):
        """Reveal the next beat (click-the-screen / Space / Enter)."""
        if not beats:
            return gr.skip(), gr.skip(), None, gr.skip(), gr.skip()
        new_idx = min((idx or 0) + 1, len(beats) - 1)
        screen, new_prev_bg, voice_path = _render_beats_screen(
            beats, new_idx, scene, bg_provider, prev_bg)
        info = f"Turn {scene.turn} · {scene.background} ({scene.mood})"
        return screen, new_idx, voice_path, info, new_prev_bg

    def on_reset():
        return _empty()

    return on_start, on_choice, on_save, on_load, on_reset, on_advance


# ═══════════════════════════════════════════════════════════════════════
# UI Assembly
# ═══════════════════════════════════════════════════════════════════════

def create_ui():
    """Build and return the Gradio VN app."""
    bg_backend = os.getenv("ARS_FABULA_BG", "preset")
    bg_provider = get_background_provider(bg_backend)
    on_start, on_choice, on_save, on_load, on_reset, on_advance = _make_callbacks(bg_provider)

    initial_scene = make_initial_scene()

    with gr.Blocks(title="Ars-Fabula VN") as app:
        # ── State ──────────────────────────────────────────────
        scene_state = gr.State(initial_scene)
        engine_ref = gr.State(None)
        cast_data = gr.State(DEFAULT_CAST)
        prev_bg_state = gr.State("")
        beats_state = gr.State([])        # ordered display beats for this turn
        beat_idx_state = gr.State(0)      # which beat is currently shown

        # ── Header ──────────────────────────────────────────────
        gr.HTML("""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
            <span style="font-size:2rem;">🌸</span>
            <div>
                <div style="font-size:1.6rem;font-weight:bold;color:#e8a87c;
                            font-family:'Georgia',serif;letter-spacing:1px;">
                    ARS-FABULA
                </div>
                <div style="color:#8888aa;font-size:0.85rem;font-style:italic;">
                    — A Slice-of-Life Visual Novel —
                </div>
            </div>
        </div>
        """)

        with gr.Row(equal_height=False):
            # ── Left: Cast / Controls ──────────────────────────
            with gr.Column(scale=1, min_width=250):
                cast_panel = gr.HTML(_cast_panel_html(DEFAULT_CAST))

                gr.Markdown("### 🎬 New Game")
                cast_preset = gr.Radio(
                    choices=[
                        ("Curated (no ComfyUI needed)", "curated"),
                        ("Generated (requires ComfyUI)", "generated"),
                    ],
                    label="Cast Source",
                    value="curated",
                )
                start_btn = gr.Button("🌸 Start Story", variant="primary", size="lg")

                scene_info = gr.Markdown(
                    "🌸 Prologue · classroom (peaceful)",
                    elem_classes="scene-info",
                )

                gr.Markdown("### 💾 Save / Load")
                save_slot = gr.Number(value=0, minimum=0, maximum=4, step=1,
                                      label="Slot", precision=0)
                with gr.Row():
                    save_btn = gr.Button("💾 Save", size="sm")
                    load_btn = gr.Button("📂 Load", size="sm")
                save_status = gr.Markdown("", elem_classes="scene-info")

                reset_btn = gr.Button("🔄 Reset", variant="stop", size="sm")

            # ── Center: VN Screen ──────────────────────────────
            with gr.Column(scale=3):
                vn_screen = gr.HTML(_vn_screen_html("", {}, [], {}))

                voice_audio = gr.Audio(
                    value=None, autoplay=True, visible=True,
                    label="🔊 Voiced line", interactive=False,
                )

                narration_log = gr.Textbox(
                    label="📜 Scene Log",
                    lines=6, max_lines=12,
                    placeholder="Scene narration will appear here...",
                    interactive=False,
                )

        # ── Hidden JS bridge for in-screen choice buttons ────────
        # visible="hidden" keeps these in the DOM (visible=False would remove
        # them and break the JS bridge)
        choice_input = gr.Textbox(visible="hidden", interactive=True,
                                  elem_id="choice-id-box")
        choice_text_input = gr.Textbox(visible="hidden", interactive=True,
                                       elem_id="choice-text-box")
        submit_choice_btn = gr.Button("Submit Choice", visible="hidden",
                                      elem_id="submit-choice-btn")
        # Hidden "next beat" trigger — clicked by the screen-click / Space-Enter
        # JS hook when there is more text to read this turn.
        advance_btn = gr.Button("Advance", visible="hidden",
                                elem_id="advance-btn")

        main_outputs = [scene_state, engine_ref, vn_screen, cast_data,
                        cast_panel, narration_log, scene_info,
                        voice_audio, prev_bg_state]
        # Turn callbacks also carry the beat list + index (read-at-your-own-pace).
        turn_outputs = main_outputs + [beats_state, beat_idx_state]

        # ── Event wiring ───────────────────────────────────────
        start_btn.click(fn=on_start, inputs=[cast_preset], outputs=turn_outputs)

        submit_choice_btn.click(
            fn=on_choice,
            inputs=[choice_input, choice_text_input, scene_state, engine_ref, prev_bg_state],
            outputs=turn_outputs,
        )

        # Advance one beat: re-render the screen, bump the index, chain the
        # crossfade (prev_bg), and play that beat's voiced line if any.
        advance_btn.click(
            fn=on_advance,
            inputs=[beats_state, beat_idx_state, scene_state, prev_bg_state],
            outputs=[vn_screen, beat_idx_state, voice_audio, scene_info, prev_bg_state],
        )

        save_btn.click(fn=on_save, inputs=[save_slot, scene_state, engine_ref],
                       outputs=[save_status])
        load_btn.click(fn=on_load,
                       inputs=[save_slot, scene_state, engine_ref, prev_bg_state],
                       outputs=turn_outputs + [save_status])

        reset_btn.click(fn=on_reset, outputs=turn_outputs)

    return app


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    port = int(os.getenv("GRADIO_PORT", "7861"))
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    app = create_ui()
    print(f"\n🌸 Ars-Fabula VN starting on http://127.0.0.1:{port}")
    print(f"   Mock mode: {os.getenv('ARS_FABULA_BACKEND', 'server') == 'mock'}")
    print(f"   Voice: {os.getenv('ARS_FABULA_VOICE', 'mock')} · "
          f"BG: {os.getenv('ARS_FABULA_BG', 'preset')}\n")
    app.launch(
        server_name="127.0.0.1",
        server_port=port,
        allowed_paths=[STATIC_DIR],   # belt-and-braces; images are data URIs anyway
        css=CSS,                      # Gradio 6: css/js/theme live on launch()
        js=ONLOAD_JS,
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="rose",
            neutral_hue="slate",
        ),
        quiet=False,
        share=False,
    )
