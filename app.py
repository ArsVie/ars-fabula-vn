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
import html as _html
import io
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


@lru_cache(maxsize=64)
def _thumb_uri_cached(path: str, mtime: float, height: int) -> str:
    try:
        from PIL import Image
        with Image.open(path) as img:
            img = img.convert("RGBA")
            w = max(1, int(img.width * height / img.height))
            img = img.resize((w, height), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        # Fall back to the full image (bigger payload, still works)
        return _data_uri_cached(path, mtime)


def thumb_data_uri(path: Optional[str], height: int = 160) -> str:
    """Downscaled data: URI — keeps the cast-panel payload small."""
    if not path or not os.path.exists(path):
        return ""
    try:
        return _thumb_uri_cached(path, os.path.getmtime(path), height)
    except OSError:
        return ""


# ── Mock VN LLM ───────────────────────────────────────────────────────

def _make_mock_engine(cast: dict[str, CastMember]) -> MockModel:
    """Create a MockModel that generates varied multi-character VN scenes.

    Exercises enter_character/exit_character across beats to verify per-beat
    presence tracking manually. Each branch produces a different character
    lineup so the tester can see characters pop in/out visually.
    """
    char_names = list(cast.keys())
    # Expect Yuki (idx 0), Haruki (idx 1), Sakura (idx 2), Ryo (idx 3)
    yuki = char_names[0] if len(char_names) > 0 else "Yuki"
    haruki = char_names[1] if len(char_names) > 1 else "Haruki"
    sakura = char_names[2] if len(char_names) > 2 else "Sakura"
    ryo = char_names[3] if len(char_names) > 3 else "Ryo"

    def handler(messages, tools):
        user_msgs = [m for m in messages if m["role"] == "user"]
        raw_choice = user_msgs[-1]["content"] if user_msgs else ""
        # Extract just the player's choice text after [CHOICE MADE]
        choice_match = __import__("re").search(r"\[CHOICE MADE\]\n(.+?)\n\nWrite the next scene", raw_choice, __import__("re").DOTALL)
        choice_text = choice_match.group(1).strip() if choice_match else ""
        choice_lower = choice_text.lower()

        # ── "talk" → Yuki + Haruki enter one-by-one ──────────────
        if "talk" in choice_lower or yuki.lower() in choice_lower:
            return (
                f'[TOOL: enter_character character="{yuki}"]\n\n'
                f'[TOOL: set_expression character="{yuki}" expression="surprised"]\n\n'
                f'[TOOL: advance_scene description="The afternoon sun filters through the classroom windows, casting warm rectangles of light across the wooden floor. The scent of chalk dust and summer lingers in the air." background="classroom" mood="cheerful"]\n\n'
                f'[TOOL: set_expression character="{yuki}" expression="smile"]\n\n'
                f'[{yuki}]: "Oh, hey! I didn\'t expect you to come over today. What\'s up?"\n\n'
                f'Her smile is warm and genuine, the afternoon light catching in her hair.\n\n'
                f'[TOOL: enter_character character="{haruki}"]\n\n'
                f'[TOOL: set_expression character="{yuki}" expression="surprised"]\n\n'
                f'The door slides open and {haruki} pokes their head in, eyebrows raising as they spot the two of you.\n\n'
                f'[TOOL: set_expression character="{haruki}" expression="laugh"]\n\n'
                f'[{haruki}]: "Am I interrupting something? Should I come back later?"\n\n'
                f'They\'re clearly teasing, a playful glint in their eye.\n\n'
                f'[TOOL: set_expression character="{yuki}" expression="embarrassed"]\n\n'
                f'[{yuki}]: "No! I mean — we were just talking. Come sit down."\n\n'
                f'The atmosphere shifts into something lighter, the three of you settling into the familiar rhythm of friends.\n\n'
                f'[TOOL: offer_choices choices=\'[{{"id":"ask","text":"\\\"So what did you want to talk about?\\\"","consequence":"get serious"}},{{"id":"tease","text":"\\\"Look who\'s getting protective~\\\"","consequence":"tease them both"}},{{"id":"group","text":"\\\"Let\'s all hang out after school!\\\"","consequence":"make plans together"}}]\']'
            )

        # ── "look" / "alone" → Yuki enters, exits, then Sakura enters ──
        # NOTE: no offer_choices — exercises the free-text fallback!
        elif "look" in choice_lower or "alone" in choice_lower:
            return (
                f'[TOOL: enter_character character="{yuki}"]\n\n'
                f'[TOOL: set_expression character="{yuki}" expression="sad"]\n\n'
                f'[TOOL: advance_scene description="The classroom feels emptier than usual. Through the window, the courtyard buzzes with laughter muffled by glass." background="classroom" mood="melancholic"]\n\n'
                f'[{yuki}]: "I\'ve been thinking a lot lately. About everything, really."\n\n'
                f'They stare out the window, their reflection overlapping with the clouds drifting past.\n\n'
                f'[TOOL: exit_character character="{yuki}"]\n\n'
                f'The door clicks shut. The room settles into silence, dust motes dancing in the slanted light.\n\n'
                f'[TOOL: enter_character character="{sakura}"]\n\n'
                f'[TOOL: set_expression character="{sakura}" expression="smile"]\n\n'
                f'[{sakura}]: "Hey. I saw {yuki} heading to the roof. They looked kinda down."\n\n'
                f'She leans against the doorframe, concern flickering behind her cheerful demeanour. The silence stretches, waiting for you to fill it.'
            )

        # ── "confess" / "love" → Rooftop with interrupted confession ──
        elif "confess" in choice_lower or "love" in choice_lower:
            return (
                f'[TOOL: enter_character character="{yuki}"]\n\n'
                f'[TOOL: set_expression character="{yuki}" expression="surprised"]\n\n'
                f'[TOOL: advance_scene description="Time seems to slow. The rooftop stretches out before you, the city glittering in the distance. Cherry blossom petals drift lazily on the warm breeze." background="rooftop" mood="romantic"]\n\n'
                f'[TOOL: set_expression character="{yuki}" expression="embarrassed"]\n\n'
                f'[{yuki}]: "I… I\'ve been wanting to tell you something for a while now."\n\n'
                f'For a long, breathless moment, neither of you speaks. The wind carries petals between you.\n\n'
                f'[TOOL: enter_character character="{haruki}"]\n\n'
                f'[TOOL: set_expression character="{yuki}" expression="surprised"]\n\n'
                f'[TOOL: set_expression character="{haruki}" expression="laugh"]\n\n'
                f'[{haruki}]: "Whoa — sorry, sorry! I didn\'t mean to walk in on… whatever this is."\n\n'
                f'They back away with their hands up, grinning, and disappear back through the door.\n\n'
                f'[TOOL: exit_character character="{haruki}"]\n\n'
                f'The moment hangs in the air again, fragile and electric.\n\n'
                f'[TOOL: set_expression character="{yuki}" expression="smile"]\n\n'
                f'[{yuki}]: "As I was saying… I really like you. I have for a long time."\n\n'
                f'The sunset paints the world in shades of gold and pink.\n\n'
                f'[TOOL: offer_choices choices=\'[{{"id":"hold_hands","text":"Take their hand","consequence":"a gentle touch","next_background":"sunset_hill","next_mood":"romantic","affection_flag":["{yuki}","confession"]}},{{"id":"hug","text":"Pull them into a hug","consequence":"embrace","next_background":"rooftop","next_mood":"romantic"}},{{"id":"speak","text":"\\\"I\'ve wanted to say it for so long too…\\\"","consequence":"vulnerable truth","next_background":"rooftop","next_mood":"romantic"}}]\']'
            )

        # ── Default / first turn → three characters gradually enter ──
        else:
            return (
                f'[TOOL: enter_character character="{yuki}"]\n\n'
                f'[TOOL: advance_scene description="The bell rings, signalling the end of another school day. Students begin packing their bags, the air filled with chatter and the shuffle of papers. Sunlight streams through the tall windows." background="classroom" mood="peaceful"]\n\n'
                f'[TOOL: set_expression character="{yuki}" expression="smile"]\n\n'
                f'[{yuki}]: "Hey! Ready to head out?"\n\n'
                f'They pack their bag with practiced ease, glancing at you expectantly.\n\n'
                f'[TOOL: enter_character character="{haruki}"]\n\n'
                f'[TOOL: set_expression character="{haruki}" expression="laugh"]\n\n'
                f'[{haruki}]: "Leaving without me? I\'m hurt, really."\n\n'
                f'They saunter over, slinging their bag over one shoulder, a familiar teasing grin on their face.\n\n'
                f'[TOOL: set_expression character="{yuki}" expression="laugh"]\n\n'
                f'[{yuki}]: "You take forever, {haruki}. We were about to send a search party."\n\n'
                f'[TOOL: enter_character character="{sakura}"]\n\n'
                f'[TOOL: set_expression character="{sakura}" expression="smile"]\n\n'
                f'[{sakura}]: "Are we doing the café thing? I\'m starving."\n\n'
                f'She appears from the hallway, already halfway out the door, flipping her hair with practiced nonchalance.\n\n'
                f'[TOOL: set_expression character="{haruki}" expression="smile"]\n\n'
                f'[{haruki}]: "Café sounds perfect. My treat — I just got paid for that tutoring gig."\n\n'
                f'The three of them look at you, waiting, the late afternoon light pooling around them like a scene from a movie.\n\n'
                f'[TOOL: offer_choices choices=\'[{{"id":"yes_cafe","text":"\\\"Sure, let\'s go!\\\"","consequence":"go to the cafe together"}},{{"id":"rain_check","text":"\\\"Maybe tomorrow? I\'ve got stuff to do.\\\"","consequence":"politely decline"}},{{"id":"walk","text":"\\\"I\'ll walk with you guys but I can\'t stay long.\\\"","consequence":"partial join"}}]\']'
            )

    return MockModel(handler=handler)


# ── LLM / engine builders ─────────────────────────────────────────────

def _try_real_llm():
    """Try to connect to the real model server. Returns client or None,
    and says exactly WHY it fell back when it does."""
    cfg = ModelConfig.from_env()
    if cfg.backend == "mock":
        print("[app] ARS_FABULA_BACKEND=mock is set — mock engine forced "
              "(unset the env var for the real model)")
        return None
    try:
        client = get_model(cfg)
        if client.health():
            print(f"[app] Connected to LLM server at {cfg.base_url}")
            return client
        base = cfg.base_url.rsplit("/v1", 1)[0]
        print(f"[app] No model server at {cfg.base_url} "
              f"(GET {base}/health got no healthy reply).")
        print("[app] Start llama-server on that port, or set "
              "ARS_FABULA_BASE_URL to where it lives.")
    except Exception as e:
        print(f"[app] LLM server check failed: {e}")
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
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400;1,600&family=Cinzel:wght@400;600&family=Noto+Sans+JP:wght@300;400&display=swap');

/* ── Tokens ── */
:root {
  --bg-void:       #0d0d1a;
  --bg-surface:    #13132a;
  --bg-raised:     #1c1c38;
  --bg-screen:     #08081a;
  --border-subtle: #252545;
  --border-mid:    #3a3a62;
  --border-accent: #e8a87c;
  --border-glow:   rgba(232,168,124,0.35);
  /* One warm text family + gold accent — cool purples are reserved for
     surfaces/borders only, so no text sinks into the dark box. */
  --text-primary:  #ede5da;
  --text-secondary:#b9b0a1;
  --text-muted:    #8f8779;
  --text-accent:   #e8a87c;
  --text-narrator: #c6bdac;
  --dialogue-bg:   rgba(8,8,26,0.92);
  --font-display:  'Cinzel', 'Georgia', serif;
  --font-dialogue: 'Cormorant Garamond', 'Georgia', serif;
  --font-ui:       'Noto Sans JP', 'Georgia', serif;
}

/* ── Global Reset ── */
.gradio-container {
  background: var(--bg-void) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-ui) !important;
  max-width: 1100px !important;   /* centered stage, not full-bleed */
  margin: 0 auto !important;
}
body { background: var(--bg-void) !important; }
*:focus { outline: 2px solid var(--border-accent) !important; outline-offset: 2px; }

/* ── VN Screen ── */
.vn-screen {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
  min-height: 380px;
  max-height: 620px;
  background: var(--bg-screen);
  border: 1px solid var(--border-mid);
  border-radius: 6px;
  overflow: hidden;
  margin-bottom: 8px;
}

/* Background layers */
.vn-bg {
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 100%;
  object-fit: cover;
}
.vn-bg.new { animation: vnBgFade 1.3s ease both; }
@keyframes vnBgFade { from { opacity: 0; } to { opacity: 1; } }

/* Sprites — anchored to the SCREEN bottom and layered BEHIND the dialogue
   box (z 2 < dialogue 4), per VN staging: the box overlaps the sprites'
   lower torso, concealing the raw bottom edge of upper-body renders.
   The container spans the full screen so percentage heights resolve. */
.vn-chars {
  position: absolute;
  top: 0; bottom: 0; left: 0;
  width: 100%;
  display: flex;
  justify-content: center;
  align-items: flex-end;     /* feet on the screen bottom */
  gap: 24px;
  pointer-events: none;
  z-index: 2;
}
.vn-sprite {
  height: 86%;               /* heads near the top, torso behind the box */
  width: auto;
  max-width: 28%;            /* 3-4 characters fit without side clipping */
  object-fit: contain;
  object-position: bottom center;
  animation: vnSpriteIn 0.5s ease both;
  transition: filter 0.35s ease, opacity 0.35s ease;
}
/* Active/inactive: the speaker is bright, the rest gently shadowed. */
.vn-sprite.active   { filter: none; opacity: 1; }
.vn-sprite.inactive {
  filter: brightness(0.6) saturate(0.85) contrast(0.95);
  opacity: 0.9;
}
@keyframes vnSpriteIn {
  from { opacity: 0.2; transform: translateY(10px) scale(0.97); }
  to   { opacity: 1;   transform: translateY(0)    scale(1); }
}

/* ── Dialogue Box ── (shorter now; tray sits centered at its bottom) */
.vn-dialogue {
  position: absolute;
  bottom: 0; left: 0; right: 0;
  background: linear-gradient(to top, rgba(8,8,26,0.97) 0%, rgba(8,8,26,0.85) 60%, transparent 100%);
  padding: 14px 28px 30px;   /* extra bottom room for the centered tray */
  min-height: 92px;
}
.vn-dialogue.has-speaker {
  border-left: 3px solid var(--border-accent);
}
.vn-name-plate {
  display: inline-block;
  font-family: var(--font-display);
  font-size: 0.9rem;
  letter-spacing: 0.08em;
  color: var(--text-accent);
  background: rgba(232,168,124,0.1);
  border: 1px solid var(--border-accent);
  border-radius: 3px;
  padding: 2px 10px;
  margin-bottom: 8px;
}
.vn-name {
  font-family: var(--font-display);
  font-weight: 600;
  color: var(--text-accent);
  font-size: 1rem;
  margin-bottom: 6px;
  letter-spacing: 0.06em;
}
.vn-text {
  font-family: var(--font-dialogue);
  color: var(--text-primary);
  font-size: 1.15rem;
  line-height: 1.7;
  margin: 0;
  text-indent: 1.6em;   /* sangría — first-line indent on dialogue/narration */
}
.vn-text.narrator {
  font-style: italic;
  color: var(--text-narrator);
}

/* Typewriter cursor */
.vn-text.typing::after {
  content: '▌';
  animation: vnBlink 0.7s step-end infinite;
  color: var(--text-accent);
  margin-left: 1px;
}
@keyframes vnBlink { 50% { opacity: 0; } }

/* Click-to-advance affordance */
.vn-screen.advancing { cursor: pointer; }
.vn-continue {
  position: absolute;
  right: 16px; bottom: 10px;
  color: var(--text-accent);
  font-size: 1rem;
  animation: vnPulse 1.2s ease infinite;
  pointer-events: none;
}
@keyframes vnPulse { 0%,100% { opacity: 0.4; } 50% { opacity: 1; } }

/* ── Choices ── (raised so the free-text bar docks directly beneath them
   in the same centered column — one interaction cluster, not two) */
.vn-choices {
  position: absolute;
  bottom: 206px;
  left: 50%;
  transform: translateX(-50%);
  width: 70%;
  display: flex;
  flex-direction: column;
  gap: 6px;
  pointer-events: auto;
}
.vn-choice-btn {
  background: rgba(19,19,42,0.88) !important;
  backdrop-filter: blur(8px);
  border: 1px solid var(--border-mid) !important;
  border-left: 3px solid var(--border-accent) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-dialogue) !important;
  font-size: 1rem !important;
  padding: 11px 18px !important;
  border-radius: 4px !important;
  text-align: left !important;
  cursor: pointer !important;
  transition: border-color 0.18s, background 0.18s, transform 0.12s !important;
}
.vn-choice-btn:hover {
  background: rgba(35,35,70,0.95) !important;
  border-left-color: var(--border-accent) !important;
  border-color: var(--border-accent) !important;
  transform: translateX(3px);
  color: #fff !important;
}

/* ── Casting beat ── */
.vn-casting {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  gap: 24px;
  background: radial-gradient(ellipse at center, #1c1c38 0%, var(--bg-screen) 70%);
}
.vn-casting .casting-title {
  font-family: var(--font-display);
  color: var(--text-accent);
  font-size: 1.1rem;
  letter-spacing: 0.15em;
  animation: vnPulse 1.6s ease infinite;
}
.vn-casting .portraits { display: flex; gap: 24px; }
.vn-portrait {
  width: 110px; height: 160px;
  border-radius: 6px;
  border: 1px solid var(--border-mid);
  background: var(--bg-surface);
  overflow: hidden;
  display: flex; align-items: center; justify-content: center;
}
.vn-portrait.pending { border-style: dashed; border-color: var(--border-subtle); }
.vn-portrait img {
  width: 100%; height: 100%;
  object-fit: cover; object-position: top;
  animation: portraitReveal 0.5s ease both;
}
@keyframes portraitReveal {
  from { opacity: 0; transform: scale(0.92) translateY(4px); }
  to   { opacity: 1;   transform: scale(1)    translateY(0); }
}
.pname {
  font-family: var(--font-ui);
  font-size: 0.78rem;
  color: var(--text-accent);
  margin-top: 4px;
  letter-spacing: 0.06em;
}
.pname.pending {
  color: var(--text-muted);
}
.portrait-wrap {
  text-align: center;
}

/* ── Cast Panel extras ── */
.cast-expressions {
  font-family: var(--font-ui);
  font-size: 0.72rem;
  color: var(--text-muted);
  margin-top: 4px;
}
.cast-panel-empty {
  padding: 8px;
  color: var(--text-muted);
  font-family: var(--font-dialogue);
  font-style: italic;
  font-size: 0.88rem;
}

/* ── Cast Cards (sidebar) ── */
.cast-card {
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-left: 3px solid var(--border-accent);
  border-radius: 5px;
  padding: 9px 12px;
  margin-bottom: 6px;
  transition: border-color 0.15s;
}
.cast-card:hover { border-color: var(--border-mid); }
.cast-card .name {
  font-family: var(--font-display);
  font-size: 0.88rem;
  color: var(--text-accent);
  letter-spacing: 0.06em;
}
.cast-card .role {
  font-family: var(--font-ui);
  font-size: 0.72rem;
  color: var(--text-muted);
  margin-top: 2px;
}
.cast-card .personality {
  font-family: var(--font-dialogue);
  font-style: italic;
  font-size: 0.82rem;
  color: var(--text-secondary);
  margin-top: 5px;
  line-height: 1.4;
}

/* ── Header ── */
.vn-header { padding: 14px 0 12px; margin-bottom: 14px; }
.vn-title-main {
  font-family: var(--font-display);
  font-size: 1.9rem;
  font-weight: 600;
  color: var(--text-accent);
  letter-spacing: 0.3em;
  text-shadow: 0 0 40px rgba(232,168,124,0.25);
}
.vn-title-sub {
  font-family: var(--font-dialogue);
  font-style: italic;
  font-size: 0.82rem;
  color: var(--text-muted);
  letter-spacing: 0.04em;
  margin-top: 2px;
}
.vn-header-rule {
  height: 1px;
  background: linear-gradient(to right, var(--border-accent), transparent);
  margin-top: 10px;
  opacity: 0.35;
}

/* ── Sidebar ── */
.sidebar-col { border-right: 1px solid var(--border-subtle); padding-right: 14px !important; }
.sidebar-section-label {
  font-family: var(--font-display);
  font-size: 0.68rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--text-accent);
  border-bottom: 1px solid var(--border-subtle);
  padding-bottom: 4px;
  margin: 16px 0 10px;
}

/* ── Scene Info HUD ── */
.scene-info p, .scene-info {
  font-family: var(--font-ui) !important;
  font-size: 0.72rem !important;
  color: var(--text-muted) !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  border-left: 2px solid var(--border-subtle);
  padding-left: 8px;
  margin: 4px 0 !important;
}

/* ── Buttons ── */
button { border-radius: 3px !important; transition: all 0.18s ease !important; }
.gradio-container button:not(.vn-choice-btn) {
  font-family: var(--font-display) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-mid) !important;
  color: var(--text-secondary) !important;
}
.gradio-container button:not(.vn-choice-btn):hover {
  border-color: var(--border-accent) !important;
  color: var(--text-accent) !important;
  background: var(--bg-raised) !important;
}
button[variant="primary"] {
  background: linear-gradient(135deg, #c07840, #e8a87c) !important;
  border-color: var(--border-accent) !important;
  color: #0d0d1a !important;
  font-weight: 600 !important;
}
button[variant="primary"]:hover {
  box-shadow: 0 0 14px var(--border-glow) !important;
}
button[variant="stop"] { border-color: #c06060 !important; color: #c06060 !important; }

/* ── Free-text input ── */
#free-input-row { margin-top: 8px; border-top: 1px solid var(--border-subtle); padding-top: 8px; }
#free-input textarea, #free-input input {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-mid) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-dialogue) !important;
  font-size: 1rem !important;
  font-style: italic !important;
  caret-color: var(--text-accent) !important;
  border-radius: 3px !important;
}
#free-input textarea:focus, #free-input input:focus {
  border-color: var(--border-accent) !important;
  box-shadow: 0 0 0 2px var(--border-glow) !important;
}
#free-input label {
  font-family: var(--font-ui) !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
}
#free-submit-btn button {
  background: linear-gradient(135deg, #c07840, #e8a87c) !important;
  border-color: var(--border-accent) !important;
  color: #0d0d1a !important;
  font-weight: 600 !important;
}
#free-submit-btn button:hover {
  box-shadow: 0 0 14px var(--border-glow) !important;
}

/* ── Narration Log ── */
.gr-textbox textarea {
  background: var(--bg-void) !important;
  border: 1px solid var(--border-subtle) !important;
  color: var(--text-narrator) !important;
  font-family: var(--font-dialogue) !important;
  font-style: italic !important;
  font-size: 0.88rem !important;
  line-height: 1.65 !important;
  border-radius: 4px !important;
}
label {
  font-family: var(--font-ui) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
}

/* ── Number input (save slot) ── */
.gr-number input {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-mid) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-ui) !important;
  border-radius: 3px !important;
  text-align: center !important;
}

/* ── Hide audio widget (audio still plays via autoplay) ── */
.gr-audio { height: 2px !important; overflow: hidden !important; opacity: 0 !important; pointer-events: none !important; margin: 0 !important; }

/* ── Gradio headings ── */
h1, h2, h3 { font-family: var(--font-display) !important; color: var(--text-accent) !important; }

/* ════════════════════════════════════════════════════════════════
   REDESIGN — everything lives inside the screen (Ren'Py grammar)
   ════════════════════════════════════════════════════════════════ */

/* Stacking order: dialogue 4 · choices 5 · tray 10 · panels 20 */
.vn-dialogue { z-index: 4; }
.vn-choices  { z-index: 5; }

/* ── Icon tray (top-right of dialogue box) ── */
.vn-icon-tray {
  position: absolute;
  bottom: 7px; left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 8px;
  align-items: center;
  z-index: 10;
}
.vn-tray-btn {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid var(--border-mid) !important;
  color: var(--text-secondary) !important;
  font-family: var(--font-ui) !important;
  font-size: 0.58rem !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  padding: 2px 8px !important;
  cursor: pointer !important;
  border-radius: 3px !important;
  transition: color 0.15s, border-color 0.15s, background 0.15s !important;
}
.vn-tray-btn:hover, .vn-tray-btn.active {
  color: var(--text-accent) !important;
  border-color: var(--border-accent) !important;
  background: rgba(232,168,124,0.1) !important;
}
.vn-icon-tray .vn-tray-confirm-row { display: none; }
.vn-icon-tray.confirming .vn-tray-normal { display: none; }
.vn-icon-tray.confirming .vn-tray-confirm-row { display: inline !important; }
.vn-reset-confirm { cursor: pointer; color: #c06060; }
.vn-reset-confirm:hover { color: #e07070; }
.vn-reset-cancel { cursor: pointer; color: var(--text-muted); }
.vn-reset-cancel:hover { color: var(--text-secondary); }

/* ── Overlay panels ── */
.vn-panel {
  position: absolute;
  z-index: 20;
  background: rgba(10, 10, 28, 0.97);
  font-family: var(--font-ui);
}
.vn-panel-title {
  font-family: var(--font-display);
  font-size: 0.72rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--text-accent);
  border-bottom: 1px solid var(--border-subtle);
  padding-bottom: 6px;
  margin-bottom: 12px;
}
.vn-panel-close {
  position: absolute;
  top: 10px; right: 12px;
  background: transparent !important;
  border: none !important;
  color: var(--text-muted) !important;
  font-size: 0.9rem !important;
  cursor: pointer !important;
  padding: 2px 6px !important;
}
.vn-panel-close:hover { color: var(--text-accent) !important; }

/* Log panel — slides up over the dialogue box */
.vn-panel-log {
  bottom: 0; left: 0; right: 0;
  height: 55%;
  border-top: 1px solid var(--border-mid);
  transform: translateY(102%);
  transition: transform 0.3s ease;
  overflow-y: auto;
  padding: 18px 28px;
}
.vn-panel-log.open { transform: translateY(0); }
.vn-log-text {
  font-family: var(--font-dialogue);
  font-style: italic;
  font-size: 0.95rem;
  line-height: 1.7;
  color: var(--text-narrator);
  white-space: pre-wrap;
}

/* Cast panel — slides in from the right */
.vn-panel-cast {
  top: 0; right: 0; bottom: 0;
  width: 40%;
  min-width: 260px;
  border-left: 1px solid var(--border-mid);
  transform: translateX(102%);
  transition: transform 0.3s ease;
  overflow-y: auto;
  padding: 18px 16px;
}
.vn-panel-cast.open { transform: translateX(0); }
.cast-panel-card {
  display: flex;
  gap: 10px;
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-left: 3px solid var(--border-accent);
  border-radius: 5px;
  padding: 8px 10px;
  margin-bottom: 8px;
}
.cast-panel-thumb {
  width: 56px; height: 80px;
  object-fit: cover; object-position: top;
  border-radius: 3px;
  flex-shrink: 0;
  background: var(--bg-raised);
}
.cast-panel-name {
  font-family: var(--font-display);
  font-size: 0.85rem;
  color: var(--text-accent);
  letter-spacing: 0.06em;
}
.cast-panel-role {
  font-size: 0.7rem;
  color: var(--text-muted);
  margin-top: 1px;
}
.cast-panel-personality {
  font-family: var(--font-dialogue);
  font-style: italic;
  font-size: 0.78rem;
  color: var(--text-secondary);
  margin-top: 4px;
  line-height: 1.35;
}

/* Save panel — centered modal */
.vn-panel-save {
  top: 50%; left: 50%;
  transform: translate(-50%, -50%) scale(0.95);
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.2s, transform 0.2s;
  width: 340px;
  border: 1px solid var(--border-mid);
  border-radius: 6px;
  padding: 22px 24px;
  box-shadow: 0 8px 60px rgba(0,0,0,0.6);
}
.vn-panel-save.open {
  opacity: 1;
  transform: translate(-50%, -50%) scale(1);
  pointer-events: auto;
}
.vn-slot-row { display: flex; gap: 8px; margin-bottom: 14px; }
.vn-slot-btn {
  flex: 1;
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-mid) !important;
  color: var(--text-secondary) !important;
  font-family: var(--font-display) !important;
  font-size: 0.8rem !important;
  padding: 7px 0 !important;
  border-radius: 3px !important;
  cursor: pointer !important;
  text-align: center !important;
}
.vn-slot-btn.active {
  border-color: var(--border-accent) !important;
  color: var(--text-accent) !important;
  background: rgba(232,168,124,0.1) !important;
}
.vn-save-actions { display: flex; gap: 10px; }
.vn-save-action {
  flex: 1;
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-mid) !important;
  color: var(--text-secondary) !important;
  font-family: var(--font-display) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  padding: 9px 0 !important;
  border-radius: 3px !important;
  cursor: pointer !important;
}
.vn-save-action:hover {
  border-color: var(--border-accent) !important;
  color: var(--text-accent) !important;
}
.vn-save-status {
  margin-top: 12px;
  font-size: 0.74rem;
  color: var(--text-muted);
  font-style: italic;
  min-height: 1.1em;
}

/* ── Title screen ── */
.vn-title-screen {
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(160deg, #0d0d1a 0%, #16162e 60%, #0a0a20 100%);
}
.vn-title-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 22px;
}
.vn-title-screen .vn-title-main { font-size: 3rem; letter-spacing: 0.35em; }
.vn-title-screen .vn-title-sub  { font-size: 0.95rem; margin-top: -14px; }
.vn-title-cast-choice {
  display: flex;
  gap: 0;
  border: 1px solid var(--border-mid);
  border-radius: 3px;
  overflow: hidden;
}
.vn-title-opt {
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  color: var(--text-muted) !important;
  font-family: var(--font-ui) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  padding: 8px 18px !important;
  cursor: pointer !important;
  transition: background 0.15s, color 0.15s !important;
}
.vn-title-opt.active {
  background: rgba(232,168,124,0.12) !important;
  color: var(--text-accent) !important;
}
.vn-title-begin {
  font-family: var(--font-display) !important;
  font-size: 0.9rem !important;
  letter-spacing: 0.22em !important;
  text-transform: uppercase !important;
  background: linear-gradient(135deg, #c07840, #e8a87c) !important;
  border: none !important;
  color: #0d0d1a !important;
  padding: 12px 44px !important;
  border-radius: 3px !important;
  cursor: pointer !important;
  font-weight: 600 !important;
  transition: box-shadow 0.2s !important;
}
.vn-title-begin:hover { box-shadow: 0 0 22px var(--border-glow) !important; }

/* ── Choice restyle (scrim + accent bar, dialogue voice) ── */
.vn-choice-btn {
  background: rgba(8,8,26,0.55) !important;
  backdrop-filter: blur(4px);
  border: none !important;
  border-left: 2px solid var(--border-accent) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-dialogue) !important;
  font-style: italic !important;
  font-size: 1.05rem !important;
  padding: 9px 16px !important;
  border-radius: 0 !important;
  transition: color 0.15s, padding-left 0.15s, background 0.15s !important;
}
.vn-choice-btn:hover {
  background: rgba(8,8,26,0.75) !important;
  color: var(--text-accent) !important;
  padding-left: 24px !important;
  transform: none;
}

/* ── Stage wrapper — the relative containing block for the canvas + the
   free-text bar. The bar overlays INSIDE the canvas frame, anchored to this
   wrapper (an earlier version had no positioned ancestor and the box flew
   across the viewport). ── */
#stage { position: relative !important; gap: 0 !important; padding: 0 !important;
         margin-bottom: 8px !important; }
/* The gr.HTML wrapper insets the canvas by 12px each side; remove it so the
   canvas spans the full stage width and the free-text bar (70% of the stage)
   lines up with the choices (70% of the canvas). */
#stage .html-container { padding: 0 !important; }
#stage .vn-screen { margin-bottom: 0 !important; }

/* ── Free-text action bar — docked inside the canvas, in the same centered
   column as the choices (which sit just above it) and styled to match the
   choice buttons, so it reads as part of the interaction cluster rather than
   a glued-on pill. Hidden until the END of a turn — revealed only on the last
   beat once the text has finished revealing, or as soon as choices appear (JS
   sets body.vn-playing; see syncStage). ── */
/* The bar is styled to be visually identical to a .vn-choice-btn (see the
   "Choice restyle" block above): the same translucent void-scrim, blur(4px),
   borderless 2px accent left bar, sharp corners and italic dialogue voice —
   so it reads as just one more option in the stack, not a separate widget. */
#free-input-row {
  display: none;              /* shown via body.vn-playing below */
  align-items: stretch;
  flex-wrap: nowrap !important;   /* Gradio rows wrap by default — keep ➤ inline */
  gap: 0;
  position: absolute;
  bottom: 140px;
  left: 50%;
  transform: translateX(-50%);
  width: 70%;
  z-index: 6;                 /* above dialogue (4) and choices (5) */
  background: rgba(8,8,26,0.55);
  backdrop-filter: blur(4px);
  border: none;
  border-left: 2px solid var(--border-accent);
  border-radius: 0;
  overflow: hidden;
  pointer-events: auto;
  transition: background 0.15s;
}
body.vn-playing #free-input-row { display: flex !important; }
#free-input-row:focus-within { background: rgba(8,8,26,0.75); }

/* The textbox fills the bar with no inner chrome — every Gradio wrapper
   background is forced transparent so only the row's scrim shows (otherwise
   the opaque .block fill makes the bar look solid). NOTE: in this Gradio
   version the <input> is nested INSIDE the <label>, so the label must stay
   laid out (only its caption span is hidden — Gradio tags it .hide). The
   empty default .wrap is collapsed so the input claims the full width. */
#free-input { flex: 8 1 auto; min-width: 0; display: flex !important; align-items: stretch; }
#free-input, #free-input .wrap, #free-input label, #free-input .label-wrap,
#free-input .input-container, #free-input .form, #free-input .block {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
#free-input > .wrap { display: none !important; }
#free-input label, #free-input .label-wrap {
  display: flex !important;
  flex: 1 1 auto; min-width: 0;
  margin: 0 !important; padding: 0 !important;
}
#free-input label > span { display: none !important; }   /* the caption text */
#free-input .input-container { display: flex !important; flex: 1 1 auto; min-width: 0; }
#free-input textarea, #free-input input {
  flex: 1 1 auto; width: 100% !important;
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  outline: none !important;
  color: var(--text-primary) !important;
  font-family: var(--font-dialogue) !important;
  font-style: italic !important;
  font-size: 1.05rem !important;
  caret-color: var(--text-accent) !important;
  padding: 9px 16px !important;
}
#free-input input::placeholder, #free-input textarea::placeholder {
  color: var(--text-muted) !important; opacity: 1;
}
/* The ➤ send button is part of the same scrim — transparent, just an accent
   glyph that brightens on hover (no filled chip, no divider). The elem_id is
   on the <button> itself in this Gradio version, so target both it and any
   wrapper to stay version-resilient. */
#free-submit-btn, #free-submit-btn button {
  flex: 0 0 auto !important;
  align-self: stretch;
  min-width: 44px !important; width: 44px !important;
  height: auto !important;
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  color: var(--text-accent) !important;
  font-size: 1.1rem !important;
  font-weight: 600 !important;
  opacity: 0.8;
  cursor: pointer !important;
  transition: color 0.15s, opacity 0.15s;
}
#free-submit-btn:hover, #free-submit-btn button:hover {
  background: transparent !important;
  color: #fff !important;
  opacity: 1;
  box-shadow: none !important;
}

/* ── Collapse the audio widget (autoplay still fires) ── */
#voice-audio {
  height: 2px !important;
  overflow: hidden !important;
  opacity: 0 !important;
  pointer-events: none !important;
  margin: 0 !important;
  padding: 0 !important;
}
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
    // Already completed (e.g. user clicked during the settle delay) — don't restart
    if (screen.getAttribute('data-typing') === '0') return;

    // Clear any running timer
    if (_typewriterTimer) { clearInterval(_typewriterTimer); _typewriterTimer = null; }

    const textEl = screen.querySelector('.vn-text');
    if (!textEl) { screen.setAttribute('data-typing', '0'); return; }

    const fullText = textEl.getAttribute('data-full-text') || textEl.textContent || '';
    if (!fullText) { screen.setAttribute('data-typing', '0'); return; }
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

// Watch for VN screen changes: start typewriter, restore overlay-panel UI
// state (the canvas HTML is replaced wholesale each beat/turn), and mirror
// the hidden save status into the save panel.
const _typewriterObserver = new MutationObserver(function() {
    // 'pending' marks a freshly rendered screen that hasn't started typing.
    // ('1' = typing now, '0' = done — neither must retrigger, or the
    // typewriter's own text mutations would restart it in a loop.)
    const screen = document.querySelector('.vn-screen');
    if (screen && screen.getAttribute('data-typing') === 'pending') {
        screen.setAttribute('data-typing', '1');
        // Small delay so the DOM settles
        setTimeout(function() { startTypewriter(screen); }, 50);
    }
    restoreUiState();
    syncSaveStatus();
    syncStage();
});

// Reveal the free-text bar only at the END of a turn: when we're on the last
// beat (data-more != "1") and the text has finished revealing (data-typing
// "0") — or as soon as choices are offered. It stays hidden mid-turn, while
// earlier beats are still being clicked through or typed out, and on the
// title/casting screens. Pure class toggle on <body>.
function syncStage() {
    const canvas = document.querySelector('.vn-screen');
    const inScene = !!(canvas && canvas.querySelector('.vn-dialogue'));
    const lastBeat = inScene && canvas.getAttribute('data-more') !== '1';
    const typingDone = inScene && canvas.getAttribute('data-typing') === '0';
    const hasChoices = inScene && !!canvas.querySelector('.vn-choices');
    const showBar = lastBeat && (typingDone || hasChoices);
    document.body.classList.toggle('vn-playing', showBar);
}

// Enter in the free-text box submits via the proven hidden-button bridge
// (not Gradio's flaky .submit). The ➤ button click works natively.
document.addEventListener('keydown', function(e) {
    if (e.key !== 'Enter') return;
    const active = document.activeElement;
    if (active && active.closest && active.closest('#free-input')) {
        e.preventDefault();
        clickHidden('#free-submit-btn');
    }
});
// Start observing once the body is ready
(function waitBody() {
    if (document.body) {
        // attributeFilter keeps class-toggle churn (restoreUiState) from
        // re-firing the observer; only data-typing flips matter here.
        _typewriterObserver.observe(document.body, {
            childList: true, subtree: true,
            attributes: true, attributeFilter: ['data-typing'],
        });
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
        // A click while a panel is open closes the panel instead of advancing.
        if (screen.getAttribute('data-more') === '1'
            && !(window._vnUi && window._vnUi.panel)
            && !e.target.closest('.vn-choices')
            && !e.target.closest('.vn-panel')
            && !e.target.closest('.vn-icon-tray')) {
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

// NOTE: the free-text box is cleared SERVER-SIDE (on_free_submit returns ""
// into it) — never with a JS timeout, which raced Gradio's value capture and
// caused submits to be read as empty and silently dropped.

// ════════════════════════════════════════════════════════════════
// REDESIGN — title screen, icon tray, overlay panels
// ════════════════════════════════════════════════════════════════

// Shared helper: write a value into a hidden gradio Textbox/Number
function setHiddenVal(rootSel, val) {
    const el = document.querySelector(rootSel + ' textarea, ' + rootSel + ' input');
    if (!el) return;
    const proto = el.tagName === 'TEXTAREA'
        ? window.HTMLTextAreaElement.prototype
        : window.HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value');
    if (setter && setter.set) { setter.set.call(el, val); }
    else { el.value = val; }
    el.dispatchEvent(new Event('input', { bubbles: true }));
}

// UI state that must survive canvas re-renders (the HTML is replaced
// wholesale every beat/turn, which would otherwise close panels and
// reset the slot pills).
window._vnUi = window._vnUi || { panel: null, slot: '0' };

function restoreUiState() {
    const screen = document.querySelector('.vn-screen');
    if (!screen) return;
    // Re-open the remembered panel
    screen.querySelectorAll('.vn-panel').forEach(function(p) {
        const isOpen = window._vnUi.panel && p.classList.contains(window._vnUi.panel);
        p.classList.toggle('open', !!isOpen);
    });
    // Re-mark the active slot pill
    screen.querySelectorAll('.vn-slot-btn').forEach(function(s) {
        s.classList.toggle('active', s.getAttribute('data-slot') === window._vnUi.slot);
    });
}

function syncSaveStatus() {
    const src = document.querySelector('#save-status-hidden');
    const dst = document.querySelector('.vn-save-status');
    if (!src || !dst) return;
    const text = (src.textContent || '').trim();
    // Write-only-if-different: setting textContent mutates childList and
    // would re-trigger the observer in a loop otherwise.
    if (dst.textContent !== text) { dst.textContent = text; }
}

function togglePanel(panelClass) {
    window._vnUi.panel = (window._vnUi.panel === panelClass) ? null : panelClass;
    restoreUiState();
}

document.addEventListener('click', function(e) {
    // ── Title screen ─────────────────────────────────────────
    const opt = e.target.closest('.vn-title-opt');
    if (opt) {
        e.preventDefault();
        document.querySelectorAll('.vn-title-opt').forEach(function(o) {
            o.classList.toggle('active', o === opt);
        });
        setHiddenVal('#preset-box', opt.getAttribute('data-preset') || 'curated');
        return;
    }
    const begin = e.target.closest('.vn-title-begin');
    if (begin) {
        e.preventDefault();
        window._vnUi.panel = null;
        // 150ms lets the preset input event reach the server first
        setTimeout(function() { clickHidden('#start-btn-hidden'); }, 150);
        return;
    }

    // ── Icon tray ────────────────────────────────────────────
    const tray = e.target.closest('.vn-tray-btn');
    if (tray) {
        // Reset confirm/cancel live INSIDE the reset tray button
        const confirmEl = e.target.closest('.vn-reset-confirm');
        const cancelEl = e.target.closest('.vn-reset-cancel');
        const trayWrap = document.querySelector('.vn-icon-tray');
        e.preventDefault();
        e.stopPropagation();
        if (confirmEl) {
            if (trayWrap) trayWrap.classList.remove('confirming');
            window._vnUi.panel = null;
            clickHidden('#reset-btn-hidden');
            return;
        }
        if (cancelEl) {
            if (trayWrap) trayWrap.classList.remove('confirming');
            return;
        }
        const action = tray.getAttribute('data-action');
        if (action === 'log')  togglePanel('vn-panel-log');
        if (action === 'cast') togglePanel('vn-panel-cast');
        if (action === 'save') togglePanel('vn-panel-save');
        if (action === 'reset' && trayWrap) trayWrap.classList.toggle('confirming');
        return;
    }

    // ── Panel close button (cast panel ✕) ────────────────────
    const close = e.target.closest('.vn-panel-close');
    if (close) {
        e.preventDefault();
        window._vnUi.panel = null;
        restoreUiState();
        return;
    }

    // ── Save panel: slot pills + actions ─────────────────────
    const slot = e.target.closest('.vn-slot-btn');
    if (slot) {
        e.preventDefault();
        window._vnUi.slot = slot.getAttribute('data-slot') || '0';
        restoreUiState();
        setHiddenVal('#save-slot-hidden', window._vnUi.slot);
        return;
    }
    const action = e.target.closest('.vn-save-action');
    if (action) {
        e.preventDefault();
        const which = action.getAttribute('data-which');
        clickHidden(which === 'save' ? '#save-btn-hidden' : '#load-btn-hidden');
        return;
    }

    // ── Click anywhere outside an open panel closes it ────────
    const scr = e.target.closest('.vn-screen');
    if (scr && window._vnUi.panel
        && !e.target.closest('.vn-panel')
        && !e.target.closest('.vn-icon-tray')) {
        window._vnUi.panel = null;
        restoreUiState();
    }
});

// Escape closes any open panel
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && window._vnUi.panel) {
        window._vnUi.panel = null;
        restoreUiState();
        const trayWrap = document.querySelector('.vn-icon-tray');
        if (trayWrap) trayWrap.classList.remove('confirming');
    }
});
"""


# ═══════════════════════════════════════════════════════════════════════
# UI Builders
# ═══════════════════════════════════════════════════════════════════════

def _title_screen_html() -> str:
    """The title card, rendered inside the VN canvas before the game starts.

    The Curated/Generated toggle and Begin button are plain HTML buttons —
    ONLOAD_JS writes the preset into the hidden #preset-box textbox and
    clicks the hidden #start-btn-hidden gradio button.
    """
    return """
    <div class="vn-screen vn-title-screen">
        <div class="vn-title-content">
            <div class="vn-title-main">ARS-FABULA</div>
            <div class="vn-title-sub">A Slice-of-Life Visual Novel</div>
            <div class="vn-title-cast-choice">
                <button class="vn-title-opt active" data-preset="generated">Generated Cast</button>
                <button class="vn-title-opt" data-preset="curated">Curated Cast</button>
            </div>
            <button class="vn-title-begin">Begin</button>
        </div>
    </div>
    """


def _cast_panel_html_inner(cast: dict[str, CastMember]) -> str:
    """Inner HTML for the in-canvas cast overlay (thumb + identity)."""
    if not cast:
        return '<div class="cast-panel-empty">No cast — start a story.</div>'
    parts = []
    for name, member in cast.items():
        char = member.character
        thumb_uri = thumb_data_uri(member.get_sprite("neutral"), height=160)
        thumb = (f'<img class="cast-panel-thumb" src="{thumb_uri}" alt="{name}">'
                 if thumb_uri else '<div class="cast-panel-thumb"></div>')
        parts.append(
            f'<div class="cast-panel-card">{thumb}'
            f'<div class="cast-panel-info">'
            f'<div class="cast-panel-name">{char.name}</div>'
            f'<div class="cast-panel-role">{char.role}</div>'
            f'<div class="cast-panel-personality">{char.personality}</div>'
            f'</div></div>'
        )
    return "\n".join(parts)


_ICON_TRAY_HTML = """
<div class="vn-icon-tray">
  <button class="vn-tray-btn" data-action="log">Log</button>
  <button class="vn-tray-btn" data-action="cast">Cast</button>
  <button class="vn-tray-btn" data-action="save">Save</button>
  <button class="vn-tray-btn" data-action="reset"><span class="vn-tray-normal">Reset</span><span
    class="vn-tray-confirm-row"><span class="vn-reset-confirm">Confirm</span>&nbsp;/&nbsp;<span
    class="vn-reset-cancel">Cancel</span></span></button>
</div>
"""


def _panels_html(log_text: str, cast_html: str) -> str:
    """The three overlay panels, rendered inside .vn-screen."""
    slots = "".join(
        f'<button class="vn-slot-btn{" active" if i == 0 else ""}" data-slot="{i}">{i}</button>'
        for i in range(5)
    )
    return f"""
    <div class="vn-panel vn-panel-log">
        <button class="vn-panel-close">✕</button>
        <div class="vn-panel-title">Scene Log</div>
        <div class="vn-log-text">{_html.escape(log_text or "")}</div>
    </div>
    <div class="vn-panel vn-panel-cast">
        <button class="vn-panel-close">✕</button>
        <div class="vn-panel-title">Cast</div>
        {cast_html}
    </div>
    <div class="vn-panel vn-panel-save">
        <button class="vn-panel-close">✕</button>
        <div class="vn-panel-title">Save / Load</div>
        <div class="vn-slot-row">{slots}</div>
        <div class="vn-save-actions">
            <button class="vn-save-action" data-which="save">Save</button>
            <button class="vn-save-action" data-which="load">Load</button>
        </div>
        <div class="vn-save-status"></div>
    </div>
    """


def _vn_screen_html(bg_path: str, cast: dict[str, CastMember],
                    present: list[str], expressions: dict[str, str],
                    dialogue_speaker: str = "", dialogue_text: str = "",
                    choices: list[Choice] | None = None,
                    narration: str = "",
                    prev_bg_path: str = "",
                    more: bool = False,
                    log_text: str = "",
                    cast_html: str = "") -> str:
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

    # Sprites — the speaking character is .active, the rest .inactive (dimmed).
    # During pure narration (no speaker) everyone stays normal.
    sprite_html = ""
    for name in present:
        member = cast.get(name)
        if not member:
            continue
        expr = expressions.get(name, "neutral")
        sprite_uri = data_uri(member.get_sprite(expr))
        if sprite_uri:
            state = ""
            if dialogue_speaker:
                state = " active" if name == dialogue_speaker else " inactive"
            sprite_html += (f'<img class="vn-sprite{state}" data-expr="{name}:{expr}" '
                            f'src="{sprite_uri}" alt="{name} ({expr})">')

    # Dialogue — name plate, then text (HUD removed)
    dialogue_html = ""
    if dialogue_text:
        if dialogue_speaker:
            dialogue_html += f'<div class="vn-name-plate">{dialogue_speaker}</div>'
        dialogue_html += f'<p class="vn-text{" narrator" if not dialogue_speaker else ""}">{dialogue_text}</p>'
    elif narration:
        dialogue_html += f'<p class="vn-text narrator">{narration}</p>'
    if more:
        dialogue_html += '<div class="vn-continue">▼</div>'
    dialogue_html += _ICON_TRAY_HTML

    # Choices — always capped with a free-form option so the player is
    # never locked to the offered choices (it reveals the input row).
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
    <div class="vn-screen{' advancing' if more else ''}" data-more="{'1' if more else '0'}" data-typing="pending">
        {bg_layers}
        <div class="vn-chars">{sprite_html}</div>
        {choices_html}
        <div class="vn-dialogue{' has-speaker' if dialogue_speaker else ''}">{dialogue_html}</div>
        {_panels_html(log_text, cast_html)}
    </div>"""


def _casting_screen_html(char_keys: list[str],
                         resolved: dict[str, CastMember],
                         message: str = "Assembling your cast…") -> str:
    """The casting-beat screen: portraits resolve one by one."""
    portraits = []
    for key in char_keys:
        # Preset keys are lowercase slugs; custom names arrive display-ready
        name = key.capitalize() if key.islower() else key
        member = resolved.get(name)
        if member:
            uri = data_uri(member.get_sprite("neutral"))
            inner = (f'<img src="{uri}" alt="{name}">' if uri
                     else f'<div class="pname">{name}</div>')
            portraits.append(
                f'<div class="portrait-wrap">'
                f'<div class="vn-portrait">{inner}</div>'
                f'<div class="pname">{name}</div></div>'
            )
        else:
            portraits.append(
                f'<div class="portrait-wrap">'
                f'<div class="vn-portrait pending">…</div>'
                f'<div class="pname pending">?</div></div>'
            )

    return f"""
    <div class="vn-screen">
        <div class="vn-casting">
            <div class="casting-title">✦ {message}</div>
            <div class="portraits">{''.join(portraits)}</div>
        </div>
    </div>
    """


def _cast_panel_html(cast: dict[str, CastMember]) -> str:
    """Build HTML for the cast info panel."""
    if not cast:
        return '<div class="cast-panel-empty">No cast yet — start a story.</div>'
    parts = ['<div style="padding:8px;">']
    for name, member in cast.items():
        char = member.character
        parts.append(
            f'<div class="cast-card">'
            f'<div class="name">{char.name}</div>'
            f'<div class="role">{char.role} · ♪ {char.voice_id}</div>'
            f'<div class="personality">{char.personality}</div>'
            f'<div class="cast-expressions">'
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


# ═══════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════

def _make_callbacks(bg_provider: BackgroundProvider):
    # All turn callbacks return/yield 11-tuples:
    # (scene, engine, screen, cast, cast_panel, log, info, audio, prev_bg,
    #  beats, beat_idx)   ← the last two drive read-at-your-own-pace.

    def _empty(scene_state=None):
        """No active scene → the title screen (also what Reset restores)."""
        scene = scene_state or make_initial_scene()
        return (scene, None,
                _title_screen_html(),
                {}, _cast_panel_html({}),
                "", "No active scene", None, "", [], 0)

    def on_start(cast_preset: str,
                 n1: str = "", c1: str = "", t1: str = "",
                 n2: str = "", c2: str = "", t2: str = "",
                 n3: str = "", c3: str = "", t3: str = ""):
        """The CASTING BEAT — a generator so portraits resolve one by one.

        Loading is the feature: yield the casting screen immediately,
        reveal each cast member as they're ready, then open the scene.

        Custom character cards (from Cast Setup) apply in generated mode:
        name + concept → the LLM designs the appearance tags; advanced
        tags override the designer. The sprite pose block is always locked.
        """
        cards = [{"name": n, "concept": c, "tags": t}
                 for n, c, t in ((n1, c1, t1), (n2, c2, t2), (n3, c3, t3))
                 if (n or "").strip()]

        if cast_preset == "generated":
            char_keys = ([card["name"].strip() for card in cards]
                         if cards else ["yuki", "haruki", "sakura"])
            loader = lambda k: None   # generated mode batches below
        else:
            char_keys = ["yuki", "haruki", "sakura"]
            loader = lambda k: load_curated_cast([k])[0]

        resolved: dict[str, CastMember] = {}

        # Beat 1 — the casting screen, empty portraits
        yield (None, None,
               _casting_screen_html(char_keys, resolved),
               {}, _cast_panel_html({}),
               "", "Casting…", None, "", [], 0)

        # Beat 2 — portraits resolve
        if cast_preset == "generated":
            # One long bake; ComfyUI does the whole batch (falls back to
            # curated internally if unreachable). Show the wait honestly.
            yield (None, None,
                   _casting_screen_html(char_keys, resolved,
                                        "Generating your cast… (this is the bake — "
                                        "everything after this is instant)"),
                   {}, _cast_panel_html({}), "", "Casting…", None, "", [], 0)
            members = generate_cast({"custom_cast": cards} if cards
                                    else {"preset": "default"})
            for m in members:
                resolved[m.character.name] = m
                yield (None, None,
                       _casting_screen_html(char_keys, resolved),
                       {}, _cast_panel_html(resolved),
                       "", "Casting…", None, "", [], 0)
        else:
            for key in char_keys:
                member = loader(key)
                if member:
                    resolved[member.character.name] = member
                time.sleep(0.45)   # let the beat land — loading as a feature
                yield (None, None,
                       _casting_screen_html(char_keys, resolved),
                       {}, _cast_panel_html(resolved),
                       "", "Casting…", None, "", [], 0)

        cast = resolved
        if not cast:
            yield _empty()
            return

        # VRAM swap-back: the bake is done, so unload ComfyUI's models and
        # bring the LLM back up (relaunch + wait for /health) before the
        # opening turn — llama-server doesn't reload lazily.
        if cast_preset == "generated":
            from cast_pipeline import comfy_free_vram
            from model_client import ensure_llm, ModelConfig
            yield (None, None,
                   _casting_screen_html(char_keys, resolved,
                                        "Cast ready — waking the storyteller…"),
                   cast, _cast_panel_html(cast), "", "Casting…", None, "", [], 0)
            comfy_free_vram()
            ensure_llm(ModelConfig.from_env())

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

        log = clean_narration(raw)
        screen, prev_bg, voice_path = _render_beats_screen(
            beats, 0, scene, bg_provider, "", log_text=log)
        yield (scene, engine, screen, cast, _cast_panel_html(cast),
               log,
               f"{scene.chapter} · {scene.background} ({scene.mood}) · {status}",
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

        log = clean_narration(raw)
        screen, new_prev_bg, voice_path = _render_beats_screen(
            beats, 0, scene, bg_provider, prev_bg, log_text=log)

        return (scene, engine, screen, scene.cast, _cast_panel_html(scene.cast),
                log,
                f"Turn {scene.turn} · {scene.background} ({scene.mood})",
                voice_path, new_prev_bg, beats, 0)

    def on_save(slot: float, scene: SceneState, engine):
        if not engine:
            return "Nothing to save — start a story first."
        engine.save(int(slot))
        return f"Saved · slot {int(slot)}"

    def on_load(slot: float, scene: SceneState, engine, prev_bg: str):
        if not engine:
            return (*_empty(scene), "Nothing to load into — start a story first.")
        try:
            scene = engine.load(int(slot))
        except FileNotFoundError:
            return (scene, engine, gr.skip(), scene.cast,
                    _cast_panel_html(scene.cast), gr.skip(),
                    f"Slot {int(slot)} is empty.", None, prev_bg, gr.skip(), gr.skip(),
                    f"Slot {int(slot)} is empty.")

        # A restored save has no per-beat tool sequence — show it as a single
        # static beat (the saved description) with its choices already visible.
        beats = [{"speaker": "", "text": scene.description or "",
                  "expressions": dict(scene.expressions),
                  "background": scene.background,
                  "present": list(scene.present), "voice": None}]
        screen, new_prev_bg, _ = _render_beats_screen(
            beats, 0, scene, bg_provider, prev_bg, log_text=scene.description or "")
        return (scene, engine, screen, scene.cast, _cast_panel_html(scene.cast),
                scene.description,
                f"Loaded · slot {int(slot)} · Turn {scene.turn} · {scene.background} ({scene.mood})",
                None, new_prev_bg, beats, 0,
                f"Loaded · slot {int(slot)}")

    def on_advance(beats, idx, scene: SceneState, prev_bg: str, log_text: str):
        """Reveal the next beat (click-the-screen / Space / Enter).

        log_text comes from the hidden narration_log sink so the in-canvas
        Scene Log panel keeps its content across beat re-renders.
        """
        if not beats:
            return gr.skip(), gr.skip(), None, gr.skip(), gr.skip()
        new_idx = min((idx or 0) + 1, len(beats) - 1)
        screen, new_prev_bg, voice_path = _render_beats_screen(
            beats, new_idx, scene, bg_provider, prev_bg, log_text=log_text or "")
        info = f"Turn {scene.turn} · {scene.background} ({scene.mood})"
        return screen, new_idx, voice_path, info, new_prev_bg

    def on_free_submit(text: str, scene: SceneState, engine, prev_bg: str):
        """Free-text action → a scene turn. Wraps on_choice and ALSO clears
        the input box server-side (returns "" into free_input) — no JS race.

        Returns the 11 turn_outputs + the cleared free_input value.
        """
        if not (text or "").strip() or not engine or not scene:
            # Nothing to do — keep the box contents, change nothing else
            return tuple(gr.skip() for _ in range(12))
        result = on_choice("__free__", text.strip(), scene, engine, prev_bg)
        return (*result, "")   # clear free_input

    def on_reset():
        return _empty()

    return on_start, on_choice, on_save, on_load, on_reset, on_advance, on_free_submit


# ═══════════════════════════════════════════════════════════════════════
# UI Assembly
# ═══════════════════════════════════════════════════════════════════════

def create_ui():
    """Build and return the Gradio VN app."""
    bg_backend = os.getenv("ARS_FABULA_BG", "preset")
    bg_provider = get_background_provider(bg_backend)
    (on_start, on_choice, on_save, on_load, on_reset, on_advance,
     on_free_submit) = _make_callbacks(bg_provider)

    initial_scene = make_initial_scene()

    with gr.Blocks(title="Ars-Fabula VN") as app:
        # ── State ──────────────────────────────────────────────
        scene_state = gr.State(initial_scene)
        engine_ref = gr.State(None)
        cast_data = gr.State(DEFAULT_CAST)
        prev_bg_state = gr.State("")
        beats_state = gr.State([])        # ordered display beats for this turn
        beat_idx_state = gr.State(0)      # which beat is currently shown

        # ── The stage: the canvas plus the free-text bar overlaid inside its
        #    frame. The wrapping Column is the positioned ancestor the bar
        #    anchors to (#stage { position: relative }), so the overlay stays
        #    docked in the canvas instead of flying across the viewport — the
        #    failure mode of an earlier absolute version with no such ancestor.
        with gr.Column(elem_id="stage"):
            # Initial value is the TITLE SCREEN.
            vn_screen = gr.HTML(_title_screen_html())

            # Free-text action bar — docked inside the canvas, in the same
            # centered column as the choices (which stack just above it) and
            # styled to match the choice buttons, so it feels like part of the
            # interaction cluster. Revealed only at the END of a turn (JS
            # toggles body.vn-playing once the last beat's text is shown or
            # choices appear), not while earlier beats are still being read.
            with gr.Row(elem_id="free-input-row"):
                free_input = gr.Textbox(
                    scale=8, lines=1, max_lines=1,
                    placeholder="Your choice…",
                    label=None, show_label=False, container=False,
                    elem_id="free-input",
                )
                free_submit = gr.Button("➤", scale=1, size="sm",
                                        elem_id="free-submit-btn")

        # ── Cast Setup — MENU ONLY. Shown on the main menu; hidden once a
        #    game starts, restored on Reset (which returns to the title). ──
        with gr.Group(visible=True, elem_id="cast-setup") as cast_setup_group:
            gr.Markdown("### ⚙ Cast Setup — customize your characters (optional)")
            gr.Markdown(
                "Fill **Name + Concept** and the story model designs the "
                "character (e.g. concept `childhood friend` → a warm, "
                "brown-haired genki design). **Advanced tags** override the "
                "designer with your own danbooru appearance tags (start with "
                "`1girl`/`1boy`; hair, eyes, clothing, accessories only).\n\n"
                "Only the **framing essentials are locked on**; body attitude "
                "is yours (add stance tags like `hand on hip`, `arms crossed`, "
                "`head tilt`). Applies to *Generated Cast* only; editing a "
                "design auto-invalidates that character's sprite cache."
            )
            card_fields = []
            for i in range(1, 4):
                with gr.Row():
                    cn = gr.Textbox(label=f"Name {i}", scale=2,
                                    placeholder="e.g. Mio")
                    cc = gr.Textbox(label="Concept", scale=3,
                                    placeholder="e.g. shy bookworm childhood friend")
                    ct = gr.Textbox(label="Advanced: appearance tags (override)", scale=4,
                                    placeholder="1girl, silver hair, long hair, blue eyes, …")
                    card_fields.extend([cn, cc, ct])

        # ── Hidden sinks (callback outputs the JS never reads) ───
        cast_panel = gr.HTML(visible=False)
        scene_info = gr.Markdown(visible=False)
        narration_log = gr.Textbox(visible=False, interactive=False)

        # CSS-collapsed, not visible=False — the <audio> element must mount
        # for autoplay to fire.
        voice_audio = gr.Audio(
            value=None, autoplay=True, visible=True,
            interactive=False, elem_id="voice-audio", container=False,
        )

        # ── Hidden JS bridges ─────────────────────────────────────
        # visible="hidden" keeps these in the DOM (visible=False would remove
        # them and break the JS bridge)
        preset_box = gr.Textbox(value="generated", visible="hidden",
                                interactive=True, elem_id="preset-box")
        start_btn = gr.Button("Begin", visible="hidden",
                              elem_id="start-btn-hidden")
        save_slot = gr.Number(value=0, minimum=0, maximum=4, step=1,
                              precision=0, visible="hidden",
                              elem_id="save-slot-hidden")
        save_btn = gr.Button("Save", visible="hidden", elem_id="save-btn-hidden")
        load_btn = gr.Button("Load", visible="hidden", elem_id="load-btn-hidden")
        reset_btn = gr.Button("Reset", visible="hidden", elem_id="reset-btn-hidden")
        save_status = gr.Markdown("", visible="hidden",
                                  elem_id="save-status-hidden")

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

        # Hidden state for free-text fallback choice id
        free_choice_id = gr.Textbox(value="__free__", visible="hidden",
                                    elem_id="free-choice-id")

        main_outputs = [scene_state, engine_ref, vn_screen, cast_data,
                        cast_panel, narration_log, scene_info,
                        voice_audio, prev_bg_state]
        # Turn callbacks also carry the beat list + index (read-at-your-own-pace).
        turn_outputs = main_outputs + [beats_state, beat_idx_state]

        # ── Event wiring ───────────────────────────────────────
        start_btn.click(fn=on_start, inputs=[preset_box] + card_fields,
                        outputs=turn_outputs)
        # Cast Setup is a main-menu tool: hide it once a game begins,
        # bring it back on Reset (which returns to the title).
        start_btn.click(fn=lambda: gr.update(visible=False),
                        outputs=[cast_setup_group])

        submit_choice_btn.click(
            fn=on_choice,
            inputs=[choice_input, choice_text_input, scene_state, engine_ref, prev_bg_state],
            outputs=turn_outputs,
        )

        # Advance one beat: re-render the screen, bump the index, chain the
        # crossfade (prev_bg), and play that beat's voiced line if any.
        advance_btn.click(
            fn=on_advance,
            inputs=[beats_state, beat_idx_state, scene_state, prev_bg_state,
                    narration_log],
            outputs=[vn_screen, beat_idx_state, voice_audio, scene_info, prev_bg_state],
        )

        save_btn.click(fn=on_save, inputs=[save_slot, scene_state, engine_ref],
                       outputs=[save_status])
        load_btn.click(fn=on_load,
                       inputs=[save_slot, scene_state, engine_ref, prev_bg_state],
                       outputs=turn_outputs + [save_status])

        reset_btn.click(fn=on_reset, outputs=turn_outputs)
        reset_btn.click(fn=lambda: gr.update(visible=True),
                        outputs=[cast_setup_group])

        # ── Free-text action (fallback when no choices, or "Write your
        #    own action…"). Clears the box server-side via the extra output. ─
        # Both the ➤ button and Enter (via JS bridge → clicks this button)
        # route here; clears the box server-side via the extra output.
        free_outputs = turn_outputs + [free_input]
        free_submit.click(
            fn=on_free_submit,
            inputs=[free_input, scene_state, engine_ref, prev_bg_state],
            outputs=free_outputs,
        )

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
        theme=gr.themes.Base(
            primary_hue="amber",
            neutral_hue="slate",
        ),
        quiet=False,
        share=False,
    )
