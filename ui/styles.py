"""CSS and onload JS for the VN UI. Pure presentation strings — no logic.

Installed via launch(css=..., js=...).
"""
import os

# Font for CHARACTER dialogue lines (the narrator keeps --font-dialogue).
# Env-overridable — pass a full CSS font-family stack, e.g.
#   ARS_FABULA_DIALOGUE_FONT="'Sazanami Gothic', monospace"
#
# 'MS Gothic' (and 'MS PGothic'/'Osaka-Mono') are Windows/mac-only and ABSENT on
# the Linux HF-Spaces deploy target, where the stack used to fall through to a
# generic 'monospace' — losing the retro-VN look. 'DotGothic16' (loaded from
# Google Fonts in the @import below) is a 16-dot bitmap-gothic webfont built to
# mimic exactly that MS-Gothic aesthetic, so it carries the look everywhere a
# local MS Gothic is unavailable while Windows users still get the real thing.
CHARACTER_FONT = os.getenv(
    "ARS_FABULA_DIALOGUE_FONT",
    "'MS Gothic', 'MS PGothic', 'DotGothic16', 'Osaka-Mono', monospace",
)

# ═══════════════════════════════════════════════════════════════════════
# CSS — VN themed (incl. scene-turn crossfade + sprite swap animations)
# ═══════════════════════════════════════════════════════════════════════

CSS = """
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400;1,600&family=Cinzel:wght@400;600&family=Noto+Sans+JP:wght@300;400&family=DotGothic16&display=swap');

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
  /* One shared panel fill for the dialogue box, choice pills and input bar
     so they're all the exact same colour + opacity. */
  --dialogue-bg:   rgba(15, 15, 25, 0.68);
  --font-display:  'Cinzel', 'Georgia', serif;
  --font-dialogue: 'Cormorant Garamond', 'Georgia', serif;
  --font-ui:       'Noto Sans JP', 'Georgia', serif;
  --font-character: __FONT_CHARACTER__;   /* injected from ARS_FABULA_DIALOGUE_FONT */
}

/* ── Global Reset ── */
.gradio-container {
  background: var(--bg-void) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-ui) !important;
  /* Size the frame to the LARGEST 16:9 that fits the viewport: as wide as
     ~90vw, but capped at 1700px AND at the width whose 16:9 height is ~88vh
     (88vh × 16/9 ≈ 156vh). The height cap is what reclaims the empty band that
     appeared below an 80%-width frame on a 16:9 screen. */
  width: min(90vw, 1900px, 156vh) !important;
  min-width: 720px !important;       /* prevent collapse on small viewports */
  margin: 0 auto !important;
}
body { background: var(--bg-void) !important; }
*:focus { outline: 2px solid var(--border-accent) !important; outline-offset: 2px; }

/* Cast Setup — a STEP shown AFTER Begin → Generated (between the title and the
   character-design phase). Hidden by default; when active it replaces the title
   content inside the canvas. JS toggles body.cast-setup-active.
   NOTE: The doubled `.vn-title-content.vn-title-content` selector beats Gradio 6's
   scoped-prefix specificity issue (same reason #cast-setup#cast-setup was needed). */
/* !important is required: Gradio 6 re-emits every custom rule with a
   `.gradio-container-... .contain` scope prefix. The scoped copy of the
   base `.vn-title-content { display:flex }` outranks this rule, while the
   scoped copy of THIS rule can never match (body is outside .contain) —
   so without !important the title stayed visible beside the cast setup. */
body.cast-setup-active .vn-title-content { display: none !important; }
body.cast-setup-active .vn-cast-setup {
  display: flex !important;
}
/* But when NOT in cast-setup-active, the cast setup is always hidden. */
.vn-cast-setup { display: none; }

/* ── VN Screen ── */
.vn-screen {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;             /* 16:9, height derived purely from width */
  min-height: 380px;                /* (no vh cap — height tracks the frame, not
                                       the page, so overlays never displace on
                                       zoom/resize) */
  background: var(--bg-screen);
  border: 1px solid var(--border-mid);
  border-radius: 6px;
  overflow: hidden;
  margin-bottom: 8px;
  container-type: inline-size;      /* lets dialogue/choice text size in cqw —
                                       i.e. scale with the screen (and thus the
                                       text box), not the page root font */
}

/* Background layers */
.vn-bg {
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 100%;
  object-fit: cover;
}
/* Only crossfade when the background ACTUALLY changed (an .old layer is present
   to fade over). Otherwise every beat re-rendered the canvas and the bg faded
   up from black again — the "blackish" flash on each advance. */
.vn-bg.old + .vn-bg.new { animation: vnBgFade 1.3s ease both; }
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
  gap: 2.4cqw;               /* scales with the canvas, not the page */
  pointer-events: none;
  z-index: 2;
}
.vn-sprite {
  /* Sized against the canvas WIDTH (cqw), the stable axis: the sprite is
     locked to the canvas's proportions and never changes on browser zoom or
     when min-height inflates the box past 16:9. 48cqw ≈ 86% of a true 16:9
     canvas's height; max-height caps it if the box is ever taller than 16:9. */
  height: 48cqw;
  max-height: 86%;
  width: auto;
  max-width: 37%;            /* must exceed the widest bake canvas's display
                                width — if this cap binds, object-fit:contain
                                letterboxes the sprite and SHRINKS that
                                character's face below the equal-face target.
                                At the 20%-face framing the widest cast member
                                (marie, ~940×1254 ⇒ ~36% display) just clears
                                this; three sprites still sum < 100% because the
                                others are far narrower (~27%), so 3-up doesn't
                                overlap. Bake canvases widen to fit wide poses
                                (cast_pipeline fig_w), so keep headroom here. */
  object-fit: contain;
  object-position: bottom center;
  /* No per-render entrance animation: the canvas HTML is rebuilt every beat,
     so an entrance keyframe replayed (and "dragged") on each advance. The
     active/inactive filter still animates below. */
  transition: filter 0.35s ease;
}
/* Active/inactive: the speaker is bright, the rest gently shadowed.
   Dimming is FILTER-only — sprites are never made translucent. */
.vn-sprite.active   { filter: none; }
.vn-sprite.inactive {
  filter: brightness(0.6) saturate(0.85) contrast(0.95);
}
/* Choice state: once the line finishes typing and the choice cluster fades in,
   EVERY sprite (active speaker included) takes the inactive dimming, so the
   semi-transparent capsules and the free-text bar read cleanly over bright
   costumes. Same gate as .vn-choices visibility; the existing 0.35s
   filter transition makes it fade in with the capsules. */
.vn-screen[data-choices="1"][data-typing="0"] .vn-sprite,
.vn-screen[data-choices="1"][data-typing="0"] .vn-sprite.active {
  filter: brightness(0.6) saturate(0.85) contrast(0.95);
}
@keyframes vnSpriteIn {
  from { opacity: 0.2; transform: translateY(10px) scale(0.97); }
  to   { opacity: 1;   transform: translateY(0)    scale(1); }
}

/* ── Dialogue Box ── (a wide, centered panel with a LOCKED height — it never
   grows or shrinks with the line length — one flat scrim, not a gradient) */
.vn-dialogue {
  position: absolute;
  bottom: 20px; left: 20%; right: 20%; /* 60% width, centered */
  height: clamp(130px, 13.5cqw, 230px); /* scales with the screen width (never
                                           with line length) so it stays in
                                           proportion to the cqw-sized text */
  background: var(--dialogue-bg);       /* shared panel colour */
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
  padding: 14px 16px 36px;
  overflow: hidden;                     /* clip rather than expand */
}
.vn-dialogue.has-speaker {
  border-left: 3px solid var(--border-accent);
}
/* Centered column — the box itself is already narrow (60% of the screen),
   so the text uses most of it. */
.vn-dialogue-inner {
  max-width: min(94%, 960px);
  margin: 0 auto;
}
.vn-name-plate {
  display: inline-block;
  font-family: var(--font-display);
  font-size: clamp(0.7rem, 1.1cqw, 1.2rem);
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
  font-size: clamp(0.75rem, 1.2cqw, 1.3rem);
  margin-bottom: 6px;
  letter-spacing: 0.06em;
}
/* Character dialogue uses --font-character (MS Gothic by default, env-set). */
.vn-text {
  font-family: var(--font-character);
  color: var(--text-primary);
  /* Scales with the screen/text-box width: ≈ the old 1.15rem on a ~1300px
     frame, smaller on regular resolutions, larger on big frames. */
  font-size: clamp(0.85rem, 1.4cqw, 1.5rem);
  line-height: 1.7;
  text-align: left;
  padding: 0 16px;
  text-indent: 0;
}
/* Narration is set in Georgia serif (distinct from the MS Gothic dialogue). */
.vn-text.narrator {
  font-family: Georgia, 'Times New Roman', serif;
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

/* Streaming loader — shown when the reader has caught up to the last beat the
   model has produced but the turn is still being written. Same ✦ pulse as the
   turn-pending indicator (body.vn-waiting), so "still generating" reads
   consistently whether a turn was just submitted or is mid-stream. */
.vn-streaming-wait {
  position: absolute;
  right: 18px; bottom: 14px;
  color: var(--text-accent);
  font-size: 1.05rem;
  z-index: 30;
  animation: vnPulse 1.2s ease infinite;
  pointer-events: none;
}

/* ── Choices ── (raised so the free-text bar docks directly beneath them
   in the same centered column — one interaction cluster, not two) */
/* Independent vertical column, centered in the open scene area ABOVE the
   dialogue box. The container spans the area but passes clicks through
   (pointer-events:none) so only the pills are interactive. */
.vn-choices {
  position: absolute;
  inset: 0;                     /* span the whole canvas … */
  display: flex;
  flex-direction: column;
  justify-content: center;      /* … so the stack sits dead-center vertically */
  align-items: center;          /* horizontal center; pills size to content */
  gap: 0.75rem;
  pointer-events: none;
}
.vn-choices .vn-choice-btn { pointer-events: auto; }
/* Choices only appear once the line has finished typing (data-typing flips to
   "0"); hidden during 'pending' and '1' so they never precede the text. They
   fade in (opacity only — no transform) so the appearance is deliberate. */
.vn-screen:not([data-typing="0"]) .vn-choices { display: none; }
.vn-screen[data-typing="0"] .vn-choices { animation: vnChoicesIn 0.22s ease both; }
@keyframes vnChoicesIn { from { opacity: 0; } to { opacity: 1; } }
/* Choices are standalone rounded capsules — solid (opaque) dark fill so they
   pop over the scene without stacking transparent layers, centered text, and
   the MS Gothic character font (--font-character). */
.vn-choice-btn {
  background: var(--dialogue-bg) !important;          /* same fill as the text box */
  border: 1px solid rgba(255,255,255,0.2) !important;
  border-radius: 30px !important;
  color: var(--text-primary) !important;
  font-family: var(--font-character) !important;     /* MS Gothic */
  font-style: normal !important;
  font-size: clamp(0.8rem, 1.25cqw, 1.35rem) !important;
  padding: 0.75em 2em !important;          /* tracks the scaled font */
  width: auto !important;                            /* size to content, not full width */
  text-align: center !important;
  cursor: pointer !important;
  transition: background 0.15s, border-color 0.15s, transform 0.12s !important;
}
.vn-choice-btn:hover {
  background: rgba(40, 40, 62, 0.82) !important;
  border-color: var(--border-accent) !important;
  transform: translateY(-1px);
  color: #fff !important;
}
/* Free-text capsule — the last item in the choice column. Same fill/shape as
   the pills, with an inline input + send arrow. */
.vn-free-capsule {
  display: flex;
  align-items: center;
  width: min(60%, 460px);
  background: var(--dialogue-bg);
  border: 1px solid rgba(255,255,255,0.2);
  border-radius: 30px;
  overflow: hidden;
  pointer-events: auto;
  transition: border-color 0.15s, background 0.15s;
}
.vn-free-capsule:focus-within {
  border-color: var(--border-accent);   /* accent the rim only — keep the fill */
}
.vn-free-text,
.vn-free-text:focus {
  flex: 1 1 auto;
  min-width: 0;
  background: transparent !important;    /* no Gradio/browser fill on focus */
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
  color: var(--text-primary);
  font-family: var(--font-character);
  font-size: clamp(0.8rem, 1.25cqw, 1.35rem);
  padding: 0.7em 0.5em 0.7em 1.5em;        /* tracks the scaled font */
}
.vn-free-text::placeholder { color: var(--text-muted); font-style: italic; }
.vn-free-go {
  flex: 0 0 auto;
  background: transparent;
  border: none;
  color: var(--text-accent);
  font-size: clamp(0.85rem, 1.3cqw, 1.4rem);
  cursor: pointer;
  padding: 0 1.1em;
}
.vn-free-go:hover { color: #fff; }

/* ── Generation-wait screen ── the app's single waiting vocabulary: every
   phase that blocks on generation (sprite bake, VRAM swap, opening turn)
   shows this centered ✦ pulse — the same star that marks turn/stream waits. */
.vn-loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  gap: 14px;
  background: radial-gradient(ellipse at center, #1c1c38 0%, var(--bg-screen) 70%);
}
.vn-loading-star {
  color: var(--text-accent);
  font-size: clamp(1.6rem, 3.4cqw, 2.6rem);
  animation: vnPulse 1.2s ease infinite;
}
.vn-loading-msg {
  font-family: var(--font-display);
  color: var(--text-accent);
  font-size: clamp(0.95rem, 1.8cqw, 1.5rem);
  letter-spacing: 0.15em;
}
.vn-loading-sub {
  font-family: var(--font-ui);
  color: var(--text-secondary);
  font-size: clamp(0.7rem, 1.05cqw, 0.95rem);
  letter-spacing: 0.08em;
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
  font-size: clamp(0.95rem, 1.8cqw, 1.5rem);
  letter-spacing: 0.15em;
  animation: vnPulse 1.6s ease infinite;
}
.vn-casting .portraits { display: flex; gap: 24px; }
.vn-portrait {
  width: clamp(82px, 9cqw, 130px); height: clamp(120px, 13cqw, 190px);
  border-radius: 6px;
  border: 1px solid var(--border-mid);
  background: var(--bg-surface);
  overflow: hidden;
  display: flex; align-items: center; justify-content: center;
}
.vn-portrait.pending { border-style: dashed; border-color: var(--border-subtle); }
/* The card currently being generated: accent border + soft pulse so the eye
   tracks which character the bake is on. */
.vn-portrait.active {
  border-color: var(--border-accent);
  box-shadow: 0 0 0 1px var(--border-accent), 0 0 18px -4px var(--border-accent);
  animation: castingCardPulse 1.4s ease-in-out infinite;
}
@keyframes castingCardPulse {
  0%, 100% { box-shadow: 0 0 0 1px var(--border-accent), 0 0 12px -6px var(--border-accent); }
  50%      { box-shadow: 0 0 0 1px var(--border-accent), 0 0 22px -2px var(--border-accent); }
}
/* The ✦ start-symbol shown inside the active card before its sprite exists. */
.casting-spinner {
  font-size: clamp(1.4rem, 2.4cqw, 2.2rem);
  color: var(--text-accent);
  animation: vnPulse 1.2s ease infinite;
}
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
  font-size: clamp(0.66rem, 0.95cqw, 0.92rem);
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

/* ── Interactive Casting Review ── */
.vn-casting-review {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  max-height: 100%;
  overflow-y: auto;
  background: radial-gradient(ellipse at center, #1c1c38 0%, var(--bg-screen) 70%);
  padding: 20px;
}
.casting-review-title {
  font-family: var(--font-display);
  color: var(--text-accent);
  font-size: 0.85rem;
  letter-spacing: 0.15em;
  margin-bottom: 18px;
}
.review-layout {
  display: flex;
  gap: 28px;
  align-items: flex-start;
  width: 100%;
  max-width: 800px;
  max-height: 100%;
  justify-content: center;
}
.review-image {
  width: clamp(150px, 18cqw, 240px);
  height: clamp(250px, 31cqw, 400px);
  border: 1px solid var(--border-mid);
  border-radius: 6px;
  overflow: hidden;
  background: var(--bg-surface);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.review-image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  object-position: top;
}
.review-no-image {
  color: var(--text-muted);
  font-style: italic;
  font-size: 0.82rem;
  text-align: center;
  padding: 16px;
}
.review-panel {
  flex: 1;
  min-width: 260px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.review-name {
  font-family: var(--font-display);
  color: var(--text-accent);
  font-size: clamp(1.15rem, 1.5cqw, 1.6rem);
  letter-spacing: 0.06em;
}
/* Designed character description — role badge + personality paragraph */
.review-desc { display: flex; flex-direction: column; gap: 6px; }
.review-role {
  align-self: flex-start;
  font-family: var(--font-ui);
  font-size: clamp(0.66rem, 0.85cqw, 0.92rem);
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text-accent);
  background: rgba(232,168,124,0.1);
  border: 1px solid var(--border-accent);
  border-radius: 3px;
  padding: 2px 10px;
}
.review-personality {
  font-family: var(--font-dialogue);
  font-style: italic;
  font-size: clamp(0.82rem, 1.05cqw, 1.15rem);
  line-height: 1.55;
  color: var(--text-secondary);
  margin: 0;
}
.review-prompt-label {
  font-family: var(--font-ui);
  font-size: 0.68rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-muted);
}
.review-prompt {
  width: 100%;
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-mid) !important;
  border-radius: 4px !important;
  color: var(--text-primary) !important;
  font-family: var(--font-dialogue) !important;
  font-size: 0.9rem !important;
  line-height: 1.5 !important;
  padding: 10px 12px !important;
  resize: vertical;
  caret-color: var(--text-accent) !important;
}
.review-prompt:focus {
  border-color: var(--border-accent) !important;
  outline: none !important;
  box-shadow: 0 0 0 2px var(--border-glow) !important;
}
/* The editable personality keeps the old read-only paragraph's voice
   (italic, secondary) so the screen still reads the same at a glance. */
.review-prompt.review-desc-edit {
  font-style: italic;
  color: var(--text-secondary) !important;
}
.review-actions {
  display: flex;
  gap: 12px;
  margin-top: 4px;
}
.review-btn {
  flex: 1;
  padding: 10px 0 !important;
  border-radius: 4px !important;
  font-family: var(--font-display) !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.1em !important;
  cursor: pointer !important;
  transition: background 0.18s ease, color 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease !important;
  border: 1px solid var(--border-mid) !important;
}
.review-btn.confirm {
  background: linear-gradient(135deg, #c07840, #e8a87c) !important;
  border-color: var(--border-accent) !important;
  color: #0d0d1a !important;
  font-weight: 600 !important;
}
.review-btn.confirm:hover {
  box-shadow: 0 0 14px var(--border-glow) !important;
}
.review-btn.redo {
  background: var(--bg-surface) !important;
  color: var(--text-secondary) !important;
}
.review-btn.redo:hover {
  border-color: var(--border-accent) !important;
  color: var(--text-accent) !important;
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
  font-size: clamp(0.88rem, 1.1cqw, 1.2rem);
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
/* Scope button transitions to colour only — `transition: all` animated layout
   geometry (transform/size/position) whenever buttons relaid out (e.g. the
   choice column appearing), which read as a dark element "dragging" in. */
button { border-radius: 3px !important; transition: background 0.18s ease, color 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease !important; }
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
  font-size: clamp(0.6rem, 0.75cqw, 0.85rem) !important;
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
  font-size: clamp(0.75rem, 0.95cqw, 1.05rem);
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
  font-size: clamp(0.9rem, 1.15cqw, 1.3rem) !important;
  cursor: pointer !important;
  padding: 2px 6px !important;
}
.vn-panel-close:hover { color: var(--text-accent) !important; }

/* Log panel — slides up over the dialogue box.
   Closed panels are visibility:hidden, NOT just translated out of the clip:
   during Gradio's canvas-HTML swap the browser could paint the off-canvas
   panels unclipped for a few frames (seen as dark log/cast sheets "dragging"
   across the page each turn). visibility doesn't depend on overflow clipping
   or compositor timing, so a closed panel can never paint. The 0s visibility
   transition is DELAYED by the slide duration on close (so the slide-out
   stays visible) and instant on open.

   Transitions are gated behind .vn-anim, which ONLY the user-toggle JS adds
   (togglePanel / close / Escape). Render-restoration (restoreUiState after a
   canvas swap re-applies .open to the freshly inserted panel) must SNAP to
   its state — otherwise every turn render replays the slide-in, dragging the
   dark sheet across the scene. */
.vn-panel-log {
  bottom: 0; left: 0; right: 0;
  height: 55%;
  border-top: 1px solid var(--border-mid);
  transform: translateY(102%);
  visibility: hidden;
  transition: none;
  overflow-y: auto;
  padding: 18px 28px;
}
.vn-panel-log.vn-anim { transition: transform 0.3s ease, visibility 0s linear 0.3s; }
.vn-panel-log.open {
  transform: translateY(0);
  visibility: visible;
}
.vn-panel-log.open.vn-anim { transition: transform 0.3s ease, visibility 0s; }
.vn-log-text {
  font-family: var(--font-dialogue);
  font-style: italic;
  font-size: clamp(0.85rem, 1.25cqw, 1.35rem);
  line-height: 1.7;
  color: var(--text-narrator);
  white-space: pre-wrap;
}
/* Player answers inside the session register (▸-prefixed lines) */
.vn-log-answer {
  font-style: normal;
  color: var(--text-accent);
}

/* Cast panel — slides in from the right */
.vn-panel-cast {
  top: 0; right: 0; bottom: 0;
  width: 40%;
  min-width: 260px;
  border-left: 1px solid var(--border-mid);
  transform: translateX(102%);
  visibility: hidden;            /* unpaintable when closed — see log panel note */
  transition: none;              /* animates only on user toggle (.vn-anim) */
  overflow-y: auto;
  padding: 18px 16px;
}
.vn-panel-cast.vn-anim { transition: transform 0.3s ease, visibility 0s linear 0.3s; }
.vn-panel-cast.open {
  transform: translateX(0);
  visibility: visible;
}
.vn-panel-cast.open.vn-anim { transition: transform 0.3s ease, visibility 0s; }
.cast-panel-card {
  display: flex;
  gap: 14px;
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-left: 3px solid var(--border-accent);
  border-radius: 5px;
  padding: 12px 14px;
  margin-bottom: 10px;
}
.cast-panel-thumb {
  width: clamp(78px, 8cqw, 130px); height: clamp(112px, 11.5cqw, 188px);
  object-fit: cover; object-position: top;
  border-radius: 3px;
  flex-shrink: 0;
  background: var(--bg-raised);
}
.cast-panel-name {
  font-family: var(--font-display);
  font-size: clamp(1rem, 1.3cqw, 1.45rem);
  color: var(--text-accent);
  letter-spacing: 0.06em;
}
.cast-panel-role {
  font-size: clamp(0.76rem, 1cqw, 1.1rem);
  color: var(--text-muted);
  margin-top: 2px;
}
.cast-panel-personality {
  font-family: var(--font-dialogue);
  font-style: italic;
  font-size: clamp(0.85rem, 1.1cqw, 1.2rem);
  color: var(--text-secondary);
  margin-top: 6px;
  line-height: 1.45;
}

/* Save panel — centered modal */
.vn-panel-save {
  top: 50%; left: 50%;
  transform: translate(-50%, -50%) scale(0.95);
  opacity: 0;
  visibility: hidden;            /* unpaintable when closed — see log panel note */
  pointer-events: none;
  transition: none;              /* animates only on user toggle (.vn-anim) */
  width: clamp(380px, 34cqw, 560px);
  max-height: 86%;
  overflow-y: auto;
  border: 1px solid var(--border-mid);
  border-radius: 6px;
  padding: 22px 24px;
  box-shadow: 0 8px 60px rgba(0,0,0,0.6);
}
.vn-panel-save.vn-anim { transition: opacity 0.2s, transform 0.2s, visibility 0s linear 0.2s; }
.vn-panel-save.open {
  opacity: 1;
  transform: translate(-50%, -50%) scale(1);
  visibility: visible;
  pointer-events: auto;
}
.vn-panel-save.open.vn-anim { transition: opacity 0.2s, transform 0.2s, visibility 0s; }
/* Rich slot rows — one full-width card per slot: name, chapter · turn,
   cast names, timestamp. Shared by the save panel and the menu load list. */
.vn-slot-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 14px;
  width: 100%;
}
.vn-slot-btn {
  display: grid !important;
  grid-template-columns: 1fr auto;
  row-gap: 2px;
  width: 100% !important;
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-mid) !important;
  border-radius: 4px !important;
  padding: 8px 12px !important;
  cursor: pointer !important;
  text-align: left !important;
  transition: border-color 0.15s, background 0.15s !important;
}
.vn-slot-btn:hover {
  border-color: var(--border-accent) !important;
}
.vn-slot-btn.active {
  border-color: var(--border-accent) !important;
  background: rgba(232,168,124,0.1) !important;
}
.vn-slot-btn.empty .vn-slot-name { color: var(--text-muted); font-style: italic; }
.vn-slot-btn.disabled { opacity: 0.45 !important; cursor: default !important; }
.vn-slot-name {
  font-family: var(--font-display);
  font-size: clamp(0.82rem, 1.05cqw, 1.15rem);
  color: var(--text-accent);
  letter-spacing: 0.05em;
  grid-column: 1;
}
.vn-slot-time {
  font-family: var(--font-ui);
  font-size: clamp(0.66rem, 0.85cqw, 0.92rem);
  color: var(--text-muted);
  grid-column: 2; grid-row: 1;
  align-self: center;
}
.vn-slot-meta {
  font-family: var(--font-ui);
  font-size: clamp(0.7rem, 0.9cqw, 1rem);
  color: var(--text-secondary);
  grid-column: 1 / -1;
}
.vn-slot-cast {
  font-family: var(--font-dialogue);
  font-style: italic;
  font-size: clamp(0.7rem, 0.9cqw, 1rem);
  color: var(--text-muted);
  grid-column: 1 / -1;
}
/* Save-name input — sits between the slots and the Save/Load actions. */
.vn-save-name {
  width: 100%;
  box-sizing: border-box;
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border-mid);
  border-radius: 3px;
  color: var(--text-primary);
  font-family: var(--font-ui);
  font-size: clamp(0.78rem, 1cqw, 1.1rem);
  padding: 7px 10px;
  margin-bottom: 10px;
  transition: border-color 0.15s;
}
.vn-save-name:focus {
  border-color: var(--border-accent);
  outline: none;
  box-shadow: 0 0 0 2px var(--border-glow);
}
.vn-save-name::placeholder { color: var(--text-muted); font-style: italic; }
.vn-save-actions { display: flex; gap: 10px; }
.vn-save-action {
  flex: 1;
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-mid) !important;
  color: var(--text-secondary) !important;
  font-family: var(--font-display) !important;
  font-size: clamp(0.72rem, 0.95cqw, 1rem) !important;
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
  font-size: clamp(0.74rem, 0.95cqw, 1rem);
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
.vn-title-screen .vn-title-main { font-size: clamp(1.7rem, 4cqw, 3.1rem); letter-spacing: 0.35em; }
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
.vn-title-actions { display: flex; gap: 14px; align-items: center; }
.vn-title-load {
  font-family: var(--font-display) !important;
  font-size: 0.9rem !important;
  letter-spacing: 0.22em !important;
  text-transform: uppercase !important;
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-mid) !important;
  color: var(--text-secondary) !important;
  padding: 12px 36px !important;
  border-radius: 3px !important;
  cursor: pointer !important;
  transition: color 0.15s, border-color 0.15s !important;
}
.vn-title-load:hover {
  color: var(--text-accent) !important;
  border-color: var(--border-accent) !important;
}
.vn-title-load.disabled {
  opacity: 0.35 !important;
  cursor: default !important;
}

/* ── Main-menu Load list — a third title-screen section, like cast setup.
   Toggled by body.menu-load-active; !important beats Gradio 6's scoped
   re-emission of the base rules (see the cast-setup note above). ── */
.vn-menu-load { display: none; }
body.menu-load-active .vn-title-content { display: none !important; }
body.menu-load-active .vn-cast-setup { display: none !important; }
body.menu-load-active .vn-menu-load {
  display: flex !important;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  width: 100%;
  height: 100%;
  padding: 28px 36px;
  overflow-y: auto;
  gap: 16px;
  background: linear-gradient(160deg, #0d0d1a 0%, #16162e 60%, #0a0a20 100%);
}
.vn-ml-title {
  font-family: var(--font-display);
  font-size: clamp(1.3rem, 2cqw, 2.1rem);
  letter-spacing: 0.28em;
  color: var(--text-accent);
  text-shadow: 0 0 30px rgba(232,168,124,0.2);
}
.vn-ml-list { width: 100%; max-width: min(620px, 80%); }
.vn-ml-back {
  font-family: var(--font-display) !important;
  font-size: clamp(0.82rem, 1.05cqw, 1.15rem) !important;
  letter-spacing: 0.18em !important;
  text-transform: uppercase !important;
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-mid) !important;
  color: var(--text-secondary) !important;
  padding: 10px 32px !important;
  border-radius: 3px !important;
  cursor: pointer !important;
}
.vn-ml-back:hover {
  color: var(--text-accent) !important;
  border-color: var(--border-accent) !important;
}

/* ── Title screen — player name input ── */
.vn-title-name-row { margin-top: 8px; }
.vn-title-name-input {
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border-mid);
  color: var(--text-primary);
  font-family: var(--font-ui);
  font-size: 0.85rem;
  letter-spacing: 0.06em;
  padding: 8px 14px;
  border-radius: 3px;
  text-align: center;
  width: clamp(160px, 20cqw, 280px);
  transition: border-color 0.15s;
}
.vn-title-name-input:focus {
  border-color: var(--border-accent);
  outline: none;
  box-shadow: 0 0 0 2px var(--border-glow);
}
.vn-title-name-input::placeholder { color: var(--text-muted); }

/* ── Cast Setup ── */
.vn-cast-setup {
  display: none;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  width: 100%;
  height: 100%;
  padding: 28px 36px;
  overflow-y: auto;
  gap: 14px;
  background: linear-gradient(160deg, #0d0d1a 0%, #16162e 60%, #0a0a20 100%);
}
.vn-cs-header { text-align: center; max-width: min(820px, 78%); }
.vn-cs-title {
  font-family: var(--font-display);
  font-size: clamp(1.3rem, 2cqw, 2.1rem);
  letter-spacing: 0.28em;
  color: var(--text-accent);
  text-shadow: 0 0 30px rgba(232,168,124,0.2);
  margin-bottom: 8px;
}
.vn-cs-desc {
  font-family: var(--font-dialogue);
  font-size: clamp(0.85rem, 1.15cqw, 1.25rem);
  line-height: 1.65;
  color: var(--text-secondary);
  text-align: left;
  margin-top: 4px;
}
.vn-cs-desc p { margin: 6px 0; }
/* ZeroGPU "generated cast is a preview here" callout — warm accent box,
   shown only on the in-process Space (see screens._zerogpu_mock_notice). */
.vn-cs-notice {
  margin: 2px 0 10px;
  background: rgba(232,168,124,0.08);
  border: 1px solid var(--border-accent);
  border-radius: 4px;
  padding: 9px 13px;
  font-family: var(--font-ui);
  font-size: clamp(0.72rem, 0.95cqw, 0.98rem);
  line-height: 1.5;
  color: var(--text-secondary);
  text-align: left;
}
.vn-cs-notice strong { color: var(--text-accent); }
.vn-cs-desc code {
  background: rgba(232,168,124,0.1);
  color: var(--text-accent);
  padding: 0 4px;
  border-radius: 2px;
  font-size: 0.92em;          /* tracks the scaled description text */
}
.vn-cs-cards {
  display: flex;
  flex-direction: column;
  gap: 10px;
  width: 100%;
  max-width: min(860px, 80%);
}
.vn-cs-card {
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: 6px;
  padding: 12px 14px;
  transition: border-color 0.15s;
}
.vn-cs-card:focus-within { border-color: var(--border-accent); }
.vn-cs-card-row {
  display: flex;
  gap: 10px;
  margin-bottom: 8px;
}
.vn-cs-field {
  display: flex;
  flex-direction: column;
  gap: 3px;
  flex: 1;
}
.vn-cs-field label {
  font-family: var(--font-ui);
  font-size: clamp(0.68rem, 0.85cqw, 0.95rem);
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--text-muted);
}
.vn-cs-input {
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border-mid);
  border-radius: 3px;
  color: var(--text-primary);
  font-family: var(--font-ui);
  font-size: clamp(0.88rem, 1.15cqw, 1.25rem);
  padding: 7px 10px;
  transition: border-color 0.15s;
  width: 100%;
  box-sizing: border-box;
}
.vn-cs-input:focus {
  border-color: var(--border-accent);
  outline: none;
  box-shadow: 0 0 0 2px var(--border-glow);
}
.vn-cs-input::placeholder { color: var(--text-muted); font-style: italic; font-size: 0.9em; }
.vn-cs-tags-row { width: 100%; }
.vn-cs-tags-input {
  font-family: monospace;
  font-size: clamp(0.8rem, 1.05cqw, 1.15rem);
}
.vn-cs-fast {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  max-width: min(860px, 80%);
  width: 100%;
  padding: 6px 0;
}
.vn-cs-fast input[type="checkbox"] {
  accent-color: var(--border-accent);
  width: 15px; height: 15px;
  cursor: pointer;
  flex-shrink: 0;
}
.vn-cs-fast span {
  font-family: var(--font-ui);
  font-size: clamp(0.78rem, 1cqw, 1.05rem);
  color: var(--text-muted);
  letter-spacing: 0.04em;
}
.vn-cs-actions {
  display: flex;
  gap: 14px;
  justify-content: center;
  width: 100%;
  max-width: min(860px, 80%);
  padding-top: 6px;
}
.vn-cs-btn {
  font-family: var(--font-display) !important;
  font-size: clamp(0.82rem, 1.05cqw, 1.15rem) !important;
  letter-spacing: 0.18em !important;
  text-transform: uppercase !important;
  padding: 10px 32px !important;
  border-radius: 3px !important;
  cursor: pointer !important;
  transition: background 0.18s ease, color 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease !important;
  border: 1px solid var(--border-mid) !important;
}
.vn-cs-btn.vn-cs-back {
  background: var(--bg-surface) !important;
  color: var(--text-secondary) !important;
}
.vn-cs-btn.vn-cs-back:hover {
  border-color: var(--border-accent) !important;
  color: var(--text-accent) !important;
}
.vn-cs-btn.vn-cs-go {
  background: linear-gradient(135deg, #c07840, #e8a87c) !important;
  border-color: var(--border-accent) !important;
  color: #0d0d1a !important;
  font-weight: 600 !important;
}
.vn-cs-btn.vn-cs-go:hover {
  box-shadow: 0 0 14px var(--border-glow) !important;
}

/* Choice pills are styled in the "rounded capsules" block above. */

/* ── Stage wrapper — the relative containing block for the canvas + the
   free-text bar. The bar overlays INSIDE the canvas frame, anchored to this
   wrapper (an earlier version had no positioned ancestor and the box flew
   across the viewport). ── */
#stage { position: relative !important; gap: 0 !important; padding: 0 !important;
         margin-bottom: 8px !important; width: 100% !important; }
/* The gr.HTML wrapper insets the canvas by 12px each side; remove it so the
   canvas spans the full stage width and the free-text bar (70% of the stage)
   lines up with the choices (70% of the canvas). */
#stage .html-container { padding: 0 !important; }
#stage .vn-screen { margin-bottom: 0 !important; }

/* ── Free-text action bar — docked inside the canvas, in the same centered
   column as the choices (which sit just above it) and styled to match the
   choice buttons, so it reads as part of the interaction cluster rather than
   a glued-on pill. Hidden until the END of a turn — revealed only on the last
   beat once the text has finished revealing, together with the choices (JS
   sets body.vn-playing; see syncStage). ── */
/* The bar is styled to be visually identical to a .vn-choice-btn (see the
   "Choice restyle" block above): the same translucent void-scrim, blur(4px),
   borderless 2px accent left bar, sharp corners and italic dialogue voice —
   so it reads as just one more option in the stack, not a separate widget. */
/* The free-text input now lives INSIDE the canvas as .vn-free-capsule (the
   last item in the centered choice column). This Gradio row is kept only as
   the hidden value/submit bridge — never shown. */
#free-input-row { display: none !important; }

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
  font-family: var(--font-character) !important;
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

/* ── Turn-pending indicator ── replaces Gradio's overlay as the "thinking"
   feedback: JS adds body.vn-waiting when a choice / free-text is submitted
   and removes it when the next screen render lands. The unpicked choices dim
   and the whole column locks (no double submits); the PICKED one (JS tags it
   .vn-chosen) stays lit and blips so you can see what you committed to. A
   pulsing ✦ inside the text box marks the wait — the same spot the
   streaming loader and the ▼ continue hint use.
   !important is load-bearing twice over: it beats the PREFIXED copies of the
   base rules (the body.…-keyed prefixed twins never match) AND the lingering
   fill-mode opacity of the vnChoicesIn animation. */
body.vn-waiting .vn-choices .vn-choice-btn:not(.vn-chosen),
body.vn-waiting .vn-free-capsule:not(.vn-chosen) { opacity: 0.35 !important; }
body.vn-waiting .vn-choices .vn-choice-btn.vn-chosen,
body.vn-waiting .vn-free-capsule.vn-chosen {
  opacity: 1 !important;
  animation: vnPulse 1.2s ease infinite;
}
body.vn-waiting .vn-choices .vn-choice-btn,
body.vn-waiting .vn-free-capsule { pointer-events: none !important; }
body.vn-waiting .vn-dialogue::after {
  content: '✦';
  position: absolute;
  right: 18px; bottom: 14px;
  color: var(--text-accent);
  font-size: 1.05rem;
  z-index: 30;
  animation: vnPulse 1.2s ease infinite;
}

/* Waiting from the TITLE screen (Generate Cast, curated Start, menu Load):
   there is no text box to host the ✦, and the menu must not stay clickable
   while generation spins up — lock it, dim it, and pulse a big centered ✦
   the user can't miss. */
body.vn-waiting .vn-title-screen { pointer-events: none !important; }
/* ::before is a static near-opaque scrim that hides the menu beneath — the
   screen visibly leaves its interactive state the moment the wait starts.
   (A child-dimming opacity rule was tried first but Gradio's style pipeline
   wouldn't let a stylesheet !important opacity land on these nodes; the
   scrim is sturdier anyway.) ::after is the pulsing ✦ on top of it. */
body.vn-waiting .vn-title-screen::before {
  content: '';
  position: absolute;
  inset: 0;
  background: rgba(13, 13, 26, 0.86);
  z-index: 40;
}
body.vn-waiting .vn-title-screen::after {
  content: '✦';
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-accent);
  font-size: 2.2rem;
  z-index: 41;
  animation: vnPulse 1.2s ease infinite;
}

/* ── Kill Gradio's queue-progress chrome inside the stage ── Gradio overlays
   each output component with .wrap.full (a cover) + .eta-bar (a dark,
   inset:0, opacity-.8 layer that GROWS across the component via a transform)
   and dims the component to opacity .2 via .pending while a callback runs.
   Over the canvas this read as the whole UI — sprites included — "dragging"
   diagonally whenever a turn was processing. The event listeners pass
   show_progress="hidden"; this is the belt-and-braces for any that don't. */
#stage .eta-bar { display: none !important; }
#stage .progress-text { display: none !important; }
#stage .wrap.full { background: transparent !important; }
#stage .pending { opacity: 1 !important; animation: none !important; }
#stage .generating { animation: none !important; border: none !important; }

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

# Inject the env-configured character-dialogue font into the --font-character token.
CSS = CSS.replace("__FONT_CHARACTER__", CHARACTER_FONT)

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
        // A fresh render arrived — the turn the user was waiting on is here.
        if (document.body.classList.contains('vn-waiting')) setWaiting(false);
        screen.setAttribute('data-typing', '1');
        // Small delay so the DOM settles
        setTimeout(function() { startTypewriter(screen); }, 50);
    }
    // A title-screen render also ends any pending wait (e.g. a menu load
    // that failed back to the title — it has no data-typing attribute).
    // Same for the casting/loading/review screens: they ARE the response to
    // the click that armed the wait, and they carry their own ✦ pulse.
    if (screen && document.body.classList.contains('vn-waiting')
        && (screen.classList.contains('vn-title-screen')
            || screen.classList.contains('vn-casting-review')
            || screen.querySelector('.vn-loading, .vn-casting'))) {
        setWaiting(false);
    }
    restoreUiState();
    syncSaveStatus();
    syncSlotRows();
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
    // Reveal only after the line finishes typing — choices and the bar appear
    // together once the text is fully shown, never before it.
    const showBar = lastBeat && typingDone;
    document.body.classList.toggle('vn-playing', showBar);
}

// Enter in the in-canvas free-text capsule submits via the hidden bridge.
document.addEventListener('keydown', function(e) {
    if (e.key !== 'Enter') return;
    const active = document.activeElement;
    if (active && active.classList && active.classList.contains('vn-free-text')) {
        e.preventDefault();
        submitFreeText(active);
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

// Write a value into a hidden Gradio textbox so the server callback reads it
// (React-style: use the prototype setter + dispatch 'input', or Gradio drops it).
const setBoxValue = (boxId, val) => {
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

// Turn-pending indicator: body.vn-waiting dims/locks the choice column and
// pulses a ✦ while the server thinks (a real-LLM turn takes seconds, and the
// Gradio progress overlay is suppressed — show_progress="hidden"). Cleared
// when the next screen render arrives; the timeout is a stuck-state guard
// (e.g. the callback errored and no re-render ever comes).
const setWaiting = (on) => {
    document.body.classList.toggle('vn-waiting', on);
    clearTimeout(window._vnWaitTimer);
    if (on) {
        window._vnWaitTimer = setTimeout(function() {
            document.body.classList.remove('vn-waiting');
        }, 120000);
    }
};

// Submit the in-canvas free-text capsule via the hidden Gradio bridge.
const submitFreeText = (input) => {
    const val = ((input && input.value) || '').trim();
    if (!val) return;
    setBoxValue('free-input', val);
    // Mark the capsule as the committed pick — it stays lit and blips while
    // the unpicked choices dim (body.vn-waiting rules).
    const cap = input && input.closest('.vn-free-capsule');
    if (cap) cap.classList.add('vn-chosen');
    setWaiting(true);
    setTimeout(() => clickHidden('#free-submit-btn'), 120);
};

document.addEventListener('click', function(e) {
    // Free-text send arrow inside the choice column
    const go = e.target.closest('.vn-free-go');
    if (go) {
        e.preventDefault();
        const cap = go.closest('.vn-free-capsule');
        submitFreeText(cap && cap.querySelector('.vn-free-text'));
        return;
    }

    const btn = e.target.closest('.vn-choice-btn');
    if (btn) {
        e.preventDefault();
        const id = btn.getAttribute('data-choice-id');
        const text = btn.getAttribute('data-choice-text');
        if (!id) return;

        setBoxValue('choice-id-box', id);
        setBoxValue('choice-text-box', text);

        // The committed pick stays lit and blips; its siblings dim.
        btn.classList.add('vn-chosen');
        setWaiting(true);
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
        // The log is a full-session register — land on the newest entry.
        if (isOpen && p.classList.contains('vn-panel-log')) {
            p.scrollTop = p.scrollHeight;
        }
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

// Mirror the hidden slot-rows HTML (refreshed by on_save) into the open
// save panel, so slot descriptions update right after a save without a
// full canvas re-render. The applied string is remembered as a JS property
// (dst._mirrorHtml) — NOT compared against live innerHTML, which diverges
// as soon as restoreUiState marks a row .active and would loop the
// observer (assign → mutate → fire → assign…), freezing the page.
function syncSlotRows() {
    const src = document.querySelector('#slots-meta-hidden');
    const dst = document.querySelector('.vn-panel-save .vn-slot-list');
    if (!src || !dst) return;
    const firstBtn = src.querySelector('.vn-slot-btn');
    if (!firstBtn) return;          // mirror not populated yet
    const html = firstBtn.parentElement.innerHTML;
    if (dst._mirrorHtml !== html) {
        dst._mirrorHtml = html;
        dst.innerHTML = html;
        restoreUiState();           // re-mark the active slot row
    }
}

// Arm the slide/fade transition on the CURRENT panel nodes. Only the
// user-toggle paths call this — render-restoration leaves panels without
// .vn-anim so a fresh canvas snaps to its remembered state instead of
// replaying the slide across the scene (the "log sheet clips over the
// scene after a choice" bug).
function animatePanels() {
    document.querySelectorAll('.vn-screen .vn-panel').forEach(function(p) {
        p.classList.add('vn-anim');
    });
}

function togglePanel(panelClass) {
    window._vnUi.panel = (window._vnUi.panel === panelClass) ? null : panelClass;
    animatePanels();
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
        // Sync the name input to the hidden Gradio textbox before starting
        const nameInput = document.querySelector('.vn-title-name-input');
        if (nameInput) setHiddenVal('#player-name-box', nameInput.value.trim() || 'Player');
        const activeOpt = document.querySelector('.vn-title-opt.active');
        const preset = activeOpt ? (activeOpt.getAttribute('data-preset') || 'curated') : 'curated';
        if (preset === 'generated') {
            // Generated → show the Cast Setup step; generation starts from its
            // own button. (Curated has no setup — start immediately.)
            document.body.classList.add('cast-setup-active');
        } else {
            // Curated start runs the casting beat + opening turn — pulse ✦
            // until the first render lands.
            setWaiting(true);
            // 150ms lets the preset input event reach the server first
            setTimeout(function() { clickHidden('#start-btn-hidden'); }, 150);
        }
        return;
    }

    // ── Cast Setup step: Generate proceeds to design; Back returns to title ──
    const csGo = e.target.closest('.vn-cs-go');
    if (csGo) {
        e.preventDefault();
        // Read all card fields from the HTML and bridge them into hidden Gradio controls
        document.querySelectorAll('.vn-cs-card').forEach(function(card) {
            const idx = card.getAttribute('data-cs-idx');
            const nameEl = card.querySelector('[data-cs-field="name"]');
            const conceptEl = card.querySelector('[data-cs-field="concept"]');
            const tagsEl = card.querySelector('[data-cs-field="tags"]');
            if (nameEl) setHiddenVal('#cs-n-' + idx, nameEl.value || '');
            if (conceptEl) setHiddenVal('#cs-c-' + idx, conceptEl.value || '');
            if (tagsEl) setHiddenVal('#cs-t-' + idx, tagsEl.value || '');
        });
        // Read fast generation checkbox
        const fastCb = document.querySelector('.vn-cs-fast-cb');
        if (fastCb) {
            // Look for the hidden Gradio checkbox's native input
            const hiddenFast = document.querySelector('#cs-fast input[type="checkbox"]');
            if (hiddenFast) {
                hiddenFast.checked = fastCb.checked;
                hiddenFast.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }
        // Hide cast setup, reveal the canvas — pulse ✦ until the casting
        // screen render arrives.
        document.body.classList.remove('cast-setup-active');
        setWaiting(true);
        // Fire the hidden Generate Cast button with a small delay for input propagation
        setTimeout(function() { clickHidden('#generate-cast-btn'); }, 150);
        return;
    }
    if (e.target.closest('.vn-cs-back')) {
        e.preventDefault();
        document.body.classList.remove('cast-setup-active');
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
        animatePanels();
        restoreUiState();
        return;
    }

    // ── Main-menu Load list ──────────────────────────────────
    const mlOpen = e.target.closest('.vn-title-load');
    if (mlOpen) {
        e.preventDefault();
        if (!mlOpen.classList.contains('disabled')) {
            document.body.classList.add('menu-load-active');
        }
        return;
    }
    if (e.target.closest('.vn-ml-back')) {
        e.preventDefault();
        document.body.classList.remove('menu-load-active');
        return;
    }
    const mlSlot = e.target.closest('.vn-menu-load .vn-slot-btn');
    if (mlSlot) {
        e.preventDefault();
        if (mlSlot.getAttribute('data-empty') === '1') return;
        window._vnUi.slot = mlSlot.getAttribute('data-slot') || '0';
        setHiddenVal('#save-slot-hidden', window._vnUi.slot);
        document.body.classList.remove('menu-load-active');
        setWaiting(true);
        setTimeout(function() { clickHidden('#menu-load-btn'); }, 150);
        return;
    }

    // ── Save panel: slot rows + actions ─────────────────────
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
        if (which === 'save') {
            // Bridge the in-panel save-name input to the hidden Gradio box
            const nameEl = document.querySelector('.vn-save-name');
            setHiddenVal('#save-name-box', nameEl ? nameEl.value.trim() : '');
        }
        clickHidden(which === 'save' ? '#save-btn-hidden' : '#load-btn-hidden');
        return;
    }

    // ── Casting Review buttons ───────────────────────────────
    const reviewConfirm = e.target.closest('.review-btn.confirm');
    const reviewRedo    = e.target.closest('.review-btn.redo');
    if (reviewConfirm || reviewRedo) {
        e.preventDefault();
        const promptEl = document.querySelector('#review-prompt-text');
        if (promptEl) setHiddenVal('#casting-prompt-box', promptEl.value);
        const descEl = document.querySelector('#review-desc-text');
        if (descEl) setHiddenVal('#casting-desc-box', descEl.value);
        // Instant ✦ feedback; the server's own loading screen takes over
        // when the callback's first yield lands (observer clears the flag).
        setWaiting(true);
        setTimeout(function() {
            clickHidden(reviewConfirm ? '#casting-confirm-btn' : '#casting-redo-btn');
        }, 120);
        return;
    }

    // ── Click anywhere outside an open panel closes it ────────
    const scr = e.target.closest('.vn-screen');
    if (scr && window._vnUi.panel
        && !e.target.closest('.vn-panel')
        && !e.target.closest('.vn-icon-tray')) {
        window._vnUi.panel = null;
        animatePanels();
        restoreUiState();
    }
});

// Escape closes any open panel
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && window._vnUi.panel) {
        window._vnUi.panel = null;
        animatePanels();
        restoreUiState();
        const trayWrap = document.querySelector('.vn-icon-tray');
        if (trayWrap) trayWrap.classList.remove('confirming');
    }
});
"""
