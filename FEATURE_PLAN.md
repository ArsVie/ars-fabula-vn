# Ars-Fabula — Feature/Polish Plan (pending implementation)

Status: PLAN ONLY — nothing here is implemented yet. Two open decisions block
the start (see bottom). Items are grouped: pipeline (1–3), UI (4–8).

## 1. Sprite height consistency
Cause: `upper body` is a loose composition tag → per-character framing varies
(chest-up vs hip-up) → head sizes differ once scaled to one on-screen height.
Decision pending: **A vs B.**
- **A (recommended): full-body gen → fixed top-crop.** Generate full body at a
  taller canvas (e.g. 512×896) with a well-defined framing tag (`full body,
  standing, full length`); FaceDetailer the face (small face OK — guide_size
  upscales the region); matte; then deterministic crop of the top ~55%,
  anchored on the alpha-bbox top (head crown). Full-body framing is
  consistent, so the fixed crop yields matching waist-up sprites with matching
  head sizes. Keep the full-body PNG for future "show more body" use.
- **B: face-anchored normalization.** Keep upper-body gen; detect the head
  (reuse ComfyUI face_yolov8n, or approximate from alpha), scale+translate all
  7 expressions so head-top-Y and head-height match across the cast, paste on a
  fixed transparent canvas. More precise, more work, adds a detector call.
- C (rejected): canvas-only normalization — fixes body height, not head size.

## 2. Eye-color (and hair) consistency
Cause: FaceDetailer re-renders the face (denoise 0.55) with only expression
tags → eye/hair color drifts. Fix: per-character **face anchor** (eye color +
hair color) prepended to every expression prompt:
`{QUALITY}. {anchor}, {expr_tags}`.
- Presets: add explicit `eye_color`/anchor field.
- LLM designer: emit `eye_color` in the JSON (reliable); regex from tags
  (`(\w+) eyes`, `(\w+) hair`) as fallback.

## 3. Artist / style tags
Add a configurable `STYLE_TAGS` block (env `ARS_FABULA_STYLE`) prepended to ALL
prompts — cast AND backgrounds — so the whole game shares one look (per the
synthesis report, artist tags are Anima's real style lever).
Decision pending: **generic style descriptors** (default, e.g. `official art,
anime screenshot, soft cel shading, detailed eyes, clean lineart`) vs **named
`@artist` tags** (user provides 1–3). Apply to backgrounds too so cast/BG match.

## 4. Free-form box — proper rebuild
Constraint: a Gradio input can't live inside the gr.HTML canvas. Solution:
overlay a real input over the canvas and reposition by mode.
- Wrap canvas + input in one `position: relative` container.
- Input always present, absolutely positioned over the canvas:
  - No choices → centered "middle of screen" prompt, soft scrim behind.
  - Choices shown → docked just below the choices block.
- Mode switch: JS observer toggles `.has-choices` on the container by checking
  for `.vn-choices` in the fresh canvas — cosmetic only.
- Submit via the PROVEN hidden-button bridge (the choice-button pattern), NOT
  Gradio's flaky `.submit`: Enter (keydown) and the ➤ button both click a
  hidden submit button; the visible textbox stays a real component so its value
  is captured natively. Server-side clear after submit (on_free_submit exists).

## 5. Dim non-speaking characters (active/inactive)
The current beat knows its speaker + present list. Renderer tags the speaking
sprite `.active` and others `.inactive` (brightness/contrast/darken down a
little). Pure-narration beats (no speaker) → everyone normal. CSS + one arg to
`_vn_screen_html` (active speaker), already available in `_render_beats_screen`.

## 6. Remove chapter label + relocate icon tray
- HUD line currently `Chapter · Location · Mood`. Drop the chapter → `Location ·
  Mood`. (Open: remove the HUD entirely instead? — awaiting confirm.)
- Move the Log/Cast/Save/Reset tray from top-right (over the scene) to the
  **bottom-right corner of the dialogue box**. It already lives inside
  `.vn-dialogue`; flip CSS `top:10px` → `bottom:10px`.

## 7. Cast Setup → menu-only
Cast Setup fields are real Gradio inputs (can't be inside the title canvas).
Make the Cast Setup group **visible only in the menu state**:
- visible at startup; hidden by `on_start` when a game begins (add it to that
  callback's outputs as `gr.update(visible=False)`); restored by `on_reset`.
- Title screen gets a "⚙ Cast Setup" affordance that opens/scrolls to it.
Reachable only from the main menu, as requested.

## 8. Narration length limit
Cause: a single narration beat can run 3–4 sentences → cluttered box.
Fix by SPLITTING, not truncating (no lost content):
- In `_segment_beats`, split any narration paragraph exceeding **2 sentences**
  (period-delimited) OR a **~240-char** cap into multiple ≤2-sentence beats the
  player clicks through.
- Also tighten the system prompt to request ≤2-sentence narration so it rarely
  triggers.
- OPEN: "sangría" — if it meant a visual paragraph **indent** on the dialogue
  text, add that as a CSS text style too (separate from the split logic).

---

## Open decisions before implementation
1. **Sprite framing:** Option A (full-body + crop) or B (face-anchored)?
2. **Style:** generic style tags or specific `@artist` tags (name them)?
3. **"sangría":** indentation (visual) or a second length limit (content)?
4. **HUD:** keep `Location · Mood`, or remove the HUD line entirely?
