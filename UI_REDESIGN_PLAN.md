# Ars-Fabula — VN UI Redesign Plan (corrected, implemented)

## Vision
One full-width VN canvas, no sidebar, no external widgets. All controls overlay
the canvas (Ren'Py grammar): background + sprites + dialogue box; Log / Cast /
Save / Reset live in a small icon tray inside the dialogue box and open panels
*over* the canvas. A title screen renders inside the canvas before the game
starts. The free-text action row is the only element below the canvas.

Gradio remains the host — it is session state + transport for one HTML canvas.
No move to actual Ren'Py: the engine, states, and callbacks stay untouched.

## Corrections folded in (vs. the original draft)
1. **Panel state survives re-renders.** The canvas HTML is replaced every
   beat/turn, which would close any open panel and reset the slot pills. JS
   keeps `window._vnUi = {panel, slot}` and a MutationObserver re-applies the
   open panel + active pill after each render. Status text is mirrored with a
   write-only-if-different guard (prevents observer loops).
2. **Title-screen JS exists.** Handlers for `.vn-title-opt` (preset toggle →
   hidden textbox) and `.vn-title-begin` (clicks hidden `#start-btn-hidden`)
   are part of ONLOAD_JS — the original draft omitted the app's entry point.
3. **`visible="hidden"` vs `visible=False`.** Anything JS must reach stays in
   the DOM with `visible="hidden"`: preset box, start/save/load/reset buttons,
   save slot number, save-status markdown. Pure output sinks the JS never
   reads (`cast_panel`, `scene_info`, `narration_log`) use `visible=False`.
4. **No hidden gr.Radio.** The preset is a hidden `gr.Textbox`
   (`#preset-box`) — the `setVal` bridge can't flip radio inputs. `on_start`
   doesn't care which component type feeds it.
5. **Payload control.** Cast-panel thumbnails use `thumb_data_uri()` (PIL
   downscale to 160px, lru-cached) instead of full-sprite data URIs. The
   `old` bg layer is only embedded when it differs (already true).
6. **Choice legibility.** Not fully transparent: a low-alpha scrim +
   `backdrop-filter` + left accent bar, dialogue font, color/indent hover.
   Panels get explicit z-index above choices and tray.

## Layout (after)
```
gr.Blocks
  gr.State ×6 (unchanged)
  vn_screen  gr.HTML — initial value: _title_screen_html()
  free-input row (always below canvas; unlabeled textbox + "→")
  hidden sinks:   cast_panel, scene_info, narration_log (visible=False)
  hidden bridges: preset-box, start-btn-hidden, save-slot-hidden,
                  save-btn-hidden, load-btn-hidden, reset-btn-hidden,
                  save-status-hidden, choice/advance bridge (all visible="hidden")
  voice_audio: rendered but CSS-collapsed (autoplay still fires; visible=False
               might not mount the <audio> element)
```

## Canvas states
- **Title** — `_title_screen_html()`: title, subtitle, Curated/Generated
  toggle, Begin. `on_reset` returns this (via `_empty()`), restoring the
  title screen after reset.
- **Casting** — unchanged (portraits resolve one by one); emoji removed.
- **Scene** — dialogue box gains: HUD line (`CHAPTER · LOCATION · MOOD`,
  passed as structured args, not re-parsed), name plate, icon tray
  (Log / Cast / Save / Reset with inline Confirm/Cancel for reset).
- **Choices** — same positioning logic, restyled (scrim + accent bar,
  dialogue font, hover indent).
- **Free text** — unlabeled row below canvas, "What do you do…" placeholder.

## Overlay panels (inside .vn-screen, z-index 20)
- **Log** — slides up from bottom (55% height); content = cleaned narration
  passed into `_vn_screen_html`; `on_advance` reads the hidden narration_log
  Textbox as an extra input so re-renders keep the log content.
- **Cast** — slides in from right (40% width); cards with downscaled neutral
  thumbs, name, role, personality; built from `scene.cast` inside
  `_render_beats_screen` (no extra plumbing).
- **Save/Load** — centered modal: slot pills 0–4 (JS → `#save-slot-hidden`),
  Save / Load buttons (JS → hidden gradio buttons), status line mirrored from
  `#save-status-hidden` by the observer.

## What does NOT change
States, callbacks (`on_start/on_choice/on_advance/on_save/on_load/on_reset`),
output tuple shapes, the beats/typewriter system, the choice/advance JS
bridge, the engine. ONLOAD_JS stays a plain-statement function body
(Gradio 6 runs it via `new Function`).

## Implementation order
CSS → `_title_screen_html` → `_cast_panel_html_inner`/thumbs →
`_vn_screen_html` (hud/log/cast/tray/panels) → `_render_beats_screen` →
callbacks (status strings de-emoji'd, `_empty` → title) → `create_ui`
flatten → ONLOAD_JS additions → smoke test all states.
