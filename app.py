"""
Ars-Fabula Visual Novel — Gradio entrypoint.

The VN UI is split across the `ui/` package:
  ui.styles     — CSS + onload JS
  ui.media      — data: URI image embedding
  ui.bootstrap  — constants, initial scene, engine construction
  ui.screens    — HTML builders for the canvas
  ui.beats      — response → paced display beats + screen render
  ui.callbacks  — Gradio event handlers (make_callbacks)

This module only assembles the gr.Blocks layout, wires events, and launches.

Run with:
    python app.py                            # production (tries real model)
    ARS_FABULA_BACKEND=mock python app.py    # test mode (no server)
    ARS_FABULA_VOICE=kokoro python app.py    # voiced lines via Kokoro
"""
from __future__ import annotations
import os

# ── Hugging Face Spaces / ZeroGPU ──────────────────────────────────────
# SPACE_ID is set by the Spaces runtime. Two Space flavours share this file:
#   • Docker/T4 Space — the Dockerfile sets ARS_FABULA_BACKEND=server and
#     ARS_FABULA_COMFY_MODE stays "http" (a real ComfyUI subprocess runs).
#   • ZeroGPU Gradio Space — backend defaults to transformers below; a
#     subprocess can't hold a GPU here, so we drive ComfyUI in-process via
#     the "embed" transport (@spaces.GPU). See comfy_embed.py.
ON_SPACES = bool(os.getenv("SPACE_ID"))
if ON_SPACES:
    os.environ.setdefault("ARS_FABULA_BACKEND", "transformers")
    if os.environ["ARS_FABULA_BACKEND"] == "transformers":
        # ZeroGPU path: live cast + live novel backgrounds, both in-process.
        os.environ.setdefault("ARS_FABULA_COMFY_MODE", "embed")
        os.environ.setdefault("ARS_FABULA_BG", "comfyui")

import gradio as gr

from providers import get_background_provider
from ui.styles import CSS, ONLOAD_JS
from ui.bootstrap import make_initial_scene, DEFAULT_CAST, STATIC_DIR
from ui.screens import _title_screen_html
from ui.callbacks import make_callbacks


def create_ui():
    """Build and return the Gradio VN app."""
    bg_backend = os.getenv("ARS_FABULA_BG", "preset")
    bg_provider = get_background_provider(bg_backend)
    (on_start, on_choice, on_save, on_load, on_reset, on_advance,
     on_free_submit, on_casting_confirm, on_casting_redo,
     on_menu_load) = make_callbacks(bg_provider)

    initial_scene = make_initial_scene()

    with gr.Blocks(title="Ars-Fabula VN") as app:
        # ── State ──────────────────────────────────────────────
        scene_state = gr.State(initial_scene)
        engine_ref = gr.State(None)
        cast_data = gr.State(DEFAULT_CAST)
        prev_bg_state = gr.State("")
        beats_state = gr.State([])        # ordered display beats for this turn
        beat_idx_state = gr.State(0)      # which beat is currently shown
        turn_streaming = gr.State(False)  # True while the model is still writing
                                          # this turn — withholds the choices/bar
        casting_state = gr.State(None)    # interactive casting review state

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

        # ── Cast Setup — MENU ONLY (renderd as HTML inside the canvas, NOT Gradio).
        #    Shown on the main menu after Begin → Generated; hidden once a game
        #    starts, restored on Reset (which returns to the title).
        #    Hidden bridge controls: the in-canvas HTML buttons set these via JS
        #    (same pattern as preset-box / player-name-box) before clicking the
        #    hidden generate button. ──
        cs_n_0 = gr.Textbox(value="", visible="hidden", interactive=True, elem_id="cs-n-0")
        cs_c_0 = gr.Textbox(value="", visible="hidden", interactive=True, elem_id="cs-c-0")
        cs_t_0 = gr.Textbox(value="", visible="hidden", interactive=True, elem_id="cs-t-0")
        cs_n_1 = gr.Textbox(value="", visible="hidden", interactive=True, elem_id="cs-n-1")
        cs_c_1 = gr.Textbox(value="", visible="hidden", interactive=True, elem_id="cs-c-1")
        cs_t_1 = gr.Textbox(value="", visible="hidden", interactive=True, elem_id="cs-t-1")
        cs_n_2 = gr.Textbox(value="", visible="hidden", interactive=True, elem_id="cs-n-2")
        cs_c_2 = gr.Textbox(value="", visible="hidden", interactive=True, elem_id="cs-c-2")
        cs_t_2 = gr.Textbox(value="", visible="hidden", interactive=True, elem_id="cs-t-2")
        cs_fast = gr.Checkbox(value=False, visible="hidden", elem_id="cs-fast")
        generate_cast_btn = gr.Button("Generate", visible="hidden",
                                      elem_id="generate-cast-btn")

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
        player_name_box = gr.Textbox(value="Player", visible="hidden",
                                     interactive=True, elem_id="player-name-box")
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
        # Save-name bridge (the in-panel name input writes here before Save) +
        # the slot-rows mirror JS copies into the open panel after a save.
        save_name_box = gr.Textbox(value="", visible="hidden", interactive=True,
                                   elem_id="save-name-box")
        slots_meta = gr.HTML("", visible="hidden", elem_id="slots-meta-hidden")
        # Main-menu Load: clicked by the title-screen load list via JS.
        menu_load_btn = gr.Button("Menu Load", visible="hidden",
                                  elem_id="menu-load-btn")

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

        # ── Casting review bridge widgets ──────────────────────
        casting_prompt_box = gr.Textbox(visible="hidden", interactive=True,
                                        elem_id="casting-prompt-box")
        casting_desc_box = gr.Textbox(visible="hidden", interactive=True,
                                      elem_id="casting-desc-box")
        casting_confirm_btn = gr.Button("Confirm", visible="hidden",
                                        elem_id="casting-confirm-btn")
        casting_redo_btn = gr.Button("Redo", visible="hidden",
                                      elem_id="casting-redo-btn")

        main_outputs = [scene_state, engine_ref, vn_screen, cast_data,
                        cast_panel, narration_log, scene_info,
                        voice_audio, prev_bg_state]
        # Turn callbacks also carry the beat list + index (read-at-your-own-pace).
        turn_outputs = main_outputs + [beats_state, beat_idx_state]
        # Start + casting confirm carry the interactive review state as well.
        start_outputs = turn_outputs + [casting_state]

        # ── Event wiring ───────────────────────────────────────
        # Two entry points into on_start, both reading the preset:
        #   • start_btn  — clicked by Begin for the CURATED path (no setup step).
        #   • generate_cast_btn — clicked from the Cast Setup step (GENERATED).
        # Cast Setup's visibility is driven entirely by JS (body.cast-setup-active
        # + CSS), so no Gradio show/hide is needed here.
        # show_progress="hidden" everywhere: Gradio's default queue overlay dims
        # the whole canvas (.pending → opacity 0.2) and slides a dark .eta-bar
        # across it via a transform — which read as the entire UI (sprites
        # included) "dragging" diagonally while a turn was processing.
        _start_inputs = [preset_box, casting_state, player_name_box,
                       cs_n_0, cs_c_0, cs_t_0,
                       cs_n_1, cs_c_1, cs_t_1,
                       cs_n_2, cs_c_2, cs_t_2,
                       cs_fast]
        start_btn.click(fn=on_start, inputs=_start_inputs, outputs=start_outputs,
                        show_progress="hidden")
        generate_cast_btn.click(fn=on_start, inputs=_start_inputs, outputs=start_outputs,
                                show_progress="hidden")

        submit_choice_btn.click(
            fn=on_choice,
            inputs=[choice_input, choice_text_input, scene_state, engine_ref,
                    prev_bg_state, narration_log],
            outputs=turn_outputs + [turn_streaming],
            show_progress="hidden",
        )

        # Advance one beat: re-render the screen, bump the index, chain the
        # crossfade (prev_bg), and play that beat's voiced line if any.
        advance_btn.click(
            fn=on_advance,
            inputs=[beats_state, beat_idx_state, scene_state, prev_bg_state,
                    narration_log, turn_streaming],
            outputs=[vn_screen, beat_idx_state, voice_audio, scene_info, prev_bg_state],
            show_progress="hidden",
        )

        save_btn.click(fn=on_save,
                       inputs=[save_slot, save_name_box, narration_log,
                               scene_state, engine_ref],
                       outputs=[save_status, slots_meta], show_progress="hidden")
        load_btn.click(fn=on_load,
                       inputs=[save_slot, scene_state, engine_ref, prev_bg_state],
                       outputs=turn_outputs + [save_status],
                       show_progress="hidden")
        menu_load_btn.click(fn=on_menu_load, inputs=[save_slot],
                            outputs=turn_outputs + [save_status],
                            show_progress="hidden")

        reset_btn.click(fn=on_reset, outputs=turn_outputs, show_progress="hidden")

        # ── Casting review — confirm advances to next character ──
        casting_confirm_btn.click(
            fn=on_casting_confirm,
            inputs=[casting_state, casting_prompt_box, casting_desc_box,
                    scene_state, engine_ref, prev_bg_state],
            outputs=start_outputs,
            show_progress="hidden",
        )
        # ── Casting review — redo regenerates current base ────
        redo_outputs = [vn_screen, scene_info, casting_state, casting_prompt_box]
        casting_redo_btn.click(
            fn=on_casting_redo,
            inputs=[casting_state, casting_prompt_box, casting_desc_box],
            outputs=redo_outputs,
            show_progress="hidden",
        )

        # ── Free-text action (fallback when no choices, or "Write your
        #    own action…"). Clears the box server-side via the extra output. ─
        # Both the ➤ button and Enter (via JS bridge → clicks this button)
        # route here; clears the box server-side via the extra output.
        free_outputs = turn_outputs + [turn_streaming, free_input]
        free_submit.click(
            fn=on_free_submit,
            inputs=[free_input, scene_state, engine_ref, prev_bg_state,
                    narration_log],
            outputs=free_outputs,
            show_progress="hidden",
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

    if ON_SPACES:
        # ZeroGPU wants the model loaded (and placed on cuda, under its
        # emulation layer) at startup, not lazily inside a request.
        from model_client import preload_transformers
        preload_transformers()
        # Embed mode has no Dockerfile to bake the diffusion weights in — stage
        # them now (idempotent) so the first cast bake doesn't stall on a
        # multi-GB download mid-@spaces.GPU.
        if os.getenv("ARS_FABULA_COMFY_MODE") == "embed":
            try:
                import comfy_embed
                comfy_embed.stage_models()
            except Exception as e:
                print(f"[space] comfy model staging failed ({e}); "
                      f"cast will fall back to curated")
        port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    app = create_ui()
    print(f"\n🌸 Ars-Fabula VN starting on http://127.0.0.1:{port}")
    print(f"   Mock mode: {os.getenv('ARS_FABULA_BACKEND', 'server') == 'mock'}")
    print(f"   Voice: {os.getenv('ARS_FABULA_VOICE', 'mock')} · "
          f"BG: {os.getenv('ARS_FABULA_BG', 'preset')}\n")
    app.launch(
        server_name="0.0.0.0" if ON_SPACES else "127.0.0.1",
        server_port=port,
        allowed_paths=[STATIC_DIR],   # belt-and-braces; images are data URIs anyway
        css=CSS,                      # Gradio 6: css/js/theme live on launch()
        js=ONLOAD_JS,
        theme=gr.themes.Base(
            primary_hue="amber",
            neutral_hue="slate",
        ),
        # CSR, never SSR. Gradio's CSS auto-prefixer boosts every custom rule's
        # specificity with a `.gradio-container…contain` prefix ONLY in CSR; in
        # SSR (which HF Spaces enables by default) that prefix is absent, so the
        # app's own selectors (e.g. .vn-title-main) lose the cascade to Gradio's
        # component CSS (.prose *, button) and every gold accent silently drops
        # to the neutral theme colour. Forcing CSR makes the Space render
        # identically to local. Harmless locally (local already runs CSR).
        ssr_mode=False,
        quiet=False,
        share=False,
    )
