# Interactive Casting Review — Implementation Plan

> **For Hermes:** Use subagent-driven-development to implement this plan task-by-task.

**Goal:** Replace the all-at-once casting beat with an interactive per-character review loop, giving the user prompt visibility, editing, and confirm/redo control over each base character before expression baking proceeds.

**Architecture:** The `on_start` Gradio generator yields intermediate review screens — one per character — with the appearance prompt text exposed as an editable widget, a generated base image, and confirm/redo buttons. Expression baking (FaceDetailer) runs only after all bases are confirmed. The fast/non-fast mode difference is collapsed to *only* base resolution (same FaceDetailer params for both).

**Tech Stack:** Gradio (existing `gr.State` + `yield` generator pattern), ComfyUI API (existing `_comfy_submit`/`_comfy_wait`), existing `cast_pipeline.py` modules.

---

## Current State

- `cast_pipeline._generate_base_with_face_gate()` generates a full-body base with face-detection retry
- `cast_pipeline._generate_expression()` bakes all 7 expressions via FaceDetailer
- `app.py:on_start` generator calls the pipeline for ALL characters at once, yields the cast screen only after everything is done
- No user visibility into or control over per-character results before finalization
- `_FAST_EXPRESSION_PARAMS` dict currently only applies when `FAST_GENERATION=True`; the standard path still uses the old heavier FaceDetailer values (guide_size=384, max_size=768, steps=20, sammpler=SAMPLER/dpm_2_ancestral, denoise=0.55, etc.)

---

## Proposed Approach

### Phase 1: Normalize Expression Params

Make `_FAST_EXPRESSION_PARAMS` the **default** for ALL modes, rename/merge into the standard config. The only difference between fast and non-fast becomes base canvas resolution (384×672 vs 512×896).

**Why:** The validated expression sheet already uses these lighter params (euler, 15 steps, guide_size=256). There's no quality reason to use heavier values, and keeping two param sets is maintenance overhead for no benefit.

**Changes:**
- Remove `_FAST_EXPRESSION_PARAMS` dict
- Update the inline `_fd_inputs` dict in `_generate_expression` to use the lighter params directly (guide_size=256, max_size=384, steps=15, sammpler_name="euler", denoise=0.5, feather=15, bbox_threshold=0.5, bbox_dilation=25, bbox_crop_factor=2.0, sam_bbox_expansion=0)
- Remove the `if FAST_GENERATION: _fd_inputs.update(...)` conditional
- Update docstrings/comments to reflect that expression params are unified

### Phase 2: Interactive Per-Character Review Loop

Extract base generation out of the bulk pipeline and into an interactive sequence in the `on_start` generator:

1. **design_character runs upfront** for all characters (LLM or heuristic → appearance tags + prompt)
2. **Per-character loop:** for each character:
   a. Generate the full-body base (existing `_generate_base_with_face_gate`)
   b. Yield a Gradio screen showing:
      - The generated base image
      - The full prompt text (editable textbox) — QUALITY + STYLE_TAGS + appearance + SPRITE_POSE_TAGS
      - **Confirm** button — accept this base, proceed to next character
      - **Redo** button — re-run base gen with the edited prompt (same seed or new)
   c. Wait for user interaction before advancing
3. After all bases confirmed → run expression baking (FaceDetailer for all 7 expressions) as a silent batch, same as today
4. VRAM swap → VN starts

### Phase 3: VRAM Consideration

Since base gen and expression baking both need the UNET loaded:
- Phase 2's per-character loop keeps the UNET loaded throughout (no VRAM swaps between characters)
- After all bases confirmed, expression baking can happen immediately (UNET still hot)
- VRAM swap (UNET unload → LLM reload) only happens once, after all baking is done

This means the LLM is **not** available during the interactive review loop. The UI during this phase is purely Gradio-driven with pre-computed data — no LLM calls needed until the VN starts.

---

## Files Likely to Change

| File | What Changes |
|------|-------------|
| `cast_pipeline.py` | Remove `_FAST_EXPRESSION_PARAMS`, unify FaceDetailer config, remove conditional update, potentially expose a `generate_base_review(character_key, prompt, seed)` entry point |
| `app.py` | Rewrite `on_start` generator to yield review screens per character; add hidden buttons for confirm/redo; manage per-character state (which char, accepted prompts, seeds) |
| `providers.py` | No change (background gen is independent) |
| `vn_contracts.py` | Possibly no change |
| `tests/test_unit.py` | Update any tests referencing the fast-mode expression conditional |

---

## Verification

- Unit tests: `pytest -m "not live" -v` — 74 existing tests must still pass
- Manual: Run `ARS_FABULA_BACKEND=mock python app.py` with a small cast — review screens should appear per character with editable prompts, confirm advances, redo regenerates

---

## Risks & Open Questions

- **Gradio hidden-button bridge pattern** is already used for advance/choice interactions. The confirm/redo buttons can follow the same pattern (hidden `gr.Button` with `visible="hidden"`, triggered by visible buttons via JS click bridge)
- **Prompt editing UX** — should the prompt be a plain textbox or a structured form (separate fields for appearance tags, style tags, pose)? Plain textbox is simpler and more flexible; structured form is discoverable. Start with plain textbox.
- **Face gate retry** — should redo also trigger face detection retry? Yes, the existing `_generate_base_with_face_gate` handles this internally.
- **Seed management** — redo should use a fresh seed (increment from last) so the user sees a genuinely different result, not the exact same image with a tweaked prompt.
- **Caching** — fast and non-fast sprites already cache to separate `_fast` directories. The review step doesn't invalidate that — the cache key hashes the prompt, so prompt edits auto-invalidate.
