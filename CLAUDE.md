# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Ars-Fabula is a slice-of-life anime visual novel PoC: a Gradio web UI driven by an LLM that narrates scenes, controls character expressions/backgrounds, and offers branching choices. Image and voice generation are pluggable (local ComfyUI / Kokoro / Chatterbox), with mock fallbacks so the whole app runs with no external servers.

This machine's real stack: **llama-server (Gemma 12B, Windows)** for the LLM + **ComfyUI (WSL)** for image gen, sharing one 8GB GPU — see "VRAM management" below.

## Running

```bash
python app.py                              # production; tries the real model server, falls back to mock
ARS_FABULA_BACKEND=mock python app.py      # force mock LLM (no server)
ARS_FABULA_REGEN=1 python app.py           # force re-bake of generated sprites (ignore cache)
ARS_FABULA_VOICE=kokoro python app.py      # voiced lines via Kokoro (default: mock = silent metadata)
ARS_FABULA_BG=comfyui python app.py        # live background gen for non-preset keys (default: preset)
python app.py 7890                          # override port (also GRADIO_PORT env; default 7861)
```

For real cast generation you must have **ComfyUI running** (WSL, `:8188`). The LLM is auto-managed (started/stopped) by the app during the casting beat. The app degrades gracefully: if the LLM server, ComfyUI, or a TTS server is unreachable, each falls back to a mock / previously-baked / curated / placeholder path. **It always runs and always renders** — images are embedded as base64 data URIs (no Gradio file-route dependency).

### Environment variables
Model: `ARS_FABULA_BACKEND` (`server`|`mock`), `ARS_FABULA_BASE_URL` (default `http://localhost:8110/v1`), `ARS_FABULA_MODEL`, `ARS_FABULA_API_KEY`, `ARS_FABULA_TEMPERATURE`, `ARS_FABULA_MAX_TOKENS`, `ARS_FABULA_TIMEOUT`.
Generation: `ARS_FABULA_COMFY` (ComfyUI URL), `ARS_FABULA_SAMPLER` (default `dpm_2_ancestral`), `ARS_FABULA_SCHEDULER` (`simple`), `ARS_FABULA_STYLE` (shared style tags, default `anime style, cel shading, detailed`), `ARS_FABULA_SPRITE_FRAC` (upper-body crop fraction, 0.62), `ARS_FABULA_REGEN` (force re-bake), `ARS_FABULA_QUALITY`, `ARS_FABULA_NEGATIVE`, `ARS_FABULA_FAST_WIDTH`/`ARS_FABULA_FAST_HEIGHT` (fast-mode base resolution, default 384×672). Fast generation (lower base resolution, quicker bake; self-heal/steps unchanged) is a Cast Setup checkbox → `cast_pipeline.set_fast_generation`; fast sprites cache under `static/sprites/generated/<key>_fast/`.
VRAM/LLM lifecycle: `ARS_FABULA_LLM_MANAGE` (`auto`|`win`|`ollama`|`none`), `ARS_FABULA_LLAMA_EXE_WIN`, `ARS_FABULA_LLAMA_MODEL_WIN`, `ARS_FABULA_LLAMA_ARGS`, `ARS_FABULA_LLM_STOP_CMD`, `ARS_FABULA_LLM_START_CMD`, `ARS_FABULA_LLM_UNLOAD_URL`.
Media: `ARS_FABULA_VOICE` (`mock`|`kokoro`|`chatterbox`), `ARS_FABULA_BG` (`preset`|`comfyui`).
Traces: `ARS_FABULA_TRACE` (`1`/on default, `0` disables), `ARS_FABULA_TRACE_DIR` (default `logs/traces`) — per-turn prompt→response JSONL for fine-tuning (real models only; mock skipped). See `trace_log.py`.

### External services
- LLM: any OpenAI-compatible server at `ARS_FABULA_BASE_URL` (here: llama-server on Windows `:8110`).
- ComfyUI: `http://127.0.0.1:8188` (WSL) — needs Anima UNET (`anima-base-v1.0`) + Qwen CLIP/VAE, Impact-Pack (FaceDetailer, UltralyticsDetectorProvider `bbox/face_yolov8n.pt`, SAM), and `rembg`+`onnxruntime` in the venv (`~/.u2net/isnet-anime.pth`).
- Kokoro TTS `:11436` · Chatterbox TTS `:11437`.

## Testing

```bash
pytest -m "not live" -v        # deterministic spine, no servers (unit + mock tiers)
pytest -m unit -v ; pytest -m mock -v
```
Markers + fixtures (`scene`, `cast`, `cast_list`, `engine`, `mock_model`) live in `tests/conftest.py` (no pytest.ini/pyproject). NOTE: `cast_pipeline` now returns `list[CastMember]` (frozen contract); `vn_contracts.cast_to_dict()` builds `SceneState`'s name→member dict.

## Tools

```bash
python tools/generate_assets.py        # placeholder bgs + curated chibi sprite sets (Pillow)
python tools/cast_contact_sheet.py [--generated]   # cast_report.md — the "friend test" surface / batch check
python tools/comfy_doctor.py [--clear-queue]       # layer-by-layer ComfyUI connection diagnosis
python tools/sampler_ab.py [--char --tries --samplers]   # A/B one char across samplers (fixed seed/prompt)
python tools/sampler_sweep.py          # same, over every OTHER installed sampler (prefixed outputs)
python tools/rembg_cast.py [--force --no-crop]     # re-run matte+crop over baked sprites (no ComfyUI)
python tools/recrop_sprites.py         # re-apply the upper-body crop to baked sprites
# tools/Start_Gemma_Headless.bat       — menuless llama-server launcher the app mirrors
# tools/workflows/anima_expression_sheet.json — the validated FaceDetailer workflow (source of truth)
```

## Architecture

Layered so the engine never depends on a concrete model, image generator, or TTS — only interfaces. This is what makes mock-fallback possible and keeps everything testable without servers.

**`vn_contracts.py` — the frozen data contract.** `Character` (immutable identity), `CastMember` (character + `expression_map`), `SceneState` (mutable session; includes `expressions` dict + `present` list), `Choice`, `ActionResult`, `SaveData` (JSON snapshot with `from_scene`/`apply_to`). `STANDARD_EXPRESSIONS` (7) is the invariant every cast member must satisfy. `cast_to_dict(list)` bridges the pipeline's list to the scene's dict. **Do not change these shapes casually.**

**`vn_engine.py` — the turn loop.** `VNEngine.run_turn(text)`: build prompt (system + `[STATE]` banner) → `ModelClient` → parse `[TOOL: ...]` calls from response text (bracketed text, NOT structured tool-calling) → route to handlers that mutate `SceneState` → enforce cast-lock → selectively voice the emotional beat → return `(raw_response, scene)`. Also `save(slot)`/`load(slot)` (JSON), `apply_choice_effects`.

**The `[TOOL: ...]` parser** (`parse_tool_calls` / `_parse_tool_args` / `_scan_balanced`) is bracket-and-quote-aware: dialogue prose has apostrophes ("I've") and `offer_choices` carries nested JSON, so the scanner tracks only double-quoted strings (escape-aware) + bracket depth — single quotes are NOT delimiters. Tools (`_TOOL_HANDLERS`): `advance_scene`, `set_background`, `set_expression`, `enter_character`, `exit_character`, `offer_choices`, `save_checkpoint`.

**Cast-lock**: the LLM may only reference characters that have sprites. `_find_cast_violations` detects non-cast `[Name]:` speakers, `_clean_noncast_dialogue` strips them, and the violation is surfaced as an engine note. Enforced regardless of model output.

**`model_client.py` — model independence + VRAM lifecycle.** `ModelClient` ABC → `LlamaServerClient` / `MockModel`. `get_model`/`ModelConfig.from_env` factory. **VRAM management** (LLM and diffusion can't co-reside on 8GB): `release_vram()` stops the LLM before a bake (Windows: `taskkill llama-server.exe` via WSL interop; also ollama keep_alive=0 / custom cmd/url hooks); `ensure_llm()` relaunches it (executes `llama-server.exe` directly via its WSL `/mnt/c/...` path — NOT `cmd start`, which mangled args — then polls `/health`).

**`cast_pipeline.py` — character generation.** Returns `list[CastMember]` (GENERATE or CURATED, same type). GENERATE pipeline per character: **full-body** Anima base gen at 512×896 (`build_identity_prompt` = QUALITY + STYLE_TAGS + appearance + `SPRITE_POSE_TAGS`; flat tags only — **weighted `(tag:weight)` is unreliable on Anima**) → **face-detect gate** (`face_yolov8n`, retry-on-miss) → **FaceDetailer per expression** (the validated `anima_expression_sheet.json`, denoise 0.55, with a per-character eye/hair **anchor** so colours don't drift; `EXPRESSION_RETRIES` self-heals misses like laugh) → **rembg `isnet-anime`** matte (`new_session("isnet-anime")` + `remove(..., session=...)`) → **`_frame_upper_body`**: anchored top-crop (head-crown + fixed `UPPER_BODY_FRAC`) so the whole cast lines up at consistent heights. Sprites cache to `static/sprites/generated/<key>/`; reused unless `ARS_FABULA_REGEN`. Custom casts: `build_custom_specs` turns UI cards into specs; `design_character` asks the LLM (or a heuristic) to design appearance tags from a concept; cache key hashes the prompt so edits auto-invalidate. `_comfy_*` HTTP helpers use a proxy-bypassing session and fetch outputs via `/view` (not a guessed output dir).

**`providers.py` — backgrounds & voice.** `BackgroundProvider` (Preset / ComfyUI; live gen waits + fetches via `/view`, uses the UNET trio + STYLE_TAGS, only outside `PRESET_BACKGROUNDS`, only at beat boundaries) and `VoiceProvider` (Kokoro / Chatterbox / Mock; `speak(text, character, emotion, voice_id)`). Voicing is **selective** — first dialogue line of a character whose expression changed this turn.

**`app.py` — Gradio UI.** Per-session `gr.State`. Key structure:
- **Casting beat** (`on_start`, a generator): yields the casting screen, reveals portraits one by one, then runs the VRAM swap (stop LLM → ComfyUI bakes → `comfy_free_vram()` /free → relaunch+`ensure_llm`) and opens the scene.
- **Beats system**: a turn's raw response is split by `_segment_beats` into ordered display beats (one narration chunk or one dialogue line, each tagged with the expressions/background/present in effect). Narration is capped to **≤2 sentences / ~240 chars** per beat (`_chunk_narration`) — long prose paces across click-through beats. `_render_beats_screen` renders beat `idx`; `on_advance` (hidden `#advance-btn`, fired by click/Space/Enter) steps through; choices appear only on the last beat. A **typewriter** reveals text (JS, two-stage click).
- **Sprites**: the speaking character is `.active`, others `.inactive` (dimmed). Data URIs; cast-panel thumbnails downscaled (`thumb_data_uri`).
- **Free-text bar**: a docked pill directly under the canvas, shown only during a scene (`body.vn-playing`); Enter and the ➤ button both submit via the hidden-button **bridge** (`on_free_submit`, clears server-side). (An earlier absolute-overlay version distorted the frame and was removed.)
- **Overlay panels** (inside the canvas): Log / Cast / Save, plus the icon **tray** (Log/Cast/Save/Reset) centered at the bottom of the dialogue box. JS observer re-applies open-panel state across re-renders. Hidden bridge controls use `visible="hidden"` (not `False`).
- **Title screen / menu**: rendered as canvas HTML; **Cast Setup** is a menu-only group (hidden by `on_start`, restored by `on_reset`). No HUD line (removed).
- **Output tuples**: `main_outputs` is a **9-tuple** `(scene, engine, screen, cast, cast_panel, log, info, audio, prev_bg)`; turn callbacks use `turn_outputs` = main_outputs + `[beats_state, beat_idx_state]` (**11**); the free-text callback adds `free_input` (**12**). Keep these in sync when adding outputs.

## Data flow per turn

`choice/➤/Enter → JS bridge → on_choice/on_free_submit → apply_choice_effects → engine.run_turn → [model → parse tools → mutate SceneState → cast-lock → voice] → _segment_beats → _render_beats_screen → 11/12-tuple → Gradio`, then `on_advance` walks remaining beats.

## Saves

JSON slots (0–4) under `saves/` via `SaveData.from_scene`/`apply_to` (`engine.save`/`engine.load`). The cast is intentionally NOT serialized (locked for the session); only `cast_names` are stored to sanity-check a load.

## Planning docs

`FEATURE_PLAN.md` and `UI_REDESIGN_PLAN.md` capture design decisions; update them when reworking the relevant areas.
