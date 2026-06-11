# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Ars-Fabula is a slice-of-life anime visual novel PoC: a Gradio web UI driven by an LLM that narrates scenes, controls character expressions/backgrounds, and offers branching choices. Image and voice generation are pluggable (local ComfyUI / Kokoro / Chatterbox), with mock fallbacks so the whole app runs with no external servers.

This machine's real stack: **llama-server (Gemma 4 26B-A4B MoE + MTP, WSL-native atomic fork)** for the LLM + **ComfyUI (WSL)** for image gen, sharing one 8GB GPU — see "VRAM management" below. Falls back to **Gemma 12B** (Windows `llama-server.exe`) when the 26B binary isn't available.

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
Model: `ARS_FABULA_BACKEND` (`server`|`mock`), `ARS_FABULA_BASE_URL` (default `http://localhost:8110/v1`), `ARS_FABULA_MODEL` (default `gemma-26b-a4b`), `ARS_FABULA_API_KEY`, `ARS_FABULA_TEMPERATURE`, `ARS_FABULA_MAX_TOKENS`, `ARS_FABULA_TIMEOUT`.
Generation: `ARS_FABULA_COMFY` (ComfyUI URL), `ARS_FABULA_SAMPLER` (default `dpm_2_ancestral`), `ARS_FABULA_SCHEDULER` (`simple`), `ARS_FABULA_STYLE` (shared style tags, default = light-novel/official-art set: `official art, light novel illustration style, clean lineart, soft cel shading, pastel color palette, soft lighting, shiny detailed eyes, smooth shading`), `ARS_FABULA_SEED` (fixed generation seed, default **42** — all presets + custom specs use it; face-gate/expression retries still bump it locally to self-heal), `ARS_FABULA_REGEN` (force re-bake), `ARS_FABULA_QUALITY`, `ARS_FABULA_NEGATIVE`, `ARS_FABULA_FAST_WIDTH`/`ARS_FABULA_FAST_HEIGHT` (fast-mode base resolution, default 384×672). Sprite framing (see `_frame_upper_body`): `ARS_FABULA_FACE_TARGET` (face-anchored: target face height ÷ base-image height, default 0.135), `ARS_FABULA_SPRITE_CANVAS` (output-canvas size ÷ base size, default 1.4), `ARS_FABULA_SPRITE_FRAC` (upper-body crop fraction 0.62 — **only the fallback** crop when no face mask exists). Fast generation (lower base resolution, quicker bake; self-heal/steps unchanged) is a Cast Setup checkbox → `cast_pipeline.set_fast_generation`; fast sprites cache under `static/sprites/generated/<key>_fast/`.
UI: `ARS_FABULA_DIALOGUE_FONT` (CSS font-family stack for CHARACTER dialogue + choices, default `'MS Gothic', 'MS PGothic', 'Osaka-Mono', monospace`; narration is fixed to Georgia serif).
VRAM/LLM lifecycle: `ARS_FABULA_LLM_MANAGE` (`auto`|`wsl`|`win`|`ollama`|`none`), `ARS_FABULA_LLAMA_BIN`, `ARS_FABULA_LLAMA_MODEL`, `ARS_FABULA_LLAMA_DRAFTER`, `ARS_FABULA_LLAMA_LD_PATH`, `ARS_FABULA_LLAMA_ARGS`, `ARS_FABULA_LLAMA_EXE_WIN`, `ARS_FABULA_LLAMA_MODEL_WIN`, `ARS_FABULA_LLM_STOP_CMD`, `ARS_FABULA_LLM_START_CMD`, `ARS_FABULA_LLM_UNLOAD_URL`.
Media: `ARS_FABULA_VOICE` (`mock`|`kokoro`|`chatterbox`), `ARS_FABULA_BG` (`preset`|`comfyui`).
Traces: `ARS_FABULA_TRACE` (`1`/on default, `0` disables), `ARS_FABULA_TRACE_DIR` (default `logs/traces`) — per-turn prompt→response JSONL for fine-tuning (real models only; mock skipped). See `trace_log.py`.

### External services
- LLM: any OpenAI-compatible server at `ARS_FABULA_BASE_URL` (here: atomic-fork llama-server on WSL `:8110`, Gemma 4 26B-A4B MoE + MTP).
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
# tools/Start_Gemma_Headless.bat       — menuless 12B llama-server launcher (Windows)
# tools/start_gemma_server.sh          — headless 26B atomic-fork launcher (WSL-native, MTP + CPU MoE)
# tools/workflows/anima_expression_sheet.json — the validated FaceDetailer workflow (source of truth)
```

## Architecture

Layered so the engine never depends on a concrete model, image generator, or TTS — only interfaces. This is what makes mock-fallback possible and keeps everything testable without servers.

**`vn_contracts.py` — the frozen data contract.** `Character` (immutable identity), `CastMember` (character + `expression_map`), `SceneState` (mutable session; includes `expressions` dict + `present` list), `Choice`, `ActionResult`, `SaveData` (JSON snapshot with `from_scene`/`apply_to`). `STANDARD_EXPRESSIONS` (7) is the invariant every cast member must satisfy. `cast_to_dict(list)` bridges the pipeline's list to the scene's dict. **Do not change these shapes casually.**

**`vn_engine.py` — the turn loop.** `VNEngine.run_turn(text)`: build prompt (system + `[STATE]` banner) → `ModelClient` → parse `[TOOL: ...]` calls from response text (bracketed text, NOT structured tool-calling) → route to handlers that mutate `SceneState` → enforce cast-lock → selectively voice the emotional beat → return `(raw_response, scene)`. Also `save(slot)`/`load(slot)` (JSON), `apply_choice_effects`.

**The `[TOOL: ...]` parser** (`parse_tool_calls` / `_parse_tool_args` / `_scan_balanced`) is bracket-and-quote-aware: dialogue prose has apostrophes ("I've") and `offer_choices` carries nested JSON, so the scanner tracks only double-quoted strings (escape-aware) + bracket depth — single quotes are NOT delimiters. Tools (`_TOOL_HANDLERS`): `advance_scene`, `set_background`, `set_expression`, `enter_character`, `exit_character`, `offer_choices`, `save_checkpoint`.

**Cast-lock**: the LLM may only reference characters that have sprites. `_find_cast_violations` detects non-cast `[Name]:` speakers, `_clean_noncast_dialogue` strips them, and the violation is surfaced as an engine note. Enforced regardless of model output.

**`model_client.py` — model independence + VRAM lifecycle.** `ModelClient` ABC → `LlamaServerClient` / `MockModel`. `get_model`/`ModelConfig.from_env` factory. **VRAM management** (LLM and diffusion can't co-reside on 8GB): `release_vram()` stops the LLM before a bake (WSL-native: `pkill -f 'build/bin/llama-server'`; Windows: `taskkill llama-server.exe` via WSL interop; also ollama keep_alive=0 / custom cmd/url hooks); `ensure_llm()` relaunches it (WSL-native: atomic-fork binary with `LD_LIBRARY_PATH`, MTP drafter, `--cpu-moe`, `-c 32000`; Windows: executes `llama-server.exe` directly via its WSL `/mnt/c/...` path — NOT `cmd start`, which mangled args — then polls `/health`). Auto-detects available binary: `_llm_manage_mode` returns `wsl` (atomic fork), `win` (Windows .exe), or falls back.

**`cast_pipeline.py` — character generation.** Returns `list[CastMember]` (GENERATE or CURATED, same type). Preset cast (`PRESET_CHARACTERS`): **yuki**, **marie** (blonde ojou-sama — replaced the old `haruki` boy), **sakura**, **ryo**; the default trio is `["yuki","marie","sakura"]`. All presets + custom specs share a fixed `SEED` (default 42). GENERATE pipeline per character: **full-body** Anima base gen at 512×896 (`build_identity_prompt` = QUALITY + STYLE_TAGS + appearance + `SPRITE_POSE_TAGS`; flat tags only — **weighted `(tag:weight)` is unreliable on Anima**) → **face-detect gate** (`face_yolov8n`, retry-on-miss; **saves `<char_dir>/_facegate_mask.png`** = the detected-face mask, reused below) → **FaceDetailer per expression** (the validated `anima_expression_sheet.json`, with a per-character eye/hair **anchor** so colours don't drift; `EXPRESSION_RETRIES` self-heals misses like laugh; the `neutral` expression prompt is `expressionless, calm…` — bare "neutral expression" read grumpy on Anima — and the base pose uses `calm expression`) → **rembg `isnet-anime`** matte → **`_frame_upper_body`**: **face-anchored** framing — reads `_facegate_mask.png`, scales each sprite so the detected face is a constant size (`ARS_FABULA_FACE_TARGET`), then composites the figure **feet-down onto a shared-size canvas** (`ARS_FABULA_SPRITE_CANVAS`× base). Because heads are a constant size on a shared baseline, **taller figures (smaller face) reach higher → render taller**; all sprites share one pixel size so the UI's uniform height-lock preserves the gap. Falls back to the old alpha-crown + `UPPER_BODY_FRAC` crop when no mask exists. Sprites cache to `static/sprites/generated/<key>/`; reused unless `ARS_FABULA_REGEN`. Custom casts: `build_custom_specs` turns UI cards into specs; `design_character` asks the LLM (or a heuristic) to design appearance tags from a concept; cache key hashes the prompt so edits auto-invalidate. `_comfy_*` HTTP helpers use a proxy-bypassing session and fetch outputs via `/view` (not a guessed output dir).

**`providers.py` — backgrounds & voice.** `BackgroundProvider` (Preset / ComfyUI; live gen waits + fetches via `/view`, uses the UNET trio + STYLE_TAGS, only outside `PRESET_BACKGROUNDS`, only at beat boundaries) and `VoiceProvider` (Kokoro / Chatterbox / Mock; `speak(text, character, emotion, voice_id)`). Voicing is **selective** — first dialogue line of a character whose expression changed this turn.

**`app.py` + `ui/` package — Gradio UI.** As of this branch the old ~2700-line `app.py` is split into a `ui/` package; `app.py` is now a thin entrypoint (`create_ui()` builds the `gr.Blocks` layout + wires events; `__main__` launches). Nothing imports from `app.py` — it's a pure leaf. Modules (acyclic: styles/media are leaves → screens → beats → callbacks → app):
- **`ui/styles.py`** — the `CSS` and `ONLOAD_JS` strings (presentation only). `CSS` injects `ARS_FABULA_DIALOGUE_FONT` into a `--font-character` token via a `.replace()` after the string. **Gradio 6 runs `launch(js=…)` as a FUNCTION BODY**, not an eval'd expression — keep it plain statements (no `()=>{}` wrapper) or the listener never attaches. **Gradio 6 also duplicates every custom CSS rule with a `.gradio-container.gradio-container-<ver> .contain ` prefix** (+1 id +3 classes of specificity); `body.<state>`-keyed rules only exist unprefixed, so they must out-specify the PREFIXED copy of any competing base rule — hence the doubled-id selectors (`#cast-setup#cast-setup`, `#stage#stage`). Closed overlay panels are `visibility:hidden` (not just translated off-canvas) so they can never paint during a render swap; all event wirings in `app.py` pass `show_progress="hidden"` so Gradio's queue overlay (`.eta-bar`/`.pending`) never plays over the canvas.
- **`ui/media.py`** — `data_uri` / `thumb_data_uri` (base64 embedding, lru-cached).
- **`ui/bootstrap.py`** — constants (`PROJECT_DIR`, `STATIC_DIR`, `DEFAULT_CAST`), `make_initial_scene`, the mock engine, `_try_real_llm`, `build_engine`.
- **`ui/screens.py`** — all HTML builders (`_title_screen_html`, `_casting_screen_html`, `_review_screen_html`, `_vn_screen_html`, panels, cast panel).
- **`ui/beats.py`** — `clean_narration`, `_split_markers`, `_chunk_narration`, `_segment_beats`, `_render_beats_screen`.
- **`ui/callbacks.py`** — `make_callbacks(bg_provider)` → all `on_*` handlers.

Key behaviors:
- **Casting beat** (`on_start`, a generator): yields the casting screen, reveals portraits, then the VRAM swap (stop LLM → ComfyUI bakes → `comfy_free_vram()` → relaunch+`ensure_llm`) and opens the scene.
- **Beats system**: `_segment_beats` splits a turn into ordered display beats (one narration chunk or dialogue line, tagged with the expressions/background/present in effect). Narration capped to **≤2 sentences / ~240 chars** (`_chunk_narration`). `_render_beats_screen` renders beat `idx`; `on_advance` (hidden `#advance-btn`, fired by click/Space/Enter) steps through. A **typewriter** reveals text; the interaction cluster appears only when the line finishes (`data-typing="0"`).
- **Dialogue box**: fixed height (180px), ~88% width centered, single flat fill `var(--dialogue-bg)` (no gradient). Fonts: **character dialogue + choices = `--font-character` (MS Gothic, env-set); narration = Georgia serif**.
- **Choices + free-text are ONE centered column inside the canvas** (`.vn-choices`, `position:absolute; inset:0; justify-content:center`): each offered choice is a rounded capsule, and a **free-text capsule (`.vn-free-capsule`, an in-canvas `<input>` + ➤) is the last item** — wired to the hidden Gradio `#free-input`/`#free-submit-btn` bridge via JS (`setBoxValue`+`clickHidden`). The old separate Gradio `#free-input-row` bar is kept only as the hidden bridge carrier (`display:none`).
- **Frame sizing**: `.gradio-container` width = `min(90vw, 1700px, 156vh)` (≈10% side padding, capped, and height-aware so the 16:9 `.vn-screen` fills vertical space without overflow). The canvas is `aspect-ratio:16/9` (width-driven, no `vh` max-height).
- **Sprites**: speaker `.active`, others `.inactive` (dimmed). Data URIs; cast-panel thumbnails downscaled.
- **Overlay panels** (inside the canvas): Log / Cast / Save + the icon **tray** (Log/Cast/Save/Reset). JS observer re-applies open-panel state across re-renders. Hidden bridge controls use `visible="hidden"` (not `False`).
|- **Title → Cast Setup → design flow**: the title is canvas HTML (Generated/Curated toggle + name + Begin). Begin with **Generated** reveals the **Cast Setup step** inside the canvas (`body.cast-setup-active` shows `.vn-cast-setup`, hides `.vn-title-content`), with **Generate Cast** / **Back** buttons styled as VN HTML; Begin with **Curated** starts immediately. `generate_cast_btn` (hidden) and the hidden `start_btn` both fire `on_start` (preset decides the path). Cast Setup visibility is JS-class/CSS driven (no Gradio show/hide). Blank cards → default trio.
- **Output tuples**: `main_outputs` is a **9-tuple** `(scene, engine, screen, cast, cast_panel, log, info, audio, prev_bg)`; turn callbacks use `turn_outputs` = main_outputs + `[beats_state, beat_idx_state]` (**11**); the free-text callback adds `free_input` (**12**). Keep these in sync when adding outputs.

## Data flow per turn

`choice/➤/Enter (in-canvas, `ui/screens.py`) → JS bridge (`ui/styles.py`) → on_choice/on_free_submit (`ui/callbacks.py`) → apply_choice_effects → engine.run_turn → [model → parse tools → mutate SceneState → cast-lock → voice] → _segment_beats → _render_beats_screen (`ui/beats.py`) → 11/12-tuple → Gradio`, then `on_advance` walks remaining beats.

## Saves

JSON slots (0–4) under `saves/` via `SaveData.from_scene`/`apply_to` (`engine.save`/`engine.load`). The cast is intentionally NOT serialized (locked for the session); only `cast_names` are stored to sanity-check a load.

## Planning docs

`FEATURE_PLAN.md` and `UI_REDESIGN_PLAN.md` capture design decisions; update them when reworking the relevant areas. **`NEXT_SESSION.md` is the session handoff — read it first; it tracks current state, known issues/regressions, and deferred work.**
