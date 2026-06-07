# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Ars-Fabula is a slice-of-life anime visual novel PoC: a Gradio web UI driven by an LLM that narrates scenes, controls character expressions/backgrounds, and offers branching choices. Image and voice generation are pluggable (local ComfyUI / Kokoro / Chatterbox), with mock fallbacks so the whole app runs with no external servers.

## Running

```bash
python app.py                              # production; tries the real model server, falls back to mock
ARS_FABULA_BACKEND=mock python app.py      # force mock LLM (no server)
ARS_FABULA_VOICE=kokoro python app.py      # voiced lines via Kokoro (default: mock = silent metadata)
ARS_FABULA_BG=comfyui python app.py        # live background gen for non-preset keys (default: preset)
python app.py 7890                          # override port (also GRADIO_PORT env; default 7861)
```

The app degrades gracefully: if the LLM server, ComfyUI, or a TTS server is unreachable, each falls back to a mock/curated/placeholder path. **It always runs and always renders** — images are embedded as base64 data URIs (no Gradio file-route dependency).

### Environment variables (model)
- `ARS_FABULA_BACKEND` — `server` (default) | `mock`
- `ARS_FABULA_BASE_URL` — OpenAI-compatible endpoint (default `http://localhost:8111/v1`)
- `ARS_FABULA_MODEL`, `ARS_FABULA_API_KEY`, `ARS_FABULA_TEMPERATURE`, `ARS_FABULA_MAX_TOKENS`, `ARS_FABULA_TIMEOUT`

### External services (all optional)
- LLM: any OpenAI-compatible `llama-server` at `ARS_FABULA_BASE_URL`
- ComfyUI: `http://127.0.0.1:8188` — character + background generation
- Kokoro TTS: `http://localhost:11436` · Chatterbox TTS: `http://localhost:11437`

## Testing

```bash
pytest -v                      # all
pytest -m unit -v              # deterministic contract/engine unit tests
pytest -m mock -v              # full engine loop via MockModel (no server)
pytest -m "not live" -v        # everything that doesn't need a model server
pytest tests/test_unit.py::TestSceneState::test_affection_flags -v   # single test
```

Markers (registered in `tests/conftest.py`): `unit` (no server), `mock` (full engine via `MockModel`, no server), `live` (skipped unless a model server is reachable). There is no pytest.ini/pyproject — markers live in `conftest.py`, and fixtures (`scene`, `cast`, `engine`, `mock_model`) are defined there too.

## Asset generation tools

```bash
python tools/generate_assets.py        # create placeholder backgrounds + curated sprite sets (needs Pillow)
python tools/cast_contact_sheet.py     # build cast_report.md — the human "friend test" review surface
```

## Architecture

The codebase is layered so the engine never depends on a concrete model, image generator, or TTS — only on interfaces. This is what makes the mock-fallback behavior possible and keeps everything testable without servers.

**`vn_contracts.py` — the frozen data contract.** Every other module codes against these shapes: `Character` (immutable identity), `CastMember` (character + `expression_map`), `SceneState` (the full mutable session), `Choice`, `ActionResult`, `SaveData`. `STANDARD_EXPRESSIONS` (7 expressions) is the invariant every cast member must satisfy. **Do not change these shapes casually** — they are the shared interface across all modules and tests.

**`vn_engine.py` — the turn loop.** `VNEngine.run_turn(choice_text)` is the heart of the system:
1. Build prompt (system + `[STATE]` banner + choice) via `vn_prompt.build_full_prompt`
2. Send to the `ModelClient`
3. Parse `[TOOL: ...]` calls out of the response text (the model emits tool calls as bracketed text, NOT as structured JSON tool-calling)
4. Route each tool call to a handler that mutates `SceneState`
5. Enforce **cast-lock** (see below)
6. Selectively voice the emotional beat, then return `(narration, scene)`

**The `[TOOL: ...]` parser is the trickiest code here.** `parse_tool_calls` / `_parse_tool_args` / `_scan_balanced` hand-roll a bracket-and-quote-aware scanner because dialogue prose contains apostrophes ("I've", "don't") and the choices argument carries nested JSON. The scanner tracks only double-quoted strings (escape-aware) and bracket depth — single quotes are NOT string delimiters at the top level. Tools: `advance_scene`, `set_background`, `set_expression`, `offer_choices`, `save_checkpoint` (routed via `_TOOL_HANDLERS`).

**Cast-lock** is a core design constraint: the LLM may only reference characters that have sprites. `_find_cast_violations` detects non-cast `[Name]:` speakers, `_clean_noncast_dialogue` strips them (replacing with "*(someone speaks)*"), and the violation is surfaced as an engine note in the narration. The system prompt (`vn_prompt.py`) states this rule emphatically, but the engine enforces it regardless of what the model does.

**`model_client.py` — model independence.** `ModelClient` (ABC) → `LlamaServerClient` (real, OpenAI-compatible) and `MockModel` (scripted via `responses=[...]` or `handler=lambda messages, tools: ...`). `get_model(cfg)` / `ModelConfig.from_env()` is the factory. `MockModel.calls` records every invocation for test assertions.

**`cast_pipeline.py` — character generation.** Two modes returning the same `list[CastMember]` (frozen contract): GENERATE (ComfyUI pipeline) and CURATED (load pre-made sprites). The ComfyUI pipeline is: anime base gen → **anime face-detect gate** (`yolov8x6_animeface`, retry-on-miss) → **ExpressionEditor warp** (AdvancedLivePortrait — warps the same image so identity is preserved by construction, unlike inpainting) → rembg `isnet-anime` alpha matte. Falls back to curated, then to colored placeholder sprites, so the "every member has all 7 expressions" invariant always holds. Note: AdvancedLivePortrait/InsightFace is non-commercial licensed (fine for PoC).

**`providers.py` — backgrounds & voice.** `BackgroundProvider` (Preset / ComfyUI) and `VoiceProvider` (Kokoro / Chatterbox / Mock), each with factory functions. Live background gen only fires for keys outside `PRESET_BACKGROUNDS`, and by design only at beat boundaries (casting/pre-staging), never mid-scene-turn. Voicing is **selective** — only the first dialogue line of a character whose expression changed this turn is voiced (a shipped-VN convention, not full voicing).

**`app.py` — Gradio UI.** Per-session state via `gr.State` (scene, engine, cast, prev background). Key UX intent baked into the code:
- **The casting beat is a generator** (`on_start`) — it yields the casting screen, then reveals portraits one by one as they resolve. Loading is treated as a feature, not a stall.
- **The scene turn** (`on_choice`) does a double-buffered background crossfade + animated sprite swap (CSS in the `CSS` constant).
- In-screen choice buttons are wired via a JS click bridge (`ONLOAD_JS`) that writes into hidden textboxes and clicks a hidden submit button — `gr.HTML` strips `<script>`, so the page-load hook is the reliable path. The hidden controls use `visible="hidden"` (not `False`, which would remove them from the DOM and break the bridge).
- Main callbacks return/yield a **9-tuple** matching `main_outputs`: `(scene, engine, screen, cast, cast_panel, log, info, audio, prev_bg)`. Keep this tuple in sync across all callbacks if you add outputs.

## Data flow per turn

`choice click → JS bridge → on_choice → engine.apply_choice_effects(choice) → engine.run_turn(text) → [model → parse tools → mutate SceneState → cast-lock → voice] → _render_turn → 9-tuple → Gradio`

## Saves

JSON slots (0–4) under `saves/` via `SaveData` (`engine.save`/`engine.load`). The cast is intentionally NOT serialized (it's locked/immutable for the session); only `cast_names` are stored to sanity-check a load against the current cast.
