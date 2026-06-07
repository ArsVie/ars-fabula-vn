# Ars-Fabula VN — Session Handoff

_Last updated: 2026-06-07. Read this + `CLAUDE.md` before starting._

## Current state — what works

The app runs end-to-end in both mock and real-LLM modes:

```bash
ARS_FABULA_BACKEND=mock   python app.py        # no server
ARS_FABULA_BACKEND=server python app.py        # real llama-server on :8111
```

Verified working (mock + real model on `127.0.0.1:8111`):

- **App launches** clean on Gradio **6.16** (`visible="hidden"`, `gr.skip()`, `gr.Number` all valid).
- **Casting beat** → opening scene renders (bg + 3 sprites as base64 data URIs).
- **JS choice bridge** works. A click on an in-screen choice button writes the
  hidden textboxes and submits → a real scene turn runs.
- **Read-at-your-own-pace**: each turn is split into ordered **beats** (one narration
  paragraph or one dialogue line each), revealed one at a time. Click the screen /
  **Space** / **Enter** advances; choices appear only on the **final** beat.
  Sprite expression + background change on the correct beat.
- **Typewriter effect** (NEW): text reveals character-by-character (35ms/char).
  First click during typing completes the text instantly; second click advances.
  Respects `prefers-reduced-motion`.
- **Character enter/exit** (NEW): characters start off-screen. The model calls
  `enter_character` when they appear and `exit_character` when they leave. Per-beat
  presence tracking means characters pop in/out on the correct beat.
- **Save / Load / Reset** work (`saves/slot_N.json`).
- **Tool/engine markers stripped** from player-facing text (dialogue box + Scene Log).
- **Robust tool parsing** — tolerates loose real-model output (bare `[…]` JSON,
  unquoted keys, numeric ids, apostrophes in single-quoted values).
- **71 tests pass**: `pytest -m "not live" -q`.

### Key gotchas learned (don't re-discover these)

1. **Gradio 6 runs `launch(js=...)` as a FUNCTION BODY**, not an eval'd expression.
   The string must be plain statements — an `() => {…}` arrow wrapper is constructed
   and discarded, so the listener never attaches. See the comment above `ONLOAD_JS`
   in `app.py`. This bit us once already; keep it plain-body.
2. **`visible="hidden"`** (not `False`) keeps the bridge controls in the DOM. Clicking
   a `display:none` button and dispatching a synthetic `input` event on a `display:none`
   textbox both still reach the Gradio backend — verified.
3. **The mock always hits the "talk" branch** (the `[STATE]` banner always contains
   "Yuki", which the mock matches), so mock turns look identical. Use the **real model**
   to see varied multi-beat turns. Not a bug.
4. When relaunching the app in a shell, `pkill -f "app.py"` will kill the launching
   shell too (its command line contains "app.py"). Use `pkill -f "[a]pp.py"`.

## Architecture quick map (see CLAUDE.md for detail)

- `vn_contracts.py` — frozen dataclasses (`Character`, `CastMember`, `SceneState`,
  `Choice`, `SaveData`). **Don't change these shapes casually.**
- `vn_engine.py` — `VNEngine.run_turn()`: prompt → model → parse `[TOOL: …]` → mutate
  `SceneState` → cast-lock → return `(raw_narration, scene)`.
- `vn_prompt.py` — the system prompt (tool list, cast-lock rules, format).
- `app.py` — Gradio UI. Turn callbacks return an **11-tuple**:
  `(scene, engine, screen, cast, cast_panel, log, info, audio, prev_bg, beats, beat_idx)`.
  The beat machinery lives in `_split_markers` / `_segment_beats` /
  `_render_beats_screen` / `on_advance`.

---

## TODO for next session

### 1. Clean up: remove unused `_route_tool` import in `_segment_beats`

The `_segment_beats` function in `app.py` imports `_route_tool` from `vn_engine`
but never uses it (it was left behind from the enter/exit refactor). Remove the
unused import.

### 2. Richer mock with varied enter/exit patterns

The mock currently only calls `enter_character` once per branch and
`exit_character` only in the "look" branch. Consider making the mock exercise
multi-character scenes — e.g. have Yuki enter first, then Haruki enter mid-scene,
then have one of them exit. This would exercise per-beat presence tracking in
manual UI testing.

---

## Deferred (still valid)

- **LLM streaming (#1 from earlier):** SSE `stream:true` from llama-server + a live
  tool-call filter that suppresses output inside `[TOOL: …]` spans. Deferred. Would cut
  the first-paint wait on the real model.
- **ComfyUI generation path** (`cast_pipeline._generate_*`) and **Kokoro voice** were
  never live-verified — both servers were unreachable this session. Verify if/when
  ComfyUI (`:8188`) and Kokoro (`:11436`) are running.
- The mock's deterministic branching makes manual UI testing repetitive; a richer mock
  that varies branches and exercises enter/exit/expression tools more thoroughly.
