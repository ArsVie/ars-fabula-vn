# Ars-Fabula VN — Session Handoff

_Last updated: 2026-06-07. Read this + `CLAUDE.md` before starting._

## Current state — what works

The app runs end-to-end in both mock and real-LLM modes:

```bash
ARS_FABULA_BACKEND=mock   python app.py        # no server
ARS_FABULA_BACKEND=server python app.py        # real llama-server on :8111
```

Verified working this session (mock + real model on `127.0.0.1:8111`):

- **App launches** clean on Gradio **6.16** (`visible="hidden"`, `gr.skip()`, `gr.Number` all valid).
- **Casting beat** → opening scene renders (bg + 3 sprites as base64 data URIs).
- **JS choice bridge** works. A click on an in-screen choice button writes the
  hidden textboxes and submits → a real scene turn runs.
- **Read-at-your-own-pace** (added this session): each turn is split into ordered
  **beats** (one narration paragraph or one dialogue line each), revealed one at a
  time. Click the screen / **Space** / **Enter** advances; choices appear only on
  the **final** beat. Sprite expression + background change on the correct beat.
- **Save / Load / Reset** work (`saves/slot_N.json`).
- **Tool/engine markers stripped** from player-facing text (dialogue box + Scene Log).
- **Robust tool parsing** — tolerates loose real-model output (bare `[…]` JSON,
  unquoted keys, numeric ids, apostrophes in single-quoted values).
- **61 tests pass**: `pytest -m "not live" -q`.

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

### 1. Characters should appear when they first speak, not always be on-screen

**Problem:** `SceneState.present` is set to `list(cast.keys())` at scene creation
(`make_initial_scene`, the `SceneState(...)` in `on_start`, and `reset_scene`) and is
never changed. So every cast sprite renders every turn. We want characters to **enter**
the scene (appear) and **exit** (leave), driven by the story.

**Plan:**

1. **Engine tools.** Add two handlers in `vn_engine.py` and register them in
   `_TOOL_HANDLERS`:
   - `enter_character character="<name>"` → cast-lock check, then append to
     `scene.present` if not already there.
   - `exit_character character="<name>"` → remove from `scene.present`.
   (Mirror `_handle_set_expression` for the cast-lock validation pattern.)
2. **Prompt.** Document both tools in `vn_prompt.py` (`VN_SYSTEM_PROMPT` and the
   `_SHORT` variant) and tell the model: characters not yet introduced should NOT be
   `present`; call `enter_character` the first time someone appears/speaks, and
   `exit_character` when they leave. Cast-lock still applies (only cast names).
3. **Initial presence.** Start scenes with a smaller `present` (e.g. empty or just the
   POV-adjacent character) instead of the whole cast. Decide the opening default.
4. **Beat integration (important).** The new beat splitter must track presence per beat
   the same way it already tracks expressions/background. In `app.py`
   `_segment_beats()`, maintain a running `present` list and apply
   `enter_character`/`exit_character` as they're encountered, storing a `present`
   snapshot on each beat. Then `_render_beats_screen` should pass `beat["present"]`
   to `_vn_screen_html(present=…)` instead of `scene.present`. This makes characters
   pop in/out **on the right beat**, consistent with the expression timing already done.
5. **Mock.** Update `_make_mock_engine` in `app.py` to emit some `enter_character`/
   `exit_character` calls so the feature is demoable without the real model.
6. **Tests.** Add unit tests in `tests/test_unit.py` (handlers mutate `present`,
   cast-lock rejects unknown names) mirroring the existing `TestToolRouting` /
   `TestExpressionState` style.

**Watch out:** `SaveData` snapshots `present` already, so save/load is fine. Sprites
render from `present` in `_vn_screen_html`; an empty `present` = no sprites (valid).

### 2. Progressive disclosure of text (typewriter within a beat)

**Problem:** within a beat the full text appears instantly. Want it to type out
progressively so you read at a natural pace.

**Recommended approach — client-side, no server round-trips:**

- Add a CSS/JS **typewriter** effect to the `.vn-text` element in `ONLOAD_JS` /
  `CSS` (`app.py`). When a new beat renders, reveal its text character-by-character
  (e.g. wrap chars in spans or step a substring on a timer).
- **Two-stage click (standard VN):** if text is still typing, the first
  click/Space **completes** the current beat's text instantly; a click/Space on a
  fully-typed beat **advances** to the next beat (the existing `#advance-btn` path).
  The `data-more="1"` flag already gates advancing; add an "is-typing" flag so the
  first interaction just finishes typing.
- Keep it **purely frontend** — do NOT stream from the server (we deliberately chose
  "read-at-pace only", not LLM streaming). The full beat text is already in the DOM;
  the typewriter just controls reveal speed.
- Speed should be configurable-ish (a constant is fine, e.g. ~25–40 ms/char) and
  respect `prefers-reduced-motion` if easy.

**Note:** this is the per-beat version of the "streaming typewriter" option discussed
earlier. It does NOT reduce the real model's ~15s first-paint wait — that would need
the separate LLM-streaming layer (SSE + live `[TOOL:]` stripping), which is **not done**
and was explicitly deferred. The beat structure is already compatible with adding it later.

---

## Optional / deferred

- **LLM streaming (#1 from earlier):** SSE `stream:true` from llama-server + a live
  tool-call filter that suppresses output inside `[TOOL: …]` spans. Deferred. Would cut
  the first-paint wait on the real model.
- **ComfyUI generation path** (`cast_pipeline._generate_*`) and **Kokoro voice** were
  never live-verified — both servers were unreachable this session. Verify if/when
  ComfyUI (`:8188`) and Kokoro (`:11436`) are running.
- The mock's deterministic branching makes manual UI testing repetitive; consider a
  richer mock that varies branches and exercises enter/exit/expression tools.
