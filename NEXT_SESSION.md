# Ars-Fabula VN — Session Handoff

_Last updated: 2026-06-11. Read this + `CLAUDE.md` before starting._

## ✅ FIXED 2026-06-11 — the two open regressions (root causes found live)

Both were diagnosed by running the app (mock mode) under a driven browser and
inspecting the live CSSOM/DOM. Fixes are on `rewire-model-gemma26b`.

### 1. Blank page / Cast Setup never appeared — FIXED
- **Root cause:** Gradio 6 injects every custom CSS rule **twice** — verbatim AND
  prefixed with `.gradio-container.gradio-container-<ver> .contain ` (specificity
  1 id + 3 classes). The prefixed copy of any `body.…`-keyed rule can never match
  (body isn't inside `.contain`), so the unprefixed
  `body.cast-setup-active #cast-setup { display:block !important }` (0,1,1,1) lost
  to the **prefixed** copy of `#cast-setup { display:none !important }` (0,1,3,0).
  Result: `#stage` hidden by the body rule, `#cast-setup` still hidden → blank.
- **Fix (`ui/styles.py`):** doubled-id selectors (`#cast-setup#cast-setup`,
  `#stage#stage`) on the body-keyed rules → specificity (0,2,…) beats any
  single-id prefixed rule. Verified live: Begin → Generated shows Cast Setup,
  ← Back returns to the title, ✨ Generate Cast proceeds to the casting beat.

### 2. Dark elements "dragging" across the screen during a turn — FIXED (two causes)
- **Cause A — Gradio's queue-progress overlay:** while a callback runs, Gradio
  covers each output component with `.wrap.full` + **`.eta-bar`** (inset:0, dark
  `--background-fill-secondary`, opacity .8, **grows across the component via a
  transform**) and dims the component to opacity .2 via `.pending`. Over the
  canvas, with a real LLM taking seconds, this is a dark sheet sliding across the
  whole UI, sprites included. **Fix:** `show_progress="hidden"` on every event
  wiring in `app.py`, plus belt-and-braces CSS killing `.eta-bar`/`.pending`/
  `.generating`/`.wrap.full` inside `#stage`.
- **Cause B — the overlay panels painting unclipped:** the closed Scene Log /
  Cast / Save panels are parked off-canvas at `translate(102%)` and relied on
  `.vn-screen { overflow:hidden }` to be invisible; during the canvas HTML swap
  the browser can paint them unclipped for a few frames (user screenshot: log +
  cast cards sliding outside the canvas). **Fix (`ui/styles.py`):** closed panels
  are now `visibility:hidden` (instant on open, delayed past the slide on close)
  — a closed panel is unpaintable regardless of clipping/compositor timing.
- Verified live: panels open/close via the tray, choices + free-text capsule
  submit and run turns, 74 non-live tests pass.
- **Follow-up — turn-pending indicator:** with the Gradio overlay suppressed,
  a real-LLM turn (seconds) had NO feedback, which read as "clicks don't
  work". Added `body.vn-waiting` (set by the choice/free-text JS bridge,
  cleared when the next render lands, 120s stuck-guard timeout): the choice
  column dims to 0.35 + locks (`pointer-events:none` — also prevents double
  submits) and a pulsing ✦ shows bottom-right of the canvas. The dim/lock
  rules need `!important` — both to beat the Gradio-prefixed base rules and
  the lingering fill-mode opacity of the `vnChoicesIn` animation. Verified
  against the real llama-server backend: dim+✦ during the wait, cleared on
  the next turn's render.

> NOTE: `.claude/launch.json` was added for browser-driven verification
> (`preview_start` → mock app). The port number in it is scratch state.

---

## Current state — what works (when rendering isn't blocked by #1 above)

```bash
ARS_FABULA_BACKEND=mock   python app.py        # no server
ARS_FABULA_BACKEND=server python app.py        # real llama-server on :8110
ARS_FABULA_REGEN=1        python app.py        # re-bake sprites (needed to see new framing)
```

- **`app.py` split into a `ui/` package** this session: `ui/styles.py` (CSS+JS), `ui/media.py` (data URIs), `ui/bootstrap.py` (constants/engine), `ui/screens.py` (HTML), `ui/beats.py` (beat segmentation/render), `ui/callbacks.py` (`make_callbacks`). `app.py` is now a thin entrypoint. Nothing imports from `app.py`. **74 tests pass** (`pytest -m "not live" -q`).
- **Cast:** `haruki` (boy) was **removed** from `PRESET_CHARACTERS` and replaced with **`marie`** (blonde ojou-sama). Default trio is `["yuki","marie","sakura"]`. Curated mode loader is defensive (`(load_curated_cast([k]) or [None])[0]`) so a missing key skips instead of crashing. (Curated on-disk art for `marie` does not exist → placeholders until baked. The `haruki` *curated* sprite folder + `tools/generate_assets.py` chibi entry were intentionally left alone.)
- **Fixed seed:** all presets + custom specs use `SEED` (default 42, `ARS_FABULA_SEED`). Retries still bump locally.
- **Style/expression tuning:** `STYLE_TAGS` default → light-novel/official-art set. `neutral` expression prompt is now `expressionless, calm…` (bare "neutral expression" read grumpy on Anima); base pose uses `calm expression`.
- **Face-anchored sprite framing** (`_frame_upper_body`): uses the face-gate mask (`_facegate_mask.png`) to scale each sprite to a constant face size and composite feet-down onto a shared-size canvas, so **taller figures (smaller face) render taller**. Tunables: `ARS_FABULA_FACE_TARGET` (0.135), `ARS_FABULA_SPRITE_CANVAS` (1.4). Falls back to the old alpha-crown + `UPPER_BODY_FRAC` crop when no mask. **Needs `ARS_FABULA_REGEN=1` to see** — and changes framing from waist-up to full-figure-on-canvas, so on-canvas zoom may need tuning.
- **UI redesign (this session):** fixed-height (180px), ~88%-wide, flat-fill dialogue box; choices as centered rounded **capsules**; **free-text input moved INTO the canvas** as the 4th item of the choice column (`.vn-free-capsule`, wired to the hidden `#free-input` bridge); frame sized `min(90vw,1700px,156vh)` 16:9; **character dialogue + choices in MS Gothic** (`ARS_FABULA_DIALOGUE_FONT`), **narration in Georgia serif**. (See issues above — the in-canvas choice/setup work is where the two regressions live.)

### Key gotchas learned (don't re-discover these)

1. **Gradio 6 runs `launch(js=…)` as a FUNCTION BODY**, not an eval'd expression. Plain statements only — an `()=>{}` wrapper is constructed and discarded and the listener never attaches. A thrown error early in `ONLOAD_JS` stops every later `addEventListener` from running.
1b. **Gradio 6 duplicates every `launch(css=…)` rule with a scoped prefix** `.gradio-container.gradio-container-<ver> .contain ` (adds 1 id + 3 classes of specificity). Any selector keyed off `body.<class>` only exists unprefixed, so it must out-specify the PREFIXED copies of competing rules — use a doubled id (`#x#x`) when a body-state rule has to override a base rule on the same element.
1c. **Gradio's queue progress overlay** (`.pending` opacity-dim + `.eta-bar` dark transform-growing sheet) plays over output components on every event unless the listener passes `show_progress="hidden"`. Use it on all wirings whose output is the canvas.
2. **`visible="hidden"`** (not `False`) keeps bridge controls in the DOM. Setting a `display:none` input's value + dispatching `input`, and clicking a `display:none` button, both still reach the backend.
3. The in-canvas free-text + choice buttons are plain HTML wired to hidden Gradio controls via `setBoxValue`/`clickHidden` in `ui/styles.py`. The Gradio `#free-input-row` is now display:none and used only as the bridge carrier.
4. When relaunching the app in a shell, use `pkill -f "[a]pp.py"` (a plain `app.py` pattern kills the launching shell too).

## Deferred (still valid)

- **LLM streaming:** SSE `stream:true` + live tool-call filter to cut first-paint wait.
- **ComfyUI / Kokoro** generation + voice paths are still only partially live-verified.
- **Per-character sprite-entrance animation:** the per-beat sprite re-animation was removed to fight issue #2; a "animate only newly-entered characters" version (server tags new vs continuing sprites) was offered but not built.
- **(NEW 2026-06-11) Cast setup refactored into canvas HTML:** The cast-setup step (Begin → Generated) was converted from raw Gradio widgets (gr.Group with `⚙✨⚡` emojis, gr.Textbox, gr.Checkbox) into styled VN HTML inside the canvas — matching the title screen's visual language. The card fields bridge to hidden Gradio textboxes via JS (same `setHiddenVal` pattern as the name input). `#cast-setup` Gradio group removed; `.vn-cast-setup` CSS lives in the canvas. 74 tests still pass.
