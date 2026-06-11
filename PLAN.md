# Implementation Plan — Player Name, Dialogue Box Text, Fixed UI Width

## 1. Player Name Input + Narrator Prompts

### Goal
Let the user input their name on the title screen. The name is stored in session state, referenced in the system prompt, and serialized in saves. The mock engine and real LLM both treat the player as an implicit character (the protagonist the cast addresses).

### Files touched
`vn_contracts.py`, `vn_prompt.py`, `vn_engine.py`, `app.py`

### 1.1 Data contract — `vn_contracts.py`
- Add `player_name: str = "Player"` to `SceneState` (line ~72).
- Add `player_name: str = "Player"` to `SaveData` (line ~144) and wire through `from_scene` / `apply_to`.

### 1.2 Prompt — `vn_prompt.py`
- Add a `{player_name}` placeholder to `VN_SYSTEM_PROMPT` (line ~10) and `VN_SYSTEM_PROMPT_SHORT` (line ~116).
  Insert a new section after the cast list:
  ```
  ## THE PLAYER

  The player is named {player_name}. They are the protagonist — the characters speak to them directly, react to their choices, and form relationships with them. The player is not a sprite; they are the first-person perspective of the story. When characters address the player, use direct address or "you" naturally. Do not give the player a physical description unless the user explicitly provides one.
  ```
- `build_full_prompt` (line ~212) now accepts `player_name: str`. Pass it to the format call.
- `build_state_banner` (line ~177) append a line: `Player: {player_name}` so the model sees it each turn.

### 1.3 Engine — `vn_engine.py`
- `VNEngine.run_turn` (line ~565) already calls `build_full_prompt(self.scene, self.cast, player_choice_text)`. Change that call to pass `self.scene.player_name` as the third positional / named arg.
- No other engine changes needed — the cast-lock logic already ignores bracketed names that are not in the cast; the player name is not a sprite so it must NOT match the cast-lock regex. Verify that `_check_cast_lock` only matches names that are in `cast.keys()`. (It does.)

### 1.4 UI — `app.py`

#### Title screen HTML
- In `_title_screen_html` (line ~1586), insert a name input row between the cast-choice toggle and the Begin button:
  ```html
  <div class="vn-title-name-row">
      <input type="text" class="vn-title-name-input" placeholder="Your name…" maxlength="24" value="Player">
  </div>
  ```
- Add CSS for `.vn-title-name-row` and `.vn-title-name-input` in the CSS block (near the title screen CSS, line ~1031):
  ```css
  .vn-title-name-row { margin-top: 4px; }
  .vn-title-name-input {
      background: rgba(255,255,255,0.04);
      border: 1px solid var(--border-mid);
      color: var(--text-primary);
      font-family: var(--font-ui);
      font-size: 0.85rem;
      letter-spacing: 0.06em;
      padding: 8px 14px;
      border-radius: 3px;
      text-align: center;
      width: 220px;
      transition: border-color 0.15s;
  }
  .vn-title-name-input:focus {
      border-color: var(--border-accent);
      outline: none;
      box-shadow: 0 0 0 2px var(--border-glow);
  }
  .vn-title-name-input::placeholder { color: var(--text-muted); }
  ```

#### Hidden bridge
- Add a hidden Gradio Textbox in `create_ui` (near the other bridges, line ~2557):
  ```python
  player_name_box = gr.Textbox(value="Player", visible="hidden",
                               interactive=True, elem_id="player-name-box")
  ```

#### JS wiring
- In ONLOAD_JS, inside the `begin` click handler (line ~1487), read the name input and write it to the hidden box before clicking `#start-btn-hidden`:
  ```js
  const nameInput = document.querySelector('.vn-title-name-input');
  if (nameInput) setHiddenVal('#player-name-box', nameInput.value.trim() || 'Player');
  ```
- Add `setHiddenVal` to the shared helper block (line ~1429) if not already available.

#### Callback plumbing
- `on_start` signature (line ~2140) adds `player_name: str = "Player"` parameter.
- `start_btn.click` inputs (line ~2605) add `player_name_box` to the list: `[preset_box, casting_state, player_name_box] + card_fields + [fast_gen]`.
- `make_initial_scene` (line ~53) takes `player_name="Player"` and stores it on the returned `SceneState`.
- In `on_start` curated path (line ~2200), when building the scene:
  ```python
  scene = SceneState(
      cast=cast, present=[],
      background="classroom", mood="peaceful",
      turn=0, chapter="Prologue",
      player_name=player_name,
  )
  ```
- In `on_start` generated path (line ~2322), same `player_name` propagation.
- In `on_load` (line ~2410), `SaveData.apply_to` restores `player_name` (covered by contract change in 1.1).

#### Mock engine
- `_make_mock_engine` (line ~114) adds `player_name: str = "Player"` parameter.
- The mock handler strings should reference the player naturally. Replace hard-coded `"you"` with `{player_name}` where appropriate, or keep the generic "you" since the prompt will tell the model the player's name. For the mock engine specifically, inject a small player reference in the first-turn default branch:
  ```python
  f'[{yuki}]: "Hey {player_name}! Ready to head out?"\n\n'
  ```
- `build_engine` (line ~248) passes `scene.player_name` to `_make_mock_engine`.

### 1.5 Testing
- `pytest -m mock -v` should still pass after contract changes (if any test constructs `SceneState` without `player_name`, the default `"Player"` covers it).
- Manual smoke: start the app, change the name, click Begin, verify the first turn's mock text uses the name.

---

## 2. Dialogue Box Text — Centered, 80% Width, No Indent

### Goal
Remove the first-line indent (`text-indent: 1.6em`) and instead center the text block within the dialogue box, constraining it to ~80% of the box width.

### Files touched
`app.py` only (CSS)

### 2.1 CSS change
Locate `.vn-text` (line ~394):
```css
.vn-text {
  font-family: var(--font-dialogue);
  color: var(--text-primary);
  font-size: 1.15rem;
  line-height: 1.7;
  margin: 0 auto;           /* was: margin: 0 */
  text-align: center;       /* new */
  max-width: 80%;           /* new — 80% of .vn-dialogue */
  text-indent: 0;           /* remove the 1.6em indent */
}
```

Also verify `.vn-text.narrator` (line ~402) inherits the same layout. The narrator class only changes font-style and color, so it will inherit centering and 80% width automatically.

### 2.2 Rationale
- `margin: 0 auto` centers the text block horizontally inside `.vn-dialogue`.
- `max-width: 80%` keeps the text from touching the dialogue box edges, creating comfortable side padding without hardcoding pixel values.
- `text-align: center` centers each line. For multi-line narration this looks cleaner than ragged-left inside a narrow column.
- Removing `text-indent: 1.6em` eliminates the first-line indentation the user previously asked for and now wants gone.

### 2.3 Testing
- Run the app, click through the first turn, verify the text sits centered and does not hit the left/right edges of the dialogue box.

---

## 3. Fixed UI Width (~80% of Screen)

### Goal
Prevent the main Gradio container from shrinking when the Cast Setup group is hidden. The stage should always occupy roughly 80% of the viewport width, regardless of which child elements are visible.

### Files touched
`app.py` only (CSS)

### 3.1 Root cause
`.gradio-container` currently has `max-width: 1100px` (line ~298). When the Cast Setup group is hidden, the tallest content in the container disappears. The Gradio container may then collapse to the intrinsic width of the remaining narrow content (the canvas + free-text bar), because Gradio's default layout uses a flex/grid column that can shrink to fit its content if no explicit width is set.

### 3.2 CSS change
Replace the `.gradio-container` rule (line ~294-300):
```css
.gradio-container {
  background: var(--bg-void) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-ui) !important;
  width: 80% !important;           /* was: max-width: 1100px */
  max-width: 1400px !important;    /* safety cap on ultra-wide monitors */
  min-width: 720px !important;     /* prevent collapse on small viewports */
  margin: 0 auto !important;
}
```

### 3.3 Stage reinforcement
Also add an explicit width on the stage wrapper so it always fills the container:
```css
#stage {
  width: 100% !important;
}
```
(This is already mostly true because `gr.Column` expands, but the explicit rule makes it bulletproof.)

### 3.4 `.vn-screen` behavior
`.vn-screen` already has `width: 100%` (line ~305). With the container now fixed at 80% of viewport, the canvas will always be 80% of the screen width (minus the container's own padding, which is minimal). The `aspect-ratio: 16 / 9` rule preserves the visual proportions.

### 3.5 Testing
1. Open the app on the title screen. Resize the browser window. The canvas should stay at ~80% of window width.
2. Click Begin. The Cast Setup group hides. The canvas should NOT shrink — it should remain the same width.
3. Click Reset. The Cast Setup group reappears. Width should still be ~80%.
4. Test on a 1920px-wide monitor: expect ~1536px wide (80% of 1920). On a 1366px laptop: expect ~1093px wide. On a small 1024px window: capped at `min-width: 720px`.

---

## 4. Implementation Order

1. **CSS changes first** (items 2 & 3) — they are purely cosmetic and carry no risk. Run a smoke test.
2. **Contract changes** (`vn_contracts.py`) — add `player_name` to `SceneState` and `SaveData`.
3. **Prompt changes** (`vn_prompt.py`) — add player section to system prompt and pass the name through `build_full_prompt`.
4. **Engine changes** (`vn_engine.py`) — pass `player_name` into `build_full_prompt`.
5. **UI changes** (`app.py`) — title screen input, hidden bridge, JS wiring, callback propagation, mock engine update.
6. **Run tests** `pytest -m mock -v` to verify no regressions from contract changes.
7. **Manual smoke test** — verify name input, centered text, and fixed width across title → scene → reset.

---

## 5. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| `SceneState` default `player_name` might break tests that assert exact field count | `pytest -m mock -v` first; default value covers missing args |
| `max-width: 80%` on `.vn-text` might make very short single words look weird | `min-width: 60%` on `.vn-text` as fallback if needed |
| `width: 80%` on `.gradio-container` could overflow on mobile | `min-width: 720px` and `max-width: 1400px` keep it in bounds |
| Player name in system prompt might leak into cast-lock if the model puts `[PlayerName]:` as a speaker | `_check_cast_lock` only matches names in `cast.keys()`, so a non-cast player name is safely ignored |
| Mock engine references to `player_name` require f-string evaluation at runtime | Use closure capture; pass it into the handler as a default arg or format just before returning |
