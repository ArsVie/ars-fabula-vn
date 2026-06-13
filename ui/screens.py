"""HTML builders for the VN canvas — title, casting, review, cast panel,
and the main scene screen.
"""
from __future__ import annotations
import html as _html

from vn_contracts import CastMember, Choice
from vn_engine import list_saves

from .media import data_uri, thumb_data_uri


def _slot_rows_html(selectable_empty: bool = True) -> str:
    """Rich save-slot rows: name, chapter · turn, cast and timestamp.

    Used by both the in-game save panel (all slots selectable) and the
    main-menu load list (empty slots rendered but not clickable)."""
    rows = []
    for meta in list_saves():
        i = meta["slot"]
        if meta.get("empty"):
            cls = "vn-slot-btn empty" + ("" if selectable_empty else " disabled")
            rows.append(
                f'<button class="{cls}" data-slot="{i}" data-empty="1">'
                f'<span class="vn-slot-name">Slot {i + 1} — Empty</span>'
                f'</button>'
            )
            continue
        name = _html.escape(meta["name"] or f"Slot {i + 1}")
        cast = _html.escape(", ".join(meta["cast_names"]) or "—")
        when = _html.escape(meta["saved_at"] or "")
        chapter = _html.escape(meta["chapter"] or "")
        rows.append(
            f'<button class="vn-slot-btn" data-slot="{i}" data-empty="0">'
            f'<span class="vn-slot-name">{name}</span>'
            f'<span class="vn-slot-meta">{chapter} · Turn {meta["turn"]}</span>'
            f'<span class="vn-slot-cast">{cast}</span>'
            f'<span class="vn-slot-time">{when}</span>'
            f'</button>'
        )
    return "\n".join(rows)


def _title_screen_html() -> str:
    """The title card + cast-setup step + load list, rendered inside the canvas.

    Three sections, toggled by body classes:
      - `.vn-title-content` — the title screen (visible by default).
      - `.vn-cast-setup`    — the cast-setup step (body.cast-setup-active).
      - `.vn-menu-load`     — the load-save list (body.menu-load-active).

    ONLOAD_JS bridges the HTML inputs to hidden Gradio controls so the
    card fields reach the backend when Generate Cast is clicked.
    """
    has_saves = any(not m.get("empty") for m in list_saves())
    load_btn = ('<button class="vn-title-load">Load</button>' if has_saves
                else '<button class="vn-title-load disabled" disabled>Load</button>')
    return f"""
    <div class="vn-screen vn-title-screen">
        <div class="vn-title-content">
            <div class="vn-title-main">ARS-FABULA</div>
            <div class="vn-title-sub">A Slice-of-Life Visual Novel</div>
            <div class="vn-title-cast-choice">
                <button class="vn-title-opt active" data-preset="generated">Generated Cast</button>
                <button class="vn-title-opt" data-preset="curated">Curated Cast</button>
            </div>
            <div class="vn-title-name-row">
                <input type="text" class="vn-title-name-input" placeholder="Your name…" maxlength="24" value="Player">
            </div>
            <div class="vn-title-actions">
                <button class="vn-title-begin">Start</button>
                {load_btn}
            </div>
        </div>
        <div class="vn-menu-load">
            <div class="vn-ml-title">Load Save</div>
            <div class="vn-slot-list vn-ml-list">{_slot_rows_html(selectable_empty=False)}</div>
            <button class="vn-ml-back">Back</button>
        </div>
        <div class="vn-cast-setup">
            <div class="vn-cs-header">
                <div class="vn-cs-title">Cast Setup</div>
                <div class="vn-cs-desc">
                    <p>Leave every card blank to generate the default cast
                    (Yuki, Marie, Sakura). Otherwise fill <strong>Name + Concept</strong>
                    and the story model designs the character (e.g. concept
                    <em>childhood friend</em> becomes a warm, brown-haired genki design).</p>
                    <p><strong>Advanced tags</strong> override the designer with your own
                    Danbooru appearance tags (start with <code>1girl</code>/<code>1boy</code>;
                    hair, eyes, clothing, accessories only). Only the framing essentials
                    are locked on — body attitude is yours (add stance tags like
                    <code>hand on hip</code>, <code>arms crossed</code>).
                    Editing a design auto-invalidates that character's sprite cache.</p>
                </div>
            </div>
            <div class="vn-cs-cards">
                <div class="vn-cs-card" data-cs-idx="0">
                    <div class="vn-cs-card-row">
                        <div class="vn-cs-field vn-cs-field-name">
                            <label>Name</label>
                            <input class="vn-cs-input vn-cs-name" data-cs-field="name" data-cs-idx="0"
                                   placeholder="e.g. Mio" maxlength="24">
                        </div>
                        <div class="vn-cs-field vn-cs-field-concept">
                            <label>Concept</label>
                            <input class="vn-cs-input vn-cs-concept" data-cs-field="concept" data-cs-idx="0"
                                   placeholder="e.g. shy bookworm childhood friend" maxlength="512">
                        </div>
                    </div>
                    <div class="vn-cs-tags-row">
                        <input class="vn-cs-input vn-cs-tags-input" data-cs-field="tags" data-cs-idx="0"
                               placeholder="Advanced tags: 1girl, silver hair, long hair, blue eyes, ..." maxlength="256">
                    </div>
                </div>
                <div class="vn-cs-card" data-cs-idx="1">
                    <div class="vn-cs-card-row">
                        <div class="vn-cs-field vn-cs-field-name">
                            <label>Name</label>
                            <input class="vn-cs-input vn-cs-name" data-cs-field="name" data-cs-idx="1"
                                   placeholder="e.g. Rin" maxlength="24">
                        </div>
                        <div class="vn-cs-field vn-cs-field-concept">
                            <label>Concept</label>
                            <input class="vn-cs-input vn-cs-concept" data-cs-field="concept" data-cs-idx="1"
                                   placeholder="e.g. tsundere rival" maxlength="512">
                        </div>
                    </div>
                    <div class="vn-cs-tags-row">
                        <input class="vn-cs-input vn-cs-tags-input" data-cs-field="tags" data-cs-idx="1"
                               placeholder="Advanced tags: 1girl, silver hair, long hair, blue eyes, ..." maxlength="256">
                    </div>
                </div>
                <div class="vn-cs-card" data-cs-idx="2">
                    <div class="vn-cs-card-row">
                        <div class="vn-cs-field vn-cs-field-name">
                            <label>Name</label>
                            <input class="vn-cs-input vn-cs-name" data-cs-field="name" data-cs-idx="2"
                                   placeholder="e.g. Aoi" maxlength="24">
                        </div>
                        <div class="vn-cs-field vn-cs-field-concept">
                            <label>Concept</label>
                            <input class="vn-cs-input vn-cs-concept" data-cs-field="concept" data-cs-idx="2"
                                   placeholder="e.g. mysterious senpai" maxlength="512">
                        </div>
                    </div>
                    <div class="vn-cs-tags-row">
                        <input class="vn-cs-input vn-cs-tags-input" data-cs-field="tags" data-cs-idx="2"
                               placeholder="Advanced tags: 1girl, silver hair, long hair, blue eyes, ..." maxlength="256">
                    </div>
                </div>
            </div>
            <label class="vn-cs-fast">
                <input type="checkbox" class="vn-cs-fast-cb">
                <span>Fast generation (lower resolution — quicker bake, softer detail)</span>
            </label>
            <div class="vn-cs-actions">
                <button class="vn-cs-btn vn-cs-back" data-action="cast-setup-back">Back</button>
                <button class="vn-cs-btn vn-cs-go" data-action="cast-setup-go">Generate Cast</button>
            </div>
        </div>
    </div>
    """


def _cast_panel_html_inner(cast: dict[str, CastMember]) -> str:
    """Inner HTML for the in-canvas cast overlay (thumb + identity)."""
    if not cast:
        return '<div class="cast-panel-empty">No cast — start a story.</div>'
    parts = []
    for name, member in cast.items():
        char = member.character
        thumb_uri = thumb_data_uri(member.get_sprite("neutral"), height=280)
        thumb = (f'<img class="cast-panel-thumb" src="{thumb_uri}" alt="{name}">'
                 if thumb_uri else '<div class="cast-panel-thumb"></div>')
        parts.append(
            f'<div class="cast-panel-card">{thumb}'
            f'<div class="cast-panel-info">'
            f'<div class="cast-panel-name">{char.name}</div>'
            f'<div class="cast-panel-role">{char.role}</div>'
            f'<div class="cast-panel-personality">{char.personality}</div>'
            f'</div></div>'
        )
    return "\n".join(parts)


_ICON_TRAY_HTML = """
<div class="vn-icon-tray">
  <button class="vn-tray-btn" data-action="log">Log</button>
  <button class="vn-tray-btn" data-action="cast">Cast</button>
  <button class="vn-tray-btn" data-action="save">Save</button>
  <button class="vn-tray-btn" data-action="reset"><span class="vn-tray-normal">Reset</span><span
    class="vn-tray-confirm-row"><span class="vn-reset-confirm">Confirm</span>&nbsp;/&nbsp;<span
    class="vn-reset-cancel">Cancel</span></span></button>
</div>
"""





def _loading_screen_html(message: str, sub: str = "") -> str:
    """A full-canvas generation-wait screen: the pulsing ✦ plus a message.

    The single waiting vocabulary of the app — every phase that blocks on
    generation (model turn, sprite bake, VRAM swap) shows this same pulse,
    matching the in-scene ✦ used for streaming/turn waits.
    """
    sub_html = f'<div class="vn-loading-sub">{_html.escape(sub)}</div>' if sub else ""
    return f"""
    <div class="vn-screen">
      <div class="vn-loading">
        <div class="vn-loading-star">✦</div>
        <div class="vn-loading-msg">{_html.escape(message)}</div>
        {sub_html}
      </div>
    </div>
    """


def _review_screen_html(key: str, preset: dict, base_path: str,
                         char_num: int, total: int) -> str:
    """Per-character review screen during interactive casting.

    Shows the generated base image alongside the full prompt text and the
    character description (both editable) and Confirm/Redo buttons. The JS
    bridge writes the textarea values to the hidden #casting-prompt-box /
    #casting-desc-box before clicking the hidden confirm/redo Gradio buttons.
    """
    name = preset.get("name", key.capitalize())
    prompt = preset.get("identity_prompt", "")
    img_uri = data_uri(base_path) if base_path else ""

    img_html = (f'<img src="{img_uri}" alt="{name} base">'
                if img_uri
                else '<div class="review-no-image">Generation failed — try Redo</div>')

    # The designed character: role (display) + personality (editable — it is
    # re-sent to the story model every turn and shapes the character's voice).
    role = _html.escape(preset.get("role", "") or "")
    personality = preset.get("personality", "") or ""
    role_html = f'<div class="review-desc"><span class="review-role">{role}</span></div>' if role else ""

    return f"""
    <div class="vn-screen vn-casting-review">
      <div class="casting-review-title">Cast Review · {char_num}/{total}</div>
      <div class="review-layout">
        <div class="review-image">{img_html}</div>
        <div class="review-panel">
          <div class="review-name">{name}</div>
          {role_html}
          <div class="review-prompt-label">Personality — their voice in the story</div>
          <textarea class="review-prompt review-desc-edit" id="review-desc-text" rows="3">{_html.escape(personality)}</textarea>
          <div class="review-prompt-label">Prompt — edit and Redo to regenerate</div>
          <textarea class="review-prompt" id="review-prompt-text" rows="6">{_html.escape(prompt)}</textarea>
          <div class="review-actions">
            <button class="review-btn confirm" data-action="casting-confirm">&#10003; Confirm</button>
            <button class="review-btn redo" data-action="casting-redo">&#8635; Redo</button>
          </div>
        </div>
      </div>
    </div>
    """


def _log_html(log_text: str) -> str:
    """Escape the session register; player-answer lines (▸-prefixed, written
    by the turn callbacks) get their own accent styling."""
    lines = []
    for line in (log_text or "").split("\n"):
        esc = _html.escape(line)
        if line.startswith("▸ "):
            esc = f'<span class="vn-log-answer">{esc}</span>'
        lines.append(esc)
    return "\n".join(lines)


def _panels_html(log_text: str, cast_html: str) -> str:
    """The three overlay panels, rendered inside .vn-screen."""
    slots = _slot_rows_html()
    return f"""
    <div class="vn-panel vn-panel-log">
        <button class="vn-panel-close">✕</button>
        <div class="vn-panel-title">Scene Log</div>
        <div class="vn-log-text">{_log_html(log_text)}</div>
    </div>
    <div class="vn-panel vn-panel-cast">
        <button class="vn-panel-close">✕</button>
        <div class="vn-panel-title">Cast</div>
        {cast_html}
    </div>
    <div class="vn-panel vn-panel-save">
        <button class="vn-panel-close">✕</button>
        <div class="vn-panel-title">Save / Load</div>
        <div class="vn-slot-list">{slots}</div>
        <input class="vn-save-name" type="text" maxlength="48"
               placeholder="Save name (optional)…">
        <div class="vn-save-actions">
            <button class="vn-save-action" data-which="save">Save</button>
            <button class="vn-save-action" data-which="load">Load</button>
        </div>
        <div class="vn-save-status"></div>
    </div>
    """


def _vn_screen_html(bg_path: str, cast: dict[str, CastMember],
                    present: list[str], expressions: dict[str, str],
                    dialogue_speaker: str = "", dialogue_text: str = "",
                    choices: list[Choice] | None = None,
                    narration: str = "",
                    prev_bg_path: str = "",
                    more: bool = False,
                    waiting: bool = False,
                    log_text: str = "",
                    cast_html: str = "") -> str:
    """Build the VN screen HTML: crossfading bg layers, sprites, dialogue, choices.

    `more=True` marks the screen as having further beats to read this turn:
    it adds a ▼ continue hint and a `data-more="1"` flag the JS click/keyboard
    hook uses to advance the reader one beat at a time.

    `waiting=True` means the reader has caught up to the last beat the model has
    produced SO FAR but the turn is still streaming: show the ✦ loading pulse
    (the same indicator used while a submitted turn is processing) and withhold
    the choices / free-text bar until the model stops writing. It still flags
    `data-more="1"` so a click re-checks for newly streamed beats.
    """
    bg_layers = ""
    prev_uri = data_uri(prev_bg_path) if prev_bg_path and prev_bg_path != bg_path else ""
    curr_uri = data_uri(bg_path)
    if prev_uri:
        bg_layers += f'<img class="vn-bg old" src="{prev_uri}" alt="">'
    if curr_uri:
        bg_layers += f'<img class="vn-bg new" src="{curr_uri}" alt="">'

    # Sprites — the speaking character is .active, the rest .inactive (dimmed).
    # During pure narration (no speaker) everyone stays normal.
    sprite_html = ""
    for name in present:
        member = cast.get(name)
        if not member:
            continue
        expr = expressions.get(name, "neutral")
        sprite_uri = data_uri(member.get_sprite(expr))
        if sprite_uri:
            state = ""
            if dialogue_speaker:
                state = " active" if name == dialogue_speaker else " inactive"
            sprite_html += (f'<img class="vn-sprite{state}" data-expr="{name}:{expr}" '
                            f'src="{sprite_uri}" alt="{name} ({expr})">')

    # Dialogue — name plate, then text (HUD removed)
    dialogue_html = ""
    if dialogue_text or narration:
        dialogue_html += '<div class="vn-dialogue-inner">'
        if dialogue_text:
            if dialogue_speaker:
                dialogue_html += f'<div class="vn-name-plate">{dialogue_speaker}</div>'
            dialogue_html += f'<p class="vn-text{" narrator" if not dialogue_speaker else ""}">{dialogue_text}</p>'
        elif narration:
            dialogue_html += f'<p class="vn-text narrator">{narration}</p>'
        dialogue_html += '</div>'
    if waiting:
        # Caught up to the streaming front — pulse the ✦ loader INSIDE the
        # text box (where the ▼ continue hint normally sits, and where the
        # post-choice wait pulse renders too — body.vn-waiting
        # .vn-dialogue::after). One indicator, one place.
        dialogue_html += '<div class="vn-streaming-wait">✦</div>'
    elif more:
        dialogue_html += '<div class="vn-continue">▼</div>'
    dialogue_html += _ICON_TRAY_HTML

    # Interaction cluster — shown on the LAST beat ONLY once the turn has
    # finished streaming (not while there are more beats to read, and not while
    # the model is still writing): the offered choices (if any) plus a free-text
    # capsule as the final item, so the whole set reads as one centered column.
    # It lives INSIDE the canvas, so it scales with the frame (not the page) and
    # centers with the choices; the typing-done CSS gate hides it until the line
    # ends.
    choices_html = ""
    if not more and not waiting:
        parts = ['<div class="vn-choices">']
        for c in (choices or []):
            safe_text = c.text.replace('"', '&quot;')
            parts.append(
                f'<button class="vn-choice-btn" '
                f'data-choice-id="{c.id}" data-choice-text="{safe_text}">'
                f'{c.text}</button>'
            )
        # Free-text capsule — always the last item, so the player is never
        # locked to the offered choices. Wired to the hidden Gradio bridge by JS.
        parts.append(
            '<div class="vn-free-capsule">'
            '<input class="vn-free-text" type="text" autocomplete="off" '
            'placeholder="Your choice…">'
            '<button class="vn-free-go" data-action="free-send" '
            'aria-label="Send">➤</button>'
            '</div>'
        )
        parts.append('</div>')
        choices_html = "\n".join(parts)

    # `data-more="1"` also while waiting so a click re-checks for new beats; the
    # bar-reveal JS keys off data-more!=="1", so the free bar stays hidden.
    more_flag = "1" if (more or waiting) else "0"
    return f"""
    <div class="vn-screen{' advancing' if more else ''}" data-more="{more_flag}" data-waiting="{'1' if waiting else '0'}" data-choices="{'1' if choices_html else '0'}" data-typing="pending">
        {bg_layers}
        <div class="vn-chars">{sprite_html}</div>
        {choices_html}
        <div class="vn-dialogue{' has-speaker' if dialogue_speaker else ''}">{dialogue_html}</div>
        {_panels_html(log_text, cast_html)}
    </div>"""


def _casting_screen_html(char_keys: list[str],
                         resolved: dict[str, CastMember],
                         message: str = "Assembling your cast…",
                         sub: str = "") -> str:
    """The casting-beat screen: portraits resolve one by one.

    `sub` adds a secondary status line under the portraits (same voice as the
    ✦ loading screen's sub-text). It lets this ONE screen carry the whole
    cast→opening-scene wait — cards stay visible while the ✦ pulses and the
    sub-line narrates — instead of swapping to a separate loading screen.
    """
    portraits = []
    for key in char_keys:
        # Preset keys are lowercase slugs; custom names arrive display-ready
        name = key.capitalize() if key.islower() else key
        member = resolved.get(name)
        if member:
            uri = data_uri(member.get_sprite("neutral"))
            inner = (f'<img src="{uri}" alt="{name}">' if uri
                     else f'<div class="pname">{name}</div>')
            portraits.append(
                f'<div class="portrait-wrap">'
                f'<div class="vn-portrait">{inner}</div>'
                f'<div class="pname">{name}</div></div>'
            )
        else:
            portraits.append(
                f'<div class="portrait-wrap">'
                f'<div class="vn-portrait pending">…</div>'
                f'<div class="pname pending">?</div></div>'
            )

    sub_html = f'<div class="vn-loading-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="vn-screen">
        <div class="vn-casting">
            <div class="casting-title">✦ {message}</div>
            <div class="portraits">{''.join(portraits)}</div>
            {sub_html}
        </div>
    </div>
    """


def _cast_panel_html(cast: dict[str, CastMember]) -> str:
    """Build HTML for the cast info panel."""
    if not cast:
        return '<div class="cast-panel-empty">No cast yet — start a story.</div>'
    parts = ['<div style="padding:8px;">']
    for name, member in cast.items():
        char = member.character
        parts.append(
            f'<div class="cast-card">'
            f'<div class="name">{char.name}</div>'
            f'<div class="role">{char.role} · ♪ {char.voice_id}</div>'
            f'<div class="personality">{char.personality}</div>'
            f'<div class="cast-expressions">'
            f'Expressions: {", ".join(member.expression_map.keys())}'
            f'</div></div>'
        )
    parts.append('</div>')
    return "\n".join(parts)
