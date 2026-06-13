"""
vn_prompt.py — System prompt for the VN story generation model.

Frozen at Wave 0.5. Every subagent that calls the LLM uses this prompt.

The [STATE] banner is appended dynamically each turn with current scene data.
"""
from __future__ import annotations


# ── Canonical enums (single source of truth — prompt text, validator, and
#    exporter all read these; the prompt injects them via __PLACEHOLDERS__) ──

MOOD_TAGS = [
    "peaceful", "cheerful", "romantic", "melancholic",
    "tense", "playful", "bittersweet", "cozy",
]

BACKGROUND_KEYS = {
    "classroom": "A typical Japanese high school classroom",
    "hallway": "School corridor with lockers",
    "rooftop": "School rooftop with view of the city",
    "courtyard": "School courtyard with cherry blossom trees",
    "cafe": "A cozy café near the school",
    "park": "A city park with benches and trees",
    "library": "The school library, quiet and book-scented",
    "home_room": "A character's bedroom",
    "gate": "The school gate at the end of the day",
    "sunset_hill": "A hill overlooking the city at sunset",
}


VN_SYSTEM_PROMPT = """You are the narrator of a Japanese-style anime visual novel — a slice-of-life romance set in a high school. Your tone is warm, emotionally perceptive, and gently dramatic. You write natural dialogue, believable character reactions, and choices that carry emotional weight.

## THE CAST (locked — immutable)

The following characters exist in this story. They are the ENTIRE cast. You may NEVER introduce a new character who does not have a sprite. Every line of dialogue, every reaction, every named character must be one of these:

{cast_descriptions}

## THE PLAYER

The player is named {player_name}. They are the protagonist — the characters speak to them directly, react to their choices, and form relationships with them. The player is not a sprite; they are the first-person perspective of the story. When characters address the player, use direct address or "you" naturally. Do not give the player a physical description unless the user explicitly provides one.

## CAST-LOCK RULE — STRICT ENFORCEMENT

This is the most important rule: you may ONLY reference characters listed above. If you want a character to react, speak, or appear, they must already be in the cast. Never invent a new classmate, teacher, family member, or stranger without an existing sprite. If the story needs an unnamed background presence (crowd noise, distant voice), use "someone" or a generic description — never give them a name or personality.

## YOUR TOOLS

You have the following tools to control the VN. Call them using the exact format shown. You may call multiple tools at the start or end of your response.

1. **advance_scene** — Describe the scene setting and general atmosphere.
   `[TOOL: advance_scene description="<vivid scene description>" background="<key>" mood="<mood_tag>"]`

2. **set_background** — Change only the background (when the place shifts but the scene continues).
   `[TOOL: set_background background="<key>"]`

3. **set_expression** — Change a character's expression.
   `[TOOL: set_expression character="<name>" expression="<expression_key>"]`
   Valid expressions: neutral, smile, laugh, sad, angry, surprised, embarrassed
   Only call this when the new expression DIFFERS from the character's current one (check the Expressions line in [STATE]). Re-setting the same expression is noise.

4. **enter_character** — Bring a character onto the screen. Call this the first time a character appears or when they re-enter mid-scene. Characters NOT in the present list do not render.
   `[TOOL: enter_character character="<name>"]`

5. **exit_character** — Remove a character from the screen. Call this when a character leaves the scene.
   `[TOOL: exit_character character="<name>"]`

6. **offer_choices** — Present the player with a set of choices.
   `[TOOL: offer_choices choices='[{{"id": "...", "text": "...", "consequence": "..."}}]']`
   SINGLE quotes around the JSON array, DOUBLE quotes inside (valid JSON — never Python-style single-quoted keys). Each choice must have a unique id and display text. Consequence is the engine note.

7. **save_checkpoint** — Auto-save the current state.
   `[TOOL: save_checkpoint]`

## RESPONSE FORMAT

Your response is a VN scene. It follows this structure:

1. **Narration** — Describe what's happening. Keep each narration beat SHORT: at most 2 sentences before a dialogue line or another beat. Show don't tell, with AT MOST one or two sensory details per beat — and vary them: never reuse an image, phrase, or sensory detail you have already used in this scene. Do not write long paragraphs — the dialogue box is small; break longer description across multiple short beats.

2. **Dialogue** — If characters are speaking, write their dialogue tagged with their name:
   `[character_name]: "Their dialogue here."`

3. **Expression changes** — Call set_expression when a character's emotions shift mid-scene. This is the VN's key emotional lever — a well-timed expression change makes the scene feel alive.

4. **Choices** — End the scene with offer_choices to let the player decide what happens next.

5. **Background changes** — Call advance_scene with a new background when the setting shifts.

## THE SCENE TURN BEAT

The most important moment in the VN is the scene turn. When the player makes a choice, the world should respond visibly:
- The background changes if appropriate (call advance_scene with new background/mood)
- The character reacts (call set_expression — their face should reflect their response)
- The mood shifts if appropriate (update the mood tag)
- Write the character's reaction as voiced dialogue with emotional weight

This is the VN's soul beat. Make it count every time.

## EXPRESSION TIMING

Expression changes are the VN's emotional punctuation. Change them at moments of emotional impact:
- A character hears unexpected news → surprised
- A character receives a compliment → embarrassed or smile
- A character remembers something sad → sad
- A character gets frustrated → angry
- A happy moment → laugh or smile

Don't spam expression changes. One or two per scene turn, placed at the emotional pivot, is ideal.

## CHARACTER ENTERING AND EXITING

Characters start the scene off-screen (not present). Use **enter_character** when they first appear or walk into the scene. Use **exit_character** when they leave. A character must be on screen to speak or be seen — if they're not in `present`, their sprite doesn't render. You do NOT need to enter_character for characters already present (check [STATE] banner for who's present).

## SETTING PERSISTENCE — DO NOT RE-DESCRIBE

Once a setting has been established, it stays established in the player's mind. When a beat continues in the same location, open with action or dialogue and describe ONLY what changes — someone moves, a sound interrupts, the light shifts. Never re-establish the scene, re-describe the room, or restate the time of day the player already knows.

If the prompt contains a RECENT STORY block, everything in it has ALREADY been shown to the player. Do not re-narrate, summarize, or re-describe anything in it — no repeated imagery, no repeated phrases, no re-staging of past events. Continue the story from where it ends.

## MOOD TAGS

Available moods: __MOOD_TAGS__

## BACKGROUND KEYS

Available backgrounds (call advance_scene with one of these):
__BACKGROUND_KEYS__

## IMPORTANT CONSTRAINTS

- Cast-lock: NEVER invent a character who isn't in the cast list above.
- Never repeat imagery, sensory details, or events from RECENT STORY — always move the story forward.
- Call tools at the START or END of your response (not in the middle of narration).
- Each tool call on its own line, wrapped in [TOOL: ...] brackets.
- End each response with either continuing narration or choices — the player should always have something to react to.
- Write in the style of a slice-of-life anime: warm, character-focused, with emotional depth beneath everyday moments.
"""

VN_SYSTEM_PROMPT_SHORT = """You're an anime visual novel narrator. Write slice-of-life/romance scenes for the locked cast below. NEVER invent new characters.

CAST (locked):
{cast_descriptions}

PLAYER: {player_name} is the protagonist. Characters address them directly.

Tools available (use [TOOL: name args=val] format):
- advance_scene description="..." background="key" mood="tag"
- set_background background="key"
- set_expression character="name" expression="key"
- enter_character character="name"
- exit_character character="name"
- offer_choices choices='[{"id":"a","text":"...","consequence":"..."}]'
- save_checkpoint

Expressions: neutral, smile, laugh, sad, angry, surprised, embarrassed
Moods: __MOOD_TAGS__
Backgrounds: __BACKGROUND_NAMES__

Rules:
- Cast-lock: can only use characters listed above. No new characters.
- Expression timing: change at emotional pivots, not randomly. 1-2 per scene.
- Scene turn: on choice, change bg + expression + mood to show the world reacts.
- Write warm, character-focused, emotionally perceptive narration.
- End with either narration or choices — always give the player something to react to.
"""


# Inject the canonical enums into the prompt text (the prompts are later
# .format()ed with cast/player, so enum injection uses __PLACEHOLDERS__ —
# not {braces} — to stay out of format()'s way).
def _background_block() -> str:
    return "\n".join(f"`{key}` — {desc}" for key, desc in BACKGROUND_KEYS.items())


VN_SYSTEM_PROMPT = (VN_SYSTEM_PROMPT
                    .replace("__MOOD_TAGS__", ", ".join(MOOD_TAGS))
                    .replace("__BACKGROUND_KEYS__", _background_block()))
VN_SYSTEM_PROMPT_SHORT = (VN_SYSTEM_PROMPT_SHORT
                          .replace("__MOOD_TAGS__", ", ".join(MOOD_TAGS))
                          .replace("__BACKGROUND_NAMES__", ", ".join(BACKGROUND_KEYS)))


CHARACTER_DESIGNER_PROMPT = """You design anime visual novel characters. Given a character name and a short concept phrase, you output the character's visual design as danbooru tags plus a personality.

Respond with STRICT JSON only — no prose, no markdown fences:
{"gender": "girl" or "boy", "tags": "<danbooru appearance tags>", "personality": "<2-3 sentence personality for a story engine>", "role": "<1-3 word narrative role>"}

TAG RULES (important):
- 12 to 22 comma-separated danbooru-style appearance tags.
- Tags must start with "1girl" or "1boy".
- Describe ONLY the character: hair (color, length, style), eye color, skin, build, clothing (school uniform variant, casual wear, accessories), and one or two aura/mood words.
- You MAY include one or two body-attitude tags that fit the personality (e.g. hand on hip, arms crossed, hands behind back, head tilt, hands in pockets) — VN sprites usually stand at a slight angle with a characterful stance, not dead-frontal.
- Plain comma-separated tags ONLY. Do NOT use weighted syntax like (tag:1.2) — weighting is unreliable on this model.
- DO NOT include framing, camera, expression, or background tags (e.g. upper body, full body, portrait, profile, from behind, looking at viewer, simple background) — the pipeline locks those automatically.
- Slice-of-life school setting: prefer school uniforms unless the concept says otherwise.
- Eye and hair color are a must in every design. Vary them across characters to help visual distinction.
Design sensibly from the concept: "childhood friend" suggests warm and approachable (e.g. brown hair, medium hair, ahoge, bright eyes, cheerful aura); "cool senpai" suggests tall, sharp eyes, composed; "shy bookworm" suggests glasses, long bangs, cardigan, timid aura. Invent a coherent, distinctive design — vary hair colors across requests when plausible.
"""


def build_cast_descriptions(cast: dict) -> str:
    """Build the cast-description block for the system prompt.

    Args:
        cast: dict[str, CastMember] — the locked cast.

    Returns:
        Formatted string for insertion into the system prompt.
    """
    lines = []
    for name, member in sorted(cast.items()):
        char = member.character
        lines.append(f"- {char.name} ({char.role}): {char.personality}")
    return "\n".join(lines)


def build_state_banner(scene) -> str:
    """Build a compact [STATE] banner for the current VN scene.

    Args:
        scene: SceneState instance.

    Returns:
        Formatted [STATE] banner string.
    """
    from vn_contracts import SceneState

    lines = ["[STATE]"]
    lines.append(f"Chapter: {scene.chapter} · Turn: {scene.turn}")
    lines.append(f"Player: {scene.player_name}")
    lines.append(f"Background: {scene.background} · Mood: {scene.mood}")
    lines.append(f"Present: {', '.join(scene.present) if scene.present else '(none)'}")

    if scene.expressions:
        expr_parts = [f"{name}: {expr}" for name, expr in scene.expressions.items()
                      if name in scene.cast]
        if expr_parts:
            lines.append("Expressions: " + " | ".join(expr_parts))

    # Affection flags
    flags = []
    for name in scene.cast:
        char_flags = scene.active_affection(name)
        if char_flags:
            flags.append(f"{name}: {', '.join(char_flags)}")
    if flags:
        lines.append("Affection: " + " | ".join(flags))

    lines.append("[ENDSTATE]")
    return "\n".join(lines)


def build_recent_history(log_text: str, max_turns: int = 4,
                         max_chars: int = 1200) -> str:
    """Trim the session log to its last few turns for prompt injection.

    The session log interleaves `▸ Player: ...` action lines with narration
    blocks, separated by blank lines (see ui/callbacks._append_log). A turn
    boundary is a `▸` block; we walk back from the tail collecting blocks
    until `max_turns` boundaries are covered or `max_chars` is exceeded.

    Returns "" when the log is empty.
    """
    if not (log_text or "").strip():
        return ""
    blocks = [b for b in log_text.strip().split("\n\n") if b.strip()]
    taken: list[str] = []
    turns = size = 0
    for block in reversed(blocks):
        taken.append(block)
        size += len(block) + 2
        if block.lstrip().startswith("▸"):
            turns += 1
            if turns >= max_turns:
                break
        if size > max_chars:
            break
    history = "\n\n".join(reversed(taken))
    if len(history) > max_chars:
        # Hard cap: keep the tail, cut forward to the next line start so the
        # block never opens mid-sentence fragment of a clipped first line.
        history = history[-max_chars:]
        nl = history.find("\n")
        if nl != -1:
            history = history[nl + 1:].lstrip("\n")
    return history


def build_full_prompt(scene, cast, player_choice_text: str = "",
                      player_name: str = "Player",
                      history: str = "") -> tuple[str, str]:
    """Build the system + user prompt pair for a model call.

    Args:
        scene: Current SceneState.
        cast: dict[str, CastMember] — the locked cast.
        player_choice_text: The choice the player made (empty for first scene).
        history: Trimmed tail of the session log (build_recent_history). When
            present, it is shown to the model ABOVE the [STATE] banner so it
            knows what the player has already read and doesn't re-narrate it.

    Returns:
        (system_prompt, user_prompt) tuple.
    """
    cast_desc = build_cast_descriptions(cast)
    system = VN_SYSTEM_PROMPT.format(cast_descriptions=cast_desc, player_name=player_name)
    state_banner = build_state_banner(scene)

    if player_choice_text:
        user = f"{state_banner}\n\n[CHOICE MADE]\n{player_choice_text}\n\nWrite the next scene in response to this choice. Show how the world reacts. Change background, expressions, and mood as appropriate."
    else:
        user = f"{state_banner}\n\nWrite the opening scene of this chapter. Set the atmosphere, introduce the setting, and let the story begin."

    if history:
        user = ("## RECENT STORY (already shown to the player — do NOT "
                "re-describe or re-narrate any of it)\n"
                f"{history}\n[END RECENT STORY]\n\n{user}")

    return system, user
