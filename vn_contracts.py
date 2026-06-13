"""
vn_contracts.py — Wave 0.5 freeze: data structures for the VN PoC.

Every subagent codes against these shapes. Immutable where noted.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ── Standard expression set every cast member must have ──────────────
# Every generated or curated character gets these 7 expressions.
# The sprite-path map keys are these strings.
STANDARD_EXPRESSIONS = [
    "neutral", "smile", "laugh", "sad", "angry", "surprised", "embarrassed",
]


# ── Character / Cast ─────────────────────────────────────────────────

@dataclass(frozen=True)
class Character:
    """Immutable character identity — the soul of a cast member."""
    name: str
    identity_prompt: str              # e.g. "1girl, long silver hair, red eyes, ..."
    seed: int                         # Anima seed for reproducibility
    personality: str                  # 2-3 sentence personality for the LLM to use
    voice_id: str                     # key into VoiceProvider voices
    role: str = "student"             # narrative role: student, teacher, senpai, etc.


@dataclass
class CastMember:
    """A locked character + their rendered expression sprites.

    The expression_map is a dict: expr_name -> absolute or relative path to RGBA PNG.
    Every member must have at least the STANDARD_EXPRESSIONS set.
    """
    character: Character
    expression_map: dict[str, str] = field(default_factory=dict)
    # Placeholder-only nominal canvas (real sprites are read from disk). Kept in
    # step with the bake: H0 = base 896 × SPRITE_CANVAS_MULT 1.4 ≈ 1254 tall, and
    # a representative widened canvas ≈ 720 wide (cast_pipeline._frame_upper_body).
    sprite_width: int = 720
    sprite_height: int = 1254

    def get_sprite(self, expression: str) -> Optional[str]:
        """Get the sprite path for a given expression. Falls back to neutral."""
        if expression in self.expression_map:
            return self.expression_map[expression]
        return self.expression_map.get("neutral")

    def has_expression(self, expression: str) -> bool:
        return expression in self.expression_map


def cast_to_dict(members: list["CastMember"]) -> dict[str, "CastMember"]:
    """Build the name -> CastMember dict used by SceneState from the
    list[CastMember] that the cast pipeline returns (frozen contract:
    generate_cast(prefs) -> list[CastMember])."""
    return {m.character.name: m for m in members}


# ── Cast cards — serializable character snapshots keyed by hash ───────
# A "cast card" is the full JSON form of a cast (identity + sprite paths),
# written once per unique cast and referenced from saves by fingerprint, so
# a save can be loaded from the main menu without an engine or a re-bake.

def member_to_dict(member: "CastMember") -> dict:
    c = member.character
    return {
        "name": c.name,
        "identity_prompt": c.identity_prompt,
        "seed": c.seed,
        "personality": c.personality,
        "voice_id": c.voice_id,
        "role": c.role,
        "expression_map": dict(member.expression_map),
        "sprite_width": member.sprite_width,
        "sprite_height": member.sprite_height,
    }


def member_from_dict(d: dict) -> "CastMember":
    character = Character(
        name=d["name"],
        identity_prompt=d.get("identity_prompt", ""),
        seed=d.get("seed", 0),
        personality=d.get("personality", ""),
        voice_id=d.get("voice_id", ""),
        role=d.get("role", "student"),
    )
    return CastMember(
        character=character,
        expression_map=dict(d.get("expression_map", {})),
        sprite_width=d.get("sprite_width", 512),
        sprite_height=d.get("sprite_height", 768),
    )


def cast_fingerprint(cast: dict[str, "CastMember"]) -> str:
    """Stable short hash of a cast's identity (name + prompt + seed per
    member, order-independent). Saves reference their cast card by this."""
    import hashlib
    import json as _json
    ident = sorted(
        (m.character.name, m.character.identity_prompt, m.character.seed)
        for m in cast.values()
    )
    blob = _json.dumps(ident, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


# ── Scene state ──────────────────────────────────────────────────────

@dataclass
class AffectionFlag:
    """A named binary flag for affection/like tracking."""
    character_name: str
    flag: str          # e.g. "first_meeting", "confided_in", "gave_gift"
    value: bool = True


@dataclass
class SceneState:
    """The full mutable state of a VN session.

    Everything a scene turn needs: who's present, what's the mood,
    what choices matter, where we are.
    """
    # Cast
    cast: dict[str, CastMember] = field(default_factory=dict)   # name -> CastMember
    present: list[str] = field(default_factory=list)             # which characters are on screen

    # Scene
    background: str = "classroom"    # key into BackgroundProvider
    description: str = ""            # current scene narration text
    dialogue: list[dict] = field(default_factory=list)  # [{"speaker","text","expression"}]
    mood: str = "peaceful"           # current mood tag for music/atmosphere
    expressions: dict[str, str] = field(default_factory=dict)  # char name -> current expression key

    # Progression
    turn: int = 0
    choices: list[Choice] = field(default_factory=list)
    affection_flags: list[AffectionFlag] = field(default_factory=list)
    player_name: str = "Player"

    # Save slot metadata
    save_slot: Optional[int] = None
    chapter: str = "Prologue"

    def active_affection(self, character_name: str) -> list[str]:
        """Get all affection flags set for a character."""
        return [af.flag for af in self.affection_flags
                if af.character_name == character_name and af.value]

    def set_affection(self, character_name: str, flag: str):
        """Set a named affection flag for a character."""
        for af in self.affection_flags:
            if af.character_name == character_name and af.flag == flag:
                af.value = True
                return
        self.affection_flags.append(AffectionFlag(character_name, flag))

    def has_affection(self, character_name: str, flag: str) -> bool:
        return any(af.character_name == character_name
                   and af.flag == flag and af.value
                   for af in self.affection_flags)


# ── Choice ────────────────────────────────────────────────────────────

@dataclass
class Choice:
    """A player-facing choice in the VN."""
    id: str                    # unique within the scene
    text: str                  # displayed text
    consequence: str = ""      # brief description of what happens (for the engine)
    affection_flag: Optional[tuple[str, str]] = None  # (char_name, flag) if choice triggers affection
    next_background: Optional[str] = None
    next_mood: Optional[str] = None
    expression_changes: dict[str, str] = field(default_factory=dict)  # char -> expression key


# ── Tool results ─────────────────────────────────────────────────────

@dataclass
class ActionResult:
    """Result of a VN tool execution."""
    success: bool = True
    message: str = ""
    data: dict = field(default_factory=dict)


# ── Save/Load ─────────────────────────────────────────────────────────

@dataclass
class SaveData:
    """JSON-serializable snapshot for saving/loading.

    The cast itself is NOT serialized (it's locked and immutable for the
    session); only cast_names are stored so a load can sanity-check that
    the save matches the current cast.
    """
    chapter: str
    turn: int
    background: str
    mood: str
    description: str
    present: list[str]
    cast_names: list[str]
    dialogue_snapshot: list[dict]
    affection_snapshot: list[tuple[str, str, bool]]
    expressions: dict[str, str] = field(default_factory=dict)
    choices_snapshot: list[dict] = field(default_factory=list)
    player_name: str = "Player"
    # Slot metadata (shown in the save/load UIs)
    save_name: str = ""              # player-given save title
    saved_at: str = ""               # local "YYYY-MM-DD HH:MM"
    cast_hash: str = ""              # fingerprint of the cast card this save uses
    log_text: str = ""               # full-session scene log (turns + player answers)

    @classmethod
    def from_scene(cls, scene: SceneState) -> "SaveData":
        return cls(
            chapter=scene.chapter,
            turn=scene.turn,
            background=scene.background,
            mood=scene.mood,
            description=scene.description,
            present=list(scene.present),
            cast_names=list(scene.cast.keys()),
            dialogue_snapshot=list(scene.dialogue),
            affection_snapshot=[(af.character_name, af.flag, af.value)
                                for af in scene.affection_flags],
            expressions=dict(scene.expressions),
            choices_snapshot=[{
                "id": c.id, "text": c.text, "consequence": c.consequence,
                "affection_flag": list(c.affection_flag) if c.affection_flag else None,
                "next_background": c.next_background,
                "next_mood": c.next_mood,
                "expression_changes": dict(c.expression_changes),
            } for c in scene.choices],
            player_name=scene.player_name,
        )

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SaveData":
        data = dict(data)
        data["affection_snapshot"] = [tuple(t) for t in data.get("affection_snapshot", [])]
        # Tolerate unknown keys so newer saves still open in older code.
        from dataclasses import fields as _fields
        allowed = {f.name for f in _fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def apply_to(self, scene: SceneState) -> SceneState:
        """Restore this snapshot into an existing scene (cast is kept)."""
        scene.chapter = self.chapter
        scene.turn = self.turn
        scene.background = self.background
        scene.mood = self.mood
        scene.description = self.description
        scene.present = list(self.present)
        scene.dialogue = list(self.dialogue_snapshot)
        scene.affection_flags = [AffectionFlag(c, f, v)
                                 for c, f, v in self.affection_snapshot]
        scene.expressions = dict(self.expressions)
        scene.player_name = self.player_name
        scene.choices = []
        for cd in self.choices_snapshot:
            choice = Choice(
                id=cd["id"], text=cd["text"],
                consequence=cd.get("consequence", ""),
                next_background=cd.get("next_background"),
                next_mood=cd.get("next_mood"),
                expression_changes=cd.get("expression_changes") or {},
            )
            if cd.get("affection_flag"):
                choice.affection_flag = (cd["affection_flag"][0], cd["affection_flag"][1])
            scene.choices.append(choice)
        return scene
