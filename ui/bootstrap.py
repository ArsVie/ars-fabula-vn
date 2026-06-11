"""App bootstrap: constants, the initial scene, and engine construction
(real LLM if reachable, mock otherwise).
"""
from __future__ import annotations
import os

from vn_contracts import SceneState, CastMember, cast_to_dict
from vn_engine import VNEngine
from cast_pipeline import load_curated_cast
from providers import get_voice_provider
from model_client import get_model, ModelConfig, MockModel


# ── Constants ─────────────────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(PROJECT_DIR, "static")

# Default cast (loaded at startup) — pipeline returns list (frozen contract);
# SceneState wants the name->member dict.
DEFAULT_CAST = cast_to_dict(load_curated_cast())


def make_initial_scene(cast=None, player_name: str = "Player") -> SceneState:
    if cast is None:
        cast = DEFAULT_CAST
    return SceneState(
        cast=cast,
        present=[],  # characters enter via enter_character as they appear
        background="classroom",
        mood="peaceful",
        turn=0,
        chapter="Prologue",
        player_name=player_name,
    )



# ── Mock VN LLM ───────────────────────────────────────────────────────

def _make_mock_engine(cast: dict[str, CastMember], player_name: str = "Player") -> MockModel:
    """Create a MockModel that generates varied multi-character VN scenes.

    Exercises enter_character/exit_character across beats to verify per-beat
    presence tracking manually. Each branch produces a different character
    lineup so the tester can see characters pop in/out visually.
    """
    char_names = list(cast.keys())
    # Expect Yuki (idx 0), Haruki (idx 1), Sakura (idx 2), Ryo (idx 3)
    yuki = char_names[0] if len(char_names) > 0 else "Yuki"
    haruki = char_names[1] if len(char_names) > 1 else "Haruki"
    sakura = char_names[2] if len(char_names) > 2 else "Sakura"
    ryo = char_names[3] if len(char_names) > 3 else "Ryo"

    def handler(messages, tools):
        user_msgs = [m for m in messages if m["role"] == "user"]
        raw_choice = user_msgs[-1]["content"] if user_msgs else ""
        # Extract just the player's choice text after [CHOICE MADE]
        choice_match = __import__("re").search(r"\[CHOICE MADE\]\n(.+?)\n\nWrite the next scene", raw_choice, __import__("re").DOTALL)
        choice_text = choice_match.group(1).strip() if choice_match else ""
        choice_lower = choice_text.lower()

        # ── "talk" → Yuki + Haruki enter one-by-one ──────────────
        if "talk" in choice_lower or yuki.lower() in choice_lower:
            return (
                f'[TOOL: enter_character character="{yuki}"]\n\n'
                f'[TOOL: set_expression character="{yuki}" expression="surprised"]\n\n'
                f'[TOOL: advance_scene description="The afternoon sun filters through the classroom windows, casting warm rectangles of light across the wooden floor. The scent of chalk dust and summer lingers in the air." background="classroom" mood="cheerful"]\n\n'
                f'[TOOL: set_expression character="{yuki}" expression="smile"]\n\n'
                f'[{yuki}]: "Oh, hey! I didn\'t expect you to come over today. What\'s up?"\n\n'
                f'Her smile is warm and genuine, the afternoon light catching in her hair.\n\n'
                f'[TOOL: enter_character character="{haruki}"]\n\n'
                f'[TOOL: set_expression character="{yuki}" expression="surprised"]\n\n'
                f'The door slides open and {haruki} pokes their head in, eyebrows raising as they spot the two of you.\n\n'
                f'[TOOL: set_expression character="{haruki}" expression="laugh"]\n\n'
                f'[{haruki}]: "Am I interrupting something? Should I come back later?"\n\n'
                f'They\'re clearly teasing, a playful glint in their eye.\n\n'
                f'[TOOL: set_expression character="{yuki}" expression="embarrassed"]\n\n'
                f'[{yuki}]: "No! I mean — we were just talking. Come sit down."\n\n'
                f'The atmosphere shifts into something lighter, the three of you settling into the familiar rhythm of friends.\n\n'
                f'[TOOL: offer_choices choices=\'[{{"id":"ask","text":"\\\"So what did you want to talk about?\\\"","consequence":"get serious"}},{{"id":"tease","text":"\\\"Look who\'s getting protective~\\\"","consequence":"tease them both"}},{{"id":"group","text":"\\\"Let\'s all hang out after school!\\\"","consequence":"make plans together"}}]\']'
            )

        # ── "look" / "alone" → Yuki enters, exits, then Sakura enters ──
        # NOTE: no offer_choices — exercises the free-text fallback!
        elif "look" in choice_lower or "alone" in choice_lower:
            return (
                f'[TOOL: enter_character character="{yuki}"]\n\n'
                f'[TOOL: set_expression character="{yuki}" expression="sad"]\n\n'
                f'[TOOL: advance_scene description="The classroom feels emptier than usual. Through the window, the courtyard buzzes with laughter muffled by glass." background="classroom" mood="melancholic"]\n\n'
                f'[{yuki}]: "I\'ve been thinking a lot lately. About everything, really."\n\n'
                f'They stare out the window, their reflection overlapping with the clouds drifting past.\n\n'
                f'[TOOL: exit_character character="{yuki}"]\n\n'
                f'The door clicks shut. The room settles into silence, dust motes dancing in the slanted light.\n\n'
                f'[TOOL: enter_character character="{sakura}"]\n\n'
                f'[TOOL: set_expression character="{sakura}" expression="smile"]\n\n'
                f'[{sakura}]: "Hey. I saw {yuki} heading to the roof. They looked kinda down."\n\n'
                f'She leans against the doorframe, concern flickering behind her cheerful demeanour. The silence stretches, waiting for you to fill it.'
            )

        # ── "confess" / "love" → Rooftop with interrupted confession ──
        elif "confess" in choice_lower or "love" in choice_lower:
            return (
                f'[TOOL: enter_character character="{yuki}"]\n\n'
                f'[TOOL: set_expression character="{yuki}" expression="surprised"]\n\n'
                f'[TOOL: advance_scene description="Time seems to slow. The rooftop stretches out before you, the city glittering in the distance. Cherry blossom petals drift lazily on the warm breeze." background="rooftop" mood="romantic"]\n\n'
                f'[TOOL: set_expression character="{yuki}" expression="embarrassed"]\n\n'
                f'[{yuki}]: "I… I\'ve been wanting to tell you something for a while now."\n\n'
                f'For a long, breathless moment, neither of you speaks. The wind carries petals between you.\n\n'
                f'[TOOL: enter_character character="{haruki}"]\n\n'
                f'[TOOL: set_expression character="{yuki}" expression="surprised"]\n\n'
                f'[TOOL: set_expression character="{haruki}" expression="laugh"]\n\n'
                f'[{haruki}]: "Whoa — sorry, sorry! I didn\'t mean to walk in on… whatever this is."\n\n'
                f'They back away with their hands up, grinning, and disappear back through the door.\n\n'
                f'[TOOL: exit_character character="{haruki}"]\n\n'
                f'The moment hangs in the air again, fragile and electric.\n\n'
                f'[TOOL: set_expression character="{yuki}" expression="smile"]\n\n'
                f'[{yuki}]: "As I was saying… I really like you. I have for a long time."\n\n'
                f'The sunset paints the world in shades of gold and pink.\n\n'
                f'[TOOL: offer_choices choices=\'[{{"id":"hold_hands","text":"Take their hand","consequence":"a gentle touch","next_background":"sunset_hill","next_mood":"romantic","affection_flag":["{yuki}","confession"]}},{{"id":"hug","text":"Pull them into a hug","consequence":"embrace","next_background":"rooftop","next_mood":"romantic"}},{{"id":"speak","text":"\\\"I\'ve wanted to say it for so long too…\\\"","consequence":"vulnerable truth","next_background":"rooftop","next_mood":"romantic"}}]\']'
            )

        # ── Default / first turn → three characters gradually enter ──
        else:
            return (
                f'[TOOL: enter_character character="{yuki}"]\n\n'
                f'[TOOL: advance_scene description="The bell rings, signalling the end of another school day. Students begin packing their bags, the air filled with chatter and the shuffle of papers. Sunlight streams through the tall windows." background="classroom" mood="peaceful"]\n\n'
                f'[TOOL: set_expression character="{yuki}" expression="smile"]\n\n'
                f'[{yuki}]: "Hey {player_name}! Ready to head out?"\n\n'
                f'They pack their bag with practiced ease, glancing at you expectantly.\n\n'
                f'[TOOL: enter_character character="{haruki}"]\n\n'
                f'[TOOL: set_expression character="{haruki}" expression="laugh"]\n\n'
                f'[{haruki}]: "Leaving without me? I\'m hurt, really."\n\n'
                f'They saunter over, slinging their bag over one shoulder, a familiar teasing grin on their face.\n\n'
                f'[TOOL: set_expression character="{yuki}" expression="laugh"]\n\n'
                f'[{yuki}]: "You take forever, {haruki}. We were about to send a search party."\n\n'
                f'[TOOL: enter_character character="{sakura}"]\n\n'
                f'[TOOL: set_expression character="{sakura}" expression="smile"]\n\n'
                f'[{sakura}]: "Are we doing the café thing? I\'m starving."\n\n'
                f'She appears from the hallway, already halfway out the door, flipping her hair with practiced nonchalance.\n\n'
                f'[TOOL: set_expression character="{haruki}" expression="smile"]\n\n'
                f'[{haruki}]: "Café sounds perfect. My treat — I just got paid for that tutoring gig."\n\n'
                f'The three of them look at you, waiting, the late afternoon light pooling around them like a scene from a movie.\n\n'
                f'[TOOL: offer_choices choices=\'[{{"id":"yes_cafe","text":"\\\"Sure, let\'s go!\\\"","consequence":"go to the cafe together"}},{{"id":"rain_check","text":"\\\"Maybe tomorrow? I\'ve got stuff to do.\\\"","consequence":"politely decline"}},{{"id":"walk","text":"\\\"I\'ll walk with you guys but I can\'t stay long.\\\"","consequence":"partial join"}}]\']'
            )

    return MockModel(handler=handler)


# ── LLM / engine builders ─────────────────────────────────────────────

def _try_real_llm():
    """Try to connect to the real model server. Returns client or None,
    and says exactly WHY it fell back when it does.

    Uses ensure_llm() under the hood so the server gets a chance to
    finish loading its model before we give up — the one-shot /health
    check is too fragile when the 26B MoE takes 2-3 minutes to load."""
    from model_client import ensure_llm

    cfg = ModelConfig.from_env()
    if cfg.backend == "mock":
        print("[app] ARS_FABULA_BACKEND=mock is set — mock engine forced "
              "(unset the env var for the real model)")
        return None
    if cfg.backend == "transformers":
        # In-process model (HF Spaces / ZeroGPU) — no server, no /health.
        client = get_model(cfg)
        if client.health():
            print("[app] In-process transformers model ready")
            return client
        print("[app] transformers backend configured but no model loaded "
              "— falling back to mock")
        return None
    try:
        if ensure_llm(cfg, timeout=300):
            client = get_model(cfg)
            print(f"[app] Connected to LLM server at {cfg.base_url}")
            return client
        base = cfg.base_url.rsplit("/v1", 1)[0]
        print(f"[app] No model server at {cfg.base_url} "
              f"(GET {base}/health got no healthy reply after 300s).")
        print("[app] Start llama-server on that port, or set "
              "ARS_FABULA_BASE_URL to where it lives.")
    except Exception as e:
        print(f"[app] LLM server check failed: {e}")
    return None


def build_engine(cast: dict[str, CastMember], scene: SceneState) -> tuple[VNEngine, str]:
    """Build the VN engine: real model if reachable, mock otherwise.
    Voice provider comes from ARS_FABULA_VOICE (mock/kokoro/chatterbox)."""
    voice = get_voice_provider(os.getenv("ARS_FABULA_VOICE", "mock"))

    real = _try_real_llm()
    if real:
        return VNEngine(model=real, cast=cast, scene=scene, voice=voice), "real LLM"

    mock = _make_mock_engine(cast, player_name=scene.player_name)
    print("[app] Using MOCK VN engine (no server needed)")
    return VNEngine(model=mock, cast=cast, scene=scene, voice=voice), "mock LLM"
