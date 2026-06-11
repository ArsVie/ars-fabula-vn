"""Gradio event callbacks — casting beat, turns, save/load, advance, reset.
Built by make_callbacks(bg_provider), closing over the background provider.
"""
from __future__ import annotations
import os
import time

import gradio as gr

from vn_contracts import SceneState, CastMember
from providers import BackgroundProvider
from cast_pipeline import load_curated_cast

from .bootstrap import make_initial_scene, build_engine
from .screens import (_title_screen_html, _cast_panel_html,
                      _casting_screen_html, _review_screen_html)
from .beats import clean_narration, _segment_beats, _render_beats_screen


def make_callbacks(bg_provider: BackgroundProvider):
    # All turn callbacks return/yield 11-tuples:
    # (scene, engine, screen, cast, cast_panel, log, info, audio, prev_bg,
    #  beats, beat_idx)   ← the last two drive read-at-your-own-pace.

    def _empty(scene_state=None):
        """No active scene → the title screen (also what Reset restores)."""
        scene = scene_state or make_initial_scene()
        return (scene, None,
                _title_screen_html(),
                {}, _cast_panel_html({}),
                "", "No active scene", None, "", [], 0)

    def on_start(cast_preset: str,
                 casting_state,
                 player_name: str = "Player",
                 n1: str = "", c1: str = "", t1: str = "",
                 n2: str = "", c2: str = "", t2: str = "",
                 n3: str = "", c3: str = "", t3: str = "",
                 fast: bool = False):
        """The CASTING BEAT — generator for curated, state-machine init for generated.

        Curated mode: yields progress screens as before (casting → portraits → scene).
        Generated mode: designs characters, yields a loading screen, generates the
        first base, yields the review screen, and ends — on_casting_confirm/redo
        take over the interactive review loop from there.
        """
        cards = [{"name": n, "concept": c, "tags": t}
                 for n, c, t in ((n1, c1, t1), (n2, c2, t2), (n3, c3, t3))
                 if (n or "").strip()]

        if cast_preset == "generated":
            from cast_pipeline import (set_fast_generation, build_custom_specs,
                                       _generate_base_with_face_gate, _char_dir,
                                       PRESET_CHARACTERS)
            from model_client import release_vram

            set_fast_generation(fast)

            # Build specs: custom cards override presets
            if cards:
                specs = build_custom_specs(cards)
                if not specs:
                    specs = [(k, PRESET_CHARACTERS[k]) for k in ["yuki", "marie", "sakura"]]
            else:
                specs = [(k, PRESET_CHARACTERS[k]) for k in ["yuki", "marie", "sakura"]]

            if not specs:
                yield (*_empty(), None)
                return

            # VRAM: evict LLM so ComfyUI can load its models
            char_keys = [k for k, _ in specs]
            yield (None, None,
                   _casting_screen_html(char_keys, {},
                                        "Waking the artists…"),
                   {}, _cast_panel_html({}), "", "Preparing…", None, "", [], 0, None)

            release_vram()

            # Generate first base for review
            key, preset = specs[0]
            char_dir = _char_dir(key)
            base_path = _generate_base_with_face_gate(key, preset, char_dir)

            new_state = {"specs": specs, "idx": 0, "phase": "review", "fast": fast}

            screen = _review_screen_html(key, preset, base_path or "", 1, len(specs))
            info = f"Review: {preset.get('name', key)} · 1/{len(specs)}"

            yield (None, None, screen, {}, _cast_panel_html({}), "", info, None, "",
                   [], 0, new_state)
            return  # generator ends; on_casting_confirm / on_casting_redo takes over

        # ── Curated mode (unchanged behaviour) ──────────────────────
        char_keys = ["yuki", "marie", "sakura"]
        # load_curated_cast returns [] for any key missing from PRESET_CHARACTERS
        # — fall back to None so a bad key skips its portrait instead of crashing.
        loader = lambda k: (load_curated_cast([k]) or [None])[0]

        resolved: dict[str, CastMember] = {}

        # Beat 1 — the casting screen, empty portraits
        yield (None, None,
               _casting_screen_html(char_keys, resolved),
               {}, _cast_panel_html({}),
               "", "Casting…", None, "", [], 0, None)

        # Beat 2 — portraits resolve
        for key in char_keys:
            member = loader(key)
            if member:
                resolved[member.character.name] = member
            time.sleep(0.45)
            yield (None, None,
                   _casting_screen_html(char_keys, resolved),
                   {}, _cast_panel_html(resolved),
                   "", "Casting…", None, "", [], 0, None)

        cast = resolved
        if not cast:
            yield (*_empty(), None)
            return

        # Open the scene
        scene = SceneState(
            cast=cast, present=[],
            background="classroom", mood="peaceful",
            turn=0, chapter="Prologue",
            player_name=player_name,
        )
        engine, status = build_engine(cast, scene)
        pre_expr, pre_bg = dict(scene.expressions), scene.background
        pre_present = list(scene.present)
        raw, scene = engine.run_turn("")
        beats = _segment_beats(raw, scene.cast, pre_expr, pre_bg,
                               init_present=pre_present, voice_line=engine.last_voice)

        log = clean_narration(raw)
        screen, prev_bg, voice_path = _render_beats_screen(
            beats, 0, scene, bg_provider, "", log_text=log)
        yield (scene, engine, screen, cast, _cast_panel_html(cast),
               log,
               f"{scene.chapter} · {scene.background} ({scene.mood}) · {status}",
               voice_path, prev_bg, beats, 0, None)

    def on_casting_confirm(casting_state, casting_prompt: str,
                           _scene: SceneState, _engine, _prev_bg: str):
        """Confirm the current character's base and advance.

        If more characters remain: generate the next base, return review screen.
        If all confirmed: bake expressions, VRAM swap, start the VN scene.
        """
        if not casting_state or casting_state.get("phase") != "review":
            return (*_empty(), None)

        specs = casting_state["specs"]
        idx = casting_state["idx"]
        fast = casting_state.get("fast", False)

        # Update current character's prompt from the (possibly edited) textbox
        key, preset = specs[idx]
        if casting_prompt and casting_prompt.strip():
            preset["identity_prompt"] = casting_prompt.strip()
            # Update the seed to reflect prompt changes (so redo gets fresh results)
            from cast_pipeline import _stable_seed
            preset["seed"] = _stable_seed(preset.get("name", key), preset["identity_prompt"])

        # Advance to next character
        next_idx = idx + 1
        if next_idx < len(specs):
            # More characters — generate next base
            from cast_pipeline import (_generate_base_with_face_gate, _char_dir,
                                       set_fast_generation)
            set_fast_generation(fast)

            next_key, next_preset = specs[next_idx]
            char_dir = _char_dir(next_key)
            base_path = _generate_base_with_face_gate(next_key, next_preset, char_dir)

            new_state = {"specs": specs, "idx": next_idx, "phase": "review", "fast": fast}

            screen = _review_screen_html(next_key, next_preset, base_path or "",
                                         next_idx + 1, len(specs))
            info = f"Review: {next_preset.get('name', next_key)} · {next_idx + 1}/{len(specs)}"

            return (None, None, screen, {}, _cast_panel_html({}), "", info, None, "",
                    [], 0, new_state)

        # All confirmed — bake expressions, swap VRAM, start the VN
        from cast_pipeline import (_generate_one_via_comfy, _char_dir,
                                   comfy_free_vram, set_fast_generation)
        from model_client import ensure_llm, ModelConfig

        set_fast_generation(fast)

        # Bake expressions for all characters (UNET still hot from base gen)
        members = []
        for skey, spreset in specs:
            char_dir = _char_dir(skey)
            base = os.path.join(char_dir, f"{skey}_base.png")
            try:
                member = _generate_one_via_comfy(skey, spreset,
                                                 base_path=(base if os.path.exists(base) else ""))
                members.append(member)
            except Exception as e:
                print(f"  [cast] Expression bake failed for {skey}: {e}")
                # Fall back to placeholder
                from cast_pipeline import _placeholder_member
                members.append(_placeholder_member(skey, spreset))

        cast = {m.character.name: m for m in members}
        if not cast:
            return (*_empty(), None)

        # VRAM swap: unload ComfyUI, load LLM
        comfy_free_vram()
        ensure_llm(ModelConfig.from_env(), timeout=300)

        # Start the VN
        scene = SceneState(
            cast=cast, present=[],
            background="classroom", mood="peaceful",
            turn=0, chapter="Prologue",
            player_name=_scene.player_name if _scene else "Player",
        )
        engine, status = build_engine(cast, scene)
        pre_expr, pre_bg = dict(scene.expressions), scene.background
        pre_present = list(scene.present)
        raw, scene = engine.run_turn("")
        beats = _segment_beats(raw, scene.cast, pre_expr, pre_bg,
                               init_present=pre_present, voice_line=engine.last_voice)

        log = clean_narration(raw)
        screen, new_prev_bg, voice_path = _render_beats_screen(
            beats, 0, scene, bg_provider, "", log_text=log)

        return (scene, engine, screen, cast, _cast_panel_html(cast),
                log,
                f"{scene.chapter} · {scene.background} ({scene.mood}) · {status}",
                voice_path, new_prev_bg, beats, 0, None)

    def on_casting_redo(casting_state, casting_prompt: str):
        """Regenerate the current character's base with the edited prompt."""
        if not casting_state or casting_state.get("phase") != "review":
            return (_title_screen_html(), "No active review", None, "")

        from cast_pipeline import (set_fast_generation, _generate_base_with_face_gate,
                                   _char_dir, build_identity_prompt)

        specs = casting_state["specs"]
        idx = casting_state["idx"]
        fast = casting_state.get("fast", False)
        key, preset = specs[idx]

        set_fast_generation(fast)

        # Update prompt from the textbox
        if casting_prompt and casting_prompt.strip():
            preset["identity_prompt"] = casting_prompt.strip()

        # Bump seed for a genuinely different result
        preset["seed"] = preset.get("seed", 42) + 1

        char_dir = _char_dir(key)
        base_path = _generate_base_with_face_gate(key, preset, char_dir)

        screen = _review_screen_html(key, preset, base_path or "",
                                     idx + 1, len(specs))
        info = f"Review: {preset.get('name', key)} · {idx + 1}/{len(specs)}"

        # Clear the prompt box so it shows the new prompt on next load
        return (screen, info, casting_state, preset["identity_prompt"])

    def on_choice(choice_id: str, choice_text: str,
                  scene: SceneState, engine, prev_bg: str):
        """A choice click → THE SCENE TURN."""
        if not engine or not scene or not choice_id:
            return _empty(scene)

        for choice in scene.choices:
            if choice.id == choice_id:
                engine.apply_choice_effects(choice)
                break

        # Snapshot the visual state going INTO the turn so the beat splitter
        # can replay expression/background changes in order.
        pre_expr, pre_bg = dict(scene.expressions), scene.background
        pre_present = list(scene.present)
        raw, scene = engine.run_turn(choice_text or choice_id)
        beats = _segment_beats(raw, scene.cast, pre_expr, pre_bg,
                               init_present=pre_present, voice_line=engine.last_voice)

        log = clean_narration(raw)
        screen, new_prev_bg, voice_path = _render_beats_screen(
            beats, 0, scene, bg_provider, prev_bg, log_text=log)

        return (scene, engine, screen, scene.cast, _cast_panel_html(scene.cast),
                log,
                f"Turn {scene.turn} · {scene.background} ({scene.mood})",
                voice_path, new_prev_bg, beats, 0)

    def on_save(slot: float, scene: SceneState, engine):
        if not engine:
            return "Nothing to save — start a story first."
        engine.save(int(slot))
        return f"Saved · slot {int(slot)}"

    def on_load(slot: float, scene: SceneState, engine, prev_bg: str):
        if not engine:
            return (*_empty(scene), "Nothing to load into — start a story first.")
        try:
            scene = engine.load(int(slot))
        except FileNotFoundError:
            return (scene, engine, gr.skip(), scene.cast,
                    _cast_panel_html(scene.cast), gr.skip(),
                    f"Slot {int(slot)} is empty.", None, prev_bg, gr.skip(), gr.skip(),
                    f"Slot {int(slot)} is empty.")

        # A restored save has no per-beat tool sequence — show it as a single
        # static beat (the saved description) with its choices already visible.
        beats = [{"speaker": "", "text": scene.description or "",
                  "expressions": dict(scene.expressions),
                  "background": scene.background,
                  "present": list(scene.present), "voice": None}]
        screen, new_prev_bg, _ = _render_beats_screen(
            beats, 0, scene, bg_provider, prev_bg, log_text=scene.description or "")
        return (scene, engine, screen, scene.cast, _cast_panel_html(scene.cast),
                scene.description,
                f"Loaded · slot {int(slot)} · Turn {scene.turn} · {scene.background} ({scene.mood})",
                None, new_prev_bg, beats, 0,
                f"Loaded · slot {int(slot)}")

    def on_advance(beats, idx, scene: SceneState, prev_bg: str, log_text: str):
        """Reveal the next beat (click-the-screen / Space / Enter).

        log_text comes from the hidden narration_log sink so the in-canvas
        Scene Log panel keeps its content across beat re-renders.
        """
        if not beats:
            return gr.skip(), gr.skip(), None, gr.skip(), gr.skip()
        new_idx = min((idx or 0) + 1, len(beats) - 1)
        screen, new_prev_bg, voice_path = _render_beats_screen(
            beats, new_idx, scene, bg_provider, prev_bg, log_text=log_text or "")
        info = f"Turn {scene.turn} · {scene.background} ({scene.mood})"
        return screen, new_idx, voice_path, info, new_prev_bg

    def on_free_submit(text: str, scene: SceneState, engine, prev_bg: str):
        """Free-text action → a scene turn. Wraps on_choice and ALSO clears
        the input box server-side (returns "" into free_input) — no JS race.

        Returns the 11 turn_outputs + the cleared free_input value.
        """
        if not (text or "").strip() or not engine or not scene:
            # Nothing to do — keep the box contents, change nothing else
            return tuple(gr.skip() for _ in range(12))
        result = on_choice("__free__", text.strip(), scene, engine, prev_bg)
        return (*result, "")   # clear free_input

    def on_reset():
        return _empty()

    return (on_start, on_choice, on_save, on_load, on_reset, on_advance, on_free_submit,
            on_casting_confirm, on_casting_redo)
