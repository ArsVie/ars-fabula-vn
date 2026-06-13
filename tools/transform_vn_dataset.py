"""
transform_vn_dataset.py — real VN transcripts -> SFT turns in our protocol.

The DeepSeek transform harness from next_session.md ("the agreed data
strategy"): use our traces as the *formula* (the [STATE] + [TOOL:] +
cast-lock + choice protocol) and a strong external LLM to transform real
VN transcript chunks into that protocol — professional pacing in the
targets, our parsers as a free validator.

Per chunk, the transform model returns a delimited envelope:
  === META ===        JSON: cast (2-4, varied per example), player_name,
                      background/mood (canonical enums), present,
                      expressions, chapter, turn
  === HISTORY ===     a RECENT STORY block in session-log format
  === CHOICE ===      the player choice this turn answers
  === COMPLETION ===  the assistant turn in our exact protocol

We then build (system, user) OURSELVES via vn_prompt.build_full_prompt —
prompt-side validity is constructed, not hoped for — and gate the
completion through vn_validate.validate_turn. On failure: one repair
call (reasons fed back), then drop.

Usage:
    export OPENCODE_API_KEY=sk-...
    python tools/transform_vn_dataset.py --limit 100              # pilot
    python tools/transform_vn_dataset.py --limit 3000 --workers 12
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import sys
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vn_contracts import Character, CastMember, SceneState, STANDARD_EXPRESSIONS  # noqa: E402
from vn_prompt import MOOD_TAGS, BACKGROUND_KEYS, build_full_prompt  # noqa: E402
from vn_validate import validate_turn  # noqa: E402

SCHEMA_VERSION = "transform-v1"
API_BASE = os.environ.get("OPENCODE_API_BASE", "https://opencode.ai/zen/go/v1")
API_KEY = os.environ.get("OPENCODE_API_KEY", "")
MODEL = os.environ.get("ARS_FABULA_TRANSFORM_MODEL", "deepseek-v4-flash")

_usage_lock = threading.Lock()
_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


# ── Transform prompt ──────────────────────────────────────────────────

TRANSFORM_SYSTEM = """You convert raw visual-novel transcript excerpts into training examples for a small VN-narrator model. The target model narrates scenes for a LOCKED CAST and drives the UI through bracketed tool calls. You must output ONE example per request, in the exact envelope format below.

## TARGET PROTOCOL (what the COMPLETION section must follow)

The completion is one VN turn: short narration beats, dialogue lines, and tool calls.

- Narration beats: AT MOST 2 sentences each, separated by blank lines. Show don't tell, at most one sensory detail per beat. NEVER long paragraphs. Narration is BARE prose — never prefix it with "Narration:" or any label.
- Dialogue lines: `Name: "Their line."` — Name MUST be a cast member (never the player; the player is addressed as "you").
- Tool calls, each alone on its own line, at the start or end of the response (never mid-narration):
  [TOOL: advance_scene description="..." background="<bg_key>" mood="<mood>"]
  [TOOL: set_background background="<bg_key>"]
  [TOOL: set_expression character="<CastName>" expression="<expr>"]
  [TOOL: enter_character character="<CastName>"]
  [TOOL: exit_character character="<CastName>"]
  [TOOL: offer_choices choices='[{"id": "...", "text": "...", "consequence": "..."}]']
  [TOOL: save_checkpoint]
- Valid expressions: __EXPRESSIONS__
- Valid moods: __MOODS__
- Valid background keys: __BACKGROUNDS__
- enter_character before a character (not already present) speaks; set_expression only when an expression CHANGES (1-2 per turn, at the emotional pivot).
- The turn MUST end with one offer_choices call: exactly 3 choices, unique snake_case ids, choice text in the player's voice, consequence = one short engine note. Use SINGLE quotes around the JSON array and double quotes inside, exactly as shown above. The JSON must be valid — escape any apostrophes carefully (prefer phrasing without apostrophes inside choice text).
- Optionally one save_checkpoint after offer_choices.
- CAST-LOCK: only cast members may be named, speak, or receive tool calls. Rewrite any other transcript character into the cast or drop them. Background extras stay nameless ("someone", "a passing student").

## YOUR JOB

Given a raw VN transcript excerpt, ADAPT it (do not copy verbatim) into one mid-session turn:

1. Pick a cast of 2-4 characters inspired by the excerpt but RENAMED with fresh Japanese given names (vary names across the cast; never reuse the source's names). Give each a 1-3 word role and a 2-3 sentence personality consistent with how they act.
2. Pick one transcript character to become the unnamed player-protagonist ("you").
3. Re-set the scene into one of the valid background keys and moods (slice-of-life school setting; adapt locations to the closest key).
4. Write HISTORY: 2-4 prior turns in session-log format — blocks separated by blank lines; each player action is a line starting with `▸ Player: `, followed by short narration/dialogue blocks adapted from the EARLIER part of the excerpt. Keep under 1100 characters.
5. Write CHOICE: the player choice (first person, imperative or quoted speech) that this turn responds to.
6. Write COMPLETION: the next turn, adapted from the LATER part of the excerpt — professional pacing, the world visibly reacting (expression change, possibly background/mood shift), ending in offer_choices. The completion must NOT re-describe anything already in HISTORY; it must move forward.

## OUTPUT ENVELOPE (exactly this, no other text, no markdown fences)

=== META ===
{"cast": [{"name": "...", "role": "...", "personality": "..."}], "player_name": "...", "background": "<bg_key>", "mood": "<mood>", "present": ["..."], "expressions": {"<CastName>": "<expr>"}, "chapter": "Chapter 1", "turn": <int 2-9>}
=== HISTORY ===
<session-log text>
=== CHOICE ===
<player choice text>
=== COMPLETION ===
<the assistant turn in the protocol>

META rules: present lists ONLY cast members currently on screen (1-3); expressions only for present members with valid expression keys; background/mood/turn must be plausible given HISTORY. player_name is a simple given name.
"""

TRANSFORM_SYSTEM = (TRANSFORM_SYSTEM
                    .replace("__EXPRESSIONS__", ", ".join(STANDARD_EXPRESSIONS))
                    .replace("__MOODS__", ", ".join(MOOD_TAGS))
                    .replace("__BACKGROUNDS__", ", ".join(BACKGROUND_KEYS)))

# A trimmed real turn from our own traces, shown as a style exemplar for the
# COMPLETION section only (pacing + tool placement), not as content to copy.
COMPLETION_EXEMPLAR = '''The heavy tension is suddenly broken by footsteps approaching the classroom door.

[TOOL: enter_character character="Yui"]

Yui: "Hark! Do you hear it? The silent scream of a tragedy unfolding!"

[TOOL: set_expression character="Yui" expression="surprised"]

She strikes a pose in the doorway, one hand pressed against her forehead.

Rin: "Good grief... it is just a study session."

[TOOL: set_expression character="Rin" expression="angry"]

Aoi tilts her head, her gaze shifting from the window to the newcomer.

Aoi: "The stage is quite lively today..."

[TOOL: advance_scene description="The classroom feels smaller now, filled with overwhelming energy." background="classroom" mood="playful"]

[TOOL: offer_choices choices='[{"id": "join_in", "text": "Ask what drama she means.", "consequence": "She becomes more animated."}, {"id": "push_back", "text": "Tell her she is being too loud.", "consequence": "She takes it as a challenge."}, {"id": "stay_quiet", "text": "Just watch the chaos unfold.", "consequence": "The tension heightens."}]']
[TOOL: save_checkpoint]'''


def build_transform_user(chunk: str) -> str:
    return (f"Here is a style exemplar for the COMPLETION section "
            f"(pacing/format only — do not copy its content):\n"
            f"-----\n{COMPLETION_EXEMPLAR}\n-----\n\n"
            f"RAW VN TRANSCRIPT EXCERPT:\n-----\n{chunk}\n-----\n\n"
            f"Produce the output envelope now.")


# ── API ───────────────────────────────────────────────────────────────

def call_model(messages: list[dict], max_tokens: int = 10000,
               temperature: float = 0.8, retries: int = 3) -> str:
    """deepseek-v4-flash is a reasoning model: reasoning_content shares the
    completion budget, so max_tokens must leave room for thinking + envelope.
    A finish_reason of 'length' means a truncated envelope — retry, don't
    waste a repair call on it."""
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={"model": MODEL, "messages": messages,
                      "max_tokens": max_tokens, "temperature": temperature},
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage") or {}
            with _usage_lock:
                _usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                _usage["completion_tokens"] += usage.get("completion_tokens", 0)
                _usage["calls"] += 1
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
            if choice.get("finish_reason") == "length" or not content.strip():
                raise RuntimeError(
                    f"truncated/empty content (finish_reason="
                    f"{choice.get('finish_reason')})")
            return content
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"model call failed after {retries} tries: {last_err}")


# ── Envelope parsing / record building ────────────────────────────────

_SECTION_RE = re.compile(r"^===\s*(META|HISTORY|CHOICE|COMPLETION)\s*===\s*$",
                         re.MULTILINE)


def parse_envelope(text: str) -> dict:
    parts = _SECTION_RE.split(text)
    # parts: [pre, name, body, name, body, ...]
    sections = {}
    for i in range(1, len(parts) - 1, 2):
        sections[parts[i]] = parts[i + 1].strip()
    missing = {"META", "HISTORY", "CHOICE", "COMPLETION"} - set(sections)
    if missing:
        raise ValueError(f"envelope missing sections: {sorted(missing)}")
    meta_text = sections["META"].strip()
    meta_text = re.sub(r"^```(json)?|```$", "", meta_text, flags=re.MULTILINE).strip()
    meta = json.loads(meta_text)

    def _strip_narration_label(text: str) -> str:
        # Engine narration is bare prose; "Narration:" labels must never
        # reach training targets or they leak into gameplay output.
        return re.sub(r"(?m)^\s*Narration:\s*", "", text)

    return {"meta": meta,
            "history": _strip_narration_label(sections["HISTORY"]),
            "choice": sections["CHOICE"],
            "completion": _strip_narration_label(sections["COMPLETION"])}


def meta_problems(meta: dict) -> list[str]:
    probs = []
    cast = meta.get("cast") or []
    if not (2 <= len(cast) <= 4):
        probs.append(f"cast size {len(cast)} not in 2-4")
    names = [c.get("name", "") for c in cast]
    if len(set(names)) != len(names) or not all(names):
        probs.append("cast names missing or duplicated")
    if meta.get("background") not in BACKGROUND_KEYS:
        probs.append(f"background {meta.get('background')!r} not a valid key")
    if meta.get("mood") not in MOOD_TAGS:
        probs.append(f"mood {meta.get('mood')!r} not a valid mood")
    present = meta.get("present") or []
    if not present or not set(present) <= set(names):
        probs.append("present empty or contains non-cast names")
    for who, expr in (meta.get("expressions") or {}).items():
        if who not in names or expr not in STANDARD_EXPRESSIONS:
            probs.append(f"bad expression entry {who}={expr}")
    player = meta.get("player_name") or ""
    if not player or player in names:
        probs.append("player_name missing or collides with cast")
    return probs


def build_record(env: dict, source: str) -> tuple[dict | None, list[str]]:
    """Construct (system,user) ourselves, validate completion. Returns
    (record, reasons) — record None when rejected."""
    meta = env["meta"]
    probs = meta_problems(meta)
    if probs:
        return None, ["meta: " + "; ".join(probs)]

    cast = {}
    for c in meta["cast"]:
        ch = Character(name=c["name"], identity_prompt="", seed=0,
                       personality=c.get("personality", ""), voice_id="",
                       role=c.get("role", "student"))
        cast[ch.name] = CastMember(character=ch)

    scene = SceneState(cast=cast)
    scene.present = list(meta["present"])
    scene.background = meta["background"]
    scene.mood = meta["mood"]
    scene.expressions = dict(meta.get("expressions") or {})
    scene.player_name = meta["player_name"]
    scene.chapter = str(meta.get("chapter") or "Chapter 1")
    try:
        scene.turn = max(1, int(meta.get("turn", 3)))
    except (TypeError, ValueError):
        scene.turn = 3

    system, user = build_full_prompt(
        scene, cast, player_choice_text=env["choice"].strip(),
        player_name=scene.player_name, history=env["history"].strip())

    completion = env["completion"].strip()
    result = validate_turn(system, completion, history=user)
    if not result.ok:
        return None, result.reasons
    record = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": completion},
        ],
        "model": MODEL,
        "meta": {"source": source, "schema_version": SCHEMA_VERSION,
                 "transform_model": MODEL, "stats": result.stats},
    }
    return record, []


# ── Chunking ──────────────────────────────────────────────────────────

def iter_chunks(in_dir: Path, chunk_chars: int, min_chars: int,
                seed: int = 7) -> list[tuple[str, str]]:
    """Split every .txt under in_dir into excerpt chunks. Returns a
    deterministically shuffled list of (source_id, chunk_text)."""
    chunks: list[tuple[str, str]] = []
    files = sorted(in_dir.rglob("*.txt"))
    if not files:
        raise SystemExit(f"no .txt files under {in_dir} — is the dataset downloaded?")
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        step = chunk_chars
        for i, start in enumerate(range(0, len(text) - min_chars, step)):
            piece = text[start:start + chunk_chars]
            if len(piece) >= min_chars:
                chunks.append((f"{f.name}#c{i}", piece))
    rng = random.Random(seed)
    rng.shuffle(chunks)
    return chunks


# ── Worker ────────────────────────────────────────────────────────────

def transform_one(source: str, chunk: str) -> tuple[dict | None, list[str], str]:
    """One chunk -> (record|None, reject_reasons, raw_output)."""
    messages = [{"role": "system", "content": TRANSFORM_SYSTEM},
                {"role": "user", "content": build_transform_user(chunk)}]
    raw = call_model(messages)
    try:
        env = parse_envelope(raw)
        record, reasons = build_record(env, source)
    except (ValueError, json.JSONDecodeError) as e:
        record, reasons = None, [f"envelope: {e}"]
    if record:
        return record, [], raw

    # Repair loop: feed reasons back, one retry, then drop.
    messages += [
        {"role": "assistant", "content": raw},
        {"role": "user", "content":
            "Your output was rejected by the validator for these reasons:\n- "
            + "\n- ".join(reasons)
            + "\nOutput the corrected FULL envelope again (all four sections, "
              "same format, no other text)."},
    ]
    raw2 = call_model(messages, temperature=0.5)
    try:
        env2 = parse_envelope(raw2)
        record2, reasons2 = build_record(env2, source)
    except (ValueError, json.JSONDecodeError) as e:
        return None, reasons + [f"repair envelope: {e}"], raw2
    if record2:
        record2["meta"]["repaired"] = True
        return record2, [], raw2
    return None, reasons + ["repair: " + "; ".join(reasons2)], raw2


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="VN transcripts -> protocol SFT data")
    ap.add_argument("--in-dir", default="data/raw_vn")
    ap.add_argument("--out", default="data/sft/transformed.jsonl")
    ap.add_argument("--rejects", default="data/sft/transformed_rejects.jsonl")
    ap.add_argument("--limit", type=int, default=100,
                    help="number of NEW accepted examples to produce")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk-chars", type=int, default=4000)
    ap.add_argument("--min-chars", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if not API_KEY:
        print("error: OPENCODE_API_KEY not set", file=sys.stderr)
        return 2

    out_path, rej_path = Path(args.out), Path(args.rejects)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_sources: set[str] = set()
    for p in (out_path, rej_path):
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    done_sources.add(json.loads(line)["meta"]["source"]
                                     if p is out_path
                                     else json.loads(line)["source"])
                except (json.JSONDecodeError, KeyError):
                    pass

    chunks = [(s, c) for s, c in
              iter_chunks(Path(args.in_dir), args.chunk_chars, args.min_chars,
                          args.seed)
              if s not in done_sources]
    print(f"chunks available: {len(chunks)} (after skipping "
          f"{len(done_sources)} already attempted)")

    accepted = rejected = 0
    write_lock = threading.Lock()
    stop = threading.Event()

    def work(item):
        if stop.is_set():
            return None
        source, chunk = item
        try:
            record, reasons, raw = transform_one(source, chunk)
        except RuntimeError as e:
            record, reasons, raw = None, [f"api: {e}"], ""
        return source, record, reasons, raw

    with open(out_path, "a", encoding="utf-8") as out_f, \
         open(rej_path, "a", encoding="utf-8") as rej_f, \
         ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, item) for item in chunks[:args.limit * 3]]
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            source, record, reasons, raw = res
            with write_lock:
                if record:
                    accepted += 1
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                else:
                    rejected += 1
                    rej_f.write(json.dumps(
                        {"source": source, "reasons": reasons,
                         "raw": raw[-3000:]}, ensure_ascii=False) + "\n")
                    rej_f.flush()
                total = accepted + rejected
                if total % 10 == 0 or accepted >= args.limit:
                    print(f"  accepted {accepted} / rejected {rejected} "
                          f"(tokens: {_usage['prompt_tokens']}p/"
                          f"{_usage['completion_tokens']}c)")
                if accepted >= args.limit:
                    stop.set()

    print(f"\ndone: accepted {accepted}, rejected {rejected}")
    print(f"usage: {_usage['calls']} calls, "
          f"{_usage['prompt_tokens']} prompt + "
          f"{_usage['completion_tokens']} completion tokens")
    print(f"output: {out_path} (+ rejects in {rej_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
