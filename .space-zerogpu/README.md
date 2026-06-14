---
title: Ars Fabula VN
emoji: 🌸
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 6.16.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: An AI-driven visual novel — a small LLM writes every scene
models:
  - google/gemma-4-12b-it
  - Abiray/Anima-base-v1.0-GGUF
  - circlestone-labs/Anima
  - Bingsu/adetailer
tags:
  - build-small-hackathon
---

# 🌸 Ars Fabula VN

An AI-driven visual novel where every scene is written live by a **small
LLM**. The model narrates, voices a three-character cast, swaps
expressions and backgrounds, and offers branching choices — through a
plain-text tool-call protocol (`[TOOL: set_expression …]`) that any small
instruct model can speak. No fine-tune, no function-calling API.

**Track:** Thousand Token Wood · **Hackathon:** Build Small

## How to play

1. Press **Begin** on the title screen.
2. Read at your own pace — click / Space / Enter advances each beat.
3. Pick a choice button, or type any action you want in the free-text bar.
4. Save/Load slots persist for your session.

## What's running

| Piece | On this Space (ZeroGPU) | At home |
|---|---|---|
| Scene writer | `google/gemma-4-12b-it` (bf16) via transformers, `@spaces.GPU` | Gemma 4 26B-A4B MoE Q4_K_M GGUF on stock llama.cpp with MTP speculative decoding, on an 8 GB GPU |
| Character sprites | Live ComfyUI pipeline, **embedded in-process** under `@spaces.GPU`: face-gated base gen → expression warping → background removal | Same pipeline, ComfyUI as a subprocess over HTTP |
| Backgrounds | Preset library + live txt2img for novel keys (same embedded pipeline) | Preset + live txt2img for novel scene keys |

The engine is model-independent: the same code drives a mock model (for
tests), a llama.cpp server, or an in-process transformers model — picked
by `ARS_FABULA_BACKEND`. The casting/background pipeline is likewise
transport-independent (`ARS_FABULA_COMFY_MODE`): at home it talks to a
ComfyUI **subprocess** over HTTP; here on ZeroGPU — where a subprocess
can't hold a GPU — it drives ComfyUI's executor **in-process**, one image
per `@spaces.GPU` call (see `comfy_embed.py`). ComfyUI and its custom nodes
are vendored as git submodules under `third_party/`; the diffusion weights
are downloaded at startup. On an 8 GB card the LLM and the diffusion stack
can't co-reside, so the home setup stops/relaunches servers around image
bakes; on ZeroGPU none of that is needed — the GPU is granted per call.
There's also a Docker variant (see `Dockerfile` on the repo) that runs
the full 26B + live ComfyUI casting on a dedicated T4.

## Run it locally

```bash
pip install -r requirements.txt gradio==6.16.0
bash tools/start_gemma_server_stock.sh &   # stock llama.cpp + MTP
python app.py
ARS_FABULA_BACKEND=mock python app.py      # no model needed
```
