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
  - google/gemma-4-26B-A4B-it-qat-q4_0-unquantized
tags:
  - build-small-hackathon
---

# 🌸 Ars Fabula VN

An AI-driven visual novel where every scene is written live by a **small
local-class LLM**. The model narrates, voices a three-character cast,
swaps expressions and backgrounds, and offers branching choices — through
a plain-text tool-call protocol (`[TOOL: set_expression …]`) that any
small instruct model can speak. No fine-tune, no function-calling API.

**Track:** Thousand Token Wood · **Hackathon:** Build Small

## How to play

1. Press **Begin** on the title screen.
2. Read at your own pace — click / Space / Enter advances each beat.
3. Pick a choice button, or type any action you want in the free-text bar.
4. Save/Load slots persist for your session.

## What's running

| Piece | On this Space (ZeroGPU) | At home |
|---|---|---|
| Scene writer | Gemma 4 26B-A4B (QAT weights, 4-bit bitsandbytes) via transformers, `@spaces.GPU` | The same model as a Q4_K_M GGUF on a custom llama.cpp fork (MTP speculative decoding) on an 8 GB GPU |
| Character sprites | Curated set (pre-baked with the pipeline below) | Live ComfyUI pipeline: face-gated base gen → expression warping → background removal |
| Backgrounds | Preset library (same pipeline) | Preset + live txt2img for novel scene keys |

The engine is model-independent: the same code drives a mock model (for
tests), a llama.cpp server, or an in-process transformers model — picked
by `ARS_FABULA_BACKEND`. On an 8 GB card the LLM and the diffusion stack
can't co-reside, so the home setup stops/relaunches servers around image
bakes; on ZeroGPU none of that is needed — the GPU is granted per call.

## Run it locally

```bash
pip install -r requirements.txt gradio==6.16.0
python app.py                          # in-process model: ARS_FABULA_BACKEND=transformers
ARS_FABULA_BACKEND=mock python app.py  # no model needed
```
