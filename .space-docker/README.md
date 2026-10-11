---
title: Ars Fabula VN
emoji: 🌸
colorFrom: pink
colorTo: indigo
sdk: docker
app_port: 7860
suggested_hardware: t4-medium
pinned: false
license: apache-2.0
short_description: An AI-driven visual novel — a small LLM writes every scene
models:
  - unsloth/gemma-4-26B-A4B-it-GGUF
  - circlestone-labs/Anima
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

This Space runs the full home stack on a single T4 (16 GB):

| Piece | How |
|---|---|
| Scene writer | **Gemma 4 26B-A4B** Q4_K_M GGUF on stock llama.cpp, with **MTP speculative decoding** (the multi-token-prediction head merged upstream June 2026), MoE experts on CPU (`--cpu-moe`), ~5.5 GB VRAM |
| Character casting | Live **ComfyUI** pipeline: anima (Q5 GGUF UNET) base gen with a face-detector gate → per-expression FaceDetailer warping → background removal — design your own cast on the title screen |
| Backgrounds | Preset library + live anima txt2img for scene keys the writer invents |

The engine is model-independent: the same code drives a mock model (for
tests), a llama.cpp server, or an in-process transformers model (the
ZeroGPU variant) — picked by `ARS_FABULA_BACKEND`. On the dev machine's
8 GB card the LLM and diffusion can't co-reside, so the app knows how to
stop/relaunch servers around image bakes; on the T4 they fit together and
that machinery idles as a fallback.

## Run it locally

```bash
pip install -r requirements.txt gradio==6.16.0
bash tools/start_gemma_server_stock.sh &   # stock llama.cpp + MTP
python app.py
ARS_FABULA_BACKEND=mock python app.py      # no model needed
```
