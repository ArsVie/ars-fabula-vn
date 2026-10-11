#!/bin/bash
# tools/space_llm_start.sh — (re)launch llama-server inside the Docker
# Space. Used at boot by start_space.sh and as ARS_FABULA_LLM_START_CMD
# by the app's VRAM-swap fallback (ensure_llm).
#
# Same validated config as tools/start_gemma_server_stock.sh, with paths
# coming from the Dockerfile ENV block. T4 notes: SM75 — if flash-attn
# misbehaves on Turing, set ARS_FABULA_LLAMA_FA=off.
set -euo pipefail

BIN="${ARS_FABULA_LLAMA_BIN:-/usr/local/bin/llama-server}"
MODEL="${ARS_FABULA_LLAMA_MODEL:-/app/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf}"
DRAFTER="${ARS_FABULA_LLAMA_DRAFTER:-/app/models/MTP/gemma-4-26B-A4B-it-Q8_0-MTP.gguf}"
FA="${ARS_FABULA_LLAMA_FA:-on}"
PORT=8110

ARGS=(-m "$MODEL")
if [ -f "$DRAFTER" ]; then
    ARGS+=(--model-draft "$DRAFTER" --spec-type draft-mtp --spec-draft-n-max 4)
fi
# --no-mmap is load-bearing: with --cpu-moe, mmap'd expert weights page on
# every token (measured 4 tok/s vs 30 on the dev machine). -t 8 = the
# Space's 8 vCPUs.
ARGS+=(-ngl 99 -ngld 99 -fa "$FA" -ctk q8_0 -ctv q8_0 -c 16384
       --cpu-moe --no-mmap -t 8 -tb 8 --reasoning off -np 1
       --host 127.0.0.1 --port "$PORT")

echo "[space-llm] launching llama-server (${MODEL##*/})"
exec "$BIN" "${ARGS[@]}"
