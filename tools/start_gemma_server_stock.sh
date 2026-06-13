#!/bin/bash
# tools/start_gemma_server_stock.sh — Gemma 4 26B-A4B MoE + MTP on STOCK
# upstream llama.cpp (MTP support merged 2026-06-07, PR #23398).
#
# Differences from the atomic-fork launcher (start_gemma_server.sh):
#   --model-draft <assistant.gguf> --spec-type draft-mtp   (upstream flags)
#   default 16k context (the paid-Space T4 target budget; override via args)
#
# Env overrides (same names as the fork launcher):
#   ARS_FABULA_LLAMA_BIN      — llama-server binary path
#   ARS_FABULA_LLAMA_MODEL    — main model GGUF
#   ARS_FABULA_LLAMA_DRAFTER  — MTP assistant GGUF
#   ARS_FABULA_LLAMA_ARGS     — extra server args (appended, wins)
#   ARS_FABULA_BASE_URL       — sets the port from the URL (default 8110)
#
# Usage:
#   bash tools/start_gemma_server_stock.sh           # foreground
#   nohup bash tools/start_gemma_server_stock.sh &   # detached

set -euo pipefail

# NOTE: model files must live on ext4 (~/models), NOT /mnt/c — and we run
# --no-mmap. Benchmarked on this machine (26B-A4B, cpu-moe, MTP):
#   mmap from /mnt/c:  4.2 tok/s   (experts page through 9P every token)
#   mmap from ext4:    4.1 tok/s   (page-cache churn, 16 threads on P+E cores)
#   --no-mmap, -t 8:  29.9 tok/s   (vs ~20-28 on the atomic fork)
# The drafter is the UPSTREAM-converted MTP head from unsloth's GGUF repo
# (arch `gemma4-assistant`); the fork's assistant GGUF (`gemma4_assistant`)
# does NOT load on stock llama.cpp.
LLAMA_BIN="${ARS_FABULA_LLAMA_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL="${ARS_FABULA_LLAMA_MODEL:-$HOME/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf}"
DRAFTER="${ARS_FABULA_LLAMA_DRAFTER:-$HOME/models/gemma-mtp/MTP/gemma-4-26B-A4B-it-Q8_0-MTP.gguf}"

if [[ -n "${ARS_FABULA_BASE_URL:-}" ]]; then
    PORT=$(echo "$ARS_FABULA_BASE_URL" | sed -n 's|.*:\([0-9]*\)/v1.*|\1|p')
fi
PORT="${PORT:-8110}"

BASE_ARGS="-ngl 99 -ngld 99 -fa on -ctk q8_0 -ctv q8_0 -c 16384"
BASE_ARGS="$BASE_ARGS --cpu-moe --no-mmap -t 8 -tb 8 --reasoning off -np 1"
BASE_ARGS="$BASE_ARGS --host 0.0.0.0 --port $PORT"

EXTRA_ARGS="${ARS_FABULA_LLAMA_ARGS:-}"

if [ ! -f "$LLAMA_BIN" ]; then
    echo "[stock-server] ERROR: binary not found: $LLAMA_BIN" >&2
    echo "[stock-server] build with: cmake -B build -DGGML_CUDA=ON && cmake --build build -t llama-server" >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "[stock-server] ERROR: model not found: $MODEL" >&2
    exit 1
fi
if [ ! -f "$DRAFTER" ]; then
    echo "[stock-server] WARNING: drafter not found: $DRAFTER — starting without MTP" >&2
    DRAFTER=""
fi

echo "[stock-server] Starting upstream llama-server…"
echo "  Binary:  $LLAMA_BIN"
echo "  Model:   $MODEL"
echo "  Drafter: ${DRAFTER:-(none — MTP off)}"
echo "  Port:    $PORT"

CMD=("$LLAMA_BIN" "-m" "$MODEL")
if [ -n "$DRAFTER" ]; then
    CMD+=("--model-draft" "$DRAFTER" "--spec-type" "draft-mtp" "--spec-draft-n-max" "4")
fi
# shellcheck disable=SC2086
CMD+=($BASE_ARGS $EXTRA_ARGS)

exec "${CMD[@]}"
