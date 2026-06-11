#!/bin/bash
# tools/start_gemma_server.sh — Headless Gemma 4 26B-A4B MoE + MTP server
#
# Launches the atomic-llama-cpp-turboquant fork with the 26B MoE model,
# MTP drafter, CPU MoE offload, and no reasoning tokens.
# Override paths via environment variables:
#   ARS_FABULA_LLAMA_BIN      — llama-server binary path
#   ARS_FABULA_LLAMA_MODEL    — main model GGUF
#   ARS_FABULA_LLAMA_DRAFTER  — MTP drafter GGUF
#   ARS_FABULA_LLAMA_ARGS     — extra server args
#   ARS_FABULA_LLAMA_LD_PATH  — LD_LIBRARY_PATH for the binary
#   ARS_FABULA_BASE_URL       — sets the port from the URL (default 8110)
#
# Usage:
#   bash tools/start_gemma_server.sh          # foreground (for testing)
#   bash tools/start_gemma_server.sh &        # background
#   nohup bash tools/start_gemma_server.sh &  # detached (for VRAM swap)

set -euo pipefail

# ── Paths (env-overridable) ──────────────────────────────────────────────
LLAMA_BIN="${ARS_FABULA_LLAMA_BIN:-/mnt/c/Users/vruizes/atomic-llama-cpp-turboquant/build/bin/llama-server}"
MODEL="${ARS_FABULA_LLAMA_MODEL:-/mnt/c/Users/vruizes/Downloads/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf}"
DRAFTER="${ARS_FABULA_LLAMA_DRAFTER:-/mnt/c/Users/vruizes/Downloads/gemma-4-26B-A4B-it-assistant.Q4_K_M.gguf}"
LD_PATH="${ARS_FABULA_LLAMA_LD_PATH:-/mnt/c/Users/vruizes/atomic-llama-cpp-turboquant/build/bin}"

# Extract port from ARS_FABULA_BASE_URL, default 8110
if [[ -n "${ARS_FABULA_BASE_URL:-}" ]]; then
    PORT=$(echo "$ARS_FABULA_BASE_URL" | sed -n 's|.*:\([0-9]*\)/v1.*|\1|p')
fi
PORT="${PORT:-8110}"

# ── Args ─────────────────────────────────────────────────────────────────
# Start with base args, then append any user overrides
BASE_ARGS="--spec-type mtp --draft-block-size 3 --draft-max 8 --draft-min 0"
BASE_ARGS="$BASE_ARGS -ngl 99 -ngld 99 -fa on -ctk q8_0 -ctv q8_0 -c 32000"
BASE_ARGS="$BASE_ARGS --cpu-moe --reasoning off --host 0.0.0.0 --port $PORT"

EXTRA_ARGS="${ARS_FABULA_LLAMA_ARGS:-}"

# ── Validate ────────────────────────────────────────────────────────────
if [ ! -f "$LLAMA_BIN" ]; then
    echo "[start_gemma_server] ERROR: binary not found: $LLAMA_BIN" >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "[start_gemma_server] ERROR: model not found: $MODEL" >&2
    exit 1
fi
if [ ! -f "$DRAFTER" ]; then
    echo "[start_gemma_server] WARNING: drafter not found: $DRAFTER — starting without MTP" >&2
    DRAFTER=""
fi

# ── Launch ───────────────────────────────────────────────────────────────
export LD_LIBRARY_PATH="$LD_PATH:${LD_LIBRARY_PATH:-}"

echo "[start_gemma_server] Starting atomic-fork llama-server…"
echo "  Binary:  $LLAMA_BIN"
echo "  Model:   $MODEL"
echo "  Drafter: ${DRAFTER:-(none)}"
echo "  Port:    $PORT"
echo "  Args:    $BASE_ARGS $EXTRA_ARGS"

CMD=("$LLAMA_BIN" "-m" "$MODEL")
if [ -n "$DRAFTER" ]; then
    CMD+=("--mtp-head" "$DRAFTER")
fi
# shellcheck disable=SC2086
CMD+=($BASE_ARGS $EXTRA_ARGS)

exec "${CMD[@]}"
