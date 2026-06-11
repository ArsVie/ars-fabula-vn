#!/usr/bin/env bash
# Deploy the working tree to the Build Small hackathon Space.
#
# Needs a valid HF token with write access to build-small-hackathon:
#   export HF_TOKEN=hf_...        (or: hf auth login)
#   bash tools/deploy_space.sh [space-id]
set -euo pipefail

SPACE_ID="${1:-build-small-hackathon/ars-fabula-vn}"
export PATH="$HOME/.local/bin:$PATH"

echo "── deploying as: $(hf auth whoami)"

hf repos create "$SPACE_ID" --type space --space-sdk gradio --exist-ok

# Request ZeroGPU hardware (flavor id is still 'zero-a10g'). If the API
# call fails, flip it manually: Space → Settings → Space hardware → ZeroGPU.
curl -sf -X POST "https://huggingface.co/api/spaces/${SPACE_ID}/hardware" \
  -H "Authorization: Bearer ${HF_TOKEN:-$(cat ~/.cache/huggingface/token)}" \
  -H "Content-Type: application/json" \
  -d '{"flavor": "zero-a10g"}' \
  && echo "── ZeroGPU hardware requested" \
  || echo "── WARNING: hardware request failed — set ZeroGPU in Space settings"

# Runtime files only: no git/venv/caches, no dev tools or tests, no
# generated sprite scratch, no NTFS alternate-data-stream droppings.
hf upload "$SPACE_ID" . . --type space \
  --exclude ".git/*" \
  --exclude ".venv/*" \
  --exclude "**/__pycache__/*" \
  --exclude ".agents/*" \
  --exclude ".claude/*" \
  --exclude "logs/*" \
  --exclude "saves/*" \
  --exclude "tools/*" \
  --exclude "tests/*" \
  --exclude "_gen_sprites.py" \
  --exclude "static/sprites/_sampler_test/*" \
  --exclude "static/sprites/generated/*" \
  --exclude "static/voice/*" \
  --exclude "*Zone.Identifier*" \
  --exclude "*PG\$Secure*" \
  --commit-message "Deploy Ars Fabula VN (ZeroGPU)"

echo "── done: https://huggingface.co/spaces/${SPACE_ID}"
