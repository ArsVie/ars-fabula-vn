#!/usr/bin/env bash
# Deploy the working tree to the Build Small hackathon Space (Docker SDK,
# paid T4-medium hardware: 16GB VRAM / 30GB RAM / 8 vCPU, $0.60/hr).
#
# Needs a valid HF token with write access to build-small-hackathon, and
# billing enabled for paid hardware:
#   export HF_TOKEN=hf_...        (or: hf auth login)
#   bash tools/deploy_space.sh [space-id]
set -euo pipefail

SPACE_ID="${1:-build-small-hackathon/ars-fabula-vn}"
export PATH="$HOME/.local/bin:$PATH"

echo "── deploying as: $(hf auth whoami)"

# sleep-time 1800: auto-pause after 30 idle minutes (~$2-5/day during the
# hack window instead of $14.40/day always-on). Cold start ≈ image pull +
# ~2-4 min of model loading.
hf repos create "$SPACE_ID" --type space --space-sdk docker \
    --flavor t4-medium --sleep-time 1800 --exist-ok

# Runtime files only: no git/venv/caches, no dev tools or tests (the
# tools/ scripts the container needs ARE included), no generated sprite
# scratch, no NTFS alternate-data-stream droppings.
hf upload "$SPACE_ID" . . --type space \
  --exclude ".git/*" \
  --exclude ".venv/*" \
  --exclude "**/__pycache__/*" \
  --exclude ".agents/*" \
  --exclude ".claude/*" \
  --exclude "logs/*" \
  --exclude "saves/*" \
  --exclude "tests/*" \
  --exclude "runs/*" \
  --exclude "data/*" \
  --exclude "tools/_*" \
  --exclude "tools/modal_train.py" \
  --exclude "tools/transform_vn_dataset.py" \
  --exclude "tools/merge_train_data.py" \
  --exclude "tools/eval_generations.py" \
  --exclude "tools/build_sft_dataset.py" \
  --exclude "CLAUDE.md" \
  --exclude "FIELD_NOTES.md" \
  --exclude "next_session.md" \
  --exclude "_gen_sprites.py" \
  --exclude "static/sprites/_sampler_test/*" \
  --exclude "static/sprites/generated/*" \
  --exclude "static/voice/*" \
  --exclude "*Zone.Identifier*" \
  --exclude "*PG\$Secure*" \
  --commit-message "Deploy Ars Fabula VN (Docker, T4)"

echo "── done: https://huggingface.co/spaces/${SPACE_ID}"
echo "── watch the build: https://huggingface.co/spaces/${SPACE_ID}?logs=build"
