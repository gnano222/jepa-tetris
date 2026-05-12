#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${GITHUB_REPO:-https://github.com/YOUR_USER/jepa-tetris.git}"
REPO_DIR="/workspace/jepa-tetris"

# Pull or clone
if [ -d "$REPO_DIR/.git" ]; then
  echo "==> Pulling latest code..."
  git -C "$REPO_DIR" pull --ff-only
else
  echo "==> Cloning repo..."
  git clone "$REPO_URL" "$REPO_DIR"
fi

# Install project in editable mode (deps already in image)
pip install -q -e "$REPO_DIR/"

# Ensure target dirs exist on volume
mkdir -p /workspace/data /workspace/checkpoints /workspace/results

# Symlink volume directories into repo so existing commands work unchanged
ln -sfn /workspace/data      "$REPO_DIR/data"
ln -sfn /workspace/checkpoints "$REPO_DIR/checkpoints"
ln -sfn /workspace/results     "$REPO_DIR/results"

cd "$REPO_DIR"

# Training parameters (set via pod env vars or fall back to defaults)
BUFFER="${JEPA_BUFFER:-data/buffer.npz}"
STEPS="${JEPA_STEPS:-50000}"
HORIZON="${JEPA_HORIZON:-4}"
RUN="${JEPA_RUN:-pod_run}"
OUT="${JEPA_OUT:-checkpoints/jepa.pt}"

echo "==> Starting JEPA training: steps=$STEPS horizon=$HORIZON buffer=$BUFFER"
python -m jepa_tetris.train \
  --buffer "$BUFFER" \
  --steps  "$STEPS"  \
  --horizon-h "$HORIZON" \
  --out "$OUT" \
  --run "$RUN"

echo "==> Training complete."
if [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "==> Stopping pod..."
  runpodctl pod stop "$RUNPOD_POD_ID"
else
  echo "==> RUNPOD_POD_ID not set; skipping pod stop (not running on RunPod)."
fi
