#!/usr/bin/env bash
set -euo pipefail

# This script runs INSIDE the pod after the git repo has already been
# cloned/updated by the docker_args bootstrap in runpod_pod.py.
# It only handles: pip install, symlinks, training, pod stop.

REPO_DIR="/workspace/jepa-tetris"
BRANCH="${JEPA_BRANCH:-main}"

echo "==> Branch: $BRANCH"
pip install -q -e "$REPO_DIR/"

# Ensure target dirs exist on volume
mkdir -p /workspace/data /workspace/checkpoints /workspace/results

# Symlink volume directories into repo so existing commands work unchanged
ln -sfn /workspace/data        "$REPO_DIR/data"
ln -sfn /workspace/checkpoints "$REPO_DIR/checkpoints"
ln -sfn /workspace/results     "$REPO_DIR/results"

cd "$REPO_DIR"

# Training parameters (set via pod env vars or fall back to defaults)
BUFFER="${JEPA_BUFFER:-data/buffer.npz}"
STEPS="${JEPA_STEPS:-50000}"
HORIZON="${JEPA_HORIZON:-4}"
RUN="${JEPA_RUN:-pod_run}"
OUT="${JEPA_OUT:-checkpoints/jepa.pt}"
EXTRA_ARGS="${JEPA_EXTRA_ARGS:-}"

echo "==> Starting JEPA training: steps=$STEPS horizon=$HORIZON buffer=$BUFFER extra='$EXTRA_ARGS'"
# shellcheck disable=SC2086
python -m jepa_tetris.train \
  --buffer "$BUFFER" \
  --steps  "$STEPS"  \
  --horizon-h "$HORIZON" \
  $EXTRA_ARGS \
  --out "$OUT" \
  --run "$RUN"

echo "==> Training complete."
if [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "==> Stopping pod..."
  runpodctl pod stop "$RUNPOD_POD_ID"
else
  echo "==> RUNPOD_POD_ID not set; skipping pod stop (not running on RunPod)."
fi
