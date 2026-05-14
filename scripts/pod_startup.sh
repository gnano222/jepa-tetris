#!/usr/bin/env bash
set -euo pipefail

# This script runs INSIDE the pod after the git repo has already been
# cloned/updated by the docker_args bootstrap in runpod_pod.py.
# It only handles: pip install, GPU compat, symlinks, training, pod stop.

REPO_DIR="/workspace/jepa-tetris"
BRANCH="${JEPA_BRANCH:-main}"

echo "==> Branch: $BRANCH"

stop_pod() {
    if [ -n "${RUNPOD_POD_ID:-}" ]; then
        echo "==> Stopping pod $RUNPOD_POD_ID..."
        runpodctl stop pod "$RUNPOD_POD_ID" || true
        sleep 30
    fi
}

# Start SSH daemon so results can be pulled via rsync at any point during training.
if [ -n "${PUBLIC_KEY:-}" ]; then
  mkdir -p ~/.ssh
  echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
  chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
fi
service ssh start || true

# Blackwell GPU (compute capability >= 10.0) requires PyTorch with cu128 support.
# Older images ship PyTorch built for sm_89/sm_90 and will silently crash on B100/B200.
GPU_CC_MAJOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | awk -F. '{print $1}' | sort -n | tail -1 || echo "0")
if [ "${GPU_CC_MAJOR:-0}" -ge "10" ]; then
    echo "==> Blackwell GPU (sm_${GPU_CC_MAJOR}x) detected — upgrading PyTorch to cu128..."
    pip install -q --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
        && echo "==> PyTorch upgrade OK" \
        || { echo "==> ERROR: PyTorch cu128 upgrade failed. CUDA runtime may be too old."; stop_pod; exit 1; }
fi

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
  --run "$RUN" &
TRAIN_PID=$!

# Watchdog: if training hasn't written a single log line within 10 minutes
# (first entry at step log_every=100), the run is stuck — kill the pod rather
# than burning GPU hours in a silent crash loop.
(
    sleep 600
    kill -0 "$TRAIN_PID" 2>/dev/null || exit 0  # already finished cleanly
    if ! find /workspace/results -name "train_log.jsonl" -size +0c 2>/dev/null | grep -q .; then
        echo "==> WATCHDOG: no training progress after 10 min — stopping pod"
        kill "$TRAIN_PID" 2>/dev/null || true
        sleep 5
        runpodctl stop pod "${RUNPOD_POD_ID:-}" 2>/dev/null || true
        sleep 30
    fi
) &
WATCHDOG_PID=$!

wait "$TRAIN_PID" || true
kill "$WATCHDOG_PID" 2>/dev/null || true

echo "==> Training complete."
stop_pod
