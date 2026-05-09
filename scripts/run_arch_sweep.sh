#!/usr/bin/env bash
# Predictor architecture sweep on the winning data recipe (mixed_default).
# Holds buffer + eval buffer constant; varies only the predictor MLP.
set -euo pipefail

DATA_BUFFER=${DATA_BUFFER:-data/sweep/mixed_default.npz}
EVAL_BUFFER=${EVAL_BUFFER:-data/sweep/_eval.npz}
TRAIN_STEPS=${TRAIN_STEPS:-50000}
PROBE_STEPS=${PROBE_STEPS:-15000}
EVAL_N=${EVAL_N:-3000}
ROLLOUT_K=${ROLLOUT_K:-4}
HORIZONS=${HORIZONS:-1,2,4,8,16}

mkdir -p checkpoints/arch_sweep results/arch_sweep

if [ ! -f "$DATA_BUFFER" ]; then
  echo "ERROR: training buffer $DATA_BUFFER not found. Run scripts/run_sweep.sh first." >&2
  exit 1
fi
if [ ! -f "$EVAL_BUFFER" ]; then
  echo "ERROR: eval buffer $EVAL_BUFFER not found. Run scripts/run_sweep.sh first." >&2
  exit 1
fi

# variant_name  predictor_residual  predictor_hidden  predictor_depth
VARIANTS=(
  "baseline    no   256 2"
  "residual    yes  256 2"
  "wide        yes  512 2"
  "deep        yes  256 3"
)

for entry in "${VARIANTS[@]}"; do
  read -r name residual hidden depth <<< "$entry"
  echo ""
  echo "=== arch variant: $name (residual=$residual hidden=$hidden depth=$depth) ==="
  mkdir -p "results/arch_sweep/$name"

  if [ -f "results/arch_sweep/$name/eval.json" ]; then
    echo "[$name] eval.json exists; skipping"
    continue
  fi

  jepa_path="checkpoints/arch_sweep/${name}_jepa.pt"
  probe_path="checkpoints/arch_sweep/${name}_probe.pt"

  res_flag=""
  if [ "$residual" = "yes" ]; then
    res_flag="--predictor-residual"
  fi

  # 1. train JEPA
  if [ ! -f "$jepa_path" ]; then
    echo "[$name] training JEPA -> $jepa_path"
    python -m jepa_tetris.train \
      --buffer "$DATA_BUFFER" --steps "$TRAIN_STEPS" --rollout-k "$ROLLOUT_K" \
      --predictor-hidden "$hidden" --predictor-depth "$depth" $res_flag \
      --out "$jepa_path" --log-file "results/arch_sweep/$name/train_log.jsonl" --seed 0
  else
    echo "[$name] jepa checkpoint exists; skipping train"
  fi

  # 2. train probe (architecture-agnostic — only uses encoder)
  if [ ! -f "$probe_path" ]; then
    echo "[$name] training probe -> $probe_path"
    python -m jepa_tetris.train_probe \
      --jepa "$jepa_path" --buffer "$DATA_BUFFER" \
      --steps "$PROBE_STEPS" --pos-weight 1 \
      --out "$probe_path" --seed 0
  else
    echo "[$name] probe checkpoint exists; skipping train"
  fi

  # 3. multistep eval against held-out buffer
  echo "[$name] multistep eval (held-out)"
  python scripts/multistep_accuracy.py \
    --jepa "$jepa_path" --buffer "$EVAL_BUFFER" --probe "$probe_path" \
    --horizons "$HORIZONS" --n "$EVAL_N" \
    --out "results/arch_sweep/$name/eval.json" --seed 0
done

echo ""
echo "✅ arch sweep complete"
