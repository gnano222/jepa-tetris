#!/usr/bin/env bash
# Sweep across data-collection variants: collect -> train JEPA -> probe -> eval.
set -euo pipefail

SEED=${SEED:-1}
EPISODES=${EPISODES:-4000}
CAPACITY=${CAPACITY:-200000}
TRAIN_STEPS=${TRAIN_STEPS:-50000}
PROBE_STEPS=${PROBE_STEPS:-15000}
EVAL_N=${EVAL_N:-3000}
EVAL_BUFFER=${EVAL_BUFFER:-data/sweep/_eval.npz}
ROLLOUT_K=${ROLLOUT_K:-4}
HORIZONS=${HORIZONS:-1,2,4,8,16}

mkdir -p data/sweep checkpoints/sweep results/sweep

# ---- fixed held-out eval buffer ----
if [ ! -f "$EVAL_BUFFER" ]; then
  echo "=== building held-out eval buffer at $EVAL_BUFFER ==="
  python -m jepa_tetris.data.collect --episodes 500 --capacity 50000 \
      --policy mixed --epsilon 0.4 --prime-prob 0.4 \
      --out "$EVAL_BUFFER" --seed 999
else
  echo "=== eval buffer already exists at $EVAL_BUFFER (skipping) ==="
fi

# variant_name policy epsilon_arg prime_prob
VARIANTS=(
  "random          random     ''            0.0"
  "heuristic       heuristic  --epsilon=0.0 0.0"
  "mixed_03        mixed      --epsilon=0.3 0.0"
  "mixed_07        mixed      --epsilon=0.7 0.0"
  "random_prime    random     ''            0.4"
  "heuristic_prime heuristic  --epsilon=0.0 0.4"
  "mixed_default   mixed      --epsilon=0.4 0.4"
)

for entry in "${VARIANTS[@]}"; do
  read -r name policy eps_arg prime <<< "$entry"
  echo ""
  echo "=== variant: $name ==="
  mkdir -p "results/sweep/$name"

  if [ -f "results/sweep/$name/eval.json" ]; then
    echo "[$name] eval.json already exists; skipping variant"
    continue
  fi

  buffer_path="data/sweep/${name}.npz"
  jepa_path="checkpoints/sweep/${name}_jepa.pt"
  probe_path="checkpoints/sweep/${name}_probe.pt"

  # 1. collect
  if [ ! -f "$buffer_path" ]; then
    echo "[$name] collecting -> $buffer_path"
    if [ "$eps_arg" = "''" ] || [ -z "$eps_arg" ]; then
      python -m jepa_tetris.data.collect \
        --episodes "$EPISODES" --capacity "$CAPACITY" \
        --policy "$policy" --prime-prob "$prime" \
        --out "$buffer_path" --seed "$SEED"
    else
      python -m jepa_tetris.data.collect \
        --episodes "$EPISODES" --capacity "$CAPACITY" \
        --policy "$policy" "$eps_arg" --prime-prob "$prime" \
        --out "$buffer_path" --seed "$SEED"
    fi
  else
    echo "[$name] buffer exists; skipping collect"
  fi

  # 2. train JEPA
  if [ ! -f "$jepa_path" ]; then
    echo "[$name] training JEPA -> $jepa_path"
    python -m jepa_tetris.train \
      --buffer "$buffer_path" --steps "$TRAIN_STEPS" --rollout-k "$ROLLOUT_K" \
      --out "$jepa_path" --log-file "results/sweep/$name/train_log.jsonl" --seed 0
  else
    echo "[$name] jepa checkpoint exists; skipping train"
  fi

  # 3. train probe
  if [ ! -f "$probe_path" ]; then
    echo "[$name] training probe -> $probe_path"
    python -m jepa_tetris.train_probe \
      --jepa "$jepa_path" --buffer "$buffer_path" \
      --steps "$PROBE_STEPS" --pos-weight 1 \
      --out "$probe_path" --seed 0
  else
    echo "[$name] probe checkpoint exists; skipping train"
  fi

  # 4. diagnose buffer
  echo "[$name] diagnosing buffer"
  python scripts/diagnose.py \
    --buffer "$buffer_path" --jepa "$jepa_path" --probe "$probe_path" \
    --out "results/sweep/$name/buffer_stats.json"

  # 5. multistep eval against held-out buffer
  echo "[$name] multistep eval (held-out)"
  python scripts/multistep_accuracy.py \
    --jepa "$jepa_path" --buffer "$EVAL_BUFFER" --probe "$probe_path" \
    --horizons "$HORIZONS" --n "$EVAL_N" \
    --out "results/sweep/$name/eval.json" --seed 0
done

echo ""
echo "✅ sweep complete"
