# Comparing counterfactual vs single-action JEPA training

A controlled head-to-head between the two predictor-training objectives:

* **Single-action:** the existing `train.py` path. For each `(s, a, s')`,
  predict `z(s')` from `(z(s), a)`. One predictor call per state.
* **Counterfactual:** the `--counterfactual` path. For each `(s,
  next_states[A], a_executed)`, predict `z(s'_a)` for *every* action `a` and
  back-propagate through all four MSE terms. Four predictor calls per state.

The CF objective should give the predictor stronger structural pressure to
distinguish actions, but it costs ~4× the compute per step. The comparison
below holds *compute* equal: CF runs N steps, single-action runs 4N steps.
The only other difference between the two runs is the loss function — they
read the same transitions, the same hyperparameters, and the same seed.

## Pre-flight

What's already in place after this prep work:

* [scripts/cf_to_single.py](../scripts/cf_to_single.py) — derives a single-action
  `.npz` from a CF `.npz` (extracts `next_states[a_executed]` per row). Lets
  both training paths consume identical transitions without code changes.
* [scripts/compare_runs.py](../scripts/compare_runs.py) — reads the JSON
  outputs of `causality_diagnostic.py` and `multistep_accuracy.py` and emits
  one Markdown report comparing the runs side-by-side.
* [tests/test_cf_to_single.py](../tests/test_cf_to_single.py) — verifies the
  conversion picks the executed branch as `s_next` and preserves all info
  fields.

## Suggested parameters

| | value | notes |
|---|---|---|
| training data | 100k CF rows | one `cf_buffer.npz`, ~10 min to collect |
| eval data | 5k held-out single-action rows | different seed; needed by `multistep_accuracy.py` |
| seed | 0 | both training runs |
| batch size | 128 | both |
| latent dim | 64 | both |
| rollout K | 4 | both |
| LR | 3e-4 | both |
| EMA τ | 0.99 | both |
| **CF steps** | **N = 12 500** | the compute-equivalent point |
| **single steps** | **4N = 50 000** | so wall-clock is comparable |

## Runbook

```bash
source .venv/bin/activate

# 1. Collect 100k CF training rows (~10 min on M-series).
python -m jepa_tetris.data.collect --counterfactual --capacity 100000 \
    --episodes 5000 --policy mixed --epsilon 0.4 --prime-prob 0.4 \
    --out data/cf_train_100k.npz --seed 1

# 2. Derive the single-action view of the SAME transitions.
python scripts/cf_to_single.py \
    --in  data/cf_train_100k.npz \
    --out data/single_train_100k.npz

# 3. Collect a small held-out single-action eval buffer (different seed).
python -m jepa_tetris.data.collect --capacity 5000 --episodes 200 \
    --policy mixed --epsilon 0.4 --prime-prob 0.4 \
    --out data/eval_held_out.npz --seed 999

# 4. Train the counterfactual model (N = 12 500 steps).
python -m jepa_tetris.train --counterfactual \
    --buffer data/cf_train_100k.npz \
    --steps 12500 --rollout-k 4 --batch-size 128 --lr 3e-4 \
    --latent-dim 64 \
    --out checkpoints/jepa_cf_compare.pt --run cf_compare --seed 0

# 5. Train the single-action baseline (4N = 50 000 steps).
python -m jepa_tetris.train \
    --buffer data/single_train_100k.npz \
    --steps 50000 --rollout-k 4 --batch-size 128 --lr 3e-4 \
    --latent-dim 64 \
    --out checkpoints/jepa_single_compare.pt --run single_compare --seed 0

# 6. Causality diagnostic on both checkpoints.
python scripts/causality_diagnostic.py \
    --jepa checkpoints/jepa_cf_compare.pt --n 500 --seed 0 \
    --out results/compare_cf_causality.json
python scripts/causality_diagnostic.py \
    --jepa checkpoints/jepa_single_compare.pt --n 500 --seed 0 \
    --out results/compare_single_causality.json

# 7. Multistep latent accuracy on both checkpoints (same eval buffer).
python scripts/multistep_accuracy.py \
    --jepa checkpoints/jepa_cf_compare.pt \
    --buffer data/eval_held_out.npz --n 2000 \
    --horizons 1,2,4,8,16 --seed 0 \
    --out results/compare_cf_multistep.json
python scripts/multistep_accuracy.py \
    --jepa checkpoints/jepa_single_compare.pt \
    --buffer data/eval_held_out.npz --n 2000 \
    --horizons 1,2,4,8,16 --seed 0 \
    --out results/compare_single_multistep.json

# 8. Side-by-side comparison report.
python scripts/compare_runs.py \
    --label single  --causality results/compare_single_causality.json \
                    --multistep results/compare_single_multistep.json \
    --label cf      --causality results/compare_cf_causality.json \
                    --multistep results/compare_cf_multistep.json \
    --out results/compare_cf_vs_single.md
```

## What the report shows

* **Causality (M1/M2/M4)** — the direct measure of action distinguishability.
  M1 = top-1 retrieval, M2 = Spearman calibration, M4 = no-op vs non-no-op
  ratio. CF training's primary target.
* **Per-action M1** — exposes whether one action (typically DROP) is dragging
  down the average.
* **Multistep cos_sim / mse / z_std / off-diag cov** — predictor accuracy at
  horizons 1/2/4/8/16. Tests whether the CF objective trades raw multi-step
  accuracy for sharper action contrast (or, ideally, gets both).

## Reading the result

Three plausible outcomes, each diagnostic:

| pattern | what it means |
|---|---|
| CF wins on M1/M2 *and* multistep cos_sim | CF objective is strictly better at compute parity. Adopt it. |
| CF wins on M1/M2 only | CF buys causality at the cost of raw fidelity. Useful for planners that need to distinguish actions; less so for visual rollouts. |
| Single wins everywhere | Compute parity reveals that the CF gradient is wasted relative to seeing 4× more transitions. Stick with single-action. |

A flat tie on all metrics is also a clear answer: at this scale, CF doesn't
help — and that's worth knowing before investing more in it.
