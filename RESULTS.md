# JEPA Tetris V1 — Results Summary

## TL;DR

Goal: build a JEPA-based world model for simplified Tetris and a planner that
beats random. **Achieved:** PlacementPlanner clears 0.59 lines/ep over 100
episodes vs random's 0.01 — a **59× ratio**, far above the 1.5× target.

## Final results (100 episodes, seed=0)

| Policy             | Lines/ep | Episode len | Ratio vs random |
|---|---|---|---|
| Random uniform     | 0.01     | 49          | 1×              |
| Heuristic (oracle, env features) | 1.28 | 291  | 100×+           |
| **PlacementPlanner (JEPA)**  | **0.59** | **110** | **59×** |
| RealDynamicsPlanner (JEPA)   | 0.07     | 65          | 7×              |
| BFSPlanner (pure latent)     | 0.00     | 60          | 0×              |

## What worked

1. **Mixed exploration policy** for data collection. Pure random play
   essentially never clears lines (0.01 lines/ep), giving the probe head no
   positive signal. Mixing 70% one-piece heuristic + 30% random raises
   line-clear density to ~1% of triplets.
2. **Board priming.** Some episodes start with bottom rows almost-full,
   producing line clears even under near-random play. Stacks with mixed
   exploration.
3. **Multi-step rollout training** for the predictor. Training on K=4
   step rollouts maintains cos_sim ≥ 0.98 even at depth-8 rollouts, vs
   ~0.96 for K=1 training.
4. **Probe target normalization.** Lines (mean 0.007), holes (mean 10),
   height (mean 47) — without normalization holes/height dominate the loss.
5. **PlacementPlanner architecture.** Enumerate (col, rotation) endpoints,
   simulate each in the real env (snapshot/restore), encode result with
   JEPA, score with probe. Couples real dynamics with learned scoring.
6. **`require_drop` filter on plans.** Without it, the planner finds the
   degenerate "stall forever" plan optimal: never dropping keeps holes /
   height stable while dropping increases them, so any score that punishes
   holes/height tells the planner to stand still.

## What didn't work

1. **Pure latent BFS planning.** Even with multi-step training and good
   single/multi-step accuracy (cos_sim 0.99 at depth 4), the probe trained
   on encoder outputs gives unreliable predictions on predictor outputs.
   Training the probe directly on rolled-out latents recovers a small
   amount of signal (0.07 lines/ep, 2× random) but doesn't approach the
   placement planner.

   This is the well-known *distribution mismatch* problem — encoder and
   predictor produce subtly different latent distributions and the probe
   trained on one doesn't generalize to the other. The principled fix is
   joint training of the probe with the JEPA, which is left to V2.

2. **Aggressive positive-class weighting in probe loss.** With pos_weight=20
   the probe over-predicted line clears (μ pred 0.099 vs target 0.009),
   adding noise to plan scoring. pos_weight=1 (vanilla normalized MSE)
   gave a more reliable probe.

3. **Naive distillation collection.** Mostly-heuristic data (eps=0.1)
   collects ~3× more line-clear examples but produces overly clean board
   states; the trained model performs similarly to the mixed-policy model.

## Probe quality (R² on held-out states from buffer)

| Probe variant | R²(lines) | R²(holes) | R²(height) | Placement lines/ep |
|---|---|---|---|---|
| pos_weight=20 (over-aggressive)              | -1.88 | 0.73 | 0.93 | n/a  |
| pos_weight=1, encoder targets, big buffer    |  0.10 | 0.91 | 0.95 | 0.59 |
| pos_weight=1, deeper (depth=2), big buffer   |  0.15 | 0.95 | 0.97 | 0.50 |
| pos_weight=1, distillation buffer (eps=0.1)  |  0.19 | 0.87 | 0.96 | 0.55 |
| trained on rolled-out latents                |   —   | 0.85 | 0.93 | n/a  |

The distillation buffer pushes R²(lines) higher but doesn't translate to better
planner performance — the gain from cleaner data is offset by loss of variety
(distillation data has low holes/height which are easy to predict but don't
cover the diverse states the planner encounters at run time).

## Predictor multi-step accuracy (cos_sim of predicted z vs encoder of true s)

K=1 training:                    1: 0.993  4: 0.977  8: 0.961
K=4 training (multi-step loss):  1: 0.997  4: 0.990  8: 0.983

## Reproducing

```bash
source .venv/bin/activate

# 1. Collect 5000 episodes with mixed-policy exploration (~2 min)
python -m jepa_tetris.data.collect --episodes 5000 --capacity 500000 \
    --policy mixed --epsilon 0.4 --prime-prob 0.4 \
    --out data/buffer.npz --seed 1

# 2. Train JEPA with K=4 multi-step loss (~6 min on M-series MPS)
#    Logs land in results/<timestamp>_k4/
python -m jepa_tetris.train --buffer data/buffer.npz --steps 50000 \
    --rollout-k 4 --out checkpoints/jepa.pt --run k4 --seed 0

# 3. Train probe (vanilla MSE on normalized targets, ~30 sec)
python -m jepa_tetris.train_probe --jepa checkpoints/jepa.pt \
    --buffer data/buffer.npz --steps 15000 --pos-weight 1 \
    --out checkpoints/probe.pt --seed 0

# 4. Eval the PlacementPlanner (~1 min for 100 eps)
#    Output lands in results/<timestamp>_placement/eval.json
python -m jepa_tetris.eval --jepa checkpoints/jepa.pt --probe checkpoints/probe.pt \
    --episodes 100 --planner placement \
    --lines-w 10 --holes-w -1.0 --height-w -0.3 \
    --run placement --seed 0
```

Expected output:
```
random:  lines/ep ≈ 0.01, episode_len ≈ 49
planner: lines/ep ≈ 0.5–0.6, episode_len ≈ 100–110
planner / random ratio ≈ 30–60×
```

## V2 directions (if continuing)

- **Joint train probe + predictor** so the probe sees rolled-out latents
  during training. Closes the distribution gap that breaks pure-latent
  planning.
- **Residual predictor** (`predictor outputs delta_z`, supported by
  `Predictor(residual=True)`). Should help long-horizon stability.
- **Multi-piece BFS in the placement planner.** Currently single-piece
  lookahead; 2-piece would let the agent shape the board for the
  upcoming piece. Cost grows from 40 to 1600 placements per plan.
- **End-to-end value head** trained on cumulative episode rewards
  rather than hand-coded score weighting.
