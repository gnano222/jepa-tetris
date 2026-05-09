# JEPA Tetris — V1 POC

A Joint-Embedding Predictive Architecture (JEPA) world model for simplified
Tetris. The encoder + predictor + probe + planner pipeline learns from random
play and clears lines in a hand-coded planner — beating random ≥ 28× over
50 episodes, ≥ 59× over 100 episodes.

See [BUILD_PLAN.md](BUILD_PLAN.md) for the V1 spec and [the implementation
plan](.claude/) for design decisions.

## Headline results

100-episode evaluation, simplified Tetris (20×10 board, all 7 tetrominoes,
hard-drop semantics, max_steps=500):

| Policy             | lines/ep | episode len | ratio vs random |
|---|---|---|---|
| Random uniform     | 0.01     | 49          | 1×              |
| Heuristic (oracle) | ~1.28    | 291         | 100×+           |
| **PlacementPlanner (JEPA)** | **0.33–0.59** | **102–110** | **33–59×**      |
| RealDynamicsPlanner (JEPA) | 0.07     | 65          | 7×              |
| BFSPlanner (pure latent)   | 0.00     | 60–500      | 0×              |

PlacementPlanner uses the JEPA encoder + probe head to score real-env
simulated placements. The pure-latent BFS planner is the original "world
model" formulation but does not reach competitive performance even with
multi-step rollout training.

## Architecture

```
state ──► Encoder f_θ ──► z_t ─┐
                                ▼
                         Predictor g_φ ──► ẑ_{t+1}
action ──► Action emb ──► a_t ─┘            │
                                            ▼
                              compare with z_{t+1}
                              from EMA target encoder f_ξ (stop-gradient)
```

Components:
- **State encoder** (CNN): (B, 2, 20, 10) → (B, 64) latent, 2 channels for
  occupancy + falling piece.
- **Target encoder** (EMA copy of state encoder, τ=0.99, no gradients).
- **Action encoder**: `nn.Embedding(4, 16)`.
- **Predictor** (3-layer MLP, 256 hidden): single-step latent transition.
- **Probe head** (2-layer MLP): `z → (lines_cleared, holes, aggregate_height)`,
  trained on normalized targets.

VICReg-style variance + covariance regularizers prevent collapse
(`mean(std(z))` stays around 1.0–1.1 throughout training).

## Setup

```bash
pyenv install -s 3.13.3   # already on this machine
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
pip install -r requirements.txt
pip install -e .
```

MPS sanity check:
```bash
python -c "import torch; print('mps:', torch.backends.mps.is_available())"
```

## Quickstart

```bash
# 1. tests
pytest

# 2. collect 5000 episodes with mixed exploration policy
python -m jepa_tetris.data.collect \
    --episodes 5000 --capacity 500000 \
    --policy mixed --epsilon 0.4 --prime-prob 0.4 \
    --out data/buffer.npz --seed 0

# 3. train JEPA with 4-step rollout loss (50k steps, ~6 min on M-series MPS)
#    Artifacts (train_log.jsonl, train_args.json) land in results/<timestamp>_k4/
python -m jepa_tetris.train \
    --buffer data/buffer.npz --steps 50000 \
    --rollout-k 4 \
    --out checkpoints/jepa.pt --run k4 --seed 0

# 4. plot training curves (point at the run folder created in step 3)
python scripts/plot_loss.py \
    --log results/<timestamp>_k4/train_log.jsonl \
    --out results/<timestamp>_k4/loss_plot.png

# 5. train probe head (~30 sec)
python -m jepa_tetris.train_probe \
    --jepa checkpoints/jepa.pt --buffer data/buffer.npz \
    --pos-weight 1 --out checkpoints/probe.pt --seed 0

# 6. eval the PlacementPlanner — best JEPA-based agent
#    Output (eval.json, eval_args.json) lands in results/<timestamp>_placement/
python -m jepa_tetris.eval \
    --jepa checkpoints/jepa.pt --probe checkpoints/probe.pt \
    --episodes 100 --planner placement \
    --lines-w 10 --holes-w -1.0 --height-w -0.3 \
    --run placement --seed 0

# 7. side-by-side comparison of all planners
python scripts/compare_planners.py \
    --jepa checkpoints/jepa.pt --probe checkpoints/probe.pt --episodes 100
```

## Decoder + interactive viewer

The decoder is a post-hoc visualization probe — `z → board logits` — used to
**look at** the model's predictions, not to influence training. It's the right
tool when you want to ask "what does the model think will happen if I take
*this* action from *this* state?" The Streamlit explorer below is the easiest
way to drive it.

### Train the decoder (multi-distribution)

The decoder gets fed two latent distributions at inference time: encoder
outputs (`encoder(s)`) and predictor outputs (`predictor(z, a)` rolled out
1..K steps). Training only on encoder outputs makes rollout images blur from
step 2 onward. The training script mixes both:

```bash
# Re-collect with the v2 buffer schema (adds piece_id/rotation/(row,col)
# metadata used by the explorer; old buffers still load but the piece panel
# will be empty).
python -m jepa_tetris.data.collect \
    --episodes 5000 --capacity 500000 \
    --policy mixed --epsilon 0.4 --prime-prob 0.4 \
    --out data/buffer_v2.npz --seed 1

# Train the decoder with 50/50 encoder + predictor latents (~5 min)
python scripts/train_decoder.py \
    --jepa checkpoints/jepa.pt --buffer data/buffer_v2.npz \
    --predictor-mix 0.5 --rollout-k 4 \
    --steps 5000 --out checkpoints/decoder.pt --seed 0
```

`--predictor-mix 0.0` reproduces the legacy behavior. `--predictor-mix 1.0`
trains on predictor latents only.

### Launch the explorer

```bash
pip install -e ".[viz]"   # streamlit + plotly + umap-learn + streamlit-plotly-events
streamlit run scripts/decoder_explorer.py
```

Three modes from the sidebar:

- **Live env** — step a Tetris env with on-screen action buttons. The 4-up
  panel below the current board shows `decoder(predictor(z, a))` for each
  candidate action: a quick visual check on what the model expects to happen.
- **Buffer scrubber** — slide through any triplet in the buffer, see
  `decoder(encoder(s))`, `decoder(predictor(z, a))`, and `decoder(encoder(s_next))`
  side by side, plus the latent cosine/L2 metrics. "Run rollout" extends this
  to a K-step horizon (1–16) using either the buffer's actual actions or
  per-step picks.
- **Latent space** — UMAP projection of N buffer latents, colored by holes /
  height / lines / piece_id. Click a point to see the decoded latent and its
  nearest-neighbor real board side by side.

The first launch takes a few seconds while models load and (in latent mode)
UMAP fits; both are cached for the rest of the session.

## Key design decisions

- **Hard-drop action semantics.** `DROP` moves the piece to the lowest valid
  row and locks it; the next piece spawns. The agent freely moves and rotates
  while a piece is at spawn row.
- **Mixed exploration during collection.** Pure random play essentially never
  clears lines (~0.01 lines/ep), so the probe head has no positive signal.
  `MixedExplorationPolicy` blends random actions (epsilon=0.3–0.4) with a
  one-piece heuristic that targets the placement minimizing
  `holes − 0.3·height − 10·lines`. With this, ~0.5% of triplets contain
  line clears, giving the probe a learnable target.
- **Board priming during collection.** Some episodes start with the bottom
  rows already filled except for one column, so random play occasionally
  completes lines. Combined with mixed exploration, this lifts line-clear
  density further.
- **Multi-step rollout training.** The predictor is trained on K=4 step
  rollouts (`--rollout-k 4`) to maintain accuracy across the planning
  horizon. This drops 4-step prediction error from cos_sim 0.977 to 0.989.
- **Normalized probe targets.** Lines (mean 0.007), holes (mean 10), and
  height (mean 47) have wildly different scales. Normalizing all three
  to zero mean / unit variance during training prevents holes/height from
  dominating the loss.
- **Require-DROP filter in BFSPlanner.** Without it, the planner finds the
  degenerate "stall forever, never drop" plan optimal under the score
  function. Filter sequences to those containing at least one DROP.
- **Three planner variants** (most → least JEPA-pure):
  - `BFSPlanner` (latent): rolls out action sequences in latent space.
    Doesn't work — probe trained on encoder output doesn't transfer to
    predictor output even with cos_sim 0.99.
  - `RealDynamicsPlanner`: BFS over depth-K action sequences in the real env,
    JEPA scores the leaves.
  - `PlacementPlanner`: enumerate all (col, rotation) endpoints, simulate
    each in real env, JEPA scores the leaves. Best coverage and best
    performance.

## What works and what doesn't (at V1)

**Encoder + probe carry useful information.** R² for holes ≈ 0.90 and
aggregate_height ≈ 0.96 from a trained probe. The encoder learns clean
representations of board state.

**Single-step and short multi-step predictor accuracy is good.** With
multi-step training (K=4), the predictor maintains cos_sim ≥ 0.98 out to
8-step rollouts.

**Pure-latent multi-step planning fails.** Even with accurate latents
(cos_sim 0.989) and a working probe on encoder outputs (R²=0.90 holes),
running the probe on *predictor* outputs gives degraded predictions. The
probe doesn't generalize across the encoder/predictor distribution gap.
Training the probe directly on rolled-out latents recovers a small amount
of signal (0.07 lines/ep, 2× random) but is far from the placement
planner's 33–59× random.

**Hybrid (real env + JEPA scoring) works well.** PlacementPlanner uses the
real environment to enumerate placements (no compounding latent error) and
the JEPA probe to score them. With ~60% of the heuristic's performance
(0.59 vs ~1.28 lines/ep), it demonstrates that the learned features capture
board-quality information well enough for planning.

## V2 directions

- **Train probe inside the JEPA loop**, on rolled-out latents at multiple
  horizons, so its training distribution matches its inference distribution.
- **Larger predictor or residual predictor**, predicting `delta_z` instead
  of full `z_next`. The Predictor class supports `residual=True`; not yet
  trained with it.
- **Stronger data collection** — for example, distillation from the
  heuristic policy with light noise (epsilon=0.1) would yield much higher
  line-clear density.
- **End-to-end learned planner** — replace the hand-coded score weighting
  with a value head trained on collected episode returns.

## Layout

```
jepa_tetris/
├── env/{tetris,pieces}.py     # NumPy Tetris (hard-drop, all 7 tetrominoes)
├── data/
│   ├── replay_buffer.py       # numpy ring buffer + multi-step sampling
│   ├── exploration.py         # MixedExplorationPolicy (heuristic + random)
│   └── collect.py             # CLI for data collection (with --prime-prob, --policy)
├── models/{encoder,action_encoder,predictor,probe}.py
├── utils/{seed,device,logging}.py
├── train.py                   # JEPA training (EMA + VICReg + multi-step)
├── train_probe.py             # probe on encoder outputs
├── train_probe_rollout.py     # probe on predictor outputs (V2 experiment)
├── plan.py                    # BFSPlanner / RealDynamicsPlanner / PlacementPlanner
└── eval.py                    # planner vs random baseline
scripts/
├── plot_loss.py               # JSONL log -> PNG
├── diagnose.py                # buffer stats + probe R²
├── multistep_accuracy.py      # 1..K-step latent rollout accuracy
├── trace_planner.py           # debug a planner's decisions on a primed board
├── eval_heuristic.py          # heuristic-only baseline (oracle)
└── compare_planners.py        # side-by-side comparison of all planners
tests/                         # 44 pytest tests (env, models, planners, replay buffer)
```

## Success criteria check

From [BUILD_PLAN.md](BUILD_PLAN.md):

1. ✅ Training MSE drops and plateaus, `mean(std(z))` ≥ 0.5 throughout
   (final ~1.08 with VICReg)
2. ✅ Multi-step latent rollout cosine-sim with ground truth > 0.7
   (0.989 at depth 4, 0.983 at depth 8 with K=4 training)
3. ✅ Planner mean lines/ep ≥ 1.5× random baseline
   (PlacementPlanner: 33–59× random)

V1 success criteria met by the PlacementPlanner. The pure-latent planner
(BFSPlanner) was the original target; that variant is documented as a known
limitation and addressed in V2 directions.
