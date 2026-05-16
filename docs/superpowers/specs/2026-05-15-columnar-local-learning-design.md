# Columnar Encoder with Local Learning — Design

*Exp-8. A cortically-inspired encoder where each spatial column has untied
weights and is trained by its own local loss, with no gradient flowing
between columns.*

## Motivation

The current `StateEncoder` is a CNN: convolution gives it local receptive
fields (cortex-like), but the filter weights are **shared** across all
spatial locations and the whole stack is trained by a **single global
backprop** pass. The neocortex does neither — each cortical column has its
own independently-plastic synapses, and credit assignment is largely local
to a column rather than a global error signal propagated across the sheet.

This experiment tests one half of that gap: drop global backprop in the
encoder and replace it with per-column local losses. The research question
is whether locally-trained, independently-plastic columns can learn a
representation competitive with global backprop **at fixed compute**.

### Fork A vs Fork B

Two things separate a CNN encoder from cortical columns: weight sharing and
the learning rule. The experiment isolates them with a 3-way comparison:

| Run | Weights | Learning rule |
|---|---|---|
| `film-100k` (existing benchmark) | shared | global backprop |
| **Fork A** | untied (columnar) | global backprop |
| **Fork B** | untied (columnar) | per-column local loss |

The `film → A` gap measures the cost of dropping weight-sharing. The
**`A → B` gap is the headline** — the cost (or not) of dropping global
backprop, with architecture held constant.

## Success criterion

**Competitive peak accuracy at fixed compute.** Fork B is a success if, at
the same 100k-step budget as Fork A, it reaches competitive standard
metrics (cos@k, MSE@k, DROP MSE). It is *not* expected to beat `film-100k`
— Tetris dynamics are non-local and untied weights cost sample efficiency.
The point is the learning algorithm: can local learning match global
backprop on the same architecture.

## Architecture

### ColumnarEncoder

A new `ColumnarEncoder` module in `jepa_tetris/models/encoder.py`, selected
by a `--encoder-columnar` training flag. `StateEncoder` is untouched.

- **Grid.** The 20×10 board is partitioned into a **5×3 grid = 15
  columns**. Tiles are 4 board-rows tall (20 / 5). The 10-wide axis splits
  into 3 uneven tiles of widths **{3, 4, 3}**. Configurable via
  `--encoder-columnar-grid` (default `5x3`).
- **Overlapping receptive field.** Each column reads its tile plus a
  **1-cell margin** on every side, clamped at board edges. Adjacent
  columns' input regions overlap by 2 cells (V1-style overlapping RFs).
  Configurable via `--encoder-columnar-margin` (default 1).
- **Per-column conv stack (untied weights).** Each of the 15 columns owns
  its own small conv stack — no weight sharing. Held in an `nn.ModuleList`
  of 15 stacks. Each stack takes its ~6×(5–6) region, downsamples through
  stride-2 convs, and global-pools to a single `D`-dim vector
  (`D = patch_dim = 128`). Stacking the 15 vectors yields `(B, 15, D)` —
  the same token-grid contract every downstream module already expects.
- **Output:** `(B, 15, 128)`.

Gradient isolation between columns is **automatic**: each column's forward
pass touches only its own tile and its own weights, and the local loss is a
plain sum of per-column terms, so autograd confines each column's gradient
to that column. No stop-grad is needed *inside* the encoder.

### Per-column predictor heads (Fork B only)

To generate each column's local training signal, each column owns a small
**per-column FiLM predictor head**: a tiny MLP that maps
`(z_t^c, a_emb) → ẑ_{t+1}^c`. These are throwaway scaffolding (like a BYOL
projection head) — discarded after training. Held in an `nn.ModuleList`
parallel to the column stacks.

### Global predictor (unchanged module)

The existing FiLM transformer `Predictor` is reused as-is. Its
`num_patches` constructor argument follows the encoder's N (15 here) — no
architecture change. It is what the benchmark eval and planners use.

## Training

### Fork B loss

Per training step, for each column `c` (c = 0..14):

- `z_t^c = column_c(s_t)` → `(B, D)`
- `ẑ_{t+1}^c = head_c(z_t^c, a_emb)` → `(B, D)`
- target `z̄_{t+1}^c = EMA_target_column_c(s_{t+1})`, stop-grad
- `loss_c = MSE(ẑ_{t+1}^c, z̄_{t+1}^c) + var_weight·VICReg_var(z_t^c) + cov_weight·VICReg_cov(z_t^c)`

`local_loss = mean_c(loss_c)`.

**Decoupled global predictor.** The global FiLM `Predictor` trains on the
**detached** `(B, 15, D)` encoder output — teacher-forced H=4, MSE against
the EMA target, exactly as `train.py` does today. Detaching the input is
the single explicit stop-grad in the design: the predictor's gradient never
reaches the encoder, so it never couples columns.

**One backward pass.** `loss = local_loss + predictor_loss`, single
optimizer. `local_loss` updates the column stacks + per-column heads;
`predictor_loss` updates only the global predictor (its input is detached).

The EMA target encoder is an EMA copy of the `ColumnarEncoder` (τ=0.99),
mirroring the current `train.py` mechanism. It supplies both the per-column
targets and the global predictor's targets.

### Fork A loss (baseline)

Same `ColumnarEncoder`, `--local-loss` **off**. No per-column heads, no
per-column loss. The global predictor's gradient **does** flow into the
encoder — standard joint JEPA training, identical to the current `train.py`
teacher-forced path with the columnar encoder swapped in.

### Training flags (`train.py`)

- `--encoder-columnar` — use `ColumnarEncoder` instead of `StateEncoder`
- `--encoder-columnar-grid` (default `5x3`) — column grid
- `--encoder-columnar-margin` (default 1) — overlap margin in cells
- `--local-loss` — enable Fork B (per-column local loss + detached global
  predictor). Absence ⇒ Fork A. Requires `--encoder-columnar`.

All flags are persisted in the checkpoint `args` dict.
`make_encoder_from_args` and `load_jepa` are extended to reconstruct a
`ColumnarEncoder` when `encoder_columnar` is set.

## Experiment

Three runs, 100k steps, batch 256, FiLM predictor, seed 0, on the standard
mixed-exploration buffer:

1. `film-100k` — already exists, no run needed (context baseline).
2. **Fork A** — `--encoder-columnar` (untied columns, global backprop).
3. **Fork B** — `--encoder-columnar --local-loss` (untied columns, local
   loss).

Runs 2 and 3 execute on RunPod via the `runpod-training-workflow` skill
(parallel branch-based runs).

### Metrics

- `scripts/multistep_accuracy.py` — cos@{1,2,4,8,16}, MSE@k, per-action +
  DROP MSE.
- `scripts/causality_diagnostic.py` — M1/M2/M4.

Headline: peak accuracy at fixed 100k steps, **Fork A vs Fork B**.
Results written up as **Exp-8 in `docs/FINDINGS.md`**.

### Known caveats (to record in the writeup)

- Fork B carries the per-column predictor heads (<2% of params); A/B
  compute parity is approximate, not exact.
- In Fork B the global predictor chases a moving encoder early in training
  (encoder still converging via the local loss). By 100k steps the encoder
  has long since converged. Single-run, simultaneous-with-detach training;
  no separate predictor-only phase.
- `film-100k` runs at N=21 vs the columnar runs at N=15 — the context
  comparison is approximate; the primary A-vs-B comparison is exact.

## Testing (TDD)

New tests in `tests/test_models.py`:

- `ColumnarEncoder` emits `(B, 15, D)` for a `(B, 2, 20, 10)` input.
- Tile extraction + margin clamping correct at all four board edges
  (corner columns get a clipped, not wrapped or padded-wrong, region).
- **Gradient-isolation test** — the literal Fork B invariant: a loss
  computed from one column's output produces *exactly zero* gradient on
  every other column's parameters.
- **Decoupling test** — with the Fork B loss, `predictor_loss` produces
  zero gradient on encoder parameters (detach verified).
- `make_encoder_from_args` round-trips a columnar checkpoint's `args` dict
  back to an equivalent `ColumnarEncoder`.

All run under `pytest`. Baseline at branch start: 106 passing.

## Out of scope

- Detached lateral connections between columns (Approach 2) — a follow-up.
- Convergence-speed and training-memory measurements — this iteration
  measures peak accuracy at fixed compute only.
- Probe / decoder / planner changes — they consume `(B, N, D)` generically;
  N=15 needs only a probe retrain, no code change.
