# Token-Gated Sparse Predictor (Exp-8) — Design

*Date: 2026-05-17*

## Motivation

Exp-7 tested a sparse-change prior as a *loss penalty* on the predictor's
change vector `Δ = ẑ′ − z`. It failed: a group-lasso penalty on `‖Δ‖`
rewards "change less," and the optimiser won that reward by making the
encoder blind to small-footprint actions (movement) — `ẑ_LEFT = ẑ_RIGHT =
ẑ_ROTATE = z`, action retrieval collapsed. The penalty conflated "small
latent change" with "discardable," and the encoder was free to game it.

The principle behind it still holds: actions cause **sparse, local**
change, and a predictor that exploited this would have less to learn.
The fix is to make sparsity **architectural** rather than a penalty —
the predictor *structurally* can only touch a few factors, so there is
no reward to game and no degenerate escape.

Two further realisations shape this design:

1. **The encoder already provides a meaningful factorisation.** Exp-7
   sparsified over arbitrary channel groups (entangled, meaningless). The
   encoder emits **21 patch tokens** — each a board region. Sparsity over
   *tokens* is meaningful out of the box: a movement perturbs the patches
   around the piece; DROP perturbs many. No encoder change needed, and it
   generalises (a screenshot ViT gives patch tokens; a click changes few
   regions).
2. **With no penalty, there is no collapse incentive.** The encoder is
   left untouched and unpenalised — exactly as protected as film-100k,
   which keeps M1 ≈ 0.98. Architectural sparsity needs no separate
   sufficiency anchor.

## Hypothesis

Gating the predictor so that each (state, action) may change at most `k`
of the 21 tokens — copying the rest forward unchanged — improves
convergence speed and action causality relative to film-100k, because:

1. the predictor's job shrinks from "rewrite 21 tokens" to "select ≤k and
   rewrite those" — a smaller hypothesis to fit per sample;
2. the ~15+ ungated tokens are *exact* copies, so long-horizon rollouts do
   not accumulate drift on the static part of the board;
3. forcing each action to commit to a spatial footprint is itself the
   action-understanding being chased — movement footprints should be far
   smaller than DROP footprints.

## Mechanism

The FiLM predictor is unchanged up to the transformer output `seq`
(shape `(B, N, D)`, N=21). Three additions:

```
seq          = transformer(z, FiLM(a))      # (B, N, D) — unchanged
delta        = seq                          # candidate change, as today
gate_logits  = gate_head(seq)               # (B, N) — Linear(D, 1) per token
mask         = hard_top_k(gate_logits, k)   # (B, N) binary, exactly min(k,N) ones
ẑ′          = z + mask ⊙ delta             # ungated tokens copied exactly
```

- **Gate source.** `gate_logits` come from `seq`, which already encodes the
  state conditioned on the action (via FiLM). The gate is therefore
  per-state and per-action — which patches RIGHT touches depends on where
  the piece currently is.
- **`k` is a hard architectural cap.** The predictor cannot change more
  than `k` tokens. There is **no sparsity penalty** — `k` is a fixed
  constant, not a tuned loss weight, and nothing in the loss rewards
  "change less," so the Exp-7 collapse cannot recur.
- **Adaptive footprint below `k`.** Within the `k` unlocked tokens the
  predictor still controls `delta`'s magnitude; it can write a near-zero
  change to tokens it does not need, so the *actual* footprint adapts
  below `k`. `k` only caps it.
- **Straight-through gradient.** `hard_top_k` forward = hard binary mask;
  backward = gradient flows through a `sigmoid(gate_logits)` surrogate so
  every logit is shaped, including non-selected tokens:
  `mask_st = mask + sigmoid(logits) − sigmoid(logits).detach()`.
- **Per step.** The gate is recomputed at every predictor call in a
  multi-step rollout, so every static token is an exact copy at every step.
- **Degenerate anchor.** `k = N = 21` makes the mask all-ones and
  reproduces film-100k exactly (a dead `gate_head` aside).

The gate is wired into the **FiLM path only** — the path the benchmark
uses. It is not added to the extra-token or cross-attn paths.

## Configuration

Identical to the film-100k benchmark (`results/20260514-040125_run01/`):
two-scale N=21 encoder (`stride_stages=2`, `two_scale`), FiLM predictor
(`depth=2`, `heads=4`, residual), 100k steps, batch 256, lr 3e-4,
`patch_dim=128`, `horizon_h=4`, `ar_weight=0`, `ema_tau=0.99`,
`var/cov=1.0/0.04`, seed 0, `data/buffer.npz`.

**Only the token gate is added.**

## k sweep

`k` is the one new knob — a hard cap, not a loss weight, so it cannot
induce collapse, but it can be set too tight (starving DROP) or too loose
(no useful constraint). Two runs bracket it:

| run | k | branch | checkpoint |
|---|---|---|---|
| token-gate-k6 | 6 | exp-tokengate-k6 | `checkpoints/jepa-exp-tokengate-k6.pt` |
| token-gate-k10 | 10 | exp-tokengate-k10 | `checkpoints/jepa-exp-tokengate-k10.pt` |

film-100k (`checkpoints/jepa-exp-film-100k.pt`) is the k=21 anchor; it is
already trained and not re-run.

`k=6` is a tight cap (movement needs ~2-4 tokens, so 6 is comfortable for
movement but may starve DROP, which spans the piece column plus any
cleared row); `k=10` is generous. The two together reveal whether the cap
helps and where DROP's floor is.

## Infrastructure fix (mandatory before launch)

Exp-7's two parallel pods both wrote to `checkpoints/jepa.pt` because a
stale `JEPA_OUT=checkpoints/jepa.pt` and `JEPA_RUN=run01` in `.env.runpod`
overrode the per-branch defaults in `runpod_pod.py`. The intermediate
`jepa_step*.pt` series was overwritten and the convergence curve was lost.

Before launching Exp-8, **remove the `JEPA_OUT` and `JEPA_RUN` lines from
`.env.runpod`** so `runpod_pod.py`'s per-branch defaults
(`checkpoints/jepa-<branch>.pt`, run name = branch) take effect and the
two pods write to distinct paths.

## Metrics

With the infra fix in place, the planned convergence curve is obtainable.

- **Convergence curves** — `scripts/convergence_curve.py` runs the
  multistep + causality evals across each run's `jepa_step*.pt` series;
  cos@4 and M1 vs. training steps, per k, against film-100k.
- **Final multistep eval** — `multistep_accuracy.json`, cos@{1,2,4,8,16},
  MSE, per-action MSE@1.
- **Causality** — `causality_diagnostic.py`: M1, M2, M4.
- **New training-log field `live_tokens`** — mean mask sum (tokens changed)
  per step, sliced by action. The diagnostic: expect LEFT/RIGHT/ROTATE
  footprints far below DROP's.
- **Post-training eval** — per-action live-token count added to
  `multistep_accuracy.json`.

## Code changes

All additive and gated behind `--predictor-token-gate` (default off), so
existing runs and checkpoints are unaffected.

### `jepa_tetris/models/predictor.py`

- `Predictor.__init__` gains `token_gate: bool = False` and
  `token_gate_k: int = 21`; constructs `gate_head = nn.Linear(patch_dim, 1)`
  when `token_gate` is set.
- The FiLM forward branch, after computing `delta`, computes the gate,
  the straight-through top-k mask, and returns `z + mask ⊙ delta`.
- `token_gate` is only valid with `film=True` — raise otherwise.

### `jepa_tetris/train.py`

- Flags `--predictor-token-gate` (bool) and `--token-gate-k` (int,
  default 21); validation that `--token-gate-k` ≥ 1 and that
  `--predictor-token-gate` implies `--predictor-film`.
- Pass both to `Predictor(...)`.
- Log `live_tokens` per action (mean over the batch of the mask sum,
  bucketed by `actions[:, 0]`) in the teacher-forced path.
- Add per-action live-token counts to the post-training eval JSON.

### `scripts/multistep_accuracy.py`, `scripts/causality_diagnostic.py`

- The `Predictor(...)` reconstruction reads `token_gate` and
  `token_gate_k` from `ckpt["args"]` via `.get(..., default)` — old
  checkpoints (no such key) load unchanged.

## Out of scope

- Slot / object-centric encoder (Design 2) — recorded in
  `RESEARCH_ROADMAP.md` as the follow-up; not implemented here.
- Wiring the gate into the extra-token / cross-attn / AR / CF paths.
- Any encoder or loss-function change.

## Success criteria

- **Win:** a `k` reaches a given cos@4 or M1 in fewer steps than
  film-100k, with no regression in final cos@16, and `live_tokens` shows
  sensible footprints (movement ≪ DROP).
- **Null:** the curves overlap film-100k — the cap is not a useful
  constraint at these `k`.
- **Informative loss:** `k=6` regresses with a DROP-MSE@1 spike — the cap
  starved DROP; `k=10` then locates the floor.

Findings are written up as Exp-8 in `docs/FINDINGS.md` (Exp-6 is the
columnar entry, Exp-7 the sparse-change prior; Exp-8 is the next free
log number).
