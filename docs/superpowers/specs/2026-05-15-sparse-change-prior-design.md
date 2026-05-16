# Sparse-Change Prior (Exp-6) — Design

*Date: 2026-05-15*

## Motivation

A human presses a Tetris button once and immediately knows what it does;
the JEPA needs ~100k gradient steps to reach comparable action causality.
The gap is not that the human learns the *same thing* faster — it is that
the human only has to **bind** a new label onto transformations they
already understand, while the JEPA learns world structure and
action effect together, at one slow speed, over one entangled set of
weights.

Stepping back from Tetris, a domain-general principle from how the brain
learns world models: **the change between two moments is sparse and
local.** The cortex represents the world in factored, compositional parts;
an action or motion touches only a few of those parts and leaves the rest
invariant. That factorization is the reason a single experience
generalizes — the untouched factors carry over for free.

The current JEPA does not exploit this. The predictor remaps the **entire**
latent every step (all 21×128 numbers) and the loss is uniform MSE over all
of them. It spends equal capacity on the ~95% of the representation that
did not meaningfully change and the ~5% that did.

This experiment tests whether a sparse-change prior — a penalty that forces
the predictor's change vector to touch few factors — makes the
encoder+predictor learn the action→effect mapping in **fewer steps**.

## Hypothesis

Adding a group-sparsity penalty on the predictor's change vector improves
convergence speed (cos@k and action-retrieval M1 reached per training step)
relative to the film-100k benchmark, by:

1. shrinking the predictor's effective hypothesis space — the job becomes
   "identify which few factors this action perturbs" rather than "remap
   2688 numbers"; a smaller hypothesis space needs fewer samples;
2. pressuring the encoder toward a factored latent — the only way to make
   the change block-sparse across many states and actions is to align
   semantic factors with channel blocks.

Expected secondary effects: blocks predicted as identity cannot accumulate
drift, so long-horizon cos@k should not regress (and may improve); DROP — a
genuinely global change — will *not* go sparse, which is a clean emergent
diagnostic rather than a failure.

## The loss term

Let `z` be the predictor input and `ẑ′` its output. The change is
`Δ = ẑ′ − z`. With `residual=True` (the default, used by film-100k) this is
exactly the `delta` tensor the predictor already computes internally; in
`train.py` it is recovered as `z_pred − z_in`.

Partition the `D = 128` channels into `G = 16` contiguous groups of 8. The
penalty is a **group lasso** — a sum of per-group L2 norms, which drives
*whole groups* to exactly zero rather than shrinking individual dimensions:

```
L_sparse = mean over (batch, H, tokens) of [ Σ_g ‖Δ[..., group_g]‖₂ ]
```

The same group definition (channels 0–7, 8–15, …) applies to every token,
so the penalty pressures the encoder to place the same semantic factor in
the same channel block across tokens. Grouping per-token also yields token
sparsity for free: a token the action does not touch has all 16 groups ≈ 0.

Total training loss:

```
L = L_pred + var_weight·var + cov_weight·cov + λ·L_sparse
```

With **λ = 0 this reproduces film-100k bit-for-bit** — the benchmark is a
true λ=0 anchor and the experiment is single-variable.

### Why group lasso, not plain L1

Plain L1 sparsifies individual dimensions; group lasso zeroes whole
*factors*. The factor — not the scalar dimension — is the unit of "what the
action touches," and the brain-aligned object this prior is meant to
encourage.

## Configuration

Identical to the film-100k benchmark (`results/20260514-040125_run01/`):

| field | value |
|---|---|
| buffer | `data/buffer.npz` |
| encoder | two-scale, `stride_stages=2`, `two_scale=true` → N=21 |
| predictor | FiLM, `depth=2`, `heads=4`, residual |
| steps | 100000 |
| batch_size | 256 |
| lr | 3e-4 |
| patch_dim | 128 |
| horizon_h | 4 |
| ar_weight | 0.0 |
| ema_tau | 0.99 |
| var_weight / cov_weight | 1.0 / 0.04 |
| seed | 0 |

**Only `--sparse-change-weight` (λ) changes.**

## λ sweep

λ is the one genuine unknown — an untuned weight could land in a dead zone
(too weak to bite) or fight the prediction loss (too strong). Two runs
bracket the useful regime:

| run | λ | `--out` |
|---|---|---|
| sparse-lam01 | 0.01 | `checkpoints/jepa-sparse-lam01.pt` |
| sparse-lam10 | 0.10 | `checkpoints/jepa-sparse-lam10.pt` |

film-100k (`checkpoints/jepa-exp-film-100k.pt`) is the λ=0 anchor; it is
already trained and is not re-run.

The two runs execute as parallel branch-based RunPod experiments (see the
`runpod-training-workflow` skill). Each pod has its own filesystem, so the
per-run `jepa_step{N}.pt` intermediate checkpoints do not collide.

## Metrics — convergence is the headline

`train.py` already saves `jepa_step{N}.pt` every 5000 steps. After training,
the existing `multistep_accuracy.py` and `causality_diagnostic.py` are run
on the intermediate checkpoints to produce **convergence curves** — cos@4
and M1 plotted against training steps — for each λ against film-100k. The
question "does the prior converge faster?" is answered by the curve, not a
single end-of-training number.

A new helper, `scripts/convergence_curve.py`, loops a list of checkpoints,
runs both evals on each, and writes a single JSON, so this is not 16 manual
invocations.

New `train_log.jsonl` fields, logged every `--log-every` steps:

- `sparse_loss` — the raw `L_sparse` value (before λ).
- `active_groups` — mean number of groups per token with L2 norm above a
  small threshold (1e-3), averaged over the batch. The diagnostic that
  shows whether the prior actually took.

The post-training eval JSON (`multistep_accuracy.json`) gains a per-action
active-group count, so the expected "LEFT/RIGHT/ROTATE collapse to few
groups, DROP stays dense" pattern is directly visible.

## Code changes

All changes are additive and gated behind the new λ flag (default 0), so
existing runs and checkpoints are unaffected.

### `jepa_tetris/train.py`

- Two new CLI args:
  - `--sparse-change-weight` (float, default `0.0`) — λ.
  - `--sparse-change-groups` (int, default `16`) — G; must divide
    `patch_dim`.
- New function `sparse_change_loss(delta, groups)` near `variance_loss` /
  `covariance_loss`. Input `delta` shape `(..., D)`; returns the scalar
  group-lasso penalty and the `active_groups` diagnostic.
- Wire into the teacher-forced branch (the `else` at `train.py:464`):
  compute `delta = z_pred − z_all[:, :H]`, add `λ·L_sparse` to `loss` when
  `λ > 0`.
- Add `sparse_loss` and `active_groups` to the logging record.
- Add per-action active-group counts to the post-training eval JSON.

The sparse prior applies to the teacher-forced path only (the path
film-100k uses). It is not wired into the autoregressive, counterfactual,
or local-loss branches; combining it with those is out of scope.

### `scripts/convergence_curve.py` (new)

CLI: a list of checkpoint paths (or a glob) + a buffer. For each
checkpoint, runs the multistep accuracy eval and the causality diagnostic,
collects `step`, `cos@{1,2,4,8,16}`, `M1`, `M2`, `M4`, and writes a single
JSON keyed by step. Used to build the convergence curves.

## Out of scope

- **Surprise-gated loss** — weighting transitions by prediction error or by
  how much the state changed. A complementary brain-derived idea (plasticity
  is neuromodulator-gated); recorded in `RESEARCH_ROADMAP.md` as a future
  entry, not implemented here, to keep this a clean single-variable test.
- Wiring the sparse prior into the AR / CF / local-loss training branches.
- Any change to the predictor or encoder architecture.

## Success criteria

The experiment is informative regardless of outcome:

- **Win:** at least one λ reaches a given cos@4 or M1 in fewer steps than
  film-100k (a left-shifted convergence curve), with no regression in final
  cos@16.
- **Null:** the curves overlap film-100k — the sparse prior neither helps
  nor hurts; the hypothesis that hypothesis-space shrinkage speeds
  convergence is not supported at these λ.
- **Informative loss:** both λ regress — either the penalty fights the
  prediction loss (λ too large) or the encoder resists factorization (no
  block-sparse solution exists); the `active_groups` diagnostic
  distinguishes these.

Findings are written up as Exp-6 in `docs/FINDINGS.md`.
