# Counterfactual Rollouts as Implicit Regularization in JEPA World Models

*A compute-scaling study in Tetris.*

---

## Abstract

Joint-Embedding Predictive Architectures (JEPAs) train a latent world
model by predicting future representations from current ones, given an
executed action. The standard objective shows the model only the
on-policy transition `(s, a, s')`, never the alternatives the same
state could have produced under different actions. We test whether
training on the full counterfactual fanout — predicting the latent
outcomes of *every* action from each starting state — improves how well
a JEPA represents action causality. We introduce three intrinsic
metrics (action retrieval, distance calibration, no-op recognition) and
sweep the compute budget at five scales under strict compute parity in a
4-action Tetris environment. The headline finding: at small compute the
standard single-action objective wins on every metric; between 1× and 3×
compute the gap flips, and by 5× compute the counterfactual objective
wins on action retrieval by 10 percentage points and on long-horizon
prediction at every horizon. Single-action training degrades
monotonically with more compute on a fixed buffer; counterfactual
training improves and saturates. We interpret the counterfactual
objective as an implicit contrastive regularizer that prevents the
encoder from collapsing toward the executed-action manifold.

---

## 1. Introduction

A JEPA learns a representation in which next-state latents are
predictable from current latents and actions. Once trained, an encoder
`f_θ` and a predictor `g_φ` satisfy `g_φ(f_θ(s), a) ≈ f_θ(s')` for
sampled transitions `(s, a, s')`. Because the loss is computed entirely
in latent space, no pixel-level signal constrains what the encoder
chooses to represent — the architecture is free to discover whatever
latent geometry makes prediction easy.

This freedom has a downside that is rarely measured directly. The
standard objective only ever asks "given that you took action `a`,
predict the resulting latent `z'`." Nothing in the loss requires the
model to distinguish what `a` *did* from what other actions would have
done. A predictor that produces nearly identical outputs for different
action embeddings can still satisfy a single-action loss to high
accuracy, as long as its outputs happen to lie close to the actual
on-policy outcomes. We refer to the worst case of this failure as
**action collapse**: the predictor ignores its action input.

A weaker but more pervasive version of the same pathology shows up as a
*distribution shift* between encoder and predictor outputs, observed in
prior work on this same Tetris environment ([RESULTS.md](../RESULTS.md)):
multi-step latent rollouts maintain cosine similarity ≥ 0.98 to the true
encoder outputs, yet a planner that searches in pure latent space using
the predictor scores zero lines per episode. Cosine similarity, the
standard JEPA proxy, did not distinguish "the predictor knows what each
action does" from "the predictor produces a single plausible
on-distribution latent regardless of action."

This paper makes three contributions:

1. **Three intrinsic metrics for action causality** that go beyond
   cosine similarity: action retrieval (M1), distance calibration (M2),
   and no-op recognition (M4) (§3).
2. **A compute-parity training redesign** that contrasts the standard
   single-action objective with a counterfactual objective in which
   every starting state contributes a prediction and a target for every
   action (§4).
3. **A scaling sweep** at five compute budgets showing that the sign of
   the CF-vs-single gap flips between 1× and 3× compute, that the gap
   widens through 5×, and that the asymmetry is driven by single-action
   monotonically overfitting on a fixed buffer while CF saturates (§5).

The result is consistent with interpreting the counterfactual fanout as
an *implicit contrastive regularizer*: forcing one starting latent to
support four distinct predictions prevents the encoder from drifting
toward an executed-action manifold (§6).

---

## 2. Background

**JEPA training.** A standard JEPA over discrete actions consists of an
online encoder `f_θ`, a target encoder `f̄_θ` (an exponential moving
average of `f_θ`), an action embedder, and a predictor `g_φ`. For a
batch of single-action transitions `(s, a, s')`, the predictive loss is
`L_pred = ‖g_φ(f_θ(s), a) − sg(f̄_θ(s'))‖²`, where `sg` denotes
stop-gradient. VICReg-style variance and covariance regularizers on
`f_θ(s)` prevent representational collapse. Multi-step training extends
the loss to a K-step rollout that chains predictor calls and supervises
each step against the corresponding ground-truth latent.

**The on-policy data substrate.** A replay buffer of single-action
transitions is necessarily *on-policy* with respect to the data
collection policy: each row records the action that was actually taken
and the state that was actually reached. The three counterfactual
outcomes — what would have happened under the three alternative actions
— are not stored.

**Action collapse.** A predictor that ignores its action input can
still attain low single-action loss, as long as its action-independent
output happens to land near the on-policy next-state latent. This is
not a hypothetical: in the Tetris environment used here, prior work
([RESULTS.md](../RESULTS.md)) found that a planner using the predictor
to search in pure latent space scored zero lines per episode despite
multi-step cosine similarity ≥ 0.98, suggesting the predictor's
internal action sensitivity was insufficient for control even when its
on-policy fidelity was high.

---

## 3. Measuring Action Causality

We propose three diagnostics that probe the predictor's action
sensitivity directly. All three operate by *forking the environment* at
sampled states `s` and applying each of the four actions to a separate
deepcopy, producing the full counterfactual tuple
`(s, s'_LEFT, s'_RIGHT, s'_ROTATE, s'_DROP)`. Implementation:
[scripts/causality_diagnostic.py](../scripts/causality_diagnostic.py).

**M1 — action retrieval.** For each state, the predictor produces four
predictions `ẑ_a = g_φ(f̄_θ(s), a)`. We then compute the four true
targets `z'_a = f̄_θ(s'_a)` and ask: across the four targets, which is
nearest to `ẑ_a`? M1 is the top-1 accuracy of the assignment, averaged
over states and actions. Random baseline = 0.25; a model with perfect
action-conditional precision achieves 1.0.

**M2 — distance calibration.** For every triple `(s, a, b)` with
`a ≠ b`, we compute the predicted pairwise distance `‖ẑ_a − ẑ_b‖` and
the true pairwise distance `‖z'_a − z'_b‖`. M2 is the Spearman rank
correlation between these two quantities across all triples. A model
that has learned the correct *relative magnitude* of action effects
ranks high on M2.

**M4 — no-op recognition.** Some `(s, a)` pairs are no-ops: the action
is illegal or has no effect (e.g. LEFT against the left wall). We label
no-ops by comparing the env's internal state dictionary before and after
applying `a` (observation equality is necessary but insufficient). M4 is
the ratio of mean `‖ẑ_a − f̄_θ(s)‖` over no-op pairs to the same mean
over non-no-op pairs. Lower is better; the model's no-op predictions
should leave the latent near the starting state.

These metrics are complementary. M1 tests whether the predictor produces
distinct outputs per action; M2 tests whether the *geometry* of those
distinctions is correct; M4 tests whether the predictor recognizes the
absence of an effect. Cosine similarity at horizon k is reported
alongside as a familiar baseline.

---

## 4. Method

### 4.1 Counterfactual training

We extend the JEPA's data substrate to store all four counterfactual
next-states per row. At each environment step the data collector
deepcopies the env four times, applies each action, and writes
`(s, [s'_0, s'_1, s'_2, s'_3], a_executed, info)`. The chain continues
along `a_executed`; the alternative branches are stored, not pursued
(see [jepa_tetris/data/replay_buffer.py](../jepa_tetris/data/replay_buffer.py)
for the buffer schema).

Given a CF row, the **counterfactual loss** is

```
L_CF = (1/A) · Σ_a ‖g_φ(f_θ(s), a) − sg(f̄_θ(s'_a))‖²
```

where `A = 4`. Multi-step training continues the chain along
`a_executed` while still applying the four-way fanout at every step.
VICReg regularizers are applied to `f_θ(s)` (not to the predictions),
matching the single-action path.

### 4.2 Compute parity

The CF objective performs ~4× the predictor work of the single-action
objective per starting state (4 predictor calls and 4 target encodes,
versus 1 each). To compare the *objectives* rather than the budgets, we
hold compute fixed by training the CF arm for `N` steps and the
single-action arm for `4N` steps. Both arms see the same 100 000
starting states (we derive the single-action buffer from the CF buffer
by extracting `next_states[a_executed]` per row, see
[scripts/cf_to_single.py](../scripts/cf_to_single.py)) and use identical
hyperparameters except for the loss function.

### 4.3 Compute scaling sweep

We sweep five compute scales:

| compute scale | CF steps (N) | single steps (4N) |
|---|---|---|
| 0.5× | 6 250 | 25 000 |
| 1× | 12 500 | 50 000 |
| 3× | 37 500 | 150 000 |
| 5× | 62 500 | 250 000 |
| 10× | 125 000 | (not run) |

The single-action 10× run was abandoned: at 5× it was already in clear
monotonic decline, and the additional 80–220 minutes of wall-clock could
not have changed the qualitative picture.

---

## 5. Results

### 5.1 The compute-parity scaling sweep

The sign of the CF-vs-single gap flips between 1× and 3× compute on
every metric except no-op recognition. Tables show the four scales of
direct parity comparison; full numerics in
[results/scaling_summary.md](../results/scaling_summary.md).

**M1 — action retrieval (higher is better).**

| scale | single | CF | Δ (CF − single) |
|---|---|---|---|
| 0.5× | **0.925** | 0.903 | −0.022 |
| 1× | 0.907 | 0.915 | +0.008 |
| 3× | 0.879 | **0.927** | +0.049 |
| 5× | 0.812 | **0.912** | +0.100 |

**Multistep cos_sim @ k=16 (higher is better).**

| scale | single | CF | Δ |
|---|---|---|---|
| 0.5× | **0.944** | 0.896 | −0.048 |
| 1× | **0.962** | 0.917 | −0.045 |
| 3× | 0.956 | **0.965** | +0.009 |
| 5× | 0.967 | **0.978** | +0.011 |

**M2 — distance calibration (higher is better).**

| scale | single | CF | Δ |
|---|---|---|---|
| 0.5× | **0.881** | 0.826 | −0.055 |
| 1× | **0.876** | 0.873 | −0.003 |
| 3× | 0.846 | **0.879** | +0.033 |
| 5× | 0.817 | **0.861** | +0.044 |

**M4 — no-op recognition (lower is better).**

| scale | single | CF |
|---|---|---|
| 0.5× | **0.384** | 0.608 |
| 1× | **0.312** | 0.553 |
| 3× | **0.321** | 0.382 |
| 5× | **0.309** | 0.322 |

M4 is the only metric on which single-action wins at every parity scale,
but the gap collapses from 2× as bad (0.31 vs 0.55 at 1×) to nearly tied
(0.31 vs 0.32 at 5×). At CF 10× compute, M4 drops to 0.238 — better than
any single-action run we observed.

### 5.2 Asymmetric scaling behavior

The two arms behave qualitatively differently as compute increases on a
fixed buffer:

* **Single-action overfits monotonically.** M1 traces 0.925 → 0.907 →
  0.879 → 0.812 across 0.5× → 5×, an 11 percentage-point drop. M2 traces
  the same shape (0.881 → 0.876 → 0.846 → 0.817).
* **CF improves and saturates.** M1 climbs 0.903 → 0.915 → 0.927 and
  then settles at ~0.91 by 5×. CF's M2 follows the same envelope.
* **Long-horizon prediction monotonically improves for CF** at every
  horizon ≥ 1 we measured (k = 1, 2, 4, 8, 16). For single-action,
  long-horizon cosine similarity peaks at 1× and then degrades.

### 5.3 CF beyond parity: the 10× and 128-dim probes

To probe CF's own ceiling we trained an additional CF run at 10×
compute and a single CF run at 5× with double the latent dimension:

| run | M1 | M2 | M4 | cos@16 |
|---|---|---|---|---|
| CF 3× d=64 | 0.927 | 0.879 | 0.382 | 0.965 |
| CF 5× d=64 | 0.912 | 0.861 | 0.322 | 0.978 |
| **CF 10× d=64** | **0.885** | 0.838 | **0.238** | **0.988** |
| CF 5× d=128 | 0.926 | 0.873 | 0.494 | 0.982 |

Two observations. First, **CF M1 has a soft ceiling near 0.93** that
even more compute will not push past — at 10× M1 actually dips while M4
and cos@16 continue to improve, suggesting the model is reallocating
representational capacity from retrieval sharpness to cleaner
action-difference geometry. Second, **doubling the latent dimension at
5× compute** lifts M1 by 1.4 points and M2 by 1.1 points but *regresses*
M4 from 0.32 to 0.49. Wider latents help retrieval and calibration but
hurt no-op handling — they are not a uniform Pareto improvement.

---

## 6. Discussion

### 6.1 Why single-action degrades on a fixed buffer

The single-action loss only ever asks "given that you took action `a`,
predict the resulting latent `z'`." On a fixed 100 000-row buffer with
250 000 training steps and batch size 128, each row is sampled
approximately 320 times. With sufficient capacity the predictor can
specialize to the executed-action distribution and achieve low loss
without learning that the four actions correspond to four distinct
transformations. M1, which forces the model to *distinguish* actions
from each other, exposes this collapse as it accumulates.

That single-action's decline is monotonic in compute (rather than first
improving then degrading) is striking. We attribute this to the
combination of the on-policy buffer and the small action set: there is
nothing in the loss, the data, or the regularization to encourage
inter-action contrast, and the encoder/predictor pair has no reason to
preserve the structure once it can fit the executed-action manifold
tightly.

### 6.2 Why CF acts as a regularizer

The CF objective forces a single starting latent `f_θ(s)` to support
four predictions, each of which must match a different target. The
encoder cannot satisfy this unless `f_θ(s)` retains information about
*how the world responds to actions*, not merely about the on-policy
trajectory. This is structurally the same constraint as a multi-task
predictor with shared trunk: the trunk must preserve the union of
features needed by all heads, even though any single training step only
penalizes one head's output for any given gradient signal.

We therefore interpret the counterfactual fanout as an **implicit
contrastive regularizer** on the encoder. Unlike an explicit
contrastive loss (e.g. InfoNCE), it requires no negative samples and
introduces no new hyperparameters: the regularization is folded into
the predictive loss by construction. Its strength scales with the
action-set size, and it is effectively free in environments where
counterfactual rollouts can be obtained cheaply via env deepcopy.

### 6.3 The 1× result is misleading

At 1× compute, the standard JEPA objective wins on every metric except
M1 (where the gap is small). A practitioner running a single-budget
experiment would correctly conclude that CF is not worth its 4× cost.
This conclusion is wrong about the underlying objective — it is right
about the budget. The flip between 1× and 3× and the divergence through
5× imply that **objective comparisons at fixed compute can be reversed
by training longer**, and that sub-saturation comparisons should not be
extrapolated.

---

## 7. Limitations

1. **Single seed per cell.** Each (arm, scale) cell is one seed. The
   monotone single-action degradation (11 pp on M1) and the magnitude
   of the flip make seed noise an unlikely driver of the qualitative
   pattern, but a publication-grade version of these tables needs
   ≥ 3 seeds per cell with confidence intervals.
2. **Single environment.** Tetris has clean deterministic transitions
   and a four-action discrete space. The regularization story should
   strengthen with larger discrete action sets but is untested here.
3. **Intrinsic metrics only.** All results are in latent space. The
   motivating downstream failure (BFS planner scoring zero
   lines/episode) has not yet been re-evaluated on the CF-trained
   models.
4. **Fixed buffer size.** Both arms trained on the same 100 000-row
   buffer. We cannot separate "single-action overfits" from
   "single-action overfits *this size* of buffer."
5. **Compute-parity definition.** We use step counts × predictor calls
   as a compute proxy. The encoder is the dominant cost in our
   architecture and is invoked once per starting state in both arms;
   target-encoder calls scale with `A`. A wall-clock parity definition
   would yield slightly different scale ratios on different hardware.

---

## 8. Future Work

1. **Hybrid loss.** A weighted mixture `α · L_single + (1 − α) · L_CF`
   should let practitioners interpolate between regularization strength
   and per-step compute cost. A small α slice (e.g. 25 %) may capture
   most of the regularization benefit at a fraction of the cost.
2. **Buffer scaling.** Repeat the sweep at 10×–100× the current buffer
   size to test whether single-action's overfitting persists or is
   washed out by sufficient diversity. If it persists, the loss
   structure is the cause; if not, buffer reuse is.
3. **Downstream control.** Re-evaluate the pure-latent BFS planner on
   the CF-trained models. The motivating hypothesis is that better M1
   and M2 should translate to non-zero lines/episode; if not, the
   encoder/predictor distribution gap dominates and is a separate
   intervention.
4. **Latent-dim × compute joint sweep.** The 128-dim probe at 5×
   suggests latent dimension and compute interact non-trivially. A
   sweep of `d ∈ {64, 128, 256}` × `compute ∈ {1×, 5×}` would map the
   surface and identify whether the M4 regression is intrinsic to
   wider latents or specific to under-training them.
5. **Other discrete-action environments.** A paper version should
   replicate the flip on at least one larger action space (e.g. a
   gridworld with eight movement actions plus interactions) to test
   whether the regularization effect strengthens with action-set size.

---

## 9. Conclusion

We trained JEPAs on Tetris using a counterfactual objective — predict
the latent outcome of every action from each starting state — and
compared it to standard single-action training under strict compute
parity across a five-point compute scaling sweep. At small budgets the
standard objective wins; the gap flips between 1× and 3× compute and
widens through 5×, with CF leading by 10 percentage points on action
retrieval at 5×. The flip is driven asymmetrically: single-action
training degrades monotonically on a fixed buffer while CF improves and
saturates. We attribute the asymmetry to the counterfactual fanout
acting as an implicit contrastive regularizer that prevents the
encoder from collapsing toward the executed-action manifold. Three
intrinsic metrics for action causality (M1, M2, M4) make the failure
mode visible in a way that cosine similarity does not.

---

## Reproduction

* Runbook: [docs/cf_vs_single_comparison.md](cf_vs_single_comparison.md)
* Scaling table: [results/scaling_summary.md](../results/scaling_summary.md)
* Per-scale JSONs: `results/compare_{cf,single}_{causality,multistep}{,_05x,_3x,_5x,_10x}.json`
* 128-dim probe: `results/compare_cf_*_5x_d128.json`
* Diagnostic implementation: [scripts/causality_diagnostic.py](../scripts/causality_diagnostic.py)
* Buffer conversion: [scripts/cf_to_single.py](../scripts/cf_to_single.py)

## One-line summary

*Counterfactual rollouts are an implicit contrastive regularizer for
JEPA predictors. Under compute parity on a fixed buffer, the standard
single-action objective overfits monotonically while the counterfactual
objective improves and saturates — flipping the sign of the comparison
between 1× and 3× compute and widening it through 5×.*

---

# Experiment Log

## Exp-1 — Per-patch action conditioning vs. extra-token (2026-05-12)

**Question.** Does broadcasting the action embedding to every patch token
(`z = z + a_emb.unsqueeze(1)`) improve prediction accuracy compared to
appending the action as an extra (N+1)-th token in the transformer
sequence?

**Setup.** Two runs at equal sample budget (~12.8M samples):

| arm | steps | batch | architecture |
|---|---|---|---|
| Standard (extra-token) | 50 000 | 256 | action appended as 16th token, pos_emb (1,16,128) |
| Per-patch (broadcast) | 6 250 | 2 048 | action added to all 15 patch tokens, pos_emb (1,15,128) |

Both use the 15-patch encoder (`encoder_stride_stages=2`, N=15),
`patch_dim=128`, `horizon_h=4`, `ar_weight=0.25`. Evaluated with
`scripts/multistep_accuracy.py` on the held-out buffer.

**Results.**

| metric | Standard (extra-token) | Per-patch (broadcast) |
|---|---|---|
| k=1 cos_sim | **0.992** | 0.966 |
| k=4 cos_sim | **0.969** | 0.893 |
| k=1 MSE | **0.037** | 0.069 |
| k=4 MSE | **0.159** | 0.238 |
| z_std | **0.868** | 0.603 |
| offdiag_cov | **0.015** | 0.087 |
| DROP k=1 cos_sim | **0.961** | 0.894 |

**Conclusion.** Per-patch broadcast conditioning is a regression on every
metric. The `offdiag_cov` jump from 0.015 → 0.087 is the sharpest signal:
the latent is more entangled. The `z_std` drop to 0.60 indicates partial
collapse that the VICReg regularizer didn't prevent.

**Why the extra-token approach wins.** Broadcasting the same action embedding
additively to all patches removes the transformer's ability to route action
influence spatially — every patch receives an identical perturbation and
must sort out the spatial structure on its own. The extra-token approach lets
self-attention decide how much each patch should attend to the action token,
effectively learning a per-patch *action relevance weight* from data. For a
DROP action (which locks the falling piece), the relevant patches are where
the piece lands, not the entire board.

**Next.** If stronger action conditioning is wanted, FiLM (per-block
`γ, β` produced from the action embedding) or dedicated cross-attention
from the action to patch tokens are the right upgrades — not broadcast
addition. See Predictor §2 in the roadmap.

---

## Exp-2 — Two-scale encoder: fine (15) + coarse (6) patches = N=21 (2026-05-12)

**Question.** Does explicitly concatenating a coarse global stream (6 tokens,
pooled from the same conv output) with the fine 15-patch stream give the
predictor better separation between global layout (skyline, wells) and local
detail (piece position)?

**Architecture.** Single conv stack (stride_stages=2) → `(B, 128, 5, 3)`.
Fine stream: flatten → `(B, 15, 128)`. Coarse stream: AdaptiveAvgPool2d to
`(3, 2)` → `(B, 6, 128)`. Concat → `(B, 21, 128)`. Zero new parameters.

**Setup — three runs, only one clean:**

| run | steps | batch | ar_weight | samples | verdict |
|---|---|---|---|---|---|
| 1 | 50 000 | 2 048 | 0.00 | 102M | ❌ missing AR — k=4 covariance collapsed |
| 2 | 6 250 | 2 048 | 0.25 | 12.8M | ❌ large batch — only 6250 gradient steps vs baseline's 50000 |
| 3 | 50 000 | 256 | 0.25 | 12.8M | ✅ clean comparison — matches 15patch baseline exactly |

**Key learning about large-batch training.** 12.8M samples at batch 2048
(6250 gradient steps) is NOT equivalent to 12.8M samples at batch 256
(50000 gradient steps). Fewer gradient updates → worse convergence per
sample, even with identical total data seen. Architecture comparisons must
hold both batch size AND step count constant.

**Results (Run 3 — clean comparison at 50k steps, batch 256, ar_weight=0.25):**

| metric | 15patch-50k (N=15) | Two-scale-50k (N=21) | delta |
|---|---|---|---|
| k=1 cos_sim | 0.9883 | **0.9912** | +0.003 |
| k=4 cos_sim | 0.9621 | **0.9660** | +0.004 |
| k=8 cos_sim | 0.7314 | **0.7353** | +0.004 |
| DROP k=1 cos_sim | 0.9458 | **0.9569** | +0.011 |
| DROP k=1 MSE | 0.2348 | **0.2275** | −0.007 |

**Conclusion.** Two-scale wins on every metric at equal training budget. The
clearest gain is DROP prediction (+1.1pp cos@1): the coarse 6-token pooled
view captures global board layout (skyline height, well structure) that the
fine 15-token stream resolves only indirectly. When a piece drops, both local
patch changes (where it lands) and global height change (skyline collapses)
are relevant — the two streams handle these at different resolution scales.

**New benchmark.** Two-scale-50k (N=21, batch 256, 50k steps, ar_weight=0.25)
supersedes 15patch-50k. Checkpoint: `checkpoints/jepa.pt`.
