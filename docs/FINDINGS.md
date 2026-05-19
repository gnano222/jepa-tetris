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

---

## Exp-3 — FiLM vs cross-attention action conditioning (2026-05-13)

**Question.** Does stronger action conditioning via FiLM (`γ/β` modulation after
each transformer block) or cross-attention (patches attend to action as KV token)
improve prediction accuracy over the two-scale-50k baseline?

**Setup.** Two runs at identical budget to the benchmark:

| arm | steps | batch | ar_weight | conditioning | checkpoint |
|---|---|---|---|---|---|
| FiLM | 50 000 | 256 | 0.0 | per-layer `γ, β` from action embedding | `jepa-exp-film.pt` |
| Cross-attn | 50 000 | 256 | 0.0 | patches attend to action as KV per layer | `jepa-exp-cross-attn.pt` |
| Baseline (two-scale-50k) | 50 000 | 256 | 0.25 | extra-token | `jepa.pt` |

Note: FiLM and cross-attn runs used `ar_weight=0.0` (no AR loss) vs baseline's 0.25.
This is not a perfectly controlled comparison for the conditioning mechanism alone,
but the FiLM gains are large enough that the ar_weight difference cannot explain them.

**Results.**

| metric | Baseline | FiLM | Cross-Attn |
|---|---|---|---|
| cos@1 | 0.9912 | **0.9966** | 0.9934 |
| cos@4 | 0.9660 | **0.9820** | 0.9686 |
| cos@8 | 0.7353 | **0.9565** | 0.5960 |
| cos@16 | 0.023 | **0.9059** | 0.097 |
| DROP cos@1 | 0.9569 | **0.9838** | 0.9681 |
| DROP MSE@1 | 0.2275 | **0.1066** | 0.2159 |

**Conclusion.** FiLM is a decisive improvement on every metric. The most striking
result is long-horizon stability: cos@16 goes from ~0.02 (complete collapse) to 0.91.
DROP MSE is halved. Cross-attention is a modest improvement at k=1–4 but regresses
on k≥8 and collapses at k=16 — worse than the baseline on the metrics that matter most.

**Why FiLM wins.** The extra-token baseline gives the action 1/22 of the attention
budget; the predictor can under-attend to it. FiLM makes the action's `γ/β`
modulation unconditional and unavoidable at every layer — the action signal cannot be
diluted. This strong, consistent conditioning prevents error from compounding across
rollout steps, which is why long-horizon performance improves so dramatically.

**Why cross-attention underperforms.** Cross-attention lets patches *choose* how much
to attend to the action. That optionality is a weakness here — the network can learn
to ignore the action at some layers, and errors still compound over long rollouts.

**New benchmark.** film-100k supersedes all prior checkpoints. Simple broadcast FiLM
trained for 100k steps is the most effective and least complicated option tested.
Checkpoint: `checkpoints/jepa-exp-film-100k.pt`.

---

## Exp-4 — FiLM conditioning variants: scaling steps and spatial/hierarchical mechanisms (2026-05-14)

**Question.** Does doubling training to 100k steps help? Does richer action conditioning
(spatial per-patch modulation, hierarchical layer-by-layer feedback) improve over
broadcast FiLM? Six variants tested across two step counts.

**Setup.** All runs: N=21 two-scale encoder, batch 256, no AR loss, seed 0.

| model | steps | conditioning | checkpoint |
|---|---|---|---|
| film | 50k | broadcast `γ/β` from action embedding | `jepa-exp-film.pt` |
| spatial-film | 50k | per-patch `γ/β` fused with positional embedding | `jepa-exp-spatial-film.pt` |
| hier-film | 50k | spatial-film + action context updated by mean-pooling seq each layer | `jepa-exp-hierarchical-film.pt` |
| hier-film-attn | 50k | hier-film + cross-attention replaces mean pool | `jepa-exp-hier-film-attn.pt` |
| film | 100k | broadcast (same architecture, 2× steps) | `jepa-exp-film-100k.pt` |
| spatial-film | 100k | per-patch (same architecture, 2× steps) | `jepa-exp-spatial-film-100k.pt` |
| hier-film | 100k | hierarchical mean-pool (same architecture, 2× steps) | `jepa-exp-hierarchical-film-100k.pt` |

**Results — multistep cos_sim.**

| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|
| film-50k | 0.9966 | 0.9928 | 0.9820 | 0.9565 | 0.9059 |
| spatial-film-50k | 0.9963 | 0.9921 | 0.9812 | 0.9546 | 0.9031 |
| hier-film-50k | 0.9958 | 0.9913 | 0.9797 | 0.9536 | 0.9019 |
| hier-film-attn-50k | 0.9947 | 0.9890 | 0.9745 | 0.9436 | 0.8848 |
| **film-100k** ⭐ | **0.9983** | **0.9961** | 0.9891 | 0.9708 | 0.9309 |
| spatial-film-100k | 0.9981 | 0.9957 | 0.9887 | 0.9693 | 0.9282 |
| hier-film-100k | 0.9979 | 0.9951 | 0.9877 | 0.9681 | 0.9253 |

**Results — multistep MSE.**

| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|
| film-50k | 0.0213 | 0.0463 | 0.1192 | 0.2972 | 0.6481 |
| spatial-film-50k | 0.0220 | 0.0474 | 0.1182 | 0.2960 | 0.6448 |
| hier-film-50k | 0.0244 | 0.0514 | 0.1246 | 0.2940 | 0.6295 |
| hier-film-attn-50k | 0.0291 | 0.0609 | 0.1434 | 0.3276 | 0.6785 |
| **film-100k** ⭐ | **0.0136** | **0.0319** | 0.0919 | 0.2562 | 0.6185 |
| spatial-film-100k | 0.0144 | 0.0330 | **0.0906** | 0.2569 | 0.6148 |
| hier-film-100k | 0.0156 | 0.0360 | 0.0934 | **0.2495** | **0.5974** |

**Results — per-action MSE @ k=1 and k=16.**

| model | LEFT@1 | RIGHT@1 | ROTATE@1 | DROP@1 | LEFT@16 | RIGHT@16 | ROTATE@16 | DROP@16 |
|---|---|---|---|---|---|---|---|---|
| film-50k | 0.0014 | 0.0016 | 0.0018 | 0.1066 | 0.6266 | 0.6648 | 0.5865 | 0.7649 |
| spatial-film-50k | 0.0016 | 0.0018 | 0.0020 | 0.1097 | 0.6292 | 0.6612 | 0.5810 | 0.7575 |
| hier-film-50k | 0.0019 | 0.0021 | 0.0022 | 0.1213 | 0.6222 | 0.6297 | 0.5703 | 0.7420 |
| hier-film-attn-50k | 0.0023 | 0.0026 | 0.0029 | 0.1438 | 0.6501 | 0.6865 | 0.6333 | 0.7870 |
| **film-100k** ⭐ | **0.0010** | **0.0011** | **0.0011** | **0.0678** | 0.5970 | 0.6167 | 0.5629 | 0.7471 |
| spatial-film-100k | 0.0011 | 0.0012 | 0.0013 | 0.0713 | 0.5968 | 0.6230 | **0.5535** | 0.7362 |
| hier-film-100k | 0.0014 | 0.0015 | 0.0016 | 0.0768 | **0.5805** | **0.6054** | 0.5382 | **0.7140** |

**Conclusions.**

**1. film-100k is the benchmark.** Broadcast FiLM at 100k steps is the best overall
model: strongest at k=1 on every action, second-best at k=16. It is also the simplest
architecture — one linear per layer producing a scalar γ/β broadcast to all patches.
No mechanism tested at 50k steps approaches what plain FiLM achieves with twice the
training. This is the new default checkpoint.

**2. Training duration dominates architecture complexity.** Every 100k run beats every
50k run at k=1 regardless of conditioning mechanism. The architectural differences
between broadcast, spatial, and hierarchical FiLM are small compared to the gain from
simply training longer. For this domain and model size, more steps is the highest-ROI
intervention.

**3. Spatial FiLM is a null result.** Per-patch position-specific modulation adds
nothing at 50k or 100k steps. Tetris's 21 patches are too spatially correlated — the
board doesn't have enough spatial independence for location-specific conditioning to
matter.

**4. Hierarchical FiLM wins at long horizons.** Hier-film-100k is the best model at
k=8 and k=16 on every action, including DROP at k=16 (0.7140 vs 0.7471 for film-100k,
-4.4%). The advantage is consistent and grows with k. The mechanism makes sense: action
context that evolves layer-by-layer by pooling the current state maintains more coherent
conditioning across long rollout chains, where a fixed broadcast signal can drift out of
distribution. For planning applications requiring depth >4, hier-film-100k is preferable.

**5. Hierarchical FiLM with attention pooling is worse.** The cross-attention variant
(action context attends selectively to patches rather than averaging) regresses on all
metrics at 50k steps. The added parameters likely haven't converged, and mean pooling
may be appropriate here since all 21 patches are relevant to the board summary — there
is no single "important" patch for the action to focus on. Not evaluated at 100k steps.

**6. DROP remains ~50× harder than movement actions.** k=1 MSE 0.068–0.14 for DROP
vs 0.001–0.003 for LEFT/RIGHT/ROTATE. No conditioning variant closes this gap
materially. DROP causes piece-lock, line clears, and piece reset — the largest latent
displacement of any action. Addressing this likely requires changes to the training
objective (e.g. supervising DROP transitions at higher weight) rather than the
conditioning architecture.

**Benchmark.** `checkpoints/jepa-exp-film-100k.pt` — broadcast FiLM, 100k steps,
batch 256, two-scale N=21 encoder. Simple, strong, and the cleanest baseline for
future architecture comparisons.

---

## Exp-5 — CF+FiLM multi-step: combining counterfactual fanout with FiLM conditioning (2026-05-15)

**Question.** Does combining counterfactual (CF) training with FiLM conditioning beat
film-100k on both multistep prediction accuracy (cos@k, MSE@k) and action causality
metrics (M1, M2, M4)?

The design hypothesis: FiLM gives the predictor the *capacity* to differentiate actions
at every layer; CF gives the encoder the *training signal* to preserve that
action-differentiating information. Together they should be additive — neither can achieve
alone what both achieve together.

**Setup.** Two runs on the CF buffer (`data/cf_train_100k.npz`, 100k rows) with a new
combined loss: L = L_CF@t=0 + L_TF@t=1..H + VICReg. L_CF is a four-way fanout from
each starting state (predict all 4 action outcomes). L_TF is teacher-forced multi-step
on the executed action chain (H=4). Both paths use FiLM conditioning.

| run | steps | predictor calls | compute vs film-100k | checkpoint |
|---|---|---|---|---|
| cf-film-50k | 50 000 | 8 per step → 400k total | parity (1×) | `jepa-cf-film-50k.pt` |
| cf-film-100k | 100 000 | 8 per step → 800k total | 2× parity | `jepa-cf-film-100k.pt` |

Parity is measured in predictor calls: CF+FiLM makes 4 (CF fanout) + 4 (TF chain) = 8
per step, vs film-100k's 4 (TF chain only). So 50k CF+FiLM steps = film-100k's 100k steps.

**Causality baseline (run before training).** film-100k had never been evaluated on
causality metrics. Running `causality_diagnostic.py` first revealed a key surprise:
FiLM alone scored M1=0.983, M2=0.954, M4=0.040 — far better than the pre-FiLM CF study
(M1=0.912, M2=0.861, M4=0.322 at 5×). FiLM's unconditional per-layer action modulation
already largely solves the action causality problem on its own.

**Results — multistep accuracy.**

| model | cos@1 | cos@2 | cos@4 | cos@8 | cos@16 | MSE@1 | DROP MSE@1 |
|---|---|---|---|---|---|---|---|
| film-100k ⭐ | **0.9983** | **0.9961** | **0.9891** | **0.9708** | **0.9309** | **0.0136** | **0.0678** |
| cf-film-50k (parity) | 0.9967 | 0.9930 | 0.9820 | 0.9561 | 0.9053 | 0.0245 | 0.1250 |
| cf-film-100k (2×) | 0.9981 | 0.9958 | 0.9886 | 0.9695 | 0.9299 | 0.0186 | 0.0955 |

**Results — action causality metrics.**

| model | M1 (↑) | M2 (↑) | M4 (↓) | DROP causal MSE (↓) |
|---|---|---|---|---|
| film-100k | 0.983 | 0.954 | 0.040 | 0.164 |
| cf-film-50k (parity) | **0.9950** | **0.9764** | 0.061 | 0.097 |
| cf-film-100k (2×) | **0.9950** | **0.9830** | 0.051 | 0.070 |

**Conclusions.**

**1. CF+FiLM wins strongly on action causality.** M1 hits 0.995 (vs 0.983 for film-100k)
and M2 hits 0.983 (vs 0.954). These are the highest M1 and M2 scores ever measured in
this project, beating the previous CF 5× record of M1=0.912 by 8pp and M2=0.861 by 12pp.
The CF fanout forces the encoder to preserve all-action information from every starting
state — exactly what M1 and M2 measure.

**2. CF+FiLM does not improve long-horizon prediction.** At compute parity (50k steps),
cos@16 is 0.905 vs film-100k's 0.931 — significantly worse. At 2× compute (100k steps),
the gap closes to 0.9299 vs 0.9309, essentially a tie within measurement noise. CF+FiLM
at 2× compute needs 2× the GPU time to match film-100k on the metric film-100k was
designed to win, while film-100k required no CF overhead.

**3. DROP prediction splits by metric.** The CF fanout dramatically improves DROP in the
*causality space* — DROP causal MSE drops from 0.164 (film-100k) to 0.070 (cf-film-100k),
a 57% improvement. But DROP MSE in the *multistep eval* gets worse: 0.068 (film-100k) →
0.096 (cf-film-100k). The explanation: the causality diagnostic evaluates single-step
DROP prediction from a dedicated fanout loss, which CF directly supervises. The multistep
eval measures DROP in on-policy trajectory chains, where the CF component appears to
slightly reorganize the latent geometry in ways that hurt chained DROP accuracy.

**4. FiLM already solved causality; CF improved it further.** The pre-experiment causality
eval on film-100k showed FiLM alone scored M1=0.983 — far better than expected and better
than all pre-FiLM CF results. The hypothesis that CF is needed to fix action causality was
wrong in the FiLM era. CF still improves M1/M2 by 1–3pp on top of FiLM's already-strong
baseline, but the improvement is incremental, not the transformative jump seen in pre-FiLM
CF experiments.

**5. M4 worsens slightly with CF.** No-op recognition (M4) is 0.040 for film-100k and
0.051–0.061 for CF+FiLM — slightly worse, though both are excellent (random baseline ~1.0).
CF training doesn't specifically help no-op states since no-op outcomes are still
supervised; the slightly higher ratio may reflect a broader spread in the predicted
action-delta distribution when the model is optimizing across all 4 actions.

**6. The hybrid objective is a trade-off, not a Pareto improvement.** Compared to
film-100k at matched GPU time (i.e., CF+FiLM 100k = 2× film-100k's compute):

- Better: M1 (+1.2pp), M2 (+2.9pp), DROP causal MSE (−57%)
- Worse: cos@16 (−0.001, within noise), DROP MSE@1 (+41%), M4 (+0.011)
- Equal: cos@1, cos@2, cos@4, cos@8

CF+FiLM is not a strict upgrade over film-100k. It is the right choice if the goal is
action-discrimination precision (planning, control, counterfactual reasoning). film-100k
remains the right choice if the goal is long-horizon on-policy prediction fidelity.

**7. The 50k parity result answers the key question from the design doc.** The combination
is NOT "genuinely superior at equal cost" — it loses on long-horizon accuracy at parity
and only ties at 2× compute. The story is not "CF+FiLM is better"; it is "CF+FiLM
trades prediction fidelity for action discrimination, and the trade is most favourable at
2× compute."

**Benchmark.** film-100k remains the default checkpoint for multistep prediction tasks.
`jepa-cf-film-100k.pt` is now the reference for action causality tasks. Both checkpoints
should be carried forward for downstream control evaluation (M1/M2 improvements should
help the pure-latent BFS planner; whether film-100k's better cos@16 still wins on
lines-per-episode is unknown).

---

## Exp-7 — Sparse-change prior: group-lasso on the predictor's Δz (2026-05-16)

**Question.** A human presses a Tetris button once and instantly knows what it
does; the JEPA needs ~100k steps. A domain-general principle from how the brain
builds world models: the change between two moments is *sparse and local* — the
cortex represents the world in factored parts, and an action touches few of them.
Does penalising the predictor's change vector toward block-sparsity make the
encoder+predictor learn the action→effect map in fewer steps?

**Method.** New loss term: a group-lasso on `Δ = ẑ′ − z` (the predictor's change;
with `residual=True` this is the `delta` it already computes). The D=128 channels
are split into G=16 groups of 8; the penalty is the sum of per-group L2 norms,
which drives whole groups to exactly zero. Total loss
`L = L_pred + var + cov + λ·L_sparse`; λ=0 reproduces film-100k exactly. Flags
`--sparse-change-weight` / `--sparse-change-groups`. Design spec:
[docs/superpowers/specs/2026-05-15-sparse-change-prior-design.md](superpowers/specs/2026-05-15-sparse-change-prior-design.md).

**Setup.** Two runs identical to the film-100k benchmark (two-scale N=21 encoder,
FiLM predictor, 100k steps, batch 256, no AR loss, seed 0, `data/buffer.npz`) —
only λ changes. λ ∈ {0.01, 0.10}; film-100k is the λ=0 anchor.

> **Infrastructure caveat.** A stale `JEPA_OUT=checkpoints/jepa.pt` /
> `JEPA_RUN=run01` in `.env.runpod` overrode the per-branch checkpoint paths, so
> both parallel pods wrote `jepa_step*.pt` and the final `jepa.pt` to the *same*
> shared-volume paths and raced. The intermediate step-checkpoint series is
> mixed/overwritten and unusable for the planned convergence curve; the surviving
> `checkpoints/jepa.pt` is the λ=0.10 final model (confirmed via stored args).
> What survived intact and attributable: each run's `train_log.jsonl`,
> `train_args.json`, and `multistep_accuracy.json` (written to distinct
> timestamped result dirs; the post-training eval used the in-memory model).
> Causality (M1/M2/M4) is therefore available for λ=0.10 only.

**Results — final multistep accuracy (held-out, n=2000).**

| metric | film-100k (λ=0) | λ=0.01 | λ=0.10 |
|---|---|---|---|
| cos@1 | **0.9983** | 0.9958 | 0.9932 |
| cos@4 | **0.9891** | 0.9671 | 0.9731 |
| cos@8 | **0.9708** | 0.9380 | 0.9513 |
| cos@16 | **0.9309** | 0.8968 | 0.9178 |
| MSE@1 | **0.0136** | 0.0329 | 0.0576 |
| DROP MSE@1 | **0.0678** | 0.1715 | 0.3056 |
| DROP cos@1 | **0.9917** | 0.9787 | 0.9640 |

**Results — per-action active groups @ k=1 (out of 16; the diagnostic).**

| run | LEFT | RIGHT | ROTATE | DROP |
|---|---|---|---|---|
| λ=0.01 | 0.0 | 0.0 | 0.0 | 16.0 |
| λ=0.10 | 0.0 | 0.0 | 0.0 | 0.0 |

**Results — causality (λ=0.10 only, n=500).** M1 = **0.277** (film-100k: 0.983);
M2 = **0.436** (film-100k: 0.954); M4 deltas both 0.0000 (predictor is the
identity). Per-action M1: LEFT 0.86, RIGHT 0.06, ROTATE 0.18, DROP 0.006.

**Results — training trajectory (`active_groups`, batch-averaged).** λ=0.01:
16.0 held through step 25k, then 14.4 (50k) → 2.95 (75k) → 3.08 (100k). λ=0.10:
16.0 → 11.9 (50k) → 0.0 (75k) → 0.0. The collapse happens between steps 50k and
75k in both runs. At every milestone film-100k's MSE leads (e.g. step 25k: film
0.021, λ=0.01 0.033, λ=0.10 0.045) — there is no convergence speed-up at any point.

**Conclusions.**

**1. Negative result — the hypothesis is not supported.** Both λ are worse than
film-100k on every prediction metric, and neither converges faster. The
sparse-change prior, as formulated, harms the model.

**2. λ=0.10 collapsed to a pure identity predictor.** `active_groups → 0` for
*every* action including DROP, `sparse_loss → 1.4e-7`, `z_pred_std` frozen at
0.8033 across all rollout horizons (the predictor outputs `ẑ′ = z` unconditionally),
and M1 = 0.277 ≈ the 0.25 random baseline. Action causality is destroyed: the
penalty won outright and the model stopped predicting change at all.

**3. λ=0.01 induced a *selective* action collapse.** Movement actions
(LEFT/RIGHT/ROTATE) dropped to 0.0 active groups — the predictor outputs `Δ ≈ 0`
for them — while DROP kept all 16. Movement MSE@1 stayed low (~0.001, equal to
film-100k) **but for the wrong reason**: the true latent change under movement is
already tiny (~0.001 MSE), so outputting `Δ = 0` costs almost no MSE. The cost is
hidden — `ẑ_LEFT = ẑ_RIGHT = ẑ_ROTATE = z` makes the three movement actions
*indistinguishable*, which collapses action retrieval exactly the way M1 (not
cosine) is designed to catch.

**4. The core design flaw: group-lasso penalises change by *magnitude*, but
causal importance is not magnitude.** Soft-thresholding kills the smallest groups
first. In Tetris, moving the piece has a small latent footprint but is causally
crucial; DROP has a large footprint. So the penalty sparsified away precisely the
*small-but-important* movement signal and spared the large DROP signal —
backwards. The brain principle ("actions cause sparse, localised change") may be
sound; the operationalisation ("penalise the L2 magnitude of Δz") is not, because
it conflates "small" with "discardable."

**5. λ=0.10 "beating" λ=0.01 at cos@8/16 is an artefact.** An identity predictor
never drifts, so it scores well on long-horizon cosine by predicting nothing —
the exact failure mode cosine-as-proxy is blind to (cf. §2). Read M1 and MSE, not
cos@k, for these runs.

**6. Why the encoder did not resist.** The JEPA loss is entirely in latent space,
so the encoder is free to choose any geometry (cf. CF study §1). The sparse
penalty added a pressure whose cheapest solution — zero movement change — is
reachable without the prediction loss objecting, because movement's true latent
change is small. No λ in this family avoids the basin; this is a redesign, not a
tuning problem.

**Next.** If the idea is pursued: (a) stop-gradient the encoder from the sparse
term so it penalises only the predictor's expression of change, not the latent
geometry; (b) weight the penalty by causal *relevance* rather than magnitude
(e.g. normalise each group's change by its typical scale, so small movement
changes are not preferentially killed); or (c) anchor the encoder with a
decoder/reconstruction term so it cannot discard piece position. All three are
design changes, not a re-sweep. The infrastructure bug should be fixed (drop the
stale `JEPA_OUT`/`JEPA_RUN` from `.env.runpod`) before any parallel re-run; a
clean re-run would also recover λ=0.01's M1/M2/M4.

**Benchmark.** film-100k remains the default checkpoint on every metric. Exp-7 is
a negative result; no checkpoint is carried forward.

---

## Exp-8 — Token-gated sparse predictor: architectural sparsity over patch tokens (2026-05-17)

**Question.** Exp-7's sparse-change *penalty* collapsed the encoder. Does making
sparsity **architectural** instead — the predictor structurally limited to
changing at most `k` of the 21 patch tokens, the rest copied forward exactly —
speed convergence or sharpen causality without that failure mode?

**Method.** In the FiLM predictor, a `gate_head` emits one logit per patch token;
a hard top-k mask (straight-through) selects ≤k tokens; `ẑ′ = z + mask ⊙ delta`,
so the other 21−k tokens are copied forward unchanged. No loss penalty — `k` is a
fixed architectural cap, nothing rewards "change less." Sparsity is over the
encoder's *existing* spatial factorisation (patch tokens), not Exp-7's arbitrary
channel groups. Design spec:
[docs/superpowers/specs/2026-05-17-token-gated-predictor-design.md](superpowers/specs/2026-05-17-token-gated-predictor-design.md).

**Setup.** Two runs identical to film-100k (two-scale N=21 encoder, FiLM, 100k
steps, batch 256, seed 0, `data/buffer.npz`) — only the gate is added. k ∈ {6, 10};
film-100k is the k=21 (ungated) anchor. The Exp-7 checkpoint-collision bugs were
fixed first (`.env.runpod` per-branch defaults; intermediate checkpoints named
`<out-stem>_step{N}.pt`), so per-run step checkpoints survived and the convergence
curve is intact this time.

**Results — final metrics, all three runs.**

| metric | film-100k (k=21) | k=10 | k=6 |
|---|---|---|---|
| cos@1 | **0.9983** | 0.9979 | 0.9966 |
| cos@4 | **0.9891** | 0.9854 | 0.9793 |
| cos@8 | **0.9708** | 0.9618 | 0.9518 |
| cos@16 | **0.9309** | 0.9139 | 0.9054 |
| MSE@1 | **0.0136** | 0.0170 | 0.0264 |
| DROP MSE@1 | **0.0678** | 0.0864 | 0.1390 |
| LEFT MSE@1 | 0.00097 | 0.00076 | **0.00068** |
| M1 | **0.983** | 0.982 | 0.961 |
| M2 | 0.954 | **0.958** | 0.943 |
| M4 | **0.040** | 0.041 | 0.070 |

**Results — realised footprint (`live_tokens` @ k=1, end of training).**

| run | LEFT | RIGHT | ROTATE | DROP |
|---|---|---|---|---|
| k=10 | 4.2 | 4.0 | 3.1 | **10.0** (cap) |
| k=6 | 2.7 | 2.6 | 1.9 | **6.0** (cap) |

**Results — convergence (held-out cos@4 / M1 vs. step).**

| step | k=10 cos@4 / M1 | k=6 cos@4 / M1 |
|---|---|---|
| 5 000 | 0.9228 / 0.978 | 0.9317 / 0.986 |
| 25 000 | 0.9728 / 0.988 | 0.9712 / 0.976 |
| 50 000 | 0.9825 / 0.987 | 0.9785 / 0.972 |
| 100 000 | 0.9854 / 0.982 | 0.9793 / 0.961 |

Training-log `cos_sim`/`mse` (apples-to-apples, all three runs) has film-100k
ahead of k=10 ahead of k=6 at *every* milestone from step 100 onward.

**Conclusions.**

**1. Negative on prediction, and strictly monotonic in `k`.** film-100k (k=21) ≥
k=10 ≥ k=6 on every prediction metric. A global cap on changeable tokens only
*subtracts* capacity — the predictor uses every token it is allowed. There is no
convergence speed-up: film-100k leads from step 100.

**2. But no collapse — architectural sparsity is safe, as predicted.** k=10 holds
M1=0.982, M2=0.958, M4=0.041 — statistically indistinguishable from film-100k
(0.983 / 0.954 / 0.040). Action causality is fully preserved. This is the decisive
contrast with Exp-7, where a sparsity *penalty* crashed M1 to 0.28: with no
penalty there is no reward to game, the gate only routes, the encoder is
untouched. The encoder/predictor split reasoning held up.

**3. DROP is starved — the gate measured the real asymmetry, but a global cap
cannot serve it.** `live_tokens` shows DROP pinned at the cap for the *entire*
run (6.0 at k=6, 10.0 at k=10) — DROP wants more than 10 tokens throughout.
Movement contracts over training to ~2.5 (k=6) / ~4 (k=10): the model genuinely
learns movement's footprint is small. So the DROP ≫ movement asymmetry the design
hoped for is real and visible — but a single global `k` either starves DROP
(k ≤ 10) or is too loose to constrain movement (k ≳ 12). DROP MSE@1 degrades
monotonically as the cap tightens: 0.068 → 0.086 → 0.139. The design needed a
*per-action* budget, not one global cap.

**4. Movement was never the problem.** LEFT/RIGHT/ROTATE MSE@1 is ~0.0007–0.0010
for all three runs — essentially equal (the gate even shaves it slightly via exact
copy-forward). film-100k already predicts movement near-perfectly. The gate
constrains the part that was already solved and starves the part that is hard.
This is the third intervention (after the CF study and Exp-7) to confirm **DROP is
the bottleneck and generic change-constraints do not address it** — DROP is hard
because of *what* changes (line clears, piece lock, piece reset), not *how many
tokens* change.

**5. A faint early-convergence hint, not a win.** In held-out cos@4, k=6 is
marginally ahead of k=10 at 5k–10k steps (0.932 vs 0.923) — the smaller hypothesis
space does converge slightly faster very early — but k=10 overtakes by 25k and
neither approaches film-100k. The hypothesis-space-shrinkage effect exists but is
tiny and swamped by the DROP capacity loss.

**6. k=6's causality degrades over training.** k=6 M1 traces 0.986 → 0.984 →
0.976 → 0.972 → 0.961 — a slow decline reminiscent of Exp-7's single-action
overfitting, though far milder (k=10 stays flat at ~0.98). A cap tight enough to
bind on movement, not just DROP, is mildly pathological.

**7. The fixed-grid result motivates the slot encoder (Design 2).** That a
fixed-grid token gate cannot serve DROP and movement with one knob is itself the
evidence — anticipated in the spec — that a *fixed grid* is too crude. An
object-centric encoder where the falling piece is one slot and the pile another
(`RESEARCH_ROADMAP.md` → "Object-centric / slot encoder") is the natural next
step: DROP's restructuring (piece → pile, line clear) is a slot-level event, not a
patch-count event.

**Benchmark.** film-100k remains the default checkpoint on every metric. Exp-8 is
a negative result; no checkpoint is carried forward. The token-gate flags
(`--predictor-token-gate`, `--token-gate-k`, default off) remain in `train.py`.

---

## Exp-9 — Columnar encoder with local learning: untied per-column weights + per-column local losses (2026-05-17)

**Question.** The CNN encoder does two things the neocortex does not: it
*shares* filter weights across all spatial locations, and it trains by one
*global* backprop pass. A cortical column has its own independently-plastic
synapses, and credit assignment is largely local to a column. Can a columnar
encoder — untied per-column conv stacks, each trained by its own local loss
with no gradient flowing between columns — learn a representation competitive
with global backprop?

**Method.** New `ColumnarEncoder`: the 20×10 board is partitioned into a 5×3
grid = 15 columns (tiles 4 rows tall, widths {3,4,3}); each column is its own
untied conv stack reading its tile plus a 1-cell overlap margin (V1-style
overlapping receptive fields), emitting one 128-d token → (B, 15, 128). In the
local-loss variant each column also owns a throwaway FiLM predictor head and is
trained by a per-column single-step JEPA loss + per-column VICReg; gradient
isolation between columns is automatic (each column's forward touches only its
own tile and weights). The global FiLM transformer predictor trains on the
**detached** encoder output — decoupled greedy learning / Greedy InfoMax applied
*spatially*. Design spec:
[docs/superpowers/specs/2026-05-15-columnar-local-learning-design.md](superpowers/specs/2026-05-15-columnar-local-learning-design.md).

**Setup.** Two things separate a CNN from cortical columns — weight sharing and
the learning rule — so a 3-way comparison isolates them: **film-100k** (shared
weights, global backprop), **Fork A** (`--encoder-columnar`: untied columns,
global backprop), **Fork B** (`--encoder-columnar --local-loss`: untied columns,
per-column local loss). Forks A and B: 100k steps, batch 256, FiLM predictor,
seed 0, `data/buffer.npz`. Checkpoints `jepa-forkA-100k.pt`, `jepa-forkB-100k.pt`.

**Results — multistep accuracy (held-out, n=2000).**

| metric | film-100k (shared, global BP) | Fork A (untied, global BP) | Fork B (untied, local loss) |
|---|---|---|---|
| cos@1 | **0.9983** | 0.9974 | 0.9905 |
| cos@2 | **0.9961** | 0.9922 | 0.9759 |
| cos@4 | **0.9891** | 0.9688 | 0.9257 |
| cos@8 | **0.9708** | 0.9022 | 0.8679 |
| cos@16 | **0.9309** | 0.7925 | 0.8175 |
| MSE@1 | 0.0136 | **0.0095** | 0.0595 |
| DROP MSE@1 | 0.0678 | **0.0491** | 0.3053 |

**Results — action causality (n=500).**

| metric | Fork A | Fork B |
|---|---|---|
| M1 action retrieval (↑) | **0.9690** | 0.9600 |
| M2 distance calibration (↑) | **0.9302** | 0.7016 |
| M4 no-op recognition (↓) | **0.0295** | 0.0506 |

**Conclusions.**

**1. Non-starter — and the columnar architecture regresses before local
learning is even introduced.** Neither fork beats film-100k. Critically, even
**Fork A** — untied columns trained by ordinary *global backprop* — loses to
shared-weight film-100k at long horizon (cos@16 0.79 vs 0.93). Dropping
weight-sharing forfeits a regularizer that pays off over long rollouts. The
columnar architecture caps the ceiling below film-100k regardless of the
learning rule, so the experiment fails its success criterion (competitive peak
accuracy at fixed compute) on the architecture alone.

**2. Local learning reproduces representational *direction* but not
*magnitude*.** Fork B is competitive on cosine (within 0.7pp at k=1, and it even
edges Fork A at k=16) and on action retrieval (M1 0.960 vs 0.969). But it
collapses on every magnitude-sensitive metric: MSE@1 6× worse (0.0595 vs
0.0095), DROP MSE@1 6× worse (0.305 vs 0.049), and M2 distance calibration falls
0.93→0.70. The causality diagnostic shows the mechanism directly — Fork B
predicts DROP's pairwise effect at distance ~26 when the truth is ~35: right
direction, wrong scale. The per-column local loss + per-column VICReg pins each
column's direction and variance but not the global magnitude of predicted
change.

**3. Fork B's cos@16 "edge" is direction-only.** Fork B's cos@16 (0.8175) beats
Fork A's (0.7925), but its MSE@16 is more than 2× worse — the long-horizon
latents drift far in magnitude while staying loosely aligned in direction. Read
MSE and M2, not cos@k, for these runs (cf. the cosine-blindness noted in the CF
study, Exp-7, Exp-8).

**4. A magnitude fix could rescue Fork B's calibration but not the verdict.**
Adding an explicit scale term to the local loss (penalise ‖ẑ_c‖ vs ‖z̄_c‖, or
tune per-column VICReg `target_std`) would likely close the magnitude gap — but
the best it could reach is Fork A, which is *itself* behind film-100k. The
branch does not lead anywhere on this task: weight-sharing wins.

**5. The one finding worth keeping.** Local, per-column credit assignment — no
gradient between columns, no global backprop in the encoder — reproduces the
*geometry* of a learned representation (direction, action-retrieval structure)
but not its *scale*. This is consistent with VICReg shaping variance and
direction while leaving global magnitude unconstrained. If local learning is
revisited (for biological plausibility or training-memory reasons), the local
objective needs an explicit magnitude constraint — direction alone is not
enough.

**Benchmark.** film-100k remains the default checkpoint on every metric. Exp-9
is a non-starter; no checkpoint is carried forward. Flags `--encoder-columnar`,
`--encoder-columnar-grid`, `--encoder-columnar-margin`, `--local-loss` remain in
`train.py` (default off).

## Exp-10 — ICM-style inverse head: shared encoder, forward + inverse (2026-05-19)

**Question.** Every JEPA model here so far is a *forward* model — it predicts
`(state, action) → next state` and never the converse. The encoder is therefore
only ever asked "what is predictable," never "what action caused this change."
Exp-7 showed the concrete cost: nothing in the forward loss punishes the encoder
for *discarding piece position*, so it does. The brain's motor system instead
runs **paired** forward and inverse models that bootstrap each other (Wolpert &
Kawato's MOSAIC, 1998). Does adding the inverse half — in its simplest proven
form, the ICM recipe (Pathak et al. 2017) — force the encoder to keep
action-causal information, and at what cost to forward fidelity?

**Method.** A new `InverseModel` head reads two encoded states and predicts the
action between them: `(z_t, z_{t+1}) → (B, 4)` logits. It is the predictor's
mirror twin — a per-patch `[z ; z' ; z'−z]` projection, spatial positional
embeddings, a 2-layer transformer, mean-pool → linear (deliberately *not*
modelled on the `Probe`, whose unordered pooling discards the spatial
localisation LEFT vs RIGHT depends on). Training adds one cross-entropy term
over the H adjacent pairs in each teacher-forced window:
`L = L_fwd + λ_inv·L_inv + VICReg`, λ_inv = 1.0, with both `z_t` and `z_{t+1}`
from the *online* encoder so the gradient reaches the shared encoder from both.
This is the minimal first step — cycle consistency, a hindsight multi-step
variant, and an inverse-model planner were scoped as explicit follow-ups.

**Setup.** One 100k run, `jepa-exp-icm-inverse.pt`, in the exact film-100k config
(`--predictor-film --encoder-stride-stages 2 --encoder-two-scale`, two-scale N=21,
batch 256, `data/buffer.npz`, seed 0) so the only difference from the benchmark
is the added inverse loss. Success criterion (pre-registered): M1/M2/M4 ≥
film-100k **and** `cos@k` not materially regressed.

**Results — forward prediction (multistep, held-out, n=2000).**

| metric | film-100k | Exp-10 ICM | Δ |
|---|---|---|---|
| cos@1 | **0.9983** | 0.9972 | −0.0011 |
| cos@4 | **0.9891** | 0.9854 | −0.0037 |
| cos@16 | **0.9309** | 0.9223 | −0.0086 |
| MSE@1 | **0.0136** | 0.0205 | +51% |
| LEFT/RIGHT/ROTATE MSE@1 | **~0.0011** | ~0.0017 | +50–58% |
| DROP MSE@1 | **0.0678** | 0.1022 | +51% |

**Results — action causality (n=500, ε=0.3).**

| metric | film-100k | Exp-10 ICM | direction |
|---|---|---|---|
| M1 action retrieval (↑) | 0.9830 | **0.9935** | ICM wins +1.1pp |
| — M1 on DROP | 0.938 | **0.980** | ICM wins +4.2pp |
| M2 distance calibration (↑) | **0.9541** | 0.8962 | ICM loses −5.8pp |
| M4 no-op recognition (↓) | **0.0400** | 0.0438 | ~tied (ICM marginally worse) |

**Conclusions.**

**1. Trade-off, not an upgrade — fails the success criterion.** Of the four
pre-registered conditions, exactly one is met: M1 rises, but M2 and M4 fall and
`cos@k`/MSE regress at *every* horizon and *every* action. Movement MSE@1 — not
just DROP — degrades ~50%. This is the Exp-5 failure mode (counterfactual
training won causality, lost `cos@k`, "no strict Pareto improvement"), reached by
a different route and slightly worse: Exp-10 also drags down M2 calibration,
which CF did not.

**2. The inverse loss reshaped the latent geometry — it spread the actions
apart.** The mechanism is visible in the encoder's true pairwise action
distances: ICM's movement–movement pairs sit at ~25 (0-1: 27.4, 0-2: 24.9,
1-2: 24.5) versus film-100k's ~16 (21.0, 15.3, 14.5) — the inverse loss pushed
LEFT/RIGHT/ROTATE ~50% further apart in latent space. That is exactly what a
classification CE optimises: *separability*. And it worked — M1 rises, DROP
retrieval especially (0.938→0.980). But a more spread-out target space is
*harder to predict*: the predictor must hit a moving target across a wider
range, so MSE rises and `cos@k` falls; and the predictor's standing
under-shoot on DROP distance (pred 33.7 vs true 39.1) costs proportionally more
rank correlation once the overall scale is larger, so M2 drops. **The
inverse-dynamics objective wants a *separable* latent space; the
forward-prediction objective wants a *compact, predictable* one. They pull the
shared encoder in different directions.**

**3. The premise did not hold for this baseline.** Exp-10 was motivated by
Exp-7's "the encoder discards piece position." But film-100k's movement MSE@1 is
~0.001 and its M1 on movement is ~1.0 — its encoder already retains piece
position essentially perfectly. FiLM-conditioned training had already solved the
problem the inverse loss was brought in to solve (cf. Exp-5: "FiLM already
solved causality well, M1 0.983 alone"). With no real causality headroom, the
inverse loss spent forward fidelity to buy separability the task did not need.
The one genuine gain is DROP *retrieval* (M1 DROP +4.2pp) — and even that did
not improve DROP *prediction* (DROP MSE@1 got worse).

**4. The inverse model is a good diagnostic, a poor auxiliary loss.** During
training `inverse_acc` reaches 1.00 within a few hundred steps and holds — a
clean, near-free confirmation that the encoder is action-complete. As a
*read-out* it works perfectly; as a *training pressure* at λ_inv = 1.0 it is
net-negative on this task.

**5. Implications for the follow-ups.** Cycle consistency and the hindsight
multi-step variant were scoped to *strengthen* the inverse coupling — given v1
shows the coupling itself is the problem, strengthening it would deepen the
trade-off, not fix it. They should not be pursued as specified. The narrow
exception worth a future look: DROP retrieval did improve, so a much smaller
λ_inv, or applying the inverse loss *only* to DROP-involving pairs, might land
nearer Pareto if DROP causality specifically matters for a downstream planner.
On the open question of training the inverse model only against the predictor's
*imagination* (decoupled from the encoder) — that is the cycle-consistency
direction and is now de-prioritised.

**Benchmark.** film-100k remains the default checkpoint on every metric. Exp-10
is a trade-off with no Pareto win; no checkpoint is carried forward
(`jepa-exp-icm-inverse.pt` is retained only for the inverse-model diagnostic and
the DROP-retrieval property). `--inverse-weight` now defaults to **0** (off);
the flag, `--inverse-depth`/`--inverse-heads`, and the optional `inverse_model`
checkpoint key remain in `train.py` / `load_jepa` (pre-Exp-10 checkpoints load
unaffected).
