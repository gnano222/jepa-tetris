# Research Roadmap

Notes on where this project sits in the JEPA literature and where to push
next. Companion to [BUILD_PLAN.md](BUILD_PLAN.md) and the "Future directions"
section of [README.md](README.md).

Organized by system component: **encoder**, **predictor**, **action
encoder**, **data collection**.

## Where this project sits in the JEPA family

JEPA isn't one architecture — it's a training principle: **predict in latent
space, with a stop-gradient/EMA target**. Different papers instantiate it
differently.

| Branch | Examples | What gets predicted |
|---|---|---|
| Masked self-supervised | I-JEPA, V-JEPA, A-JEPA | Masked patches/tokens from visible context |
| Action-conditioned world model | **DINO-WM, TD-JEPA, ours** | Next state's latent, given current latent + action |

We're firmly on the **world-model branch**. The training signal is
"`(z_t, a_t) → ẑ_{t+1}`," not "fill in the blanks." Closest published cousin
is DINO-WM, and V2 deliberately mirrors its architecture: patch-token
latents, transformer predictor, teacher-forced multi-step training.

### Notable contrasts with DINO-WM

- **Backbone.** DINO-WM uses frozen DINOv2 (huge, internet-pretrained); we
  train a small CNN encoder from scratch (~260K params). On a 20×10 grid
  with no texture/scale variation, a Tetris-pretrained vision model doesn't
  exist — training the encoder is the cost of admission. We share the patch
  output shape (V2: `(B, 6, 128)`) and the no-flatten convention.
- **Predictor inputs.** DINO-WM takes a history of frames (causal attention
  across time). We pass a single frame's patches + action. Tetris is Markov
  on the current board, so frame history adds no info.
- **Collapse prevention.** DINO-WM relies on the frozen-encoder regime;
  there's no online encoder to collapse. We update the encoder online with
  an EMA target, so VICReg variance + covariance regularizers carry the
  anti-collapse load (`mean(std(z)) ≈ 1.08` historically).

## Design principle: cross-domain generalization

Tetris is the testbed, not the destination. The architecture should transfer
cleanly to richer digital tasks (browser control, desktop use, other games)
without rewriting the predictor. Two interface constraints follow:

- **State `z` is always a token sequence `(B, N, D)`**, not a flat vector.
  V2 already complies (N=6); future encoders (e.g. a ViT on screenshots)
  plug in by changing N and D, with no predictor changes.
- **The predictor treats `a_emb` as an opaque `(B, E)` vector.** No knowledge
  of `num_actions` or action structure leaks past `ActionEncoder`. Swapping
  the 4-action lookup for a structured (click x/y + key code + modifier)
  encoder shouldn't touch `predictor.py`.

The Tetris-specific bits (4-action `nn.Embedding`, 20×10 conv stack) live
at the boundaries; everything in the middle stays generic. When evaluating
a proposed change, ask: *would this still make sense if `z` were a ViT
patch grid and `a` were a mouse-and-keyboard event?* If not, it probably
belongs at the boundary, not in the core.

## Encoder (`jepa_tetris/models/encoder.py`)

V2 emits a (B, N=6, D=128) patch-token grid (implemented). Open directions:

### Finer patch granularity ⬅ NEXT UP (1 of 2)
6 patches in a 3×2 grid means each patch summarises a ~7×5 chunk of the
board. A falling tetromino is 1-2 cells; line clears are non-local. The grid
is too coarse for either to register precisely — DROP MSE almost certainly
has headroom that finer patches would expose.

**Immediate target — drop one stride-2 conv:** Two stride-2 stages instead of three:
20×10 → 10×5 → **5×3** = 15 patches. Same compute (one fewer conv), 2.5×
more spatial resolution. Transformer at N=15 is still trivial (15² = 225
attention entries per head). Requires encoder rewrite (remove one stride-2
conv) + full retrain. Measure DROP MSE@1, cos@1, and cos@4 against V2-Mixed
baseline.

Further options now that 15 patches show gains:

- **Two-scale latent** ✅ DONE — fine 15 tokens + pooled coarse 6 tokens
  = N=21. Zero new parameters. See [FINDINGS.md — Exp-2](FINDINGS.md).
  Beats 15-patch on all metrics at equal budget; now the current baseline.
- **No striding.** Three stride-1 convs → 200 patches. DINO-WM convention.
  Requires bumping predictor depth (200² attention entries per head vs 225).
  Estimate ~20 min per run at batch 256 on RTX 4090.

### Deeper conv stack
Three stride-2 convs give a ~15×15 receptive field — enough to "see" the
board, but limited *depth* for compositional reasoning ("tall stack here
AND a piece that fits the well there"). The `residual_blocks` flag already
adds depth without changing the latent shape; not yet tuned.

### Asymmetric strides
A 20×10 board is taller than wide; the row axis carries most of the
meaningful structure (skyline, holes). Square `k=3, stride=2` convs compress
rows and columns at the same rate. Asymmetric strides (compress rows faster)
or column-wise pooling would preserve column structure longer. This flag
was removed in the V2 simplification — bring it back if width matters.

### ~~Columnar encoder + local learning~~ ❌ TESTED — NON-STARTER (Exp-9)
A cortically-inspired encoder: the board split into a 5×3 grid of 15 *untied*
per-column conv stacks (overlapping receptive fields), each trained by its own
per-column local loss with **no gradient between columns** — decoupled greedy
learning / Greedy InfoMax applied spatially, with the global predictor trained
on the detached encoder output. See [FINDINGS.md — Exp-9](FINDINGS.md).
Non-starter on two counts. (1) Untied per-column weights regress the
*architecture* before local learning is even introduced: Fork A (untied
columns, ordinary global backprop) loses to shared-weight film-100k at long
horizon (cos@16 0.79 vs 0.93) — weight-sharing is a regularizer worth keeping.
(2) The local-loss variant (Fork B) matches global backprop on representational
*direction* (cosine, M1 action retrieval 0.96 vs 0.97) but not *magnitude* —
MSE and DROP MSE 6× worse, M2 distance calibration collapses 0.93→0.70. The
per-column local loss + per-column VICReg pins direction and variance but not
global scale. The one finding worth keeping: local credit assignment reproduces
the *geometry* of a representation but not its *scale*; if revisited, the local
objective needs an explicit magnitude constraint. Flags `--encoder-columnar`,
`--encoder-columnar-grid`, `--encoder-columnar-margin`, `--local-loss` remain in
`train.py` (default off).

### Object-centric / slot encoder
The current encoder describes the board by a *fixed grid* of patch tokens
(N=21). A more brain-aligned factorisation describes it by *things*: a
slot-attention encoder emits a small set of "slots" that compete to each
claim one entity in the scene — one slot latches onto the falling piece,
others onto regions of the settled pile. The payoff is that sparse change
becomes *automatic*: an action updates the piece slot and leaves the pile
slots untouched, with no gate or penalty needed — the factorisation itself
carries the inductive bias. This is also the most general direction (for
computer-use the "things" become windows, buttons, icons; a click affects
one). It is the natural follow-up to the token-gated predictor: Exp-8
confirmed the rationale — a *fixed-grid* gate could not pin down what DROP
touches (it wants more tokens than any movement-tight cap allows; see
[FINDINGS.md — Exp-8](FINDINGS.md)). That failure is the evidence that a
fixed grid is too crude and genuine object slots are needed: DROP's
restructuring (piece → pile, line clear) is a slot-level event, not a
patch-count event.

Cost / risk: a full encoder rewrite; Slot Attention is notoriously fiddly
to train stably; and Tetris has only ~2 entities (piece + pile), so the
machinery may be heavy relative to the payoff *on this task* — its value
shows most on scenes with many objects. Sequencing: ⬅ NEXT UP — the
token-gated result (Exp-8) is in, and points here.

## Predictor (`jepa_tetris/models/predictor.py`)

V2 is a 2-layer transformer with `residual=True` by default (predicts Δz).
Open directions:

### ~~Per-patch action conditioning~~ ❌ TESTED — REGRESSION
See [FINDINGS.md — Exp-1](FINDINGS.md). Skip to FiLM/cross-attention if stronger conditioning is needed.

### Stronger action conditioning: FiLM / cross-attention
Per-patch broadcast addition is the lightest form of "action everywhere." If
DROP accuracy still has headroom after it lands, two stronger conditioning
schemes share the same philosophy but give the action more capacity to shape
the computation — and both generalize cleanly to non-trivial action spaces:

- **FiLM (Feature-wise Linear Modulation).** Action produces per-block
  `(γ, β)` and modulates every patch's hidden state at every transformer
  layer: `h = γ(a) ⊙ h + β(a)`. Adds ~one small Linear per block. Standard
  in V-JEPA2-AC and many video world models — the modulation MLP doesn't
  care about action dimensionality, so it transfers as-is when `a_emb`
  grows from 16-d to a structured encoding.
- **Cross-attention from action tokens to state tokens.** The action becomes
  one (or several) KV tokens that state tokens attend to in a dedicated
  cross-attn sublayer. Trivial extension of the existing transformer;
  matches Genie/Sora-style conditioning. Best fit when the action embedding
  is eventually a *sequence* (e.g. a tokenized "click(x, y)" event).

Both stack on top of per-patch broadcast; they're not exclusive. Evaluate
via per-action prediction MSE under counterfactual training (the metric most
sensitive to how well the predictor distinguishes actions — the predictor
should produce visibly different `ẑ_{t+1}` for different `a_t` on the same
`z_t`).

### Multi-step training stability
Teacher-forced K-step rollouts compound noise: at K=4 it's manageable; at
the horizons that matter for richer digital tasks (computer use is 50+
steps) it dominates. Cheap, generalizable fixes:

- **Random K per batch** (1..K_max) — predictor sees both short and long
  rollouts in the same minibatch.
- **Curriculum K** — schedule K=1 → K_max over training.
- **Stop-grad on early rollout steps** — backprop only through the most
  recent N steps; keeps gradients well-conditioned without giving up the
  long-horizon training signal.

Test on V2's current K=4 setting first; the payoff scales with horizon length.

### Cosine prediction loss as an A/B
Raw MSE rewards matching latent magnitude; cosine focuses on direction.
VICReg shapes the encoder's variance/covariance but not its global scale,
so direction is the more semantically meaningful axis to fit. Worth a
one-flag A/B on `cos_sim_k4` and per-action separation under counterfactual
training. Generalizes to any latent shape — no Tetris-specific assumptions.

### ~~Sparse-change prior~~ ❌ TESTED — REGRESSION (Exp-7)
A group-lasso penalty on the predictor's change vector `Δ = ẑ' − z`
(channels split into G groups; sum of per-group L2 norms) was meant to
shrink the predictor's hypothesis space and pressure a factored latent.
It backfired: see [FINDINGS.md — Exp-7](FINDINGS.md). Group-lasso kills the
*smallest-magnitude* groups first, and in Tetris movement has a small
latent footprint but is causally crucial — so the penalty zeroed the
movement signal (`ẑ_LEFT = ẑ_RIGHT = ẑ_ROTATE = z`, action retrieval
collapsed) while sparing the large DROP change. At λ=0.10 the predictor
became the pure identity map (M1 0.98→0.28). The brain principle (actions
cause sparse, local change) may hold; penalising change *magnitude* is the
wrong operationalisation — it conflates "small" with "discardable." Open
redesigns if revisited: stop-grad the encoder from the sparse term;
weight the penalty by causal relevance not magnitude; or anchor the
encoder with a reconstruction term. Flags `--sparse-change-weight` /
`--sparse-change-groups` remain in `train.py` (default 0 = off).

### ~~Token-gated sparse predictor~~ ❌ TESTED — REGRESSION (Exp-8)
The architectural successor to the sparse-change prior: a hard top-k gate
lets the predictor change at most `k` of the 21 patch tokens, copying the
rest forward exactly — no penalty, so no collapse. See
[FINDINGS.md — Exp-8](FINDINGS.md). It worked as designed *and still lost*:
unlike Exp-7 it preserved causality (k=10 M1=0.982, matching film-100k),
but a single global cap only subtracts capacity. The `live_tokens`
diagnostic showed DROP pinned at the cap throughout (it wants >10 tokens)
while movement contracted to ~3 — so the cap starves DROP and does nothing
for movement, which was already solved. DROP MSE@1 degraded monotonically
as `k` tightened (0.068 → 0.086 → 0.139). Confirms a recurring result:
DROP is the bottleneck and generic change-constraints do not touch it. The
fixed-grid failure motivates the slot encoder (see Encoder §). Flags
`--predictor-token-gate` / `--token-gate-k` remain in `train.py` (default
off).

### Surprise-gated loss
The brain learns from prediction *error* — plasticity is gated by
neuromodulators that turn the learning rate up for surprising or important
events and down for routine ones. JEPA training instead weights every
transition equally, so the rarest, most state-changing transitions (line
clears: ~0.5% of the buffer; see Data §"event-poor") are drowned out — a
plausible contributor to DROP prediction being ~50× harder than movement
(Exp-4). A cheap, domain-general fix: weight each transition's loss by its
prediction error, or by how much the state actually changed (latent-space
displacement `‖z' − z‖`). This rebalances *which transitions* the model
learns from — complementary to the sparse-change prior, which rebalances
*which dimensions*. Best evaluated as a one-flag A/B against film-100k.
Generalizes to any task — no Tetris-specific assumptions.

### Depth and head count
Default is `depth=2, heads=4`. With only 7 tokens and a 128-d model, this
is plenty of capacity, but it's untuned.

## Inverse model (`jepa_tetris/models/inverse_model.py`)

### ICM-style inverse head ⬅ IN PROGRESS (Exp-10)
The predictor is a *forward* model: `(state, action) → next state`. It never
asks the converse — *what action caused this change?* The brain's motor system
runs **paired** forward and inverse models that bootstrap each other (Wolpert &
Kawato's MOSAIC, 1998); the inverse half turns a goal into the motor command
that reaches it. Exp-10 adds the inverse half in its simplest proven form — the
ICM recipe (Pathak et al. 2017): a second head, `(z_t, z_{t+1}) → action`,
trained jointly over the *same* encoder. Recovering the action forces the
encoder to keep action-causal information: you cannot tell LEFT from RIGHT
without precise piece position, so the "discard piece position" failure mode of
Exp-7 becomes structurally impossible. It also turns the M1 action-retrieval
*eval* metric into a training *objective*.

`InverseModel` is the predictor's mirror twin — a per-patch `[z; z'; z'−z]`
projection, spatial positional embeddings, a shallow transformer, mean-pool →
`(B, 4)` action logits. Notably it is *not* modelled on the `Probe`: the Probe
pools an unordered patch set, which discards the spatial localisation that LEFT
vs RIGHT depends on. Wired into the teacher-forced path as a cross-entropy term
behind `--inverse-weight` (default 1.0; `0` = baseline forward-only). Risk
(per Exp-5): an extra objective can trade long-horizon `cos@k` for causality —
the run must report both. Follow-ups if it lands: cycle consistency, a
hindsight multi-step variant, and an inverse-model goal-conditioned planner.

## Action encoder (`jepa_tetris/models/action_encoder.py`)

### Currently Tetris-shaped; preserve the interface
`nn.Embedding(num_actions=4, 16)` is the right call for Tetris but doesn't
transfer: most non-game digital tasks have structured action spaces (click
x/y, key code, modifiers, scroll deltas, drag start/end). The action space
is effectively combinatorial, not a 4-way categorical.

The constraint to lock in *now*, while everything still fits the Tetris
shape: **`predictor.py` consumes `a_emb` as an opaque `(B, E)` vector and
assumes nothing else about it.** That's already true today; the goal is to
keep it true as the predictor evolves.

When the time comes to handle a richer action space, the action encoder
becomes something like:

```python
class StructuredActionEncoder(nn.Module):
    def forward(self, action) -> Tensor:  # -> (B, E)
        # e.g. embed action_type, encode (x, y) via sinusoidal pos encoding,
        # embed key codes, concat + small MLP
        ...
```

…and nothing upstream changes. A one-line shape assertion on `a_emb` in the
predictor forward locks the contract in cheaply.

### Larger `embed_dim` if FiLM lands
Today `embed_dim=16` is fine because the action is one token among seven. If
FiLM-style conditioning is adopted, the action embedding's expressiveness
matters more (it has to produce `(γ, β)` for every transformer block). 32 or
64 is a cheap bump; gating it behind a flag avoids regressing the per-patch
broadcast baseline.

## Data collection (`jepa_tetris/data/`)

### Buffer is exploration-heavy and event-poor
Mixed-exploration data has ~0.5% line-clear density. Line-clears are the
single most state-changing transition in Tetris — entire rows vanish, the
skyline collapses. A predictor that's only seen them in 1-in-200 samples
has very little signal to learn that dynamic from, which shows up as
elevated DROP MSE.

**Try:** Heuristic distillation — run a hand-coded greedy player with
light noise (epsilon=0.1) and store its trajectories. Should yield much
higher line-clear density (and a broader distribution of board shapes
near clears) without losing the random-state coverage the encoder needs
for generalization. Generalizes as a pattern: when a rare event dominates
the dynamics, oversample it in the buffer rather than reweight in the
loss.

## Current benchmark — film-50k

All future experiments must beat these numbers to represent progress.

| metric | film-50k | two-scale-50k (prev) | 15patch-100k (old champion) |
|---|---|---|---|
| cos@1 | **0.9966** | 0.9912 | 0.994 |
| cos@4 | **0.9820** | 0.9660 | 0.976 |
| cos@8 | **0.9565** | 0.7353 | — |
| cos@16 | **0.9059** | 0.023 | — |
| DROP cos@1 | **0.9838** | 0.9569 | 0.970 |
| DROP MSE@1 | **0.1066** | 0.2275 | 0.168 |

film-50k: two-scale encoder (`encoder_stride_stages=2`, N=21), FiLM action conditioning (`--predictor-film`), 50k steps, batch 256, `ar_weight=0.0`, teacher-forced H=4. Checkpoint: `checkpoints/jepa-exp-film.pt`.

Note: the k>4 collapse that plagued all previous runs is largely solved by FiLM — cos@16 went from ~0.02 to 0.91. DROP MSE halved (0.227 → 0.107). FiLM was tested alongside cross-attention (see Exp-3 in FINDINGS.md); cross-attn is worse than the two-scale baseline on k≥8 and is not recommended.

## Suggested ordering

Roughly increasing surgery, all encoder/predictor focused.

1. ~~**Retrain V2**~~ ✅ DONE — patch tokens + transformer + teacher-forced
   multi-step trained. V2-Mixed-50k is the current best baseline.
2. ~~**15-patch encoder**~~ ✅ DONE — two stride-2 stages → N=15 patches.
   15patch-100k is the current best baseline (cos@1 0.994, cos@4 0.976).
3. ~~**Per-patch action conditioning**~~ ❌ REGRESSION — broadcast addition
   worse than extra-token on every metric. See [FINDINGS.md — Exp-1](FINDINGS.md).
4. ~~**Two-scale encoder**~~ ✅ DONE — N=21 (fine 15 + coarse 6). Beats
   15-patch on all metrics at 50k steps. Now the current benchmark.
5. **No-striding encoder** — N=200. Only after two-scale verdict; requires
   predictor depth bump (3–4 layers).
6. ~~**FiLM or cross-attention action conditioning**~~ ✅ DONE — FiLM wins
   decisively (see Exp-3 in FINDINGS.md). Cross-attention regresses on k≥8.
   FiLM is now the default predictor and the new benchmark.
7. **Heuristic distillation** (Data) — higher line-clear density in the buffer.
   DROP MSE improved dramatically with FiLM (0.107) but further gains likely
   require richer training signal on rare DROP transitions.
8. **ICM-style inverse head** ⬅ IN PROGRESS — inverse-dynamics auxiliary loss
   over the shared encoder (Exp-10). See [FINDINGS.md — Exp-10](FINDINGS.md).
9. **Structured action encoder** — only when moving beyond Tetris.
