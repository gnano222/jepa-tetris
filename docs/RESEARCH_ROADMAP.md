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

Further options if 15 patches show gains:

- **No striding.** Three stride-1 convs (receptive-field only), tokenize at
  full resolution → 200 patches per board. Matches DINO-WM's convention
  (~196 patches per image). 200² = 40k attention entries per head — still
  fast on M-series.
- **Two-scale latent.** Concatenate a coarse 6-token stream and a fine
  ~50-token stream. Coarse handles global layout (skyline); fine handles
  the falling piece. Feature-pyramid style.

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

## Predictor (`jepa_tetris/models/predictor.py`)

V2 is a 2-layer transformer with `residual=True` by default (predicts Δz).
Open directions:

### Per-patch action conditioning ⬅ NEXT UP (2 of 2)
Today the action is one extra token in the sequence (alongside the 6 patches).
DINO-WM adds the action embedding to *every* patch token via broadcast addition
(`z = z + a_emb.unsqueeze(1)`), so every patch simultaneously "sees" the action
rather than attending to a separate token. This may improve DROP accuracy in
particular, since the piece lock affects multiple patches at once.

One-line change in `predictor.py` forward (remove action token from sequence,
add to all patch tokens instead). Rerun with same training recipe and compare
DROP MSE@1 and cos@1/4 against the 15-patch baseline.

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

### Depth and head count
Default is `depth=2, heads=4`. With only 7 tokens and a 128-d model, this
is plenty of capacity, but it's untuned.

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

## Current benchmark — V2-Mixed-50k

All future experiments must beat these numbers to represent progress.

| metric | value |
|---|---|
| cos@1 | 0.992 |
| cos@4 | 0.968 |
| cos@8 | 0.729 |
| cos@16 | 0.013 |
| DROP MSE@1 | 0.153 |

Training recipe: 50k steps, `--ar-weight 0.25` (teacher-forced + AR mixed loss), mixed exploration buffer (5k episodes, ε=0.4). The k>4 collapse is architectural — H=4 training horizon is a hard ceiling at inference.

## Suggested ordering

Roughly increasing surgery, all encoder/predictor focused.

1. ~~**Retrain V2**~~ ✅ DONE — patch tokens + transformer + teacher-forced
   multi-step trained. V2-Mixed-50k is the current best baseline.
2. **15-patch encoder** (Encoder §1) — drop one stride-2 conv, retrain with
   mixed loss. Measure DROP MSE@1 and cos@1/4. Primary encoder improvement.
3. **Per-patch action conditioning** (Predictor §1) — broadcast action
   embedding to all patch tokens. One-line predictor change; stack on top
   of 15-patch result or test independently.
   - **3a. Multi-step stability tricks** (Predictor §3) — random K /
     curriculum K / stop-grad on early steps. Cheap and folds into the same
     retrain; the bigger the payoff, the longer the horizon you care about.
   - **3b. FiLM or cross-attention conditioning** (Predictor §2) — only if
     per-patch broadcast leaves DROP headroom. Bigger change than 3a but
     unlocks richer action spaces later.
   - **3c. Cosine loss A/B** (Predictor §4) — orthogonal one-flag experiment;
     run alongside 3 or 3b on the same checkpoint.
4. **Heuristic distillation** (Data) — higher line-clear density in the
   buffer. Improves predictor signal on the most state-changing transition
   class; useful any time DROP MSE plateaus.
5. **Structured action encoder** (Action encoder §1) — only when actually
   moving to a non-Tetris domain. Listed here so the predictor interface
   stays opaque to action structure from now on.
