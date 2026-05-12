# Decoder

The decoder is a separately-trained module that turns a 64-dim JEPA latent
back into a `(2, 20, 10)` Tetris board so you can *see* what the model is
representing or predicting. It is a **post-hoc probe**: it never participates
in JEPA training, never sends gradients into the encoder/predictor, and can
be retrained or thrown away without affecting the model under study.

This document covers what it produces, why it's structured this way, how to
train one, and how to actually look at the results.

---

## Why have a decoder at all?

The JEPA's encoder and predictor live in latent space. The training loss
(see [docs/decoder.md](decoder.md) sibling material in `train.py`) only ever
compares latents to latents — there is no pixel-level supervision anywhere
in the model. That's deliberate: it's the whole point of joint-embedding
predictive architectures. But it means a trained encoder output is just a
64-dim vector, not something you can look at.

Three things you cannot do without a decoder:

1. **See what the predictor "imagines"** for a state-action pair before you
   commit to it. Useful for planner debugging and for sanity-checking that
   `predictor(z, DROP)` actually represents a locked piece.
2. **Watch rollout drift visually.** Latent cosine similarity at horizon 8
   tells you the predictor is on-distribution, but it doesn't tell you
   *which* parts of the predicted state went wrong.
3. **Read the latent manifold.** Pair the decoder with UMAP and you get a
   2D map of the latent space where each point can be decoded back to a
   board.

The decoder also independently confirms the encoder/predictor distribution
gap documented in `RESULTS.md` — when rollout decode quality drops faster
than rollout cosine similarity, that gap is the cause.

---

## What the decoder does

Architecturally it mirrors `StateEncoder` ([jepa_tetris/models/encoder.py](../jepa_tetris/models/encoder.py))
with `ConvTranspose2d` instead of `Conv2d`. Source: [jepa_tetris/models/decoder.py](../jepa_tetris/models/decoder.py).

```
z (B, 64)
   │
   ├─ Linear(64 → 128 * 3 * 2) → reshape to (B, 128, 3, 2)
   │
   ├─ ConvTranspose2d(128, 64, k=3, s=2, pad=1)        → (B, 64, 5, 3)
   │   GroupNorm(8, 64) + GELU
   │
   ├─ ConvTranspose2d(64, 32,  k=3, s=2, pad=1)        → (B, 32, 10, 5)
   │   GroupNorm(8, 32) + GELU
   │
   └─ ConvTranspose2d(32, 2,   k=3, s=2, pad=1)        → (B, 2, 20, 10)
                                                          ↑ logits
```

Output is **logits**, not probabilities. Channel 0 is the locked board;
channel 1 is the falling piece. Apply `sigmoid` to get per-cell occupancy
probabilities; threshold at 0.5 if you need a hard binary board.

---

## Design decision: post-hoc, not joint

The decoder's BCE loss could in principle flow back into the encoder. We do
not do this, and would rather not.

JEPA's whole thesis is "predict in representation space, not pixel space."
If reconstruction loss flowed into the encoder, it would compete directly
with two things that are currently shaping the latent:

- The **predictive MSE** that wants `z` to be next-state-predictable from
  `(z_t, a_t)`. Cells the predictor cannot reliably forecast (e.g. exact
  spawn positions of new pieces after DROP) would be redundantly preserved
  in the latent at the cost of predictor-friendly structure.
- The **VICReg variance/covariance regularizers** that want the latent to
  spread evenly across all 64 dimensions. A reconstruction objective
  pushes toward dimensions that encode pixel-locality, which often
  collapses several dims onto similar features.

Keeping the decoder post-hoc preserves the JEPA's representation as the
canonical thing under study and lets us replace the decoder freely without
re-running JEPA training.

---

## How (and when) the decoder is trained

### When

After the JEPA is trained, before any visualization. One decoder per JEPA
checkpoint — they're tied to the encoder's specific 64-dim coordinate
system, which is reset every training run.

### How

[scripts/train_decoder.py](../scripts/train_decoder.py) takes a frozen JEPA
checkpoint and a replay buffer, then minimizes

```
L = BCE_with_logits(decoder(z), s)
```

against a deliberately mixed input distribution. Every training step the
batch is split:

| fraction | latent source | target | what it teaches |
|---|---|---|---|
| `1 - predictor_mix` | `z = encoder(s)` | the same `s` | reconstruct from encoder outputs |
| `predictor_mix` | `z = predictor^d(encoder(s_0), a_1..a_d)` for random `d ∈ [1, K]` | the actual rolled-out state at depth `d` | reconstruct from predictor outputs at every horizon |

Default is `predictor_mix=0.5`, `rollout_k=4` — meaning each step splits
the batch 50/50 and the predictor branch picks a random depth in `[1, 4]`
so the decoder sees every horizon over the course of training.

### Why both distributions

The encoder and predictor produce slightly different output manifolds even
when cosine similarity is high (this is the gap that breaks `BFSPlanner` —
see `RESULTS.md`). A decoder trained only on `encoder(s)` outputs will
silently produce garbage when fed `predictor(z, a)` outputs at inference,
because it never saw inputs from that manifold during training.

`predictor_mix > 0` solves this directly: the decoder sees both manifolds
during training and learns to be sane on both. The mix should match the
horizon you'll actually use at inference — `0` is fine for one-step
visualizations, `0.5` is the safe default, `1.0` is appropriate if you
only ever decode rollouts and never raw encoder outputs.

### Buffer compatibility

`train_decoder.py` auto-detects the buffer type via
[jepa_tetris/data/buffer_adapters.py](../jepa_tetris/data/buffer_adapters.py)
and works with both:

- `ReplayBuffer` — standard `(s, a, s', info)` triplets.
- `CounterfactualReplayBuffer` — `(s, next_states[A], a_executed, info)`
  rows. The decoder only ever uses the on-policy executed branch
  (`next_states[a_executed]`) so the off-policy counterfactuals are simply
  ignored.

There is no separate decoder for counterfactual training.

### Typical training command

```bash
python scripts/train_decoder.py \
  --jepa checkpoints/jepa_cf.pt \
  --buffer data/cf_buffer.npz \
  --out checkpoints/decoder_cf.pt \
  --predictor-mix 0.5 \
  --rollout-k 2 \
  --steps 5000 \
  --batch-size 256 \
  --lr 1e-3 \
  --log-file results/decoder_cf_log.jsonl
```

What to expect: the logged `bce_enc` and `bce_pred` should both drop below
~0.05 within a few thousand steps for a well-trained JEPA on a reasonable
buffer (typical end-state: ~0.02). `binary_acc` (computed on the encoder
branch) should rise above 0.95.

If `bce_pred` plateaus far above `bce_enc`, the predictor's manifold is
too different from the encoder's for the decoder to bridge — that's
diagnostic information about the JEPA, not the decoder.

---

## How to use the decoder

Three flows, in increasing order of interactivity.

### 1. Static visualizations — `visualize_predictions.py`

[scripts/visualize_predictions.py](../scripts/visualize_predictions.py).
Renders matplotlib PNGs (and optional animated GIFs) without a UI. Two
modes.

**Compare mode** — one figure per sample, showing
`s_t | action | s_{t+1} (real) | dec(predictor(z_t, a))`:

```bash
python scripts/visualize_predictions.py \
  --checkpoint checkpoints/jepa_cf.pt \
  --decoder checkpoints/decoder_cf.pt \
  --mode compare \
  --n 4 \
  --out viz_out/compare
```

Each PNG also reports: cos(ẑ, z\*), ‖ẑ − z\*‖, decode_acc(s\*).

**Rollout mode** — k-step strip showing actual vs predicted boards at
every horizon, with per-step metrics:

```bash
python scripts/visualize_predictions.py \
  --checkpoint checkpoints/jepa_cf.pt \
  --decoder checkpoints/decoder_cf.pt \
  --mode rollout \
  --horizon 8 \
  --n 4 \
  --gif \
  --out viz_out/rollout
```

`--gif` adds an animated version next to each PNG strip — useful for
seeing predictor drift evolve over time.

### 2. Interactive exploration — `decoder_explorer.py`

[scripts/decoder_explorer.py](../scripts/decoder_explorer.py). A Streamlit
app with three tabs:

- **Live env** — step a Tetris env by clicking action buttons; every
  candidate action's predicted next-state is decoded and shown so you can
  preview your options before committing.
- **Buffer scrubber** — jump to any row in a replay buffer, compare
  `dec(enc(s))` against `dec(predictor(z, a))` side-by-side, and roll the
  predictor forward for K steps from there.
- **Latent space** — UMAP projection of N latents from the buffer, with
  click-to-decode and nearest-real-neighbor lookup.

Launch:

```bash
streamlit run scripts/decoder_explorer.py
```

Then point the sidebar inputs at your JEPA checkpoint, decoder
checkpoint, and a replay buffer.

> **Caveat:** as of this commit, `decoder_explorer.py` reads buffers via
> `ReplayBuffer.load(...)` directly and accesses `buf.s_next` / `buf.a`,
> neither of which exists on `CounterfactualReplayBuffer`. Until that's
> patched, the explorer needs a single-action buffer (e.g.
> `data/buffer_big.npz`); CF buffers will work with `train_decoder.py`
> and `visualize_predictions.py` but crash the explorer.

### 3. Programmatic — load and use directly

For bespoke analysis or custom plots:

```python
import torch
from jepa_tetris.utils.checkpoint import load_jepa, load_decoder
from jepa_tetris.utils.device import get_device

device = get_device()
bundle = load_jepa("checkpoints/jepa_cf.pt", device)
decoder = load_decoder("checkpoints/decoder_cf.pt", bundle.latent_dim, device)

# Decode an encoded state.
s = ...  # (1, 2, 20, 10) float32 tensor on `device`
with torch.no_grad():
    z = bundle.encoder(s)
    logits = decoder(z)
    probs = torch.sigmoid(logits)         # (1, 2, 20, 10) in [0, 1]
    binary_board = (probs > 0.5).float()  # hard reconstruction
```

Predicted-next-state decoding follows the same shape:

```python
with torch.no_grad():
    z = bundle.encoder(s)
    a_emb = bundle.action_encoder(torch.tensor([action_id], device=device))
    z_pred = bundle.predictor(z, a_emb)
    pred_grid = torch.sigmoid(decoder(z_pred))
```

`load_jepa` and `load_decoder` both freeze parameters and put modules in
eval mode, so you don't need additional `.eval()` / `requires_grad_(False)`
calls.

---

## Reading the metrics

When `visualize_predictions.py` reports something like
`cos(ẑ, z*)=0.99, decode_acc=0.85`, here is how to parse it.

| pattern | what it means |
|---|---|
| high cos, high decode_acc | model and decoder both healthy |
| high cos, low decode_acc | decoder undertrained, or decoder needs more `predictor_mix` |
| low cos, high decode_acc | impossible in practice; sanity check the metric pipeline |
| low cos, low decode_acc | predictor inaccurate at this horizon — JEPA issue, not decoder |

The split lets you blame the right layer. If a rollout has
`cos=0.99` but the decoded board looks nothing like the actual board,
the decoder is the bottleneck — retrain with higher `--predictor-mix`
or more steps. If `cos` is also low, retrain the JEPA at deeper `K` or
use counterfactual training.

---

## File index

- [jepa_tetris/models/decoder.py](../jepa_tetris/models/decoder.py) — the
  `StateDecoder` module.
- [scripts/train_decoder.py](../scripts/train_decoder.py) — training loop.
- [scripts/visualize_predictions.py](../scripts/visualize_predictions.py)
  — static PNG/GIF rendering.
- [scripts/decoder_explorer.py](../scripts/decoder_explorer.py) —
  Streamlit interactive viewer.
- [jepa_tetris/utils/checkpoint.py](../jepa_tetris/utils/checkpoint.py) —
  `load_decoder`, `load_jepa`.
- [jepa_tetris/data/buffer_adapters.py](../jepa_tetris/data/buffer_adapters.py)
  — buffer-type-agnostic sampling used by the trainer.
- [jepa_tetris/viz/render.py](../jepa_tetris/viz/render.py) —
  `render_board`, `render_compare`, `render_rollout`.
