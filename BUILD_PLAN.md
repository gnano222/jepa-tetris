# JEPA-Based Tetris Agent — V1 POC Plan

## Goal

Build a minimal proof-of-concept of a JEPA (Joint-Embedding Predictive Architecture) world model that learns the dynamics of a simplified Tetris environment through self-supervised exploration, then uses the learned model to plan moves.

This POC is the foundation for a longer-term direction: extending the same architecture to abstract, general-purpose computer tool use (e.g. ordering food, navigating web UIs).

---

## Scope (V1)

- **Simplified Tetris**: blocks do **not** auto-fall. The agent controls when a piece drops. Game state = board grid + current falling piece.
- **Action space**: `{left, right, rotate, drop}`.
- **No reward model, no hierarchy, no structured perception layer.** Pure JEPA world model + brute-force planner.

---

## Architecture

JEPA learns to predict the *latent representation* of the next state — not raw pixels.

```
        ┌──────────────┐
state → │   Encoder    │ ──► z_t  (latent of current state)
        └──────────────┘                │
                                        ▼
        ┌──────────────┐         ┌──────────────┐
action →│Action Encoder│ ──► a_t │  Predictor   │ ──► ẑ_{t+1}
        └──────────────┘         └──────────────┘
                                        │
                                        ▼
                              compare with z_{t+1}
                              (target encoder output
                               on actual next state)
```

### Components

| Component | Role | Suggested implementation |
|---|---|---|
| **State encoder** `f_θ` | Board grid → latent vector `z_t` | Small CNN |
| **Target encoder** `f_ξ` | Same architecture, used to encode actual next state for the loss target. EMA of `f_θ`'s weights (stop-gradient). | Same CNN, no gradient |
| **Action encoder** | Action ID → embedding | Embedding layer or small MLP |
| **Predictor** `g_φ` | `(z_t, a_t)` → `ẑ_{t+1}` | 2-3 layer MLP |

### Training objective

Latent-space prediction loss:

```
L = || g_φ(f_θ(s_t), a_t) − sg(f_ξ(s_{t+1})) ||²
```

Where `sg(·)` is stop-gradient. Use EMA updates on `f_ξ` (target encoder) to prevent representation collapse — standard JEPA / BYOL-style trick. Add VICReg-style variance/covariance regularization on `z_t` if collapse is observed.

**No decoder during JEPA training. No pixel reconstruction in the loss.** That's what makes this JEPA rather than a standard autoencoder world model.

For *visualization only*, an optional post-hoc decoder probe (`models/decoder.py`, trained via `scripts/train_decoder.py` against the frozen encoder) maps `z → board logits` so predicted latents can be rendered alongside actual states. The probe is never in the JEPA gradient path; representation learning remains pure JEPA.

---

## Data Collection

Self-supervised — no human labels needed.

- Agent plays random Tetris games.
- After every action, log the triplet: `(s_t, a_t, s_{t+1})`.
- Stream triplets to disk as a replay buffer.
- Target dataset size for V1: ~100k–500k triplets (tune based on convergence).

---

## Training Loop

1. Sample minibatch of triplets from replay buffer.
2. Encode `s_t` with `f_θ` → `z_t`.
3. Encode action with action encoder → `a_t`.
4. Predict next latent: `ẑ_{t+1} = g_φ(z_t, a_t)`.
5. Encode `s_{t+1}` with `f_ξ` (stop-gradient) → `z_{t+1}`.
6. Compute MSE loss in latent space.
7. Backprop through `f_θ`, action encoder, and `g_φ`.
8. EMA-update `f_ξ` from `f_θ`.

---

## Evaluation / Planning

Once the world model is trained, bolt on a **brute-force planner** to test it:

- Enumerate all action sequences up to depth 3–4.
- For each sequence, roll out predicted latents through `g_φ`.
- Score sequences by a simple heuristic (lines cleared, board height, holes). For V1, this heuristic can be a hand-coded function operating on the decoded board — *or* a tiny probe head trained to read line-clear count from latents.
- Execute the best sequence in the real environment, observe, repeat.

If the world model is accurate, the agent should play meaningfully better than random.

---

## Tech Stack

- **Python 3.10+**
- **PyTorch** — networks and training
- **NumPy** — Tetris environment (grid logic, collision detection)
- **Pygame** *(optional)* — for visualizing gameplay during debugging
- **Weights & Biases** *(optional)* — track loss curves, latent collapse metrics

---

## Repo Structure (suggested)

```
jepa_tetris/
├── env/
│   └── tetris.py            # Minimal NumPy Tetris (no auto-fall)
├── models/
│   ├── encoder.py           # CNN state encoder
│   ├── action_encoder.py    # Action embedding
│   ├── predictor.py         # MLP predictor
│   └── decoder.py           # Post-hoc visualization probe (z -> board logits)
├── viz/
│   └── render.py            # matplotlib render: board, compare, rollout
├── data/
│   └── replay_buffer.py     # Triplet logging + sampling
├── train.py                 # Training loop with EMA target encoder
├── plan.py                  # Brute-force planner using learned model
└── eval.py                  # Run planner, measure score vs random baseline
scripts/
├── train_decoder.py         # Fit decoder probe to frozen JEPA encoder
└── visualize_predictions.py # Compare s_t / actual s_{t+1} / predicted s_{t+1}
```

---

## Success Criteria for V1

1. **Training converges**: latent prediction loss decreases and stabilizes (without collapsing — verify variance of `z_t` stays > 0).
2. **Predictor is accurate**: rolled-out latents over a few steps stay close to ground-truth latents.
3. **Planner outperforms random play**: clears measurably more lines per game than a random-action baseline.

---

## Known Risks

- **Representation collapse** — all states map to the same latent. Mitigated by EMA target encoder and (if needed) VICReg-style regularizers.
- **Compounding prediction error** in multi-step rollouts. Acceptable for V1 with shallow planning depth (3–4); revisit for V2.
- **Random exploration coverage** — random play may not visit interesting board states often enough. If problematic, add curiosity-driven exploration in V2.

---

## Out of Scope (Future Versions)

- Auto-falling blocks / real-time dynamics
- Learned planner (replace brute force with MCTS or learned policy)
- Reward / outcome model for goal-directed behavior
- Structured perception layer (parsing screens into elements)
- Action abstraction / hierarchical planning
- Generalization to non-Tetris environments (web UIs, computer tool use)
