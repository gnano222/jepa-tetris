# CLAUDE.md

## Tone
you are an expert in neuroscience and deep learning. but you explain concepts in a really simple to understand *layman terms* for someone without neuro knowledge and limited AI experience.

You should push the envelope upon state of the art AI research and architechture. Look to build upon already established best practice research, and look to go one step further.

## Research Goal

MY RESERCH HUNCH: There is something more powerful to decode from neuroscience when building deep learning machines - something more powerful than transformers today. You must help find how to translate the power of the brain into AI architechture. Specifically i believe:
- the *learning algorithm* in brains appears to be a lot more efficient than transformer architechture. 
- brains tend to understand *causality* quicker and better than transformers.

MY GOAL: i want to build a new JEPA inspired solution for a tetris playing AI. it can see the blocks of the board, and uses JEPA self-supervise learning how to plan and play the game

Tetris will be my POC, but I'd like to eventually generalize it into other digital tasks like computer use

## Key documents

Throughout our research work, we should keep a few documents in the /docs finding up to date with latest findings:
- RESEARCH_ROADMAP.md with upcoming research areas
- FINDINGS.md with a concise and summary of research findings

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pip install -e ".[viz]"   # adds streamlit/plotly/umap for decoder_explorer

# Tests
pytest                              # all 44 tests
pytest tests/test_models.py         # single file
pytest -k test_encoder              # single test by name

# Data collection
python -m jepa_tetris.data.collect \
    --episodes 5000 --capacity 500000 \
    --policy mixed --epsilon 0.4 --prime-prob 0.4 \
    --out data/buffer.npz --seed 0

# JEPA training (artifacts land in results/<timestamp>_<run>/)
python -m jepa_tetris.train \
    --buffer data/buffer.npz --steps 50000 --horizon-h 4 \
    --out checkpoints/jepa.pt --run h4 --seed 0

# Probe training (~30 sec)
python -m jepa_tetris.train_probe \
    --jepa checkpoints/jepa.pt --buffer data/buffer.npz \
    --out checkpoints/probe.pt --seed 0

# Evaluation
python -m jepa_tetris.eval \
    --jepa checkpoints/jepa.pt --probe checkpoints/probe.pt \
    --episodes 100 --planner placement \
    --lines-w 10 --holes-w -1.0 --height-w -0.3

# Decoder training and interactive explorer
python scripts/train_decoder.py \
    --jepa checkpoints/jepa.pt --buffer data/buffer.npz \
    --predictor-mix 0.5 --rollout-k 4 --steps 5000 \
    --out checkpoints/decoder.pt
streamlit run scripts/decoder_explorer.py

# Diagnostics
python scripts/diagnose.py --jepa checkpoints/jepa.pt --buffer data/buffer.npz
python scripts/multistep_accuracy.py --jepa checkpoints/jepa.pt --buffer data/buffer.npz
python scripts/compare_planners.py --jepa checkpoints/jepa.pt --probe checkpoints/probe.pt
```

## Architecture

### Data flow

```
TetrisEnv (NumPy, hard-drop semantics)
    → MixedExplorationPolicy (heuristic + random blend)
    → ReplayBuffer / CounterfactualReplayBuffer (.npz)
    → train.py
```

The state is a `(2, 20, 10)` float32 array: channel 0 = board occupancy, channel 1 = current piece.

### JEPA model components

All four submodules are saved together in a single checkpoint and loaded via `utils/checkpoint.py:load_jepa` → `JepaBundle`.

| Module | Input → Output | Notes |
|---|---|---|
| `StateEncoder` | `(B, 2, 20, 10)` → `(B, 6, D)` | Three stride-2 convs, `patch_dim` must be divisible by 32 |
| `ActionEncoder` | `(B,)` int → `(B, D)` | `nn.Embedding(4, D)` |
| `Predictor` | `(B, 6, D)`, `(B, D)` → `(B, 6, D)` | ViT: 7-token sequence (6 patches + 1 action), predicts Δz by default |
| `target_encoder` | same as encoder | EMA copy (τ=0.99), stop-grad; produces training targets |
| `Probe` | `(B, 6, D)` → `(B, 3)` | Cross-attention pooling + MLP → (lines, holes, height) |
| `StateDecoder` | `(B, 6, D)` → `(B, 2, 20, 10)` | Mirrored ConvTranspose2d; visualization only, not part of JEPA training |

### Training loop (`train.py`)

Each step samples an `H+1` frame window. The encoder runs over all frames, then the predictor is called independently at each of `H` positions from the *real* encoded frame (teacher-forced, no autoregressive chain). Loss = `MSE(ẑ, stop_grad(target_encoder(s_next)))` + VICReg variance + covariance regularizers.

Key flags:
- `--horizon-h`: multi-step window size (default 4)
- `--autoregressive`: switch from teacher-forced to AR training
- `--ar-weight`: mix AR loss on top of teacher-forced loss
- `--counterfactual`: train against all 4 action branches per state

### Planners (`plan.py`)

Three variants, best-to-worst performance:
1. **`PlacementPlanner`** — enumerate all (col, rotation) placements in the real env, score with JEPA probe. Best.
2. **`RealDynamicsPlanner`** — BFS of depth-K action sequences in the real env, JEPA scores leaves.
3. **`BFSPlanner`** — pure latent rollout via predictor. Fails in practice (probe doesn't generalize across the encoder/predictor distribution gap).

All planners require at least one DROP in the action sequence (`require_drop=True` default) to prevent the stall exploit.

### Checkpoint format

```python
torch.save({
    "step": int,
    "encoder": state_dict,
    "target_encoder": state_dict,
    "action_encoder": state_dict,
    "predictor": state_dict,
    "args": vars(args),   # all CLI args, used to reconstruct arch on load
}, path)
```

`args` is the source of truth for model architecture. Always use `load_jepa()` / `make_encoder_from_args()` rather than constructing models manually — they read `patch_dim`, `predictor_heads`, `predictor_depth`, etc. from the stored args.

### Run artifacts

Training writes to `results/<YYYYMMDD-HHMMSS>[_<run_name>]/`:
- `train_log.jsonl` — one JSON object per `--log-every` steps
- `train_args.json` — full CLI args snapshot

Eval writes `eval.json` + `eval_args.json` to a similar timestamped folder.

### Buffer schemas

Two buffer types, distinguished by presence of `next_states` key in the `.npz`:
- `ReplayBuffer`: classic `(s, a, s_next, info)` triplets
- `CounterfactualReplayBuffer`: `(s, next_states[4], a_executed, info)` — all four action branches per step

V2 buffers add `piece_id/rotation/piece_row/piece_col` metadata (used by `decoder_explorer.py`). Old buffers load cleanly with `has_piece_meta=False`.

## Key constraints

- `patch_dim` must be divisible by 32 (GroupNorm groups=8, first stage has `patch_dim//4` channels).
- V1 checkpoints (flat 64-d latent, MLP predictor) are incompatible with V2 code; do not attempt to load them.
- The probe is trained on encoder outputs. Running it on predictor outputs degrades significantly — this is the known "distribution gap" limitation. `train_probe_rollout.py` partially addresses this.
- Device is auto-detected via `utils/device.py:get_device()` (MPS on Apple Silicon, CUDA if available, else CPU).
