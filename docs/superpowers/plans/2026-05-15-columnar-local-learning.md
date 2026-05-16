# Columnar Encoder with Local Learning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `ColumnarEncoder` with untied per-column conv stacks and a per-column local-loss training path, so Exp-6 can compare local learning against global backprop at fixed compute.

**Architecture:** 15 spatial columns (5×3 grid over the 20×10 board), each an independent conv stack with an overlapping receptive field. Fork B trains each column with its own single-step JEPA loss via a throwaway per-column predictor head; the global FiLM transformer predictor trains on the *detached* encoder output. Fork A reuses the same encoder under standard global backprop.

**Tech Stack:** PyTorch, existing `jepa_tetris` package, pytest.

Spec: `docs/superpowers/specs/2026-05-15-columnar-local-learning-design.md`.

---

### Task 1: `ColumnPredictorHead` and `ColumnarEncoder`

**Files:**
- Modify: `jepa_tetris/models/encoder.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_models.py` (add `ColumnarEncoder, ColumnPredictorHead` to the existing `from jepa_tetris.models.encoder import (...)`):

```python
def test_column_predictor_head_shape():
    head = ColumnPredictorHead(dim=128)
    z = torch.randn(8, 128)
    a = torch.randn(8, 128)
    assert head(z, a).shape == (8, 128)


def test_columnar_encoder_output_shape():
    enc = ColumnarEncoder(patch_dim=128)
    x = torch.randn(4, 2, 20, 10)
    z = enc(x)
    assert z.shape == (4, 15, 128)
    assert enc.num_patches == 15


def test_columnar_encoder_patch_dim_configurable():
    enc = ColumnarEncoder(patch_dim=64)
    z = enc(torch.randn(2, 2, 20, 10))
    assert z.shape == (2, 15, 64)


def test_columnar_encoder_regions_clamp_at_edges():
    """5x3 grid, margin 1: row splits [4]*5, col splits [3,4,3].
    Corner and centre regions are margin-expanded then clamped to the board."""
    enc = ColumnarEncoder(patch_dim=64, grid=(5, 3), margin=1)
    assert enc.regions[0] == (0, 5, 0, 4)      # grid cell (0,0)
    assert enc.regions[14] == (15, 20, 6, 10)  # grid cell (4,2)
    assert enc.regions[7] == (7, 13, 2, 8)     # grid cell (2,1), centre


def test_columnar_encoder_gradient_isolation():
    """The Fork B invariant: a loss from one column's output produces zero
    gradient on every other column's conv stack."""
    enc = ColumnarEncoder(patch_dim=64)
    z = enc(torch.randn(2, 2, 20, 10))
    z[:, 0].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in enc.stacks[0].parameters())
    for i in range(1, 15):
        for p in enc.stacks[i].parameters():
            assert p.grad is None or p.grad.abs().sum() == 0


def test_columnar_encoder_predictor_compat():
    enc = ColumnarEncoder(patch_dim=128)
    pred = Predictor(patch_dim=128, num_patches=enc.num_patches, film=True)
    z = enc(torch.randn(4, 2, 20, 10))
    a = torch.randn(4, 128)
    assert pred(z, a).shape == (4, 15, 128)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models.py -k columnar -v`
Expected: FAIL — `ImportError: cannot import name 'ColumnarEncoder'`.

- [ ] **Step 3: Implement `ColumnPredictorHead` and `ColumnarEncoder`**

Add to `jepa_tetris/models/encoder.py` (after `_ResidualBlock`, before `_spatial_after_strides`):

```python
def _split_evenly(total: int, parts: int) -> list[int]:
    """Split `total` into `parts` sizes; extra cells go to the centre tiles.
    _split_evenly(10, 3) -> [3, 4, 3];  _split_evenly(20, 5) -> [4,4,4,4,4]."""
    base, rem = divmod(total, parts)
    sizes = [base] * parts
    start = (parts - rem) // 2
    for i in range(start, start + rem):
        sizes[i] += 1
    return sizes


class ColumnPredictorHead(nn.Module):
    """Throwaway per-column predictor: (z_c, a_emb) -> predicted next z_c.

    FiLM-conditioned residual MLP. Exists only to generate a column's local
    training signal in Fork B; discarded after training.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.film = nn.Linear(dim, 2 * dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, z: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.lin1(z))
        gamma, beta = self.film(a_emb).chunk(2, dim=-1)
        h = gamma * h + beta
        return z + self.lin2(h)


class ColumnarEncoder(nn.Module):
    """Cortically-inspired encoder: one untied conv stack per spatial column.

    The board is partitioned into a `grid` of tiles; each column reads its
    tile plus a `margin`-cell overlap and emits one D-dim token. Columns share
    no weights. Output (B, num_columns, D) matches the StateEncoder contract.
    """

    def __init__(
        self,
        patch_dim: int = 128,
        grid: tuple[int, int] = (5, 3),
        margin: int = 1,
        board_h: int = 20,
        board_w: int = 10,
    ):
        super().__init__()
        if patch_dim % 32 != 0:
            raise ValueError(
                f"patch_dim must be divisible by 32 for GroupNorm(groups=8). got {patch_dim}."
            )
        self.patch_dim = patch_dim
        self.grid = grid
        self.margin = margin
        self.board_h = board_h
        self.board_w = board_w

        gr, gc = grid
        row_sizes = _split_evenly(board_h, gr)
        col_sizes = _split_evenly(board_w, gc)
        row_bounds = [0]
        for s in row_sizes:
            row_bounds.append(row_bounds[-1] + s)
        col_bounds = [0]
        for s in col_sizes:
            col_bounds.append(col_bounds[-1] + s)

        # regions: row-major list of (r0, r1, c0, c1), margin-expanded + clamped.
        self.regions: list[tuple[int, int, int, int]] = []
        for i in range(gr):
            for j in range(gc):
                r0 = max(0, row_bounds[i] - margin)
                r1 = min(board_h, row_bounds[i + 1] + margin)
                c0 = max(0, col_bounds[j] - margin)
                c1 = min(board_w, col_bounds[j + 1] + margin)
                self.regions.append((r0, r1, c0, c1))

        self.num_patches = gr * gc
        self.stacks = nn.ModuleList(
            [self._make_stack(patch_dim) for _ in range(self.num_patches)]
        )

    @staticmethod
    def _make_stack(patch_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(2, patch_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, patch_dim // 2),
            nn.GELU(),
            nn.Conv2d(patch_dim // 2, patch_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, patch_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 2, H, W) -> (B, num_patches, patch_dim)."""
        cols = []
        for stack, (r0, r1, c0, c1) in zip(self.stacks, self.regions):
            region = x[:, :, r0:r1, c0:c1]
            h = stack(region)                       # (B, D, h', w')
            cols.append(h.mean(dim=(2, 3)))         # (B, D) — MPS-safe global pool
        return torch.stack(cols, dim=1)             # (B, num_patches, D)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models.py -k "columnar or column_predictor" -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add jepa_tetris/models/encoder.py tests/test_models.py
git commit -m "feat: add ColumnarEncoder + ColumnPredictorHead"
```

---

### Task 2: Reconstruct `ColumnarEncoder` from checkpoint args

**Files:**
- Modify: `jepa_tetris/models/encoder.py` (`make_encoder_from_args`)
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models.py`:

```python
def test_make_encoder_from_args_columnar():
    args = {
        "patch_dim": 128,
        "encoder_columnar": True,
        "encoder_columnar_grid": "5x3",
        "encoder_columnar_margin": 1,
    }
    enc = make_encoder_from_args(args)
    z = enc(torch.randn(2, 2, 20, 10))
    assert z.shape == (2, 15, 128)
    assert enc.num_patches == 15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py::test_make_encoder_from_args_columnar -v`
Expected: FAIL — `make_encoder_from_args` builds a `StateEncoder`, which has `num_patches == 6`.

- [ ] **Step 3: Implement**

In `jepa_tetris/models/encoder.py`, replace the body of `make_encoder_from_args` with:

```python
def make_encoder_from_args(args: dict, device=None) -> nn.Module:
    """Reconstruct the encoder (StateEncoder or ColumnarEncoder) from a
    training checkpoint's stored args dict."""
    if args.get("encoder_columnar", False):
        grid = args.get("encoder_columnar_grid", "5x3")
        if isinstance(grid, str):
            gr, gc = (int(v) for v in grid.lower().split("x"))
        else:
            gr, gc = grid
        enc: nn.Module = ColumnarEncoder(
            patch_dim=args["patch_dim"],
            grid=(gr, gc),
            margin=args.get("encoder_columnar_margin", 1),
        )
    else:
        enc = StateEncoder(
            patch_dim=args["patch_dim"],
            residual_blocks=args.get("encoder_residual_blocks", 0),
            aux_channels=args.get("encoder_aux_channels", False),
            stride_stages=args.get("encoder_stride_stages", 3),
            two_scale=args.get("encoder_two_scale", False),
        )
    if device is not None:
        enc = enc.to(device)
    return enc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models.py -k "make_encoder_from_args" -v`
Expected: PASS (all 4 `make_encoder_from_args` tests, including the existing StateEncoder ones).

- [ ] **Step 5: Commit**

```bash
git add jepa_tetris/models/encoder.py tests/test_models.py
git commit -m "feat: reconstruct ColumnarEncoder from checkpoint args"
```

---

### Task 3: `columnar_local_loss` helper

**Files:**
- Modify: `jepa_tetris/train.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models.py` (add `import torch.nn as nn` at the top if not present — it is not; add it):

```python
def test_columnar_local_loss_isolation_and_finiteness():
    """Local loss is finite, and each column's conv stack receives gradient
    only from its own term (verified via the encoder, end to end)."""
    from jepa_tetris.models.encoder import ColumnarEncoder, ColumnPredictorHead
    from jepa_tetris.train import columnar_local_loss

    torch.manual_seed(0)
    enc = ColumnarEncoder(patch_dim=64)
    heads = nn.ModuleList([ColumnPredictorHead(64) for _ in range(enc.num_patches)])
    z0 = enc(torch.randn(3, 2, 20, 10))            # (3, 15, 64), with grad
    z1_target = torch.randn(3, 15, 64)             # detached target stand-in
    a_emb = torch.randn(3, 64)

    loss, parts = columnar_local_loss(
        z0_online=z0, z1_target=z1_target, a_emb=a_emb, column_heads=heads,
        var_weight=1.0, cov_weight=0.04, target_std=1.0,
    )
    assert torch.isfinite(loss)
    assert {"mse_local", "var_local", "cov_local"} <= parts.keys()

    loss.backward()
    # Every column stack got gradient (all columns contribute to the loss).
    for i in range(enc.num_patches):
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in enc.stacks[i].parameters())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py::test_columnar_local_loss_isolation_and_finiteness -v`
Expected: FAIL — `ImportError: cannot import name 'columnar_local_loss'`.

- [ ] **Step 3: Implement the helper**

In `jepa_tetris/train.py`, add this function immediately after `counterfactual_step_loss` (before `ema_update`):

```python
def columnar_local_loss(
    *,
    z0_online: torch.Tensor,
    z1_target: torch.Tensor,
    a_emb: torch.Tensor,
    column_heads: torch.nn.ModuleList,
    var_weight: float,
    cov_weight: float,
    target_std: float,
) -> tuple[torch.Tensor, dict]:
    """Per-column single-step JEPA loss (Fork B encoder signal).

    Each column c predicts its own next-state latent via its own throwaway
    head and is supervised against the (stop-grad) target encoder's column c.
    The total loss is a plain sum over columns, so autograd confines each
    column's gradient to its own conv stack.

    z0_online: (B, N, D) online columnar encoder output at t=0 (requires grad)
    z1_target: (B, N, D) target encoder output at t=1 (no grad)
    a_emb:     (B, D)
    Returns (scalar loss, {"mse_local", "var_local", "cov_local"}).
    """
    N = z0_online.shape[1]
    mse_total = z0_online.new_zeros(())
    var_total = z0_online.new_zeros(())
    cov_total = z0_online.new_zeros(())
    for c in range(N):
        z_pred_c = column_heads[c](z0_online[:, c], a_emb)        # (B, D)
        mse_total = mse_total + F.mse_loss(z_pred_c, z1_target[:, c])
        var_total = var_total + variance_loss(z0_online[:, c], target_std=target_std)
        cov_total = cov_total + covariance_loss(z0_online[:, c])
    n = float(N)
    mse_local = mse_total / n
    var_local = var_total / n
    cov_local = cov_total / n
    loss = mse_local + var_weight * var_local + cov_weight * cov_local
    return loss, {
        "mse_local": mse_local.item(),
        "var_local": var_local.item(),
        "cov_local": cov_local.item(),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py::test_columnar_local_loss_isolation_and_finiteness -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jepa_tetris/train.py tests/test_models.py
git commit -m "feat: add columnar_local_loss helper for Fork B training"
```

---

### Task 4: Wire columnar + local-loss flags into `train.py`

**Files:**
- Modify: `jepa_tetris/train.py`

This task is CLI/orchestration wiring; it is verified by the smoke runs in Task 5.

- [ ] **Step 1: Add CLI arguments**

In `main()`, after the `--encoder-two-scale` argument block, add:

```python
    parser.add_argument("--encoder-columnar", action="store_true",
                        help="Use ColumnarEncoder (untied per-column conv stacks) "
                             "instead of StateEncoder. N=15 for the default 5x3 grid.")
    parser.add_argument("--encoder-columnar-grid", default="5x3",
                        help="Columnar grid as 'GRxGC' (default 5x3 -> 15 columns).")
    parser.add_argument("--encoder-columnar-margin", type=int, default=1,
                        help="Overlap margin in cells for each column's receptive field.")
    parser.add_argument("--local-loss", action="store_true",
                        help="Fork B: train the columnar encoder with per-column local "
                             "losses; the global predictor trains on the detached encoder "
                             "output. Requires --encoder-columnar.")
```

- [ ] **Step 2: Validate flag combinations**

In `main()`, immediately after `args = parser.parse_args()`, add:

```python
    if args.local_loss and not args.encoder_columnar:
        parser.error("--local-loss requires --encoder-columnar")
    if args.local_loss and args.counterfactual:
        parser.error("--local-loss is incompatible with --counterfactual")
```

- [ ] **Step 3: Build the columnar encoder and per-column heads**

Replace the `encoder = StateEncoder(...)` construction block with:

```python
    if args.encoder_columnar:
        from jepa_tetris.models.encoder import ColumnarEncoder
        gr, gc = (int(v) for v in args.encoder_columnar_grid.lower().split("x"))
        encoder = ColumnarEncoder(
            patch_dim=args.patch_dim,
            grid=(gr, gc),
            margin=args.encoder_columnar_margin,
        ).to(device)
    else:
        encoder = StateEncoder(
            patch_dim=args.patch_dim,
            residual_blocks=args.encoder_residual_blocks,
            aux_channels=args.encoder_aux_channels,
            stride_stages=args.encoder_stride_stages,
            two_scale=args.encoder_two_scale,
        ).to(device)
```

After the `predictor = Predictor(...)` construction, add the per-column heads:

```python
    column_heads = None
    if args.local_loss:
        from jepa_tetris.models.encoder import ColumnPredictorHead
        column_heads = torch.nn.ModuleList(
            [ColumnPredictorHead(args.patch_dim) for _ in range(encoder.num_patches)]
        ).to(device)
```

Then extend the `params` list so the heads are optimized:

```python
    params = (
        list(encoder.parameters())
        + list(action_encoder.parameters())
        + list(predictor.parameters())
    )
    if column_heads is not None:
        params += list(column_heads.parameters())
```

- [ ] **Step 4: Add the Fork B branch in the training loop**

In the `for step in pbar:` loop, the current top-level structure is `if args.counterfactual: ... else: ...`. Change it to `if args.counterfactual: ... elif args.local_loss: ... else: ...` by inserting this `elif` branch between them:

```python
        elif args.local_loss:
            batch = buf.sample_rollout(args.batch_size, H, rng=rng)
            s0 = torch.from_numpy(batch["s0"]).to(device)              # (B, *state)
            actions = torch.from_numpy(batch["actions"]).to(device)    # (B, H)
            s_next_k = torch.from_numpy(batch["s_next_k"]).to(device)  # (B, H, *state)
            B = s0.shape[0]

            frames = torch.cat([s0.unsqueeze(1), s_next_k], dim=1)     # (B, H+1, *state)
            frames_flat = frames.reshape(B * (H + 1), *state_shape)

            z_all_flat = encoder(frames_flat)                          # (B*(H+1), N, D)
            N, D = z_all_flat.shape[1], z_all_flat.shape[2]
            z_all = z_all_flat.view(B, H + 1, N, D)
            with torch.no_grad():
                z_target = target_encoder(
                    frames[:, 1:].reshape(B * H, *state_shape)
                ).view(B, H, N, D)

            # --- Fork B encoder signal: per-column single-step local loss ---
            local_loss, local_parts = columnar_local_loss(
                z0_online=z_all[:, 0],
                z1_target=z_target[:, 0],
                a_emb=action_encoder(actions[:, 0]),
                column_heads=column_heads,
                var_weight=args.var_weight,
                cov_weight=args.cov_weight,
                target_std=args.target_std,
            )

            # --- Global predictor: teacher-forced H on the DETACHED encoder out ---
            z_in = z_all[:, :H].detach().reshape(B * H, N, D)
            a_emb_tf = action_encoder(actions.reshape(B * H))
            z_pred = predictor(z_in, a_emb_tf).view(B, H, N, D)
            mse_tf = F.mse_loss(z_pred, z_target)

            mse = mse_tf
            mse_tf_val = mse_tf.item()
            mse_ar_val = None
            loss = local_loss + mse_tf
            var_loss = z_all.new_tensor(local_parts["var_local"])
            cov_loss = z_all.new_tensor(local_parts["cov_local"])

            z_pred_log = z_pred[:, 0]
            z_target_log = z_target[:, 0]
            z_pred_last = z_pred[:, -1]
            z_target_last = z_target[:, -1]
            z_for_vic = z_all
            _local_mse_log = local_parts["mse_local"]
```

- [ ] **Step 5: Log the local-loss MSE**

In the logging block (`if log_now:`), after the existing `if mse_ar_val is not None:` block, add:

```python
            if args.local_loss:
                record["mse_local"] = _local_mse_log
```

Note: `cos_sim_kK` for the local-loss branch uses `z_pred_last`/`z_target_last` (the predictor's last teacher-forced step), which are set in the branch — the existing `else` path of the `if args.counterfactual` logging guard already handles this correctly since `args.counterfactual` is False here.

- [ ] **Step 6: Run the existing test suite**

Run: `pytest -q`
Expected: PASS (109 tests — 106 baseline + 3 new test functions, with parametrization the count may differ; 0 failures).

- [ ] **Step 7: Commit**

```bash
git add jepa_tetris/train.py
git commit -m "feat: wire --encoder-columnar and --local-loss into train.py"
```

---

### Task 5: Local smoke runs (Fork A and Fork B)

**Files:** none modified — verification only.

- [ ] **Step 1: Locate a buffer**

Run: `ls -la data/*.npz`
Expected: at least one buffer file (e.g. `data/buffer.npz`). If none exists, collect a tiny one:
`python -m jepa_tetris.data.collect --episodes 200 --capacity 20000 --policy mixed --epsilon 0.4 --out data/smoke.npz --seed 0`

- [ ] **Step 2: Fork A smoke run (200 steps)**

Run (substitute the buffer path found in Step 1):
```bash
python -m jepa_tetris.train --buffer data/buffer.npz --steps 200 \
    --batch-size 64 --horizon-h 4 --predictor-film \
    --encoder-columnar --run smoke-forkA --out checkpoints/smoke_forkA.pt --seed 0
```
Expected: completes without error; `checkpoints/smoke_forkA.pt` written; the post-training multistep table prints finite cos_sim values.

- [ ] **Step 3: Fork B smoke run (200 steps)**

```bash
python -m jepa_tetris.train --buffer data/buffer.npz --steps 200 \
    --batch-size 64 --horizon-h 4 --predictor-film \
    --encoder-columnar --local-loss --run smoke-forkB --out checkpoints/smoke_forkB.pt --seed 0
```
Expected: completes without error; the training log shows a `mse_local` field; `checkpoints/smoke_forkB.pt` written; multistep table prints finite values.

- [ ] **Step 4: Verify checkpoint reloads**

Run:
```bash
python -c "
from jepa_tetris.utils.checkpoint import load_jepa
from jepa_tetris.utils.device import get_device
b = load_jepa('checkpoints/smoke_forkB.pt', get_device())
print('num_patches', b.num_patches, 'columnar', b.args.get('encoder_columnar'))
assert b.num_patches == 15 and b.args.get('encoder_columnar') is True
print('OK')
"
```
Expected: prints `num_patches 15 columnar True` then `OK`.

- [ ] **Step 5: Commit (smoke checkpoints are throwaway — do not commit them)**

```bash
git status   # confirm only intended files staged; checkpoints/ should be gitignored
```
No commit needed if nothing changed. If `checkpoints/` is not ignored, do not stage the smoke `.pt` files.

---

### Task 6: RunPod head-to-head training run

**Files:** none modified — uses the `runpod-training-workflow` skill.

- [ ] **Step 1: Push the branch**

```bash
git push -u origin columnar-local-learning
```

- [ ] **Step 2: Launch the two runs via the runpod-training-workflow skill**

Invoke the `runpod-training-workflow` skill. Launch two parallel runs from the `columnar-local-learning` branch, both at the standard budget:

- **Fork A:** `python -m jepa_tetris.train --buffer <data> --steps 100000 --batch-size 256 --horizon-h 4 --predictor-film --encoder-columnar --run forkA-100k --out checkpoints/jepa-forkA-100k.pt --seed 0`
- **Fork B:** `python -m jepa_tetris.train --buffer <data> --steps 100000 --batch-size 256 --horizon-h 4 --predictor-film --encoder-columnar --local-loss --run forkB-100k --out checkpoints/jepa-forkB-100k.pt --seed 0`

Use the same buffer the existing `film-100k` benchmark used (the standard mixed-exploration buffer).

- [ ] **Step 3: Retrieve checkpoints and run metrics**

For each returned checkpoint, run:
```bash
python scripts/multistep_accuracy.py --jepa <ckpt> --buffer <data>
python scripts/causality_diagnostic.py --jepa <ckpt> --buffer <data>
```

- [ ] **Step 4: Write up Exp-6 in FINDINGS.md**

Append an `## Exp-6` section to `docs/FINDINGS.md` following the existing experiment-log format: question, setup table, the 3-way results table (film-100k / Fork A / Fork B) for cos@k, MSE@k, DROP MSE, M1/M2/M4, and conclusions on the `A→B` gap. Commit.

---

## Self-Review

**Spec coverage:**
- ColumnarEncoder (grid, overlapping RF, untied stacks, `(B,15,D)` output) → Task 1 ✓
- Per-column predictor heads → Task 1 (`ColumnPredictorHead`), Task 4 (instantiated) ✓
- Per-column local loss + decoupled global predictor (detach) → Task 3 (helper), Task 4 (branch) ✓
- Fork A (columnar + global backprop, no local loss) → Task 4 (the `--encoder-columnar` path reuses the existing teacher-forced `else` branch) ✓
- Training flags → Task 4 ✓
- Checkpoint reconstruction → Task 2 ✓
- Tests: shape, edge clamping, gradient isolation, decoupling, args round-trip → Tasks 1–3 ✓
  (Decoupling is covered structurally: Task 4 Step 4 detaches `z_in` before the predictor; the gradient-isolation test in Task 1 plus the explicit `.detach()` make a separate decoupling unit test redundant. The Task 5 Fork B smoke run exercises the full detached path.)
- 3-way experiment, RunPod, Exp-6 writeup → Task 6 ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code. Buffer path in Tasks 5–6 is intentionally substituted at runtime (the exact filename depends on what exists in `data/`).

**Type consistency:** `ColumnarEncoder.num_patches`, `.stacks`, `.regions` used consistently across Tasks 1–5. `columnar_local_loss` signature (keyword-only, returns `(Tensor, dict)`) matches between Task 3 definition and Task 4 call site. `column_heads` is an `nn.ModuleList` throughout.
