"""CNN state encoder: (B, 2, 20, 10) -> (B, N, D) patch-token grid.

The conv stack downsamples with stride-2 convs, ending in (B, D, H', W'); each
spatial cell of the final feature map becomes a patch token.

stride_stages controls spatial resolution:
  3 (default): 20x10 -> 10x5 -> 5x3 -> 3x2  → N=6  patches  (V2 baseline)
  2:           20x10 -> 10x5 -> 5x3           → N=15 patches  (finer granularity)

two_scale (requires stride_stages=2):
  Fine tokens (5x3=15) + coarse tokens (3x2=6 via adaptive avg pool) → N=21.
  Zero extra parameters; the coarse stream is a pooled view of the same feature map.

Flags:
- aux_channels: prepend hand-engineered features (column heights, holes mask,
  bumpiness) to the input; total input channels become 2 + 3 = 5
- residual_blocks: N residual blocks at each stage (after the stride-2 conv)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_aux_channels(x: torch.Tensor) -> torch.Tensor:
    """Compute 3 hand-engineered channels from raw state (B, 2, 20, 10).

    Returns (B, 3, 20, 10):
        - column height map (fraction of column occupied, broadcast over rows)
        - holes mask (empty cells with at least one occupied cell above)
        - bumpiness (|height[c] - height[c+1]|, broadcast over rows)
    """
    B, _, H, W = x.shape
    board = x[:, 0]                                            # (B, H, W)
    occupied = (board > 0).float()
    col_heights = occupied.sum(dim=1, keepdim=True) / H        # (B, 1, W) in [0,1]
    height_channel = col_heights.expand(B, H, W)               # (B, H, W)

    cum_above = torch.cummax(occupied, dim=1).values
    holes_mask = (1.0 - occupied) * cum_above                  # (B, H, W)

    diffs = torch.abs(col_heights[:, :, 1:] - col_heights[:, :, :-1])  # (B, 1, W-1)
    pad = torch.zeros(B, 1, 1, device=x.device, dtype=x.dtype)
    bumpiness = torch.cat([diffs, pad], dim=2)                 # (B, 1, W)
    bumpiness_channel = bumpiness.expand(B, H, W)

    return torch.stack([height_channel, holes_mask, bumpiness_channel], dim=1)


class _ResidualBlock(nn.Module):
    """Pre-activation Conv-GN-GELU-Conv-GN residual block at a single resolution."""

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.gn1(h)
        h = F.gelu(h)
        h = self.conv2(h)
        h = self.gn2(h)
        return F.gelu(x + h)


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


def _spatial_after_strides(h: int, w: int, strides: list[tuple[int, int]]) -> tuple[int, int]:
    for sh, sw in strides:
        # Conv2d kernel=3, padding=1, stride=s: out = ceil(in/s).
        h = (h + sh - 1) // sh
        w = (w + sw - 1) // sw
    return h, w


class StateEncoder(nn.Module):
    """Produces a (B, N, D) patch-token grid from a (B, 2, H, W) board.

    stride_stages=3 (default): channels [D//4, D//2, D], 20x10 -> 3x2, N=6 patches.
    stride_stages=2:           channels [D//2, D],       20x10 -> 5x3, N=15 patches.
    two_scale=True:            stride_stages=2 fine (15) + adaptive-pooled coarse (6) = N=21.
    """

    def __init__(
        self,
        patch_dim: int = 128,
        residual_blocks: int = 0,
        aux_channels: bool = False,
        board_h: int = 20,
        board_w: int = 10,
        stride_stages: int = 3,
        two_scale: bool = False,
    ):
        super().__init__()
        if patch_dim % 32 != 0:
            raise ValueError(
                f"patch_dim must be divisible by 32 for GroupNorm(groups=8). got {patch_dim}."
            )
        if stride_stages not in (2, 3):
            raise ValueError(f"stride_stages must be 2 or 3, got {stride_stages}")
        if two_scale and stride_stages != 2:
            raise ValueError("two_scale requires stride_stages=2")
        self.patch_dim = patch_dim
        self.stride_stages = stride_stages
        self.two_scale = two_scale
        self.use_aux_channels = aux_channels
        self.board_h = board_h
        self.board_w = board_w

        in_channels = 2 + (3 if aux_channels else 0)
        strides = [(2, 2)] * stride_stages
        if stride_stages == 2:
            channels = [patch_dim // 2, patch_dim]
        else:
            channels = [patch_dim // 4, patch_dim // 2, patch_dim]

        prev_c = in_channels
        stages: list[nn.Module] = []
        for c, stride in zip(channels, strides):
            stage_layers: list[nn.Module] = [
                nn.Conv2d(prev_c, c, kernel_size=3, stride=stride, padding=1),
                nn.GroupNorm(8, c),
                nn.GELU(),
            ]
            for _ in range(residual_blocks):
                stage_layers.append(_ResidualBlock(c, groups=8))
            stages.append(nn.Sequential(*stage_layers))
            prev_c = c
        self.conv = nn.Sequential(*stages)

        out_h, out_w = _spatial_after_strides(board_h, board_w, strides)
        self.out_spatial = (out_h, out_w)

        if two_scale:
            # Coarse stream: pool fine (5,3) map down to (3,2) — same resolution as V2 baseline.
            self.coarse_pool = nn.AdaptiveAvgPool2d((3, 2))
            self.num_patches = out_h * out_w + 3 * 2  # 15 fine + 6 coarse = 21
        else:
            self.num_patches = out_h * out_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C_in, H, W) -> (B, num_patches, patch_dim)."""
        if self.use_aux_channels:
            aux = compute_aux_channels(x)
            x = torch.cat([x, aux], dim=1)
        h = self.conv(x)                                              # (B, D, H', W')
        fine = h.flatten(2).transpose(1, 2).contiguous()             # (B, N_fine, D)
        if self.two_scale:
            # AdaptiveAvgPool2d with non-integer ratios (5->3, 3->2) is unsupported on MPS.
            h_pool = h.cpu() if h.device.type == "mps" else h
            coarse = self.coarse_pool(h_pool).to(h.device)            # (B, D, 3, 2)
            coarse = coarse.flatten(2).transpose(1, 2).contiguous()  # (B, 6, D)
            return torch.cat([fine, coarse], dim=1)                   # (B, 21, D)
        return fine


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
