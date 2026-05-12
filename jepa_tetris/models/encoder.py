"""CNN state encoder: (B, 2, 20, 10) -> (B, N, D) patch-token grid.

The conv stack downsamples 3x with stride 2, ending in (B, D, H', W'); each
spatial cell of the final feature map becomes a patch token. For the default
20x10 board, this is N = H' * W' = 3 * 2 = 6 tokens of dim D.

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


def _spatial_after_strides(h: int, w: int, strides: list[tuple[int, int]]) -> tuple[int, int]:
    for sh, sw in strides:
        # Conv2d kernel=3, padding=1, stride=s: out = ceil(in/s).
        h = (h + sh - 1) // sh
        w = (w + sw - 1) // sw
    return h, w


class StateEncoder(nn.Module):
    """Produces a (B, N, D) patch-token grid from a (B, 2, H, W) board.

    Conv channels scale with `patch_dim`: [patch_dim//4, patch_dim//2, patch_dim].
    Three stride-2 convs downsample 20x10 -> 3x2, yielding N=6 patches.
    """

    def __init__(
        self,
        patch_dim: int = 128,
        residual_blocks: int = 0,
        aux_channels: bool = False,
        board_h: int = 20,
        board_w: int = 10,
    ):
        super().__init__()
        if patch_dim % 32 != 0:
            raise ValueError(
                f"patch_dim must be divisible by 32 for GroupNorm(groups=8) on "
                f"the first conv stage (patch_dim//4 channels). got {patch_dim}."
            )
        self.patch_dim = patch_dim
        self.use_aux_channels = aux_channels
        self.board_h = board_h
        self.board_w = board_w

        in_channels = 2 + (3 if aux_channels else 0)
        strides = [(2, 2), (2, 2), (2, 2)]
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
        self.num_patches = out_h * out_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C_in, H, W) -> (B, num_patches, patch_dim)."""
        if self.use_aux_channels:
            aux = compute_aux_channels(x)
            x = torch.cat([x, aux], dim=1)
        h = self.conv(x)                                       # (B, D, H', W')
        return h.flatten(2).transpose(1, 2).contiguous()       # (B, N, D)


def make_encoder_from_args(args: dict, device=None) -> StateEncoder:
    """Reconstruct StateEncoder from a training checkpoint's stored args dict."""
    enc = StateEncoder(
        patch_dim=args["patch_dim"],
        residual_blocks=args.get("encoder_residual_blocks", 0),
        aux_channels=args.get("encoder_aux_channels", False),
    )
    if device is not None:
        enc = enc.to(device)
    return enc
