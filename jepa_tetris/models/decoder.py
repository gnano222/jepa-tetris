"""Post-hoc decoder probe: (B, N, D) patches -> board logits (B, 2, 20, 10).

Mirrors the encoder with ConvTranspose2d. Reshapes the patch grid back to a
spatial feature map and upsamples 3x. Trained with frozen JEPA weights (see
scripts/train_decoder.py); used only for visualization, never to influence
representation learning.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class StateDecoder(nn.Module):
    def __init__(
        self,
        patch_dim: int = 128,
        out_h: int = 3,
        out_w: int = 2,
    ):
        super().__init__()
        self.patch_dim = patch_dim
        self.out_h = out_h
        self.out_w = out_w
        c1 = patch_dim // 2
        c2 = patch_dim // 4
        # Spatial chain: (3, 2) -> (5, 3) -> (10, 5) -> (20, 10).
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(patch_dim, c1, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.GroupNorm(8, c1),
            nn.GELU(),
            nn.ConvTranspose2d(c1, c2, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.GroupNorm(8, c2),
            nn.GELU(),
            nn.ConvTranspose2d(c2, 2, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> (B, 2, 20, 10)."""
        B, N, D = z.shape
        if N != self.out_h * self.out_w:
            raise ValueError(
                f"Decoder expects N = out_h * out_w = {self.out_h * self.out_w}, got N={N}"
            )
        h = z.transpose(1, 2).reshape(B, D, self.out_h, self.out_w)
        return self.deconv(h)
