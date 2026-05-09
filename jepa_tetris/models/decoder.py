"""Post-hoc decoder probe: latent z -> board logits (B, 2, 20, 10).

Mirrors the encoder in encoder.py with ConvTranspose2d. Trained with frozen
JEPA weights (see scripts/train_decoder.py); used only for visualization, never
to influence representation learning.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class StateDecoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.head = nn.Linear(latent_dim, 128 * 3 * 2)
        # Spatial chain: (3, 2) -> (5, 3) -> (10, 5) -> (20, 10).
        # output_padding tuned per layer to match encoder's downsampling.
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 2, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.head(z).view(-1, 128, 3, 2)
        return self.deconv(h)
