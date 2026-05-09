"""CNN state encoder: (B, 2, 20, 10) -> (B, latent_dim)."""
from __future__ import annotations

import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),    # (B, 32, 10, 5)
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # (B, 64, 5, 3)
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (B, 128, 3, 2)
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        self.head = nn.Linear(128 * 3 * 2, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).flatten(1)
        return self.head(h)
