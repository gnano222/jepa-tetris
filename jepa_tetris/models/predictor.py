"""Latent-space predictor: (z, a_emb) -> z_next_pred.

Optionally learns the residual `z_next - z` (then outputs `z + delta`). Residual
prediction gives more stable multi-step rollouts because the predictor only
models deltas, not the full state — most actions don't change z dramatically.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        action_emb_dim: int = 16,
        hidden: int = 256,
        depth: int = 2,
        residual: bool = False,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.residual = residual
        layers: list[nn.Module] = [nn.Linear(latent_dim + action_emb_dim, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([z, a_emb], dim=-1))
        return z + delta if self.residual else delta
