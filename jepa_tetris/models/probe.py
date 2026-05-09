"""Probe head: latent -> (lines_cleared, holes, aggregate_height)."""
from __future__ import annotations

import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        hidden: int = 64,
        num_targets: int = 3,
        depth: int = 1,
    ):
        super().__init__()
        layers = [nn.Linear(latent_dim, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, num_targets))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
