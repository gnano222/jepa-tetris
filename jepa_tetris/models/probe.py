"""Probe head: (B, N, D) patches -> (B, num_targets).

A learnable query attends over the patch tokens to produce a single pooled
vector, then a small MLP outputs (lines_cleared, holes, aggregate_height).
Patch-aware in the same spirit as a [CLS]-token pooling over ViT features.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(
        self,
        patch_dim: int = 128,
        hidden: int = 64,
        num_targets: int = 3,
        depth: int = 1,
        num_heads: int = 4,
    ):
        super().__init__()
        self.patch_dim = patch_dim
        self.query = nn.Parameter(torch.zeros(1, 1, patch_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=patch_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        layers: list[nn.Module] = [nn.Linear(patch_dim, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, num_targets))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> (B, num_targets)."""
        B = z.shape[0]
        q = self.query.expand(B, -1, -1)                       # (B, 1, D)
        pooled, _ = self.attn(q, z, z, need_weights=False)     # (B, 1, D)
        return self.mlp(pooled.squeeze(1))                     # (B, num_targets)
