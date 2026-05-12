"""Latent-space predictor: ((B, N, D) patches, (B, D) action) -> (B, N, D) next patches.

A small ViT-style transformer block. Patches + action are concatenated into a
sequence of N+1 tokens, given learned positional embeddings, run through
self-attention layers; the first N output tokens are the next-state patches.
Optionally outputs the residual delta added to z (predicting Δz is typically
more stable since most actions don't change z dramatically).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(
        self,
        patch_dim: int = 128,
        num_patches: int = 6,
        num_heads: int = 4,
        depth: int = 2,
        ff_mult: int = 4,
        residual: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.patch_dim = patch_dim
        self.num_patches = num_patches
        self.residual = residual

        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches + 1, patch_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=patch_dim,
            nhead=num_heads,
            dim_feedforward=patch_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, z: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        """
        z:     (B, N, D)  current patch tokens
        a_emb: (B, D)     action token
        returns (B, N, D) next-state patch tokens.
        """
        a_token = a_emb.unsqueeze(1)                           # (B, 1, D)
        seq = torch.cat([z, a_token], dim=1)                   # (B, N+1, D)
        seq = seq + self.pos_emb
        out = self.transformer(seq)                            # (B, N+1, D)
        delta = out[:, : self.num_patches]                     # (B, N, D)
        return z + delta if self.residual else delta
