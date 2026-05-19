"""Inverse dynamics model: ((B, N, D), (B, N, D)) -> (B, NUM_ACTIONS) action logits.

The Predictor's mirror twin. Where the Predictor maps (patches, action) -> next
patches, the InverseModel maps (patches, next patches) -> the action between them.
Trained jointly over the *same* encoder as the predictor (the ICM recipe,
Pathak et al. 2017): recovering the action forces the encoder to keep
action-causal information such as precise piece position.

Unlike the Probe — which pools an unordered patch set — the inverse model keeps
spatial positional embeddings. LEFT vs RIGHT is only distinguishable by *where*
the change between the two states happened.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from jepa_tetris.env.tetris import NUM_ACTIONS


class InverseModel(nn.Module):
    def __init__(
        self,
        patch_dim: int = 128,
        num_patches: int = 6,
        num_heads: int = 4,
        depth: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.0,
        num_actions: int = NUM_ACTIONS,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.patch_dim = patch_dim
        self.num_patches = num_patches

        # Per-patch projection of [z ; z_next ; z_next - z] -> D. The change is
        # handed to the model explicitly rather than rediscovered by attention.
        self.in_proj = nn.Linear(3 * patch_dim, patch_dim)

        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches, patch_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        def _make_layer() -> nn.TransformerEncoderLayer:
            return nn.TransformerEncoderLayer(
                d_model=patch_dim,
                nhead=num_heads,
                dim_feedforward=patch_dim * ff_mult,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

        self.transformer_layers = nn.ModuleList([_make_layer() for _ in range(depth)])
        self.head = nn.Linear(patch_dim, num_actions)

    def forward(self, z: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        """
        z, z_next: (B, N, D) current and next patch tokens.
        returns (B, NUM_ACTIONS) action logits.
        """
        delta = z_next - z
        seq = self.in_proj(torch.cat([z, z_next, delta], dim=-1))  # (B, N, D)
        seq = seq + self.pos_emb
        for layer in self.transformer_layers:
            seq = layer(seq)
        pooled = seq.mean(dim=1)                                   # (B, D)
        return self.head(pooled)                                   # (B, NUM_ACTIONS)
