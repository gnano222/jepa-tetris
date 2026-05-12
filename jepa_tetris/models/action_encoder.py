"""Action embedding."""
from __future__ import annotations

import torch
import torch.nn as nn


class ActionEncoder(nn.Module):
    """Embeds discrete actions to D-dim tokens.

    Default `embed_dim` matches the encoder's default `patch_dim` so the
    action token can drop straight into the predictor's patch sequence
    without a projection layer.
    """

    def __init__(self, num_actions: int = 4, embed_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(num_actions, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.embed(a)
