"""Action embedding."""
from __future__ import annotations

import torch
import torch.nn as nn


class ActionEncoder(nn.Module):
    def __init__(self, num_actions: int = 4, embed_dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(num_actions, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.embed(a)
