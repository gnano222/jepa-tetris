"""Latent-space predictor: ((B, N, D) patches, (B, D) action) -> (B, N, D) next patches.

Action conditioning modes (mutually exclusive):
- extra-token (default): action appended as (N+1)-th token in self-attention sequence.
- film: action produces per-layer (γ, β) that modulate every patch token after each
  transformer block (V-JEPA2-AC style).
- cross-attn: patches attend to the action as a dedicated KV token after each
  self-attention block; action never participates in patch-to-patch attention.
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
        film: bool = False,
        cross_attn: bool = False,
        token_gate: bool = False,
        token_gate_k: int = 21,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if film and cross_attn:
            raise ValueError("film and cross_attn are mutually exclusive")
        if token_gate and not film:
            raise ValueError("token_gate requires film=True")
        if token_gate and token_gate_k < 1:
            raise ValueError(f"token_gate_k must be >= 1, got {token_gate_k}")
        self.patch_dim = patch_dim
        self.num_patches = num_patches
        self.residual = residual
        self.film = film
        self.cross_attn = cross_attn
        self.token_gate = token_gate
        self.token_gate_k = token_gate_k
        # Diagnostic: the most recent gate mask (B, N), detached. None until a
        # token-gated forward runs. Read by train.py for the live_tokens log.
        self._last_mask = None

        # extra-token uses N+1 positions; film/cross-attn use N (action injected externally)
        seq_len = num_patches if (film or cross_attn) else num_patches + 1
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, patch_dim))
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

        if film:
            # Per-layer linear: action -> (γ, β) of shape (B, D) each
            self.film_layers = nn.ModuleList(
                [nn.Linear(patch_dim, 2 * patch_dim) for _ in range(depth)]
            )
        elif cross_attn:
            # Per-layer cross-attention: patches (Q) attend to action token (K, V)
            self.cross_attn_layers = nn.ModuleList(
                [nn.MultiheadAttention(patch_dim, num_heads, batch_first=True)
                 for _ in range(depth)]
            )
            self.cross_attn_norms = nn.ModuleList(
                [nn.LayerNorm(patch_dim) for _ in range(depth)]
            )

        if token_gate:
            # One gate logit per token, from the transformer output.
            self.gate_head = nn.Linear(patch_dim, 1)

    def _token_gate_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """Straight-through hard top-k token mask.

        seq: (B, N, D) transformer output. Returns (B, N, 1) — a binary mask
        with exactly min(k, N) ones per row (forward), with gradient routed
        through a sigmoid surrogate so every logit is shaped (backward).
        """
        logits = self.gate_head(seq).squeeze(-1)             # (B, N)
        N = logits.shape[-1]
        k = min(self.token_gate_k, N)
        hard = torch.zeros_like(logits)
        topk_idx = logits.topk(k, dim=-1).indices            # (B, k)
        hard.scatter_(-1, topk_idx, 1.0)                     # (B, N) in {0,1}
        soft = torch.sigmoid(logits)
        mask = hard + soft - soft.detach()                   # straight-through
        self._last_mask = hard.detach()                      # diagnostic
        return mask.unsqueeze(-1)                            # (B, N, 1)

    def forward(self, z: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        """
        z:     (B, N, D)  current patch tokens
        a_emb: (B, D)     action embedding
        returns (B, N, D) next-state patch tokens.
        """
        if self.film:
            seq = z + self.pos_emb  # (B, N, D)
            for layer, film_linear in zip(self.transformer_layers, self.film_layers):
                seq = layer(seq)
                gamma, beta = film_linear(a_emb).chunk(2, dim=-1)  # (B, D) each
                seq = gamma.unsqueeze(1) * seq + beta.unsqueeze(1)
            delta = seq
            if self.token_gate:
                # Gate which tokens may change; ungated tokens copied exactly.
                return z + self._token_gate_mask(seq) * delta

        elif self.cross_attn:
            seq = z + self.pos_emb       # (B, N, D)
            a_token = a_emb.unsqueeze(1) # (B, 1, D)
            for layer, ca, norm in zip(
                self.transformer_layers, self.cross_attn_layers, self.cross_attn_norms
            ):
                seq = layer(seq)
                ca_out, _ = ca(query=norm(seq), key=a_token, value=a_token)
                seq = seq + ca_out       # residual cross-attention
            delta = seq

        else:
            # Default: extra-token — action appended to self-attention sequence
            a_token = a_emb.unsqueeze(1)                 # (B, 1, D)
            seq = torch.cat([z, a_token], dim=1)         # (B, N+1, D)
            seq = seq + self.pos_emb
            for layer in self.transformer_layers:
                seq = layer(seq)
            delta = seq[:, : self.num_patches]           # (B, N, D)

        return z + delta if self.residual else delta
