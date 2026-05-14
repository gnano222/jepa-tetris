"""Latent-space predictor: ((B, N, D) patches, (B, D) action) -> (B, N, D) next patches.

Action conditioning modes (mutually exclusive):
- extra-token (default): action appended as (N+1)-th token in self-attention sequence.
- film: action produces per-layer (γ, β) that modulate every patch token after each
  transformer block (V-JEPA2-AC style). Broadcast: same γ/β for all patches.
- spatial-film: action fused with each patch's positional embedding before computing
  (γ, β), giving each patch its own modulation. Strictly more expressive than film.
- hierarchical-film: like spatial-film but the action context is updated after each
  layer by pooling the current sequence state back into it. Each layer conditions on
  action + what prior layers have already predicted (mirrors cortical feedback hierarchy).
- hierarchical-film-attn: like hierarchical-film but replaces mean pooling with
  cross-attention — action context attends selectively to patches rather than averaging,
  so the feedback focuses on the most action-relevant regions (e.g. landing column for DROP).
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
        spatial_film: bool = False,
        hierarchical_film: bool = False,
        hierarchical_film_attn: bool = False,
        cross_attn: bool = False,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if sum([film, spatial_film, hierarchical_film, hierarchical_film_attn, cross_attn]) > 1:
            raise ValueError("film, spatial_film, hierarchical_film, hierarchical_film_attn, and cross_attn are mutually exclusive")
        self.patch_dim = patch_dim
        self.num_patches = num_patches
        self.residual = residual
        self.film = film
        self.spatial_film = spatial_film
        self.hierarchical_film = hierarchical_film
        self.hierarchical_film_attn = hierarchical_film_attn
        self.cross_attn = cross_attn

        # extra-token uses N+1 positions; all other modes use N (action injected externally)
        seq_len = num_patches if (film or spatial_film or hierarchical_film or hierarchical_film_attn or cross_attn) else num_patches + 1
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
        elif spatial_film:
            # Per-layer linear: (action + pos_emb) -> per-patch (γ, β) of shape (B, N, D) each
            self.spatial_film_layers = nn.ModuleList(
                [nn.Linear(patch_dim, 2 * patch_dim) for _ in range(depth)]
            )
        elif hierarchical_film:
            # Same structure as spatial_film; differs in forward: action context updated each layer
            self.hierarchical_film_layers = nn.ModuleList(
                [nn.Linear(patch_dim, 2 * patch_dim) for _ in range(depth)]
            )
        elif hierarchical_film_attn:
            # Like hierarchical_film but uses cross-attention to pool seq → a_ctx each layer,
            # so the feedback is selective (attends to relevant patches) rather than a mean.
            self.hierarchical_film_layers = nn.ModuleList(
                [nn.Linear(patch_dim, 2 * patch_dim) for _ in range(depth)]
            )
            self.pool_attn_layers = nn.ModuleList(
                [nn.MultiheadAttention(patch_dim, num_heads, batch_first=True)
                 for _ in range(depth)]
            )
            self.pool_attn_norms = nn.ModuleList(
                [nn.LayerNorm(patch_dim) for _ in range(depth)]
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

        elif self.spatial_film:
            seq = z + self.pos_emb  # (B, N, D)
            # Fuse action with position once per forward (pos_emb is (1, N, D))
            a_spatial = a_emb.unsqueeze(1) + self.pos_emb  # (B, N, D)
            for layer, film_linear in zip(self.transformer_layers, self.spatial_film_layers):
                seq = layer(seq)
                gamma, beta = film_linear(a_spatial).chunk(2, dim=-1)  # (B, N, D) each
                seq = gamma * seq + beta  # per-patch modulation
            delta = seq

        elif self.hierarchical_film:
            seq = z + self.pos_emb  # (B, N, D)
            a_ctx = a_emb  # (B, D) — evolves each layer
            for layer, film_linear in zip(self.transformer_layers, self.hierarchical_film_layers):
                seq = layer(seq)
                a_spatial = a_ctx.unsqueeze(1) + self.pos_emb  # (B, N, D)
                gamma, beta = film_linear(a_spatial).chunk(2, dim=-1)  # (B, N, D) each
                seq = gamma * seq + beta
                a_ctx = seq.mean(dim=1)  # pool state → enrich action context for next layer
            delta = seq

        elif self.hierarchical_film_attn:
            seq = z + self.pos_emb  # (B, N, D)
            a_ctx = a_emb  # (B, D) — evolves each layer
            for layer, film_linear, pool_attn, pool_norm in zip(
                self.transformer_layers,
                self.hierarchical_film_layers,
                self.pool_attn_layers,
                self.pool_attn_norms,
            ):
                seq = layer(seq)
                a_spatial = a_ctx.unsqueeze(1) + self.pos_emb  # (B, N, D)
                gamma, beta = film_linear(a_spatial).chunk(2, dim=-1)  # (B, N, D) each
                seq = gamma * seq + beta
                # Cross-attention pool: a_ctx queries the sequence selectively
                # instead of averaging — focuses on the patches most relevant to the action.
                a_ctx_new, _ = pool_attn(
                    query=pool_norm(a_ctx.unsqueeze(1)),  # (B, 1, D)
                    key=seq,
                    value=seq,
                )
                a_ctx = a_ctx_new.squeeze(1)  # (B, D)
            delta = seq

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
