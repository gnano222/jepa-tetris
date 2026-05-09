"""Shared loaders for JEPA + decoder checkpoints.

Used by `scripts/visualize_predictions.py` and `scripts/decoder_explorer.py`
so both consumers see the same model, eval-mode, no-grad setup.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from jepa_tetris.models.action_encoder import ActionEncoder
from jepa_tetris.models.decoder import StateDecoder
from jepa_tetris.models.encoder import StateEncoder
from jepa_tetris.models.predictor import Predictor


@dataclass
class JepaBundle:
    """All four JEPA submodules in eval mode with grads disabled."""

    encoder: StateEncoder
    target_encoder: StateEncoder
    action_encoder: ActionEncoder
    predictor: Predictor
    latent_dim: int
    args: dict


def load_jepa(path: str, device: torch.device) -> JepaBundle:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    latent_dim = args.get("latent_dim", 64)

    encoder = StateEncoder(latent_dim=latent_dim).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    target_encoder = StateEncoder(latent_dim=latent_dim).to(device)
    target_encoder.load_state_dict(ckpt["target_encoder"])
    target_encoder.eval()

    action_encoder = ActionEncoder().to(device)
    action_encoder.load_state_dict(ckpt["action_encoder"])
    action_encoder.eval()

    predictor = Predictor(
        latent_dim=latent_dim,
        action_emb_dim=action_encoder.embed_dim,
        hidden=args.get("predictor_hidden", 256),
        depth=args.get("predictor_depth", 2),
        residual=args.get("predictor_residual", False),
    ).to(device)
    predictor.load_state_dict(ckpt["predictor"])
    predictor.eval()

    for m in (encoder, target_encoder, action_encoder, predictor):
        for p in m.parameters():
            p.requires_grad_(False)

    return JepaBundle(
        encoder=encoder,
        target_encoder=target_encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        latent_dim=latent_dim,
        args=args,
    )


def load_decoder(path: str, latent_dim: int, device: torch.device) -> StateDecoder:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt_dim = ckpt.get("latent_dim", latent_dim)
    if ckpt_dim != latent_dim:
        raise ValueError(
            f"decoder latent_dim ({ckpt_dim}) != JEPA latent_dim ({latent_dim})"
        )
    decoder = StateDecoder(latent_dim=latent_dim).to(device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    return decoder
