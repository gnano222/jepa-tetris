"""Shared loaders for JEPA + decoder checkpoints.

Used by `scripts/visualize_predictions.py` and `scripts/decoder_explorer.py`
so both consumers see the same model, eval-mode, no-grad setup.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from jepa_tetris.models.action_encoder import ActionEncoder
from jepa_tetris.models.decoder import StateDecoder
from jepa_tetris.models.encoder import StateEncoder, make_encoder_from_args
from jepa_tetris.models.inverse_model import InverseModel
from jepa_tetris.models.predictor import Predictor


@dataclass
class JepaBundle:
    """All four JEPA submodules in eval mode with grads disabled.

    `inverse_model` is the optional Exp-10 ICM head — None for any checkpoint
    trained without --inverse-weight (including all pre-Exp-10 checkpoints).
    """

    encoder: StateEncoder
    target_encoder: StateEncoder
    action_encoder: ActionEncoder
    predictor: Predictor
    patch_dim: int
    num_patches: int
    args: dict
    inverse_model: InverseModel | None = None


def load_jepa(path: str, device: torch.device) -> JepaBundle:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    patch_dim = args["patch_dim"]

    encoder = make_encoder_from_args(args, device=device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    target_encoder = make_encoder_from_args(args, device=device)
    target_encoder.load_state_dict(ckpt["target_encoder"])
    target_encoder.eval()

    action_encoder = ActionEncoder(embed_dim=patch_dim).to(device)
    action_encoder.load_state_dict(ckpt["action_encoder"])
    action_encoder.eval()

    predictor = Predictor(
        patch_dim=patch_dim,
        num_patches=encoder.num_patches,
        num_heads=args.get("predictor_heads", 4),
        depth=args.get("predictor_depth", 2),
        residual=not args.get("predictor_no_residual", False),
        film=args.get("predictor_film", False),
        cross_attn=args.get("predictor_cross_attn", False),
        token_gate=args.get("predictor_token_gate", False),
        token_gate_k=args.get("token_gate_k", 21),
    ).to(device)
    predictor.load_state_dict(ckpt["predictor"])
    predictor.eval()

    modules = [encoder, target_encoder, action_encoder, predictor]

    # Exp-10 inverse model — only present if trained with --inverse-weight > 0.
    inverse_model = None
    if ckpt.get("inverse_model") is not None:
        inverse_model = InverseModel(
            patch_dim=patch_dim,
            num_patches=encoder.num_patches,
            num_heads=args.get("inverse_heads", 4),
            depth=args.get("inverse_depth", 2),
        ).to(device)
        inverse_model.load_state_dict(ckpt["inverse_model"])
        inverse_model.eval()
        modules.append(inverse_model)

    for m in modules:
        for p in m.parameters():
            p.requires_grad_(False)

    return JepaBundle(
        encoder=encoder,
        target_encoder=target_encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        patch_dim=patch_dim,
        num_patches=encoder.num_patches,
        args=args,
        inverse_model=inverse_model,
    )


def load_decoder(path: str, patch_dim: int, device: torch.device) -> StateDecoder:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt_dim = ckpt.get("patch_dim", patch_dim)
    if ckpt_dim != patch_dim:
        raise ValueError(
            f"decoder patch_dim ({ckpt_dim}) != JEPA patch_dim ({patch_dim})"
        )
    decoder = StateDecoder(patch_dim=patch_dim).to(device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    return decoder
