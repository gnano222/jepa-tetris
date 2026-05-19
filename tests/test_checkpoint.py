"""Checkpoint save/load — back-compat for the Exp-10 inverse_model field."""
import argparse

import torch

from jepa_tetris.models.action_encoder import ActionEncoder
from jepa_tetris.models.encoder import StateEncoder
from jepa_tetris.models.inverse_model import InverseModel
from jepa_tetris.models.predictor import Predictor
from jepa_tetris.train import save_checkpoint
from jepa_tetris.utils.checkpoint import load_jepa


def _make_models(patch_dim=64):
    enc = StateEncoder(patch_dim=patch_dim)
    tgt = StateEncoder(patch_dim=patch_dim)
    ae = ActionEncoder(embed_dim=patch_dim)
    pred = Predictor(patch_dim=patch_dim, num_patches=enc.num_patches)
    return enc, tgt, ae, pred


def _args(patch_dim=64):
    return argparse.Namespace(
        patch_dim=patch_dim,
        predictor_heads=4,
        predictor_depth=2,
        inverse_depth=2,
        inverse_heads=4,
    )


def test_save_checkpoint_omits_inverse_model_when_none(tmp_path):
    """A forward-only run (--inverse-weight 0) writes no inverse_model key —
    the checkpoint format is byte-identical to pre-Exp-10."""
    enc, tgt, ae, pred = _make_models()
    path = tmp_path / "jepa.pt"
    save_checkpoint(path, step=1, encoder=enc, target_encoder=tgt,
                    action_encoder=ae, predictor=pred, args=_args(),
                    inverse_model=None)
    ckpt = torch.load(path, weights_only=False)
    assert "inverse_model" not in ckpt


def test_load_jepa_handles_missing_inverse_model(tmp_path):
    """Pre-Exp-10 checkpoints (no inverse_model key) still load cleanly."""
    enc, tgt, ae, pred = _make_models()
    path = tmp_path / "jepa.pt"
    save_checkpoint(path, step=1, encoder=enc, target_encoder=tgt,
                    action_encoder=ae, predictor=pred, args=_args(),
                    inverse_model=None)
    bundle = load_jepa(str(path), torch.device("cpu"))
    assert bundle.inverse_model is None


def test_load_jepa_loads_inverse_model_when_present(tmp_path):
    """An Exp-10 checkpoint round-trips the inverse model, ready to query."""
    enc, tgt, ae, pred = _make_models()
    inv = InverseModel(patch_dim=64, num_patches=enc.num_patches)
    path = tmp_path / "jepa.pt"
    save_checkpoint(path, step=1, encoder=enc, target_encoder=tgt,
                    action_encoder=ae, predictor=pred, args=_args(),
                    inverse_model=inv)
    bundle = load_jepa(str(path), torch.device("cpu"))
    assert isinstance(bundle.inverse_model, InverseModel)
    z = torch.randn(2, enc.num_patches, 64)
    assert bundle.inverse_model(z, z).shape == (2, 4)
