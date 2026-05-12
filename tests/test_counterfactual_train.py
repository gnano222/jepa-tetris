"""Tests for the counterfactual training step.

Validates the single-step counterfactual loss: predictor receives all
NUM_ACTIONS action embeddings per state, target encoder produces all
NUM_ACTIONS targets from the corresponding counterfactual next-states, loss
is the per-(state, action) MSE averaged over actions and batch.
"""
from __future__ import annotations

import torch

from jepa_tetris.data.replay_buffer import NUM_ACTIONS
from jepa_tetris.models.action_encoder import ActionEncoder
from jepa_tetris.models.encoder import StateEncoder
from jepa_tetris.models.predictor import Predictor
from jepa_tetris.train import counterfactual_step_loss


PATCH_DIM = 32
N_DEFAULT = 6


def _build_models():
    torch.manual_seed(0)
    encoder = StateEncoder(patch_dim=PATCH_DIM)
    target_encoder = StateEncoder(patch_dim=PATCH_DIM)
    target_encoder.load_state_dict(encoder.state_dict())
    for p in target_encoder.parameters():
        p.requires_grad_(False)
    target_encoder.eval()
    action_encoder = ActionEncoder(num_actions=NUM_ACTIONS, embed_dim=PATCH_DIM)
    predictor = Predictor(patch_dim=PATCH_DIM, num_patches=encoder.num_patches, depth=1)
    return encoder, target_encoder, action_encoder, predictor


def _random_batch(batch_size: int = 4):
    s0 = torch.randn(batch_size, 2, 20, 10)
    next_states = torch.randn(batch_size, NUM_ACTIONS, 2, 20, 10)
    return s0, next_states


def test_counterfactual_step_loss_returns_finite_scalar():
    encoder, target_encoder, action_encoder, predictor = _build_models()
    s0, next_states = _random_batch(batch_size=4)
    out = counterfactual_step_loss(
        s0=s0,
        next_states=next_states,
        encoder=encoder,
        target_encoder=target_encoder,
        action_encoder=action_encoder,
        predictor=predictor,
    )
    assert torch.isfinite(out["mse"])
    assert torch.isfinite(out["z_pred_all"]).all()
    assert out["z_pred_all"].shape == (4, NUM_ACTIONS, N_DEFAULT, PATCH_DIM)


def test_counterfactual_step_loss_n_predictions():
    encoder, target_encoder, action_encoder, predictor = _build_models()
    s0, next_states = _random_batch(batch_size=3)
    out = counterfactual_step_loss(
        s0=s0,
        next_states=next_states,
        encoder=encoder,
        target_encoder=target_encoder,
        action_encoder=action_encoder,
        predictor=predictor,
    )
    assert out["n_predictions"] == 3 * NUM_ACTIONS


def test_counterfactual_step_loss_zero_when_predictor_matches_target():
    """If the predictor's outputs equal the target encoder's outputs for every
    (state, action), MSE must be exactly zero. Contract check on (state, action)
    alignment."""
    encoder, target_encoder, action_encoder, predictor = _build_models()
    s0, next_states = _random_batch(batch_size=2)

    with torch.no_grad():
        B, A = next_states.shape[:2]
        next_flat = next_states.reshape(B * A, 2, 20, 10)
        z_target_flat = target_encoder(next_flat)              # (B*A, N, D)
        N, D = z_target_flat.shape[1], z_target_flat.shape[2]
        z_target_per_action = z_target_flat.view(B, A, N, D)

    class FakePredictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, z, a_emb):
            # Reverse-map action embeddings to indices.
            embed_w = action_encoder.embed.weight              # (A, D)
            sims = a_emb @ embed_w.T
            actions = sims.argmax(dim=-1)                      # (B*A,)
            out = torch.zeros(z.shape[0], N, D)
            for i in range(z.shape[0]):
                batch_i = i // A
                a_i = int(actions[i].item())
                out[i] = z_target_per_action[batch_i, a_i]
            return out

    fake_predictor = FakePredictor()
    out = counterfactual_step_loss(
        s0=s0,
        next_states=next_states,
        encoder=encoder,
        target_encoder=target_encoder,
        action_encoder=action_encoder,
        predictor=fake_predictor,
    )
    assert out["mse"].item() < 1e-8
