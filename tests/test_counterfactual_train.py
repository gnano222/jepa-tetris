"""Tests for the counterfactual training step.

Validates the counterfactual loss function: predictor receives all NUM_ACTIONS
action embeddings per state, target encoder produces all NUM_ACTIONS targets
from the corresponding counterfactual next-states, and the loss is the
per-(state, action) MSE averaged over actions and batch.
"""
from __future__ import annotations

import torch

from jepa_tetris.data.replay_buffer import NUM_ACTIONS
from jepa_tetris.models.action_encoder import ActionEncoder
from jepa_tetris.models.encoder import StateEncoder
from jepa_tetris.models.predictor import Predictor
from jepa_tetris.train import counterfactual_step_loss


def _build_models(latent_dim: int = 16):
    torch.manual_seed(0)
    encoder = StateEncoder(latent_dim=latent_dim)
    target_encoder = StateEncoder(latent_dim=latent_dim)
    target_encoder.load_state_dict(encoder.state_dict())
    for p in target_encoder.parameters():
        p.requires_grad_(False)
    target_encoder.eval()
    action_encoder = ActionEncoder(num_actions=NUM_ACTIONS, embed_dim=8)
    predictor = Predictor(latent_dim=latent_dim, action_emb_dim=8, hidden=32, depth=1)
    return encoder, target_encoder, action_encoder, predictor


def _random_batch(batch_size: int = 4, k: int = 1):
    s0 = torch.randn(batch_size, 2, 20, 10)
    next_states_k = torch.randn(batch_size, k, NUM_ACTIONS, 2, 20, 10)
    actions_executed = torch.randint(0, NUM_ACTIONS, (batch_size, k))
    return s0, next_states_k, actions_executed


def test_counterfactual_step_loss_returns_finite_scalar_k1():
    encoder, target_encoder, action_encoder, predictor = _build_models()
    s0, next_states_k, actions_executed = _random_batch(batch_size=4, k=1)
    out = counterfactual_step_loss(
        s0=s0,
        next_states_k=next_states_k,
        actions_executed=actions_executed,
        encoder=encoder,
        target_encoder=target_encoder,
        action_encoder=action_encoder,
        predictor=predictor,
    )
    assert torch.isfinite(out["mse"])
    assert torch.isfinite(out["z_pred_all"]).all()
    assert out["z_pred_all"].shape == (4, NUM_ACTIONS, 16)


def test_counterfactual_step_loss_runs_with_k_greater_than_1():
    encoder, target_encoder, action_encoder, predictor = _build_models()
    s0, next_states_k, actions_executed = _random_batch(batch_size=3, k=4)
    out = counterfactual_step_loss(
        s0=s0,
        next_states_k=next_states_k,
        actions_executed=actions_executed,
        encoder=encoder,
        target_encoder=target_encoder,
        action_encoder=action_encoder,
        predictor=predictor,
    )
    assert torch.isfinite(out["mse"])
    # At each rollout step we evaluate all NUM_ACTIONS, so total predictions
    # across the rollout should be batch_size * k * NUM_ACTIONS.
    assert out["n_predictions"] == 3 * 4 * NUM_ACTIONS


def test_counterfactual_step_loss_drops_to_zero_when_targets_match_predictions():
    """If the predictor's outputs equal the target encoder's outputs for every
    (state, action), MSE must be exactly zero. This is a contract check —
    the loss function must not silently mis-pair states and actions."""
    encoder, target_encoder, action_encoder, predictor = _build_models()
    s0, next_states_k, actions_executed = _random_batch(batch_size=2, k=1)

    # Replace predictor's net with the identity-ish: produce z_target directly.
    # Achieve this by setting predictor weights so output = target_encoder(s'_a).
    # Simpler approach: monkey-patch predictor.forward to return the right thing.
    # Build the truth lookup ourselves.
    with torch.no_grad():
        next_flat = next_states_k.reshape(2 * 1 * NUM_ACTIONS, 2, 20, 10)
        z_target_flat = target_encoder(next_flat)  # (2 * 1 * NUM_ACTIONS, D)
    # Ensure indexing convention: target for (batch i, step t, action a) is at
    # flat index i * (k * NUM_ACTIONS) + t * NUM_ACTIONS + a.
    z_target_per_step = z_target_flat.view(2, 1, NUM_ACTIONS, -1)  # (B, K, A, D)

    # Replace predictor with a callable that returns the matching target.
    # Index lookup uses (z_chain, a_emb); we'll match purely from action_emb.
    orig_predictor_forward = predictor.forward

    class FakePredictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Need a parameter so .parameters() is non-empty for the optimizer.
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, z, a_emb):
            # Look up target by reverse-mapping action embeddings to indices.
            embed_w = action_encoder.embed.weight  # (A, E)
            # For each row in a_emb, find its index in embed_w via exact match.
            # (Not needed for general test; here we assume a_emb came from action_encoder.)
            sims = a_emb @ embed_w.T
            actions = sims.argmax(dim=-1)  # (B*A,)
            B = z.shape[0] // NUM_ACTIONS
            # We're at step t=0 (k=1). Build (B, A, D) of targets indexed by `actions`.
            out = torch.zeros(z.shape[0], z_target_per_step.shape[-1])
            for i in range(z.shape[0]):
                batch_i = i // NUM_ACTIONS
                a_i = int(actions[i].item())
                out[i] = z_target_per_step[batch_i, 0, a_i]
            return out

    fake_predictor = FakePredictor()

    out = counterfactual_step_loss(
        s0=s0,
        next_states_k=next_states_k,
        actions_executed=actions_executed,
        encoder=encoder,
        target_encoder=target_encoder,
        action_encoder=action_encoder,
        predictor=fake_predictor,
    )
    assert out["mse"].item() < 1e-8
