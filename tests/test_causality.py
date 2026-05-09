"""Tests for the causality diagnostic helpers."""
from __future__ import annotations

import numpy as np
import torch

from jepa_tetris.env.tetris import DROP, LEFT, TetrisEnv
from jepa_tetris.eval_causality import (
    enumerate_counterfactuals,
    m1_action_retrieval,
    m2_calibration_correlation,
    m4_noop_recognition,
    predict_all_actions_per_state,
)
from jepa_tetris.models.action_encoder import ActionEncoder
from jepa_tetris.models.encoder import StateEncoder
from jepa_tetris.models.predictor import Predictor


def test_enumerate_counterfactuals_shape_and_drop_changes_state():
    env = TetrisEnv(seed=0)
    env.reset()
    snap = enumerate_counterfactuals(env)
    assert snap.s.shape == (2, 20, 10)
    assert snap.s_primes.shape == (4, 2, 20, 10)
    assert snap.is_noop.shape == (4,)
    # From a fresh board, DROP locks the piece — never observationally a no-op.
    assert not snap.is_noop[DROP]


def test_enumerate_counterfactuals_does_not_mutate_original_env():
    env = TetrisEnv(seed=0)
    env.reset()
    obs_before = env.observe().copy()
    pos_before = (env.piece_row, env.piece_col, env.rotation)
    board_before = env.board.copy()
    enumerate_counterfactuals(env)
    np.testing.assert_array_equal(env.observe(), obs_before)
    np.testing.assert_array_equal(env.board, board_before)
    assert (env.piece_row, env.piece_col, env.rotation) == pos_before


def test_enumerate_counterfactuals_flags_left_noop_at_left_wall():
    env = TetrisEnv(seed=0)
    env.reset()
    while env._is_valid(env.piece_row, env.piece_col - 1, env.rotation):
        env.step(LEFT)
    snap = enumerate_counterfactuals(env)
    assert snap.is_noop[LEFT]
    assert not snap.is_noop[DROP]


def test_m1_perfect_predictions_yield_full_retrieval():
    torch.manual_seed(0)
    z_target = torch.randn(50, 4, 16)
    z_pred = z_target.clone()
    out = m1_action_retrieval(z_pred, z_target)
    assert out["top1"] == 1.0
    for a in range(4):
        assert out["per_action"][a] == 1.0


def test_m1_collapsed_predictions_yield_chance_retrieval():
    torch.manual_seed(0)
    z_target = torch.randn(400, 4, 16)
    # Predictor outputs identical vector for every action → argmin over b
    # picks the same b for every query a; each query matches with probability 1/A.
    z_pred = torch.randn(400, 1, 16).expand(-1, 4, -1).contiguous()
    out = m1_action_retrieval(z_pred, z_target)
    assert 0.20 <= out["top1"] <= 0.30


def test_m2_perfect_predictions_yield_correlation_1():
    torch.manual_seed(0)
    z_target = torch.randn(50, 4, 16)
    z_pred = z_target.clone()
    rho = m2_calibration_correlation(z_pred, z_target)
    assert rho > 0.999


def test_m2_unrelated_predictions_yield_correlation_near_0():
    torch.manual_seed(0)
    z_target = torch.randn(400, 4, 16)
    z_pred = torch.randn(400, 4, 16)
    rho = m2_calibration_correlation(z_pred, z_target)
    assert -0.10 < rho < 0.10


def test_m4_predictions_matching_z_for_noops_yield_zero_ratio():
    torch.manual_seed(0)
    z_s = torch.randn(50, 16)
    z_pred = torch.randn(50, 4, 16) + z_s.unsqueeze(1)
    is_noop = np.zeros((50, 4), dtype=bool)
    is_noop[:, 0] = True
    # Force the no-op predictions to equal z(s) exactly.
    z_pred[:, 0] = z_s
    out = m4_noop_recognition(z_s, z_pred, is_noop)
    assert out["noop_mean_delta"] < 1e-5
    assert out["non_noop_mean_delta"] > 0.1
    assert out["ratio"] < 1e-5


def test_predict_all_actions_per_state_matches_per_pair_calls():
    """Regression test for the (state, action) batching alignment.

    Earlier the script paired states and actions with mismatched repeat
    patterns, causing many (i, a) slots in the (N, A, D) tensor to actually
    hold predictions for the wrong action — silently yielding ~0 pairwise
    distances and a misleading "action collapse" diagnosis. Verify that the
    helper's batched output matches per-(state, action) calls one-by-one."""
    torch.manual_seed(0)
    encoder = StateEncoder(latent_dim=32).eval()
    action_encoder = ActionEncoder(num_actions=4, embed_dim=8).eval()
    predictor = Predictor(latent_dim=32, action_emb_dim=8, hidden=64, depth=1).eval()
    states = torch.randn(7, 2, 20, 10)

    with torch.no_grad():
        z_s, z_pred = predict_all_actions_per_state(
            states, encoder, action_encoder, predictor,
        )
        # Manual reference: one (state, action) at a time.
        for i in range(7):
            for a in range(4):
                z_i = encoder(states[i:i + 1])
                a_emb = action_encoder(torch.tensor([a]))
                expected = predictor(z_i, a_emb).squeeze(0)
                torch.testing.assert_close(z_pred[i, a], expected, rtol=1e-5, atol=1e-6)
        # And different actions should give different outputs (sanity).
        for i in range(7):
            for a in range(4):
                for b in range(a + 1, 4):
                    assert not torch.allclose(z_pred[i, a], z_pred[i, b])


def test_m4_predictions_independent_of_noop_flag_yield_ratio_near_1():
    torch.manual_seed(0)
    z_s = torch.randn(200, 16)
    z_pred = z_s.unsqueeze(1).expand(-1, 4, -1) + torch.randn(200, 4, 16) * 0.5
    is_noop = np.zeros((200, 4), dtype=bool)
    is_noop[:, 0] = True
    out = m4_noop_recognition(z_s, z_pred, is_noop)
    assert 0.7 < out["ratio"] < 1.4
