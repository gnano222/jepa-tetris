"""Causality diagnostic helpers.

Three metrics that test whether a JEPA predictor has learned action causality
in a way that's calibrated to the true environment dynamics. All three operate
on counterfactual rollouts: from the same starting state, run all four
actions in deepcopied env forks and compare predicted vs true next-latents.

- M1 (action retrieval): top-1 accuracy of identifying which action led to a
  given next-state from the predictor's outputs. Random baseline 25%.
- M2 (calibration correlation): Spearman correlation between predicted and
  true pairwise distances ‖ẑ_a − ẑ_b‖ vs ‖z(s'_a) − z(s'_b)‖.
- M4 (no-op recognition): ratio of mean ‖ẑ_a − z(s)‖ over no-op (s, a) pairs
  vs non-no-op pairs. Lower = better.

A no-op is detected by observation equality (s'_a == s); the encoder cannot
distinguish two states with identical observations, so observation equality
is the right notion of "the world didn't change as far as the model can see."
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from jepa_tetris.env.tetris import NUM_ACTIONS, TetrisEnv


def predict_all_actions_per_state(
    states: torch.Tensor,
    encoder: torch.nn.Module,
    action_encoder: torch.nn.Module,
    predictor: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the predictor on every (state, action) pair in `states × {0..A-1}`.

    Returns flat-latent forms (patch grid is folded into the final dim) so the
    M1/M2/M4 metrics treat each latent as a single vector for distance work.

    Returns:
        z_s:    (N, F)    where F = num_patches * patch_dim.
        z_pred: (N, A, F) — predictor(z_s[i], action_encoder(a)) per (i, a),
                            flattened across patches.

    Indexing convention: ``z_pred[i, a]`` is the prediction for state i under
    action a. The flat batch packs entries in row-major order: index k*A + a
    holds (state k, action a).
    """
    n = states.shape[0]
    a = next(predictor.parameters())  # device anchor
    device = a.device
    num_actions = action_encoder.embed.num_embeddings
    actions_all = torch.arange(num_actions, device=device).repeat(n)            # (N*A,)
    s_repeat = states.repeat_interleave(num_actions, dim=0)                     # (N*A, ...)
    z_s = encoder(states)                                                       # (N, P, D)
    z_repeat = encoder(s_repeat)                                                # (N*A, P, D)
    a_emb = action_encoder(actions_all)                                         # (N*A, D)
    z_pred_grid = predictor(z_repeat, a_emb)                                    # (N*A, P, D)
    z_s_flat = z_s.flatten(1)                                                   # (N, P*D)
    z_pred_flat = z_pred_grid.flatten(1)                                        # (N*A, P*D)
    z_pred = z_pred_flat.view(n, num_actions, -1)                               # (N, A, P*D)
    return z_s_flat, z_pred


@dataclass
class CounterfactualSnapshot:
    """A starting observation and all `NUM_ACTIONS` counterfactual next-observations."""

    s: np.ndarray              # (2, 20, 10)
    s_primes: np.ndarray       # (A, 2, 20, 10) — index = action ID
    is_noop: np.ndarray        # (A,) bool — True iff s'_a == s observationally


def enumerate_counterfactuals(env: TetrisEnv) -> CounterfactualSnapshot:
    """Fork `env` four times and apply each action to a fork.

    The original env is not mutated. No-op detection uses observation equality:
    z(s'_a) == z(s) iff obs(s'_a) == obs(s), since the encoder is a deterministic
    function of the observation alone.
    """
    s = env.observe().copy()
    s_primes = np.zeros((NUM_ACTIONS, *s.shape), dtype=np.float32)
    is_noop = np.zeros(NUM_ACTIONS, dtype=bool)
    for a in range(NUM_ACTIONS):
        fork = copy.deepcopy(env)
        s_prime, _ = fork.step(a)
        s_primes[a] = s_prime
        is_noop[a] = bool(np.array_equal(s_prime, s))
    return CounterfactualSnapshot(s=s, s_primes=s_primes, is_noop=is_noop)


def m1_action_retrieval(z_pred: torch.Tensor, z_target: torch.Tensor) -> dict:
    """Top-1 action-retrieval accuracy.

    For every (state i, action a), find argmin_b ‖z_pred[i, b] − z_target[i, a]‖.
    A hit is when that argmin equals a — i.e. the predictor's output for the
    true action is the closest to the true next-latent among all four candidates.

    Args:
        z_pred:   (N, A, D) predicted next-latents for each action.
        z_target: (N, A, D) encoder-derived true next-latents for each action.

    Returns:
        {"top1": float, "per_action": {action_id: float}}
    """
    N, A, _ = z_pred.shape
    diffs = z_target.unsqueeze(2) - z_pred.unsqueeze(1)   # (N, A_target, A_pred, D)
    dists = diffs.pow(2).sum(dim=-1).sqrt()               # (N, A_target, A_pred)
    pred_b = dists.argmin(dim=-1)                         # (N, A_target)
    target_a = torch.arange(A, device=z_pred.device).unsqueeze(0).expand(N, A)
    correct = (pred_b == target_a)                        # (N, A)
    return {
        "top1": float(correct.float().mean().item()),
        "per_action": {a: float(correct[:, a].float().mean().item()) for a in range(A)},
    }


def m2_calibration_correlation(z_pred: torch.Tensor, z_target: torch.Tensor) -> float:
    """Spearman correlation between predicted and true pairwise action-distances.

    For every (state, action-pair (a, b) with a < b), compute the L2 distances
    ‖z_pred[a] − z_pred[b]‖ and ‖z_target[a] − z_target[b]‖. Spearman ρ over
    all (state, pair) entries measures whether the predictor's geometry of
    action effects mirrors the true geometry.
    """
    N, A, _ = z_pred.shape
    pairs = [(a, b) for a in range(A) for b in range(a + 1, A)]
    pred_dists, true_dists = [], []
    for a, b in pairs:
        pred_dists.append((z_pred[:, a] - z_pred[:, b]).norm(dim=-1))
        true_dists.append((z_target[:, a] - z_target[:, b]).norm(dim=-1))
    pred_d = torch.stack(pred_dists, dim=1).cpu().numpy().reshape(-1)
    true_d = torch.stack(true_dists, dim=1).cpu().numpy().reshape(-1)
    return _spearman(pred_d, true_d)


def m4_noop_recognition(
    z_s: torch.Tensor,
    z_pred: torch.Tensor,
    is_noop: np.ndarray | torch.Tensor,
) -> dict:
    """Compare ‖ẑ_a − z(s)‖ on no-op (s, a) pairs vs non-no-op pairs.

    A model that understands environmental constraints (LEFT at left wall,
    ROTATE on O-piece, etc.) should produce ẑ_a ≈ z(s) on no-op actions, so
    `noop_mean_delta` should be small and `ratio = noop_mean / non_noop_mean`
    should be much less than 1.
    """
    if isinstance(is_noop, np.ndarray):
        is_noop = torch.from_numpy(is_noop).to(z_pred.device)
    deltas = (z_pred - z_s.unsqueeze(1)).norm(dim=-1)     # (N, A)
    flat_deltas = deltas.reshape(-1)
    flat_noop = is_noop.reshape(-1).bool()
    noop_d = flat_deltas[flat_noop]
    non_noop_d = flat_deltas[~flat_noop]
    noop_mean = float(noop_d.mean().item()) if noop_d.numel() > 0 else float("nan")
    non_noop_mean = (
        float(non_noop_d.mean().item()) if non_noop_d.numel() > 0 else float("nan")
    )
    ratio = (
        noop_mean / non_noop_mean
        if non_noop_d.numel() > 0 and non_noop_mean > 0
        else float("nan")
    )
    return {
        "noop_count": int(flat_noop.sum().item()),
        "non_noop_count": int((~flat_noop).sum().item()),
        "noop_mean_delta": noop_mean,
        "non_noop_mean_delta": non_noop_mean,
        "ratio": ratio,
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation with average-rank tie handling."""
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    if denom <= 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    sorted_vals = a[order]
    n = len(a)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks
