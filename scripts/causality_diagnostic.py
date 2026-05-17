"""Causality diagnostic for a JEPA Tetris checkpoint.

For ~N sampled states, fork the env four times and apply each action,
producing all four counterfactual next-observations. Encode both s and the
four s'_a with the target encoder, run the predictor on (z(s), a) for every
action, and compute three metrics:

  M1 (action retrieval)        — does argmin_b ‖ẑ_b − z(s'_a)‖ pick a?
  M2 (calibration correlation) — Spearman ρ between predicted and true
                                 pairwise distances over (state, a, b) triples.
  M4 (no-op recognition)       — ratio of ‖ẑ_a − z(s)‖ on no-op (s, a) pairs
                                 vs non-no-op pairs (lower = better).

States are drawn from fresh rollouts of MixedExplorationPolicy so the
distribution matches what training sees, without depending on a particular
buffer file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from jepa_tetris.data.exploration import MixedExplorationPolicy
from jepa_tetris.env.tetris import DROP, NUM_ACTIONS, TetrisEnv
from jepa_tetris.eval_causality import (
    enumerate_counterfactuals,
    m1_action_retrieval,
    m2_calibration_correlation,
    m4_noop_recognition,
    predict_all_actions_per_state,
)
from jepa_tetris.models.action_encoder import ActionEncoder
from jepa_tetris.models.encoder import make_encoder_from_args
from jepa_tetris.models.predictor import Predictor
from jepa_tetris.utils.device import get_device


def collect_counterfactual_snapshots(
    n_states: int,
    seed: int,
    epsilon: float,
    max_steps: int,
) -> list:
    """Run rollouts under MixedExplorationPolicy, recording one counterfactual
    snapshot per visited (non-terminal) state. Episodes are reset as needed
    until `n_states` snapshots have been collected."""
    rng = np.random.default_rng(seed)
    snapshots = []
    ep = 0
    while len(snapshots) < n_states:
        env = TetrisEnv(seed=seed + ep, max_steps=max_steps)
        env.reset()
        policy = MixedExplorationPolicy(env, rng, epsilon=epsilon)
        ep += 1
        while not env.done and len(snapshots) < n_states:
            snapshots.append(enumerate_counterfactuals(env))
            a = policy()
            env.step(a)
            if a == DROP and not env.done:
                policy.reset_target()
    return snapshots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa", required=True, help="Path to JEPA checkpoint .pt file.")
    parser.add_argument("--n", type=int, default=500,
                        help="Number of starting states to sample.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.3,
                        help="Random-action probability for the rollout policy.")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--out", default=None,
                        help="Optional JSON output path. Defaults to no file write.")
    parser.add_argument("--use-online-encoder", action="store_true",
                        help="Use the online encoder for ground-truth z(s'_a) instead of "
                             "the target (EMA) encoder. Mostly for ablation; the target "
                             "encoder matches the training loss target.")
    args = parser.parse_args()

    device = get_device()
    print(f"using device: {device}")

    ckpt = torch.load(args.jepa, map_location=device, weights_only=False)
    patch_dim = ckpt["args"]["patch_dim"]
    pred_depth = ckpt["args"].get("predictor_depth", 2)
    pred_heads = ckpt["args"].get("predictor_heads", 4)
    pred_residual = not ckpt["args"].get("predictor_no_residual", False)

    encoder_state = (
        ckpt["encoder"] if args.use_online_encoder else ckpt["target_encoder"]
    )
    encoder = make_encoder_from_args(ckpt["args"], device=device)
    encoder.load_state_dict(encoder_state)
    encoder.eval()

    action_encoder = ActionEncoder(embed_dim=patch_dim).to(device)
    action_encoder.load_state_dict(ckpt["action_encoder"])
    action_encoder.eval()

    predictor = Predictor(
        patch_dim=patch_dim,
        num_patches=encoder.num_patches,
        num_heads=pred_heads,
        depth=pred_depth,
        residual=pred_residual,
        film=ckpt["args"].get("predictor_film", False),
        cross_attn=ckpt["args"].get("predictor_cross_attn", False),
        token_gate=ckpt["args"].get("predictor_token_gate", False),
        token_gate_k=ckpt["args"].get("token_gate_k", 21),
    ).to(device)
    predictor.load_state_dict(ckpt["predictor"])
    predictor.eval()

    print(f"collecting {args.n} counterfactual snapshots ...")
    snapshots = collect_counterfactual_snapshots(
        args.n, seed=args.seed, epsilon=args.epsilon, max_steps=args.max_steps,
    )
    n = len(snapshots)
    s_arr = np.stack([snap.s for snap in snapshots])                     # (N, 2, 20, 10)
    s_primes_arr = np.stack([snap.s_primes for snap in snapshots])       # (N, A, 2, 20, 10)
    is_noop_arr = np.stack([snap.is_noop for snap in snapshots])         # (N, A)

    s_t = torch.from_numpy(s_arr).to(device)
    s_primes_t = torch.from_numpy(s_primes_arr).to(device)

    A = NUM_ACTIONS
    with torch.no_grad():
        z_s_full, z_pred = predict_all_actions_per_state(
            s_t, encoder, action_encoder, predictor,
        )                                                                 # (N,F), (N,A,F)
        z_target_full = encoder(s_primes_t.reshape(n * A, *s_t.shape[1:]))
        z_target = z_target_full.flatten(1).view(n, A, -1)                # (N, A, F)

    m1 = m1_action_retrieval(z_pred, z_target)
    m2 = m2_calibration_correlation(z_pred, z_target)
    m4 = m4_noop_recognition(z_s_full, z_pred, is_noop_arr)

    # Sanity / context numbers.
    with torch.no_grad():
        # Mean per-action prediction L2 (low = accurate, high = inaccurate).
        per_action_mse = (
            (z_pred - z_target).pow(2).mean(dim=-1).mean(dim=0).cpu().numpy().tolist()
        )
        # Mean ‖z(s'_a) − z(s'_b)‖ — the true scale of action-induced changes.
        true_pair_dists = []
        for a in range(A):
            for b in range(a + 1, A):
                true_pair_dists.append((z_target[:, a] - z_target[:, b]).norm(dim=-1).mean().item())
        # Mean ‖ẑ_a − ẑ_b‖ — the predicted scale of action-induced changes.
        pred_pair_dists = []
        for a in range(A):
            for b in range(a + 1, A):
                pred_pair_dists.append((z_pred[:, a] - z_pred[:, b]).norm(dim=-1).mean().item())

    noop_per_action_count = is_noop_arr.sum(axis=0).tolist()

    # ---- print summary ----
    encoder_label = "online" if args.use_online_encoder else "target (EMA)"
    print()
    print(f"checkpoint: {args.jepa}")
    print(f"encoder used for ground truth: {encoder_label}")
    print(f"states sampled: {n}")
    print()
    print("=== M1: action retrieval (top-1, random baseline = 25%) ===")
    print(f"  overall:     {m1['top1']:.4f}")
    for a, name in enumerate(("LEFT", "RIGHT", "ROTATE", "DROP")):
        print(f"  {name:>6}:     {m1['per_action'][a]:.4f}")
    print()
    print("=== M2: calibration correlation (Spearman ρ over pairwise distances) ===")
    print(f"  rho:         {m2:.4f}")
    print()
    print("=== M4: no-op recognition (ratio = noop_mean / non_noop_mean) ===")
    print(f"  noop_count:           {m4['noop_count']}")
    print(f"  non_noop_count:       {m4['non_noop_count']}")
    print(f"  noop_mean_delta:      {m4['noop_mean_delta']:.4f}")
    print(f"  non_noop_mean_delta:  {m4['non_noop_mean_delta']:.4f}")
    print(f"  ratio:                {m4['ratio']:.4f}")
    print()
    print("=== context ===")
    pair_names = [f"{a}-{b}" for a in range(A) for b in range(a + 1, A)]
    print("  per-action prediction MSE (lower = more accurate):")
    for a, name in enumerate(("LEFT", "RIGHT", "ROTATE", "DROP")):
        print(f"    {name:>6}: {per_action_mse[a]:.4f}")
    print("  true / predicted pairwise distances (mean over states):")
    for label, t, p in zip(pair_names, true_pair_dists, pred_pair_dists):
        print(f"    pair {label}: true={t:.4f}  pred={p:.4f}")
    print("  no-op counts per action:")
    for a, name in enumerate(("LEFT", "RIGHT", "ROTATE", "DROP")):
        print(f"    {name:>6}: {noop_per_action_count[a]}")

    if args.out is not None:
        out_obj = {
            "checkpoint": args.jepa,
            "encoder_used": encoder_label,
            "n_states": n,
            "epsilon": args.epsilon,
            "seed": args.seed,
            "M1": m1,
            "M2_spearman_rho": m2,
            "M4": m4,
            "per_action_mse": per_action_mse,
            "true_pair_dists": dict(zip(pair_names, true_pair_dists)),
            "pred_pair_dists": dict(zip(pair_names, pred_pair_dists)),
            "noop_per_action_count": noop_per_action_count,
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_obj, indent=2))
        print(f"\nwrote summary to {args.out}")


if __name__ == "__main__":
    main()
