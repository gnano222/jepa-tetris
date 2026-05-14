"""Measure latent rollout accuracy at specified horizons (cos_sim/mse/std/cov + probe R²)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from jepa_tetris.data.replay_buffer import ReplayBuffer
from jepa_tetris.models.action_encoder import ActionEncoder
from jepa_tetris.models.encoder import make_encoder_from_args
from jepa_tetris.models.predictor import Predictor
from jepa_tetris.models.probe import Probe
from jepa_tetris.utils.device import get_device


ACTION_NAMES = ["LEFT", "RIGHT", "ROTATE", "DROP"]


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum()) + 1e-12
    return 1.0 - ss_res / ss_tot


def offdiag_cov_mean_abs(z: torch.Tensor) -> float:
    # Accept either (B, D) or (B, N, D); flatten patches to (B*N, D).
    z = z.reshape(-1, z.shape[-1])
    n, d = z.shape
    if n < 2:
        return 0.0
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / (n - 1)
    abs_cov = cov.abs()
    off_sum = abs_cov.sum() - abs_cov.diag().sum()
    n_off = d * d - d
    if n_off <= 0:
        return 0.0
    return float(off_sum.item() / n_off)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jepa", required=True)
    p.add_argument("--buffer", required=True)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe", default=None,
                   help="Optional probe checkpoint (from train_probe.py) for R² eval.")
    p.add_argument("--out", default=None,
                   help="Optional JSON output path.")
    p.add_argument("--horizons", default="1,2,4,8,16",
                   help="Comma-separated list of horizons to evaluate.")
    args = p.parse_args()

    horizons = sorted({int(x) for x in args.horizons.split(",") if x.strip()})
    if not horizons or min(horizons) < 1:
        raise ValueError(f"invalid horizons: {args.horizons}")
    max_h = max(horizons)
    horizon_set = set(horizons)

    device = get_device()
    ckpt = torch.load(args.jepa, map_location=device, weights_only=False)
    patch_dim = ckpt["args"]["patch_dim"]

    encoder = make_encoder_from_args(ckpt["args"], device=device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    action_encoder = ActionEncoder(embed_dim=patch_dim).to(device)
    action_encoder.load_state_dict(ckpt["action_encoder"])
    action_encoder.eval()
    pred_depth = ckpt["args"].get("predictor_depth", 2)
    pred_heads = ckpt["args"].get("predictor_heads", 4)
    pred_residual = not ckpt["args"].get("predictor_no_residual", False)
    predictor = Predictor(
        patch_dim=patch_dim,
        num_patches=encoder.num_patches,
        num_heads=pred_heads,
        depth=pred_depth,
        residual=pred_residual,
        film=ckpt["args"].get("predictor_film", False),
        spatial_film=ckpt["args"].get("predictor_spatial_film", False),
        hierarchical_film=ckpt["args"].get("predictor_hierarchical_film", False),
        cross_attn=ckpt["args"].get("predictor_cross_attn", False),
    ).to(device)
    predictor.load_state_dict(ckpt["predictor"])
    predictor.eval()

    probe = None
    target_mean = target_std = None
    if args.probe is not None:
        probe_ckpt = torch.load(args.probe, map_location=device, weights_only=False)
        probe_depth = probe_ckpt.get("probe_depth", 1)
        probe_hidden = probe_ckpt.get("probe_hidden", 64)
        probe = Probe(patch_dim=patch_dim, num_targets=3,
                      depth=probe_depth, hidden=probe_hidden).to(device)
        probe.load_state_dict(probe_ckpt["probe"])
        probe.eval()
        target_mean = np.asarray(probe_ckpt.get("target_mean", np.zeros(3, dtype=np.float32)),
                                 dtype=np.float32)
        target_std = np.asarray(probe_ckpt.get("target_std", np.ones(3, dtype=np.float32)),
                                dtype=np.float32)

    buf = ReplayBuffer.load(args.buffer)
    rng = np.random.default_rng(args.seed)
    batch = buf.sample_rollout(args.n, k=max_h, rng=rng)
    s0 = torch.from_numpy(batch["s0"]).to(device)
    actions = torch.from_numpy(batch["actions"]).to(device)
    s_next_k = torch.from_numpy(batch["s_next_k"]).to(device)
    starts = batch["starts"]

    # bucket batch indices by first action for per-action k=1 sliced cos_sim
    a0_np = batch["actions"][:, 0]
    per_action_indices = {name: np.where(a0_np == i)[0] for i, name in enumerate(ACTION_NAMES)}

    cos_list: list[float] = []
    mse_list: list[float] = []
    std_list: list[float] = []
    offdiag_list: list[float] = []
    per_action_cos_k1: dict[str, float] = {}
    per_action_count_k1: dict[str, int] = {}
    per_action_mse: dict[str, list[float]] = {name: [] for name in ACTION_NAMES}
    probe_r2: dict[str, list[float]] = {"lines_cleared": [], "holes": [], "aggregate_height": []}

    with torch.no_grad():
        z = encoder(s0)
        z_pred = z
        for t in range(max_h):
            k = t + 1
            a_emb = action_encoder(actions[:, t])
            z_pred = predictor(z_pred, a_emb)
            if k not in horizon_set:
                continue
            z_target = encoder(s_next_k[:, t])
            cos_per = F.cosine_similarity(z_pred, z_target, dim=-1)
            mse_per = ((z_pred - z_target) ** 2).mean(dim=-1)
            cos = cos_per.mean().item()
            mse = mse_per.mean().item()
            std = z_pred.std(dim=0).mean().item()
            off = offdiag_cov_mean_abs(z_pred)
            cos_list.append(cos)
            mse_list.append(mse)
            std_list.append(std)
            offdiag_list.append(off)

            mse_per_np = mse_per.cpu().numpy()
            for name, ind in per_action_indices.items():
                per_action_mse[name].append(
                    float(mse_per_np[ind].mean()) if ind.size > 0 else float("nan")
                )

            if k == 1:
                cos_per_np = cos_per.cpu().numpy()
                for name, ind in per_action_indices.items():
                    per_action_count_k1[name] = int(ind.size)
                    per_action_cos_k1[name] = (
                        float(cos_per_np[ind].mean()) if ind.size > 0 else float("nan")
                    )

            if probe is not None:
                preds_norm = probe(z_pred).cpu().numpy()
                preds = preds_norm * target_std + target_mean
                gt_idx = starts + (k - 1)
                gt = np.stack([
                    buf.lines_cleared[gt_idx],
                    buf.holes[gt_idx],
                    buf.aggregate_height[gt_idx],
                ], axis=1)
                for j, name in enumerate(["lines_cleared", "holes", "aggregate_height"]):
                    probe_r2[name].append(r2_score(gt[:, j], preds[:, j]))

    # ---- print stdout table ----
    print(f"buffer:  {args.buffer}  (sampled n={args.n}, max_h={max_h})")
    print(f"jepa:    {args.jepa}")
    if args.probe is not None:
        print(f"probe:   {args.probe}")
    header = f"{'k':>3} | {'cos_sim':>8} | {'mse':>8} | {'z_std':>7} | {'offdiag_cov':>11}"
    if probe is not None:
        header += f" | {'R²(lines)':>10} | {'R²(holes)':>10} | {'R²(height)':>10}"
    print(header)
    print("-" * len(header))
    for i, k in enumerate(horizons):
        line = (f"{k:>3} | {cos_list[i]:>8.4f} | {mse_list[i]:>8.4f} | "
                f"{std_list[i]:>7.4f} | {offdiag_list[i]:>11.5f}")
        if probe is not None:
            line += (f" | {probe_r2['lines_cleared'][i]:>10.4f} | "
                     f"{probe_r2['holes'][i]:>10.4f} | "
                     f"{probe_r2['aggregate_height'][i]:>10.4f}")
        print(line)

    print("\n--- per-action cos_sim @ k=1 ---")
    for name in ACTION_NAMES:
        c = per_action_cos_k1.get(name, float("nan"))
        n_ = per_action_count_k1.get(name, 0)
        print(f"  {name:>6}: cos={c:.4f}  n={n_}")

    print("\n--- per-action MSE (sliced by first action) ---")
    pa_header = f"{'action':>7} | {'n':>5} | " + " | ".join(f"k={k:<5}" for k in horizons)
    print(pa_header)
    print("-" * len(pa_header))
    for name in ACTION_NAMES:
        n_ = per_action_count_k1.get(name, 0)
        vals = " | ".join(f"{v:>7.4f}" for v in per_action_mse[name])
        print(f"{name:>7} | {n_:>5} | {vals}")

    if args.out is not None:
        out_obj: dict = {
            "horizons": horizons,
            "cos_sim": cos_list,
            "mse": mse_list,
            "z_pred_std": std_list,
            "z_pred_offdiag_cov": offdiag_list,
            "per_action_cos_sim_k1": per_action_cos_k1,
            "per_action_count_k1": per_action_count_k1,
            "per_action_mse": per_action_mse,
            "n": int(args.n),
            "max_horizon_evaluated": int(max_h),
            "buffer": args.buffer,
            "jepa": args.jepa,
            "probe": args.probe,
        }
        if probe is not None:
            out_obj["probe_r2"] = {
                "lines_cleared": probe_r2["lines_cleared"],
                "holes": probe_r2["holes"],
                "aggregate_height": probe_r2["aggregate_height"],
            }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_obj, indent=2))
        print(f"\nwrote summary to {args.out}")


if __name__ == "__main__":
    main()
