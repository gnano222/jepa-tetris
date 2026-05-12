"""Diagnose the current pipeline: buffer composition, target/action distributions, probe R²."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from jepa_tetris.data.replay_buffer import ReplayBuffer
from jepa_tetris.models.encoder import make_encoder_from_args
from jepa_tetris.models.probe import Probe
from jepa_tetris.utils.device import get_device


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum()) + 1e-12
    return 1.0 - ss_res / ss_tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--buffer", required=True)
    p.add_argument("--jepa", default=None)
    p.add_argument("--probe", default=None)
    p.add_argument("--n-eval", type=int, default=4000)
    p.add_argument("--out", default=None,
                   help="Optional JSON path to dump all reported numbers.")
    args = p.parse_args()

    buf = ReplayBuffer.load(args.buffer)
    n = buf.size
    print(f"buffer: {n} triplets")

    lc = buf.lines_cleared[:n]
    holes = buf.holes[:n]
    height = buf.aggregate_height[:n]
    done = buf.done[:n]
    actions = buf.a[:n]

    # ---- buffer composition ----
    episodes_complete = int(done.sum())
    mean_episode_len = float(n / max(episodes_complete, 1))
    sample_size = min(50_000, n)
    rng_uniq = np.random.default_rng(0)
    if sample_size < n:
        idx_uniq = rng_uniq.choice(n, size=sample_size, replace=False)
    else:
        idx_uniq = np.arange(n)
    s_sample = buf.s[idx_uniq]
    s_next_sample = buf.s_next[idx_uniq]
    unique_s = len({s_sample[i].tobytes() for i in range(s_sample.shape[0])})
    unique_s_next = len({s_next_sample[i].tobytes() for i in range(s_next_sample.shape[0])})

    print("\n--- buffer composition ---")
    print(f"total_triplets:    {n}")
    print(f"episodes_complete: {episodes_complete}")
    print(f"mean_episode_len:  {mean_episode_len:.2f}")
    print(f"unique_s_states:   {unique_s} (sampled {sample_size})")
    print(f"unique_s_next:     {unique_s_next} (sampled {sample_size})")

    composition = {
        "total_triplets": int(n),
        "episodes_complete": episodes_complete,
        "mean_episode_len": mean_episode_len,
        "unique_s_states": int(unique_s),
        "unique_s_next": int(unique_s_next),
        "unique_sample_size": int(sample_size),
    }

    print("\n--- target distributions ---")
    lc_counts = np.bincount(lc.astype(int), minlength=5)[:5].tolist()
    print(f"lines_cleared: mean={lc.mean():.4f}, max={int(lc.max())}, frac>0={(lc>0).mean():.4%}, "
          f"counts: {lc_counts}")
    print(f"holes:        mean={holes.mean():.2f}, max={int(holes.max())}, std={holes.std():.2f}")
    print(f"agg_height:   mean={height.mean():.2f}, max={int(height.max())}, std={height.std():.2f}")
    print(f"done:         frac={done.mean():.4%}")

    target_distributions = {
        "lines_cleared": {
            "mean": float(lc.mean()),
            "max": int(lc.max()),
            "frac>0": float((lc > 0).mean()),
            "counts": lc_counts,
        },
        "holes": {
            "mean": float(holes.mean()),
            "max": int(holes.max()),
            "std": float(holes.std()),
        },
        "aggregate_height": {
            "mean": float(height.mean()),
            "max": int(height.max()),
            "std": float(height.std()),
        },
        "done_frac": float(done.mean()),
    }

    print("\n--- action distribution ---")
    counts = np.bincount(actions, minlength=4)
    names = ["LEFT", "RIGHT", "ROTATE", "DROP"]
    action_dist: dict = {}
    for i, name in enumerate(names):
        c = int(counts[i])
        frac = c / n if n > 0 else 0.0
        print(f"  {name}: {c} ({frac:.2%})")
        action_dist[name] = {"count": c, "frac": float(frac)}

    probe_r2: dict = {}
    if args.jepa and args.probe:
        device = get_device()
        print(f"\nloading models on {device}")
        ckpt = torch.load(args.jepa, map_location=device, weights_only=False)
        patch_dim = ckpt["args"]["patch_dim"]
        encoder = make_encoder_from_args(ckpt["args"], device=device)
        encoder.load_state_dict(ckpt["encoder"])
        encoder.eval()

        probe_ckpt = torch.load(args.probe, map_location=device, weights_only=False)
        probe_depth = probe_ckpt.get("probe_depth", 1)
        probe_hidden = probe_ckpt.get("probe_hidden", 64)
        probe = Probe(patch_dim=patch_dim, num_targets=3,
                      depth=probe_depth, hidden=probe_hidden).to(device)
        probe.load_state_dict(probe_ckpt["probe"])
        probe.eval()
        target_mean = probe_ckpt.get("target_mean", np.zeros(3, dtype=np.float32))
        target_std = probe_ckpt.get("target_std", np.ones(3, dtype=np.float32))

        rng = np.random.default_rng(0)
        idx = rng.integers(0, n, size=min(args.n_eval, n))
        s_next = torch.from_numpy(buf.s_next[idx]).to(device)
        targets = np.stack([buf.lines_cleared[idx], buf.holes[idx], buf.aggregate_height[idx]], axis=1)

        with torch.no_grad():
            z = encoder(s_next)
            preds_norm = probe(z).cpu().numpy()
            preds = preds_norm * target_std + target_mean

        print(f"\n--- probe R² (n={len(idx)}) ---")
        for i, name in enumerate(["lines_cleared", "holes", "aggregate_height"]):
            r2 = r2_score(targets[:, i], preds[:, i])
            print(f"  {name}: R²={r2:.4f}  "
                  f"target μ/σ={targets[:,i].mean():.3f}/{targets[:,i].std():.3f}  "
                  f"pred μ/σ={preds[:,i].mean():.3f}/{preds[:,i].std():.3f}")
            probe_r2[name] = {
                "r2": float(r2),
                "target_mean": float(targets[:, i].mean()),
                "target_std": float(targets[:, i].std()),
                "pred_mean": float(preds[:, i].mean()),
                "pred_std": float(preds[:, i].std()),
            }
        probe_r2["n_eval"] = int(len(idx))

    if args.out is not None:
        out_obj = {
            "buffer": args.buffer,
            "composition": composition,
            "target_distributions": target_distributions,
            "action_distribution": action_dist,
        }
        if probe_r2:
            out_obj["probe_r2"] = probe_r2
        # convenience top-level keys for the sweep aggregator
        out_obj["frac>0"] = target_distributions["lines_cleared"]["frac>0"]
        out_obj["mean_episode_len"] = composition["mean_episode_len"]
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_obj, indent=2))
        print(f"\nwrote summary to {args.out}")


if __name__ == "__main__":
    main()
