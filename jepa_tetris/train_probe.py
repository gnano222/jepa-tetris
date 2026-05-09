"""Train the probe head: latent z (from frozen JEPA encoder) -> (lines, holes, height)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from jepa_tetris.data.replay_buffer import ReplayBuffer
from jepa_tetris.models.encoder import StateEncoder
from jepa_tetris.models.probe import Probe
from jepa_tetris.utils.device import get_device
from jepa_tetris.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa", required=True)
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--out", default="checkpoints/probe.pt")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=10.0,
                        help="Loss weight on samples with lines_cleared > 0.")
    parser.add_argument("--probe-depth", type=int, default=1,
                        help="Number of hidden layers in the probe MLP.")
    parser.add_argument("--probe-hidden", type=int, default=64,
                        help="Hidden dim in the probe MLP.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"using device: {device}")

    buf = ReplayBuffer.load(args.buffer)
    print(f"loaded {buf.size} triplets")

    ckpt = torch.load(args.jepa, map_location=device, weights_only=False)
    latent_dim = ckpt["args"]["latent_dim"]

    encoder = StateEncoder(latent_dim=latent_dim).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    probe = Probe(latent_dim=latent_dim, num_targets=3,
                  depth=args.probe_depth, hidden=args.probe_hidden).to(device)
    optimizer = AdamW(probe.parameters(), lr=args.lr)

    # Per-target normalization constants (computed once from the full buffer).
    # The probe predicts normalized targets; eval.py unnormalizes at inference.
    target_mean_np = np.array([
        buf.lines_cleared[: buf.size].mean(),
        buf.holes[: buf.size].mean(),
        buf.aggregate_height[: buf.size].mean(),
    ], dtype=np.float32)
    target_std_np = np.array([
        max(buf.lines_cleared[: buf.size].std(), 1e-3),
        max(buf.holes[: buf.size].std(), 1e-3),
        max(buf.aggregate_height[: buf.size].std(), 1e-3),
    ], dtype=np.float32)
    print(f"target means={target_mean_np}, stds={target_std_np}")
    target_mean = torch.from_numpy(target_mean_np).to(device)
    target_std = torch.from_numpy(target_std_np).to(device)

    # Upweight rare positive line-clear samples in the loss.
    pos_weight = float(args.pos_weight)

    rng = np.random.default_rng(args.seed)
    pbar = tqdm(range(args.steps), desc="probe")
    for step in pbar:
        batch = buf.sample(args.batch_size, rng=rng)
        s_next = torch.from_numpy(batch["s_next"]).to(device)
        targets = torch.stack(
            [
                torch.from_numpy(batch["lines_cleared"]),
                torch.from_numpy(batch["holes"]),
                torch.from_numpy(batch["aggregate_height"]),
            ],
            dim=1,
        ).float().to(device)

        with torch.no_grad():
            z = encoder(s_next)
        pred = probe(z)
        targets_norm = (targets - target_mean) / target_std
        sample_weight = torch.where(targets[:, 0] > 0, pos_weight, 1.0).unsqueeze(1)
        loss = (((pred - targets_norm) ** 2) * sample_weight).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "probe": probe.state_dict(),
            "latent_dim": latent_dim,
            "target_mean": target_mean_np,
            "target_std": target_std_np,
            "probe_depth": args.probe_depth,
            "probe_hidden": args.probe_hidden,
        },
        args.out,
    )
    print(f"saved probe to {args.out}")


if __name__ == "__main__":
    main()
