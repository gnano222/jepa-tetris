"""Read a training JSONL log and produce a multi-panel summary PNG."""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True,
                        help="Path to training_log.jsonl (e.g. results/<run>/train_log.jsonl).")
    parser.add_argument("--out", required=True,
                        help="Path to write loss plot (e.g. results/<run>/loss_plot.png).")
    args = parser.parse_args()

    records = []
    with open(args.log) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise SystemExit(f"no records in {args.log}")

    steps = [r["step"] for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(steps, [r["mse"] for r in records])
    axes[0, 0].set_title("MSE (predictor vs target)")
    axes[0, 1].plot(steps, [r["z_std_mean"] for r in records])
    axes[0, 1].axhline(0.5, color="r", linestyle="--", label="collapse threshold")
    axes[0, 1].set_title("mean(std(z)) per dim")
    axes[0, 1].legend()
    axes[1, 0].plot(steps, [r["cos_sim"] for r in records])
    axes[1, 0].set_title("cosine_sim(z_pred, z_next_target)")
    axes[1, 1].plot(steps, [r["var_loss"] for r in records], label="var")
    axes[1, 1].plot(steps, [r["cov_loss"] for r in records], label="cov")
    axes[1, 1].set_title("VICReg terms")
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=100)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
