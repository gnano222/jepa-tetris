"""Visualize JEPA predictions: compare s_t, actual s_{t+1}, decoded predicted s_{t+1}.

Two modes:
  --mode compare  : 1-step compare, one figure per sample (matches training horizon)
  --mode rollout  : k-step recurrent latent rollout, PNG strip + animated GIF

Decoder probe weights are required (train via scripts/train_decoder.py). The
decoder is purely a visualization aid: latent diagnostics (cosine, L2) are
shown alongside the decoded grids so you can tell decoder error from predictor
error.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from jepa_tetris.env.tetris import NUM_ACTIONS, TetrisEnv
from jepa_tetris.utils.checkpoint import load_decoder, load_jepa
from jepa_tetris.utils.device import get_device
from jepa_tetris.utils.seed import set_seed
from jepa_tetris.viz import render_compare, render_rollout


def _to_grid(logits: torch.Tensor) -> np.ndarray:
    """Single-state decoder logits (1, 2, 20, 10) -> (2, 20, 10) probabilities."""
    return torch.sigmoid(logits).squeeze(0).cpu().numpy()


def _binary_acc(prob_grid: np.ndarray, true_grid: np.ndarray) -> float:
    return float(((prob_grid > 0.5).astype(np.float32) == true_grid).mean())


def _cosine(z_a: torch.Tensor, z_b: torch.Tensor) -> float:
    return float(F.cosine_similarity(z_a, z_b, dim=-1).mean().item())


def _l2(z_a: torch.Tensor, z_b: torch.Tensor) -> float:
    return float((z_a - z_b).norm(dim=-1).mean().item())


def run_compare(args, env, encoder, target_encoder, action_encoder, predictor, decoder, device, rng, out_dir):
    for i in range(args.n):
        s_t = env.reset()
        action = int(rng.integers(0, NUM_ACTIONS))
        s_t1, _ = env.step(action)

        s_t_t = torch.from_numpy(s_t).unsqueeze(0).to(device)
        s_t1_t = torch.from_numpy(s_t1).unsqueeze(0).to(device)
        a_t = torch.tensor([action], dtype=torch.long, device=device)

        with torch.no_grad():
            z_t = encoder(s_t_t)
            a_emb = action_encoder(a_t)
            z_pred = predictor(z_t, a_emb)
            z_target = target_encoder(s_t1_t)
            pred_grid = _to_grid(decoder(z_pred))
            recon_actual = _to_grid(decoder(z_target))

        metrics = {
            "cos(ẑ,z*)": _cosine(z_pred, z_target),
            "‖ẑ-z*‖": _l2(z_pred, z_target),
            "decode_acc(s*)": _binary_acc(recon_actual, s_t1),
        }
        savepath = out_dir / f"compare_{i:03d}.png"
        render_compare(s_t, s_t1, pred_grid, action, metrics=metrics, savepath=savepath)
        print(f"saved {savepath}  ({metrics})")


def run_rollout(args, env, encoder, target_encoder, action_encoder, predictor, decoder, device, rng, out_dir):
    for i in range(args.n):
        s_0 = env.reset()
        actions = [int(rng.integers(0, NUM_ACTIONS)) for _ in range(args.horizon)]

        actual_states = [s_0]
        for a in actions:
            s_next, _ = env.step(a)
            actual_states.append(s_next)
            if env.done:
                # Pad the rest of the actual rollout with the terminal state so
                # the predicted vs actual comparison stays length-aligned.
                while len(actual_states) < args.horizon + 1:
                    actual_states.append(s_next)
                break

        s_0_t = torch.from_numpy(s_0).unsqueeze(0).to(device)
        with torch.no_grad():
            z = encoder(s_0_t)
            predicted_grids = [_to_grid(decoder(z))]
            metrics_per_step = []
            z_target = target_encoder(s_0_t)
            metrics_per_step.append(
                {
                    "cos(ẑ,z*)": _cosine(z, z_target),
                    "‖ẑ-z*‖": _l2(z, z_target),
                    "decode_acc": _binary_acc(predicted_grids[0], s_0),
                }
            )

            z_curr = z
            for t, a in enumerate(actions):
                a_t = torch.tensor([a], dtype=torch.long, device=device)
                a_emb = action_encoder(a_t)
                z_curr = predictor(z_curr, a_emb)
                pred_grid = _to_grid(decoder(z_curr))
                predicted_grids.append(pred_grid)

                actual_t1 = actual_states[t + 1]
                actual_t1_t = torch.from_numpy(actual_t1).unsqueeze(0).to(device)
                z_tgt = target_encoder(actual_t1_t)
                metrics_per_step.append(
                    {
                        "cos(ẑ,z*)": _cosine(z_curr, z_tgt),
                        "‖ẑ-z*‖": _l2(z_curr, z_tgt),
                        "decode_acc": _binary_acc(pred_grid, actual_t1),
                    }
                )

        png_path = out_dir / f"rollout_{i:03d}.png"
        render_rollout(actions, actual_states, predicted_grids, metrics_per_step, savepath=png_path)
        print(f"saved {png_path}")
        if args.gif:
            gif_path = out_dir / f"rollout_{i:03d}.gif"
            render_rollout(
                actions,
                actual_states,
                predicted_grids,
                metrics_per_step,
                savepath=gif_path,
                to_gif=True,
            )
            print(f"saved {gif_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="JEPA checkpoint path")
    parser.add_argument("--decoder", required=True, help="decoder checkpoint path")
    parser.add_argument("--mode", choices=("compare", "rollout"), default="compare")
    parser.add_argument("--n", type=int, default=4, help="number of samples to render")
    parser.add_argument("--horizon", type=int, default=8, help="rollout length (rollout mode only)")
    parser.add_argument("--out", default="viz_out")
    parser.add_argument("--gif", action="store_true", help="also save animated GIF (rollout mode)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200, help="env max_steps")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"using device: {device}")

    bundle = load_jepa(args.checkpoint, device)
    decoder = load_decoder(args.decoder, bundle.latent_dim, device)

    env = TetrisEnv(seed=args.seed, max_steps=args.max_steps)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "compare":
        run_compare(args, env, bundle.encoder, bundle.target_encoder,
                    bundle.action_encoder, bundle.predictor, decoder, device, rng, out_dir)
    else:
        run_rollout(args, env, bundle.encoder, bundle.target_encoder,
                    bundle.action_encoder, bundle.predictor, decoder, device, rng, out_dir)


if __name__ == "__main__":
    main()
