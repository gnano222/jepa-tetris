"""Train the post-hoc decoder probe with mixed encoder + predictor latents.

The decoder is a visualization aid only; it never influences JEPA training.
But the OLD training procedure only ever fed it `encoder(s)` latents, while
inference (e.g. visualize_predictions.py rollouts) feeds it `predictor(z, a)`
latents at depths 1..K. The encoder/predictor distribution mismatch is the
same one that breaks BFSPlanner (see RESULTS.md), and it's why rollout
images degrade past step 1.

This script trains the decoder on a mix of both distributions:

  * Fraction `(1 - predictor_mix)` of each batch:
        z = encoder(s);   target = s
        decoder(z) is supervised against the actual board s.

  * Fraction `predictor_mix` of each batch (only if mix > 0):
        z_0 = encoder(s_0);   for d in 1..rollout_k:  z = predictor(z, a_d)
        target = s_actual_d (from buf.sample_rollout)
        At each training step we pick a random depth d uniformly in [1, K]
        so the decoder sees every horizon over the course of training.

We log `bce_enc` and `bce_pred` separately so it's easy to see whether the
predictor-distribution loss is actually decreasing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from jepa_tetris.data.replay_buffer import ReplayBuffer
from jepa_tetris.models.decoder import StateDecoder
from jepa_tetris.utils.checkpoint import load_jepa
from jepa_tetris.utils.device import get_device
from jepa_tetris.utils.logging import JsonlLogger
from jepa_tetris.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa", required=True, help="path to JEPA checkpoint")
    parser.add_argument("--buffer", required=True, help="replay buffer .npz")
    parser.add_argument("--out", default="checkpoints/decoder.pt")
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--log-file", default="decoder_log.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--source",
        choices=("s", "s_next", "both"),
        default="both",
        help="which encoder-distribution states to train on (s, s_next, or both)",
    )
    parser.add_argument(
        "--predictor-mix",
        type=float,
        default=0.5,
        help="Fraction of each batch sampled from predictor rollouts. "
             "0.0 = legacy encoder-only training; 0.5 = balanced; 1.0 = predictor only.",
    )
    parser.add_argument(
        "--rollout-k",
        type=int,
        default=4,
        help="Max rollout depth used when sampling predictor latents. Each step "
             "picks a depth uniformly in [1, K].",
    )
    args = parser.parse_args()

    if not 0.0 <= args.predictor_mix <= 1.0:
        raise ValueError(f"--predictor-mix must be in [0, 1], got {args.predictor_mix}")
    if args.rollout_k < 1:
        raise ValueError(f"--rollout-k must be >= 1, got {args.rollout_k}")

    set_seed(args.seed)
    device = get_device()
    print(f"using device: {device}")

    buf = ReplayBuffer.load(args.buffer)
    print(f"loaded {buf.size} triplets")
    if buf.size < args.batch_size:
        raise ValueError(f"buffer too small ({buf.size}) for batch_size ({args.batch_size})")

    bundle = load_jepa(args.jepa, device)
    encoder = bundle.encoder
    action_encoder = bundle.action_encoder
    predictor = bundle.predictor
    latent_dim = bundle.latent_dim

    decoder = StateDecoder(latent_dim=latent_dim).to(device)
    optimizer = AdamW(decoder.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    logger = JsonlLogger(args.log_file)

    # Split the per-step batch budget between the two distributions. We do this
    # once up front since the split is fixed across training. Both halves can be
    # zero (use --predictor-mix 0 or 1 to train on a single distribution).
    n_pred = int(round(args.batch_size * args.predictor_mix))
    n_enc = args.batch_size - n_pred
    print(f"per-step split: {n_enc} encoder samples, {n_pred} predictor samples")
    if n_pred > 0 and buf.size <= args.rollout_k:
        raise ValueError(
            f"buffer too small ({buf.size}) for rollout sampling at k={args.rollout_k}"
        )

    pbar = tqdm(range(args.steps), desc="decoder")
    for step in pbar:
        loss_terms: list[torch.Tensor] = []
        bce_enc_val: float | None = None
        bce_pred_val: float | None = None
        binary_acc_val: float | None = None

        # ----- encoder-distribution branch -----------------------------------
        if n_enc > 0:
            batch = buf.sample(n_enc, rng=rng)
            if args.source == "s":
                s_np = batch["s"]
            elif args.source == "s_next":
                s_np = batch["s_next"]
            else:
                # `both` doubles the encoder branch's effective batch but we keep
                # the call cheap by sampling once and stacking the two states.
                s_np = np.concatenate([batch["s"], batch["s_next"]], axis=0)
            s = torch.from_numpy(s_np).to(device)

            with torch.no_grad():
                z_enc = encoder(s)
            logits_enc = decoder(z_enc)
            bce_enc = F.binary_cross_entropy_with_logits(logits_enc, s)
            loss_terms.append(bce_enc)
            bce_enc_val = bce_enc.item()
            if step % args.log_every == 0:
                with torch.no_grad():
                    pred_bin = (torch.sigmoid(logits_enc) > 0.5).float()
                    binary_acc_val = (pred_bin == s).float().mean().item()

        # ----- predictor-distribution branch ---------------------------------
        if n_pred > 0:
            roll_batch = buf.sample_rollout(n_pred, k=args.rollout_k, rng=rng)
            depth = int(rng.integers(1, args.rollout_k + 1))  # uniform in [1, K]
            s0 = torch.from_numpy(roll_batch["s0"]).to(device)
            actions = torch.from_numpy(roll_batch["actions"][:, :depth]).to(device)
            target_state = torch.from_numpy(roll_batch["s_next_k"][:, depth - 1]).to(device)

            with torch.no_grad():
                z = encoder(s0)
                for t in range(depth):
                    a_emb = action_encoder(actions[:, t])
                    z = predictor(z, a_emb)
            # Decoder gets a graph; the rolled-out latent does not (we only
            # train the decoder, never the JEPA components).
            logits_pred = decoder(z)
            bce_pred = F.binary_cross_entropy_with_logits(logits_pred, target_state)
            loss_terms.append(bce_pred)
            bce_pred_val = bce_pred.item()

        # Equal-weight average of present terms (skips the missing branch when
        # predictor_mix is 0 or 1).
        loss = torch.stack(loss_terms).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0:
            record: dict = {"step": step, "loss": loss.item()}
            if bce_enc_val is not None:
                record["bce_enc"] = bce_enc_val
            if bce_pred_val is not None:
                record["bce_pred"] = bce_pred_val
            if binary_acc_val is not None:
                record["binary_acc"] = binary_acc_val
            logger.log(record)
            postfix = {"loss": f"{loss.item():.4f}"}
            if bce_enc_val is not None:
                postfix["enc"] = f"{bce_enc_val:.3f}"
            if bce_pred_val is not None:
                postfix["pred"] = f"{bce_pred_val:.3f}"
            pbar.set_postfix(**postfix)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "decoder": decoder.state_dict(),
            "latent_dim": latent_dim,
            "predictor_mix": args.predictor_mix,
            "rollout_k": args.rollout_k,
        },
        args.out,
    )
    print(f"saved decoder to {args.out}")


if __name__ == "__main__":
    main()
