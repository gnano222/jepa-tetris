"""JEPA training loop with EMA target encoder and VICReg anti-collapse regularizers."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from jepa_tetris.data.replay_buffer import (
    NUM_ACTIONS,
    CounterfactualReplayBuffer,
    ReplayBuffer,
)
from jepa_tetris.models.action_encoder import ActionEncoder
from jepa_tetris.models.encoder import StateEncoder
from jepa_tetris.models.predictor import Predictor
from jepa_tetris.utils.device import get_device
from jepa_tetris.utils.logging import JsonlLogger
from jepa_tetris.utils.run_paths import run_dir
from jepa_tetris.utils.seed import set_seed


def variance_loss(z: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(target_std - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    n, d = z.shape
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / max(n - 1, 1)
    off_diag_sq = cov.pow(2).sum() - cov.diag().pow(2).sum()
    return off_diag_sq / d


def counterfactual_step_loss(
    *,
    s0: torch.Tensor,
    next_states_k: torch.Tensor,
    actions_executed: torch.Tensor,
    encoder: torch.nn.Module,
    target_encoder: torch.nn.Module,
    action_encoder: torch.nn.Module,
    predictor: torch.nn.Module,
) -> dict:
    """Compute the counterfactual JEPA loss for one (possibly K-step) batch.

    At every rollout step t, the predictor is run on all NUM_ACTIONS action
    embeddings against the same `z_chain` to produce ẑ_{t,a} for every a; the
    target is the (stop-grad) target encoder applied to the corresponding
    `next_states_k[:, t, a]`. The chain's next latent is the predicted ẑ for
    the action that was actually executed at step t.

    Shapes:
        s0:               (B, 2, 20, 10)
        next_states_k:    (B, K, A, 2, 20, 10)
        actions_executed: (B, K) integer

    Returns:
        {"mse": scalar tensor, "z_pred_all": (B, A, D) at the FINAL step (for
         logging cos_sim / VICReg), "n_predictions": int}.
    """
    B, K, A = next_states_k.shape[:3]
    D = encoder(s0[:1]).shape[-1]
    state_shape = next_states_k.shape[3:]

    z_chain = encoder(s0)                                          # (B, D)
    mse_terms = []
    z_pred_last = None
    for t in range(K):
        # Predict all A actions from the current chain latent.
        actions_all = torch.arange(A, device=z_chain.device).repeat(B)        # (B*A,)
        z_chain_repeat = z_chain.repeat_interleave(A, dim=0)                  # (B*A, D)
        a_emb_all = action_encoder(actions_all)                               # (B*A, E)
        z_pred_flat = predictor(z_chain_repeat, a_emb_all)                    # (B*A, D)
        z_pred_all = z_pred_flat.view(B, A, D)                                # (B, A, D)

        s_next_t = next_states_k[:, t]                                        # (B, A, *)
        s_next_flat = s_next_t.reshape(B * A, *state_shape)
        with torch.no_grad():
            z_target_all = target_encoder(s_next_flat).view(B, A, D)

        mse_terms.append(F.mse_loss(z_pred_all, z_target_all))
        z_pred_last = z_pred_all

        if t < K - 1:
            # Continue the chain along the executed action's prediction.
            a_exec_t = actions_executed[:, t].to(z_chain.device).long()       # (B,)
            idx = a_exec_t.view(B, 1, 1).expand(-1, 1, D)
            z_chain = z_pred_all.gather(1, idx).squeeze(1)                    # (B, D)

    mse = torch.stack(mse_terms).mean()
    return {"mse": mse, "z_pred_all": z_pred_last, "n_predictions": B * K * A}


@torch.no_grad()
def ema_update(target: torch.nn.Module, online: torch.nn.Module, tau: float) -> None:
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.data.mul_(tau).add_(p_o.data, alpha=1.0 - tau)
    for b_t, b_o in zip(target.buffers(), online.buffers()):
        b_t.data.copy_(b_o.data)


def save_checkpoint(path: Path, *, step: int, encoder, target_encoder, action_encoder, predictor, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "encoder": encoder.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "action_encoder": action_encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "args": vars(args),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--out", default="checkpoints/jepa.pt")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--ema-tau", type=float, default=0.99)
    parser.add_argument("--var-weight", type=float, default=1.0)
    parser.add_argument("--cov-weight", type=float, default=0.04)
    parser.add_argument("--target-std", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--run", default=None,
                        help="Run name. Log goes to results/<YYYYMMDD-HHMMSS>[_<run>]/train_log.jsonl. "
                             "Ignored if --log-file is given.")
    parser.add_argument("--log-file", default=None,
                        help="Explicit training log path. Overrides --run.")
    parser.add_argument("--rollout-k", type=int, default=1,
                        help="Train predictor on K-step rollouts (1 = standard JEPA, >1 = multi-step).")
    parser.add_argument("--predictor-hidden", type=int, default=256)
    parser.add_argument("--predictor-depth", type=int, default=2,
                        help="Number of hidden Linear+GELU blocks in the predictor MLP.")
    parser.add_argument("--predictor-residual", action="store_true",
                        help="Predictor outputs delta added to z instead of replacing it.")
    parser.add_argument("--counterfactual", action="store_true",
                        help="Train against all NUM_ACTIONS counterfactual next-states per "
                             "starting state. Requires a buffer produced with --counterfactual.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.log_file is None:
        log_path = run_dir(args.run) / "train_log.jsonl"
    else:
        log_path = Path(args.log_file)

    set_seed(args.seed)
    device = get_device()
    print(f"using device: {device}")

    if args.counterfactual:
        buf = CounterfactualReplayBuffer.load(args.buffer)
        print(f"loaded {buf.size} counterfactual rows from {args.buffer}")
    else:
        buf = ReplayBuffer.load(args.buffer)
        print(f"loaded {buf.size} triplets from {args.buffer}")
    if buf.size < args.batch_size:
        raise ValueError(f"buffer too small ({buf.size}) for batch_size ({args.batch_size})")

    encoder = StateEncoder(latent_dim=args.latent_dim).to(device)
    target_encoder = copy.deepcopy(encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad_(False)
    target_encoder.eval()

    action_encoder = ActionEncoder().to(device)
    predictor = Predictor(
        latent_dim=args.latent_dim,
        action_emb_dim=action_encoder.embed_dim,
        hidden=args.predictor_hidden,
        depth=args.predictor_depth,
        residual=args.predictor_residual,
    ).to(device)

    params = (
        list(encoder.parameters())
        + list(action_encoder.parameters())
        + list(predictor.parameters())
    )
    optimizer = AdamW(params, lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps)

    rng = np.random.default_rng(args.seed)
    logger = JsonlLogger(log_path)
    (log_path.parent / "train_args.json").write_text(json.dumps(vars(args), indent=2))

    pbar = tqdm(range(args.steps), desc="train")
    for step in pbar:
        log_now = (step % args.log_every == 0)
        z_pred_first = None
        z_target_first = None
        if args.counterfactual:
            k = max(1, args.rollout_k)
            batch = buf.sample_rollout(args.batch_size, k, rng=rng)
            s0 = torch.from_numpy(batch["s0"]).to(device)
            next_states_k = torch.from_numpy(batch["next_states_k"]).to(device)
            actions_executed = torch.from_numpy(batch["actions_executed"]).to(device)

            cf_out = counterfactual_step_loss(
                s0=s0,
                next_states_k=next_states_k,
                actions_executed=actions_executed,
                encoder=encoder,
                target_encoder=target_encoder,
                action_encoder=action_encoder,
                predictor=predictor,
            )
            mse = cf_out["mse"]
            z_pred_all_last = cf_out["z_pred_all"]                     # (B, A, D)
            z = encoder(s0)                                            # for VICReg + logging

            # For cos_sim logging pick action 0 (any fixed action works).
            z_pred = z_pred_all_last[:, 0]
            with torch.no_grad():
                z_next_target = target_encoder(next_states_k[:, -1, 0])
            if log_now:
                z_pred_first = z_pred.detach()
                z_target_first = z_next_target.detach()
            # VICReg on the (online) encoder's output of s0 — prevents encoder
            # collapse the same way as the single-action path. Applying it to
            # the predicted latents instead lets the encoder drift unbounded
            # because VICReg is the only thing constraining its scale.
            var_loss = variance_loss(z, target_std=args.target_std)
            cov_loss = covariance_loss(z)
            loss = mse + args.var_weight * var_loss + args.cov_weight * cov_loss
        elif args.rollout_k <= 1:
            batch = buf.sample(args.batch_size, rng=rng)
            s = torch.from_numpy(batch["s"]).to(device)
            a = torch.from_numpy(batch["a"]).to(device)
            s_next = torch.from_numpy(batch["s_next"]).to(device)

            z = encoder(s)
            a_emb = action_encoder(a)
            z_pred = predictor(z, a_emb)

            with torch.no_grad():
                z_next_target = target_encoder(s_next)
            mse = F.mse_loss(z_pred, z_next_target)
            if log_now:
                z_pred_first = z_pred.detach()
                z_target_first = z_next_target.detach()
            var_loss = variance_loss(z, target_std=args.target_std)
            cov_loss = covariance_loss(z)
            loss = mse + args.var_weight * var_loss + args.cov_weight * cov_loss
        else:
            batch = buf.sample_rollout(args.batch_size, args.rollout_k, rng=rng)
            s0 = torch.from_numpy(batch["s0"]).to(device)
            actions = torch.from_numpy(batch["actions"]).to(device)  # (B, K)
            s_next_k = torch.from_numpy(batch["s_next_k"]).to(device)  # (B, K, *)

            z = encoder(s0)  # (B, latent_dim)
            mse_terms = []
            z_pred = z
            for t in range(args.rollout_k):
                a_emb = action_encoder(actions[:, t])
                z_pred = predictor(z_pred, a_emb)
                with torch.no_grad():
                    z_target_t = target_encoder(s_next_k[:, t])
                mse_terms.append(F.mse_loss(z_pred, z_target_t))
                if log_now and t == 0:
                    z_pred_first = z_pred.detach()
                    z_target_first = z_target_t.detach()
            mse = torch.stack(mse_terms).mean()
            with torch.no_grad():
                z_next_target = z_target_t  # for cos_sim logging
            var_loss = variance_loss(z, target_std=args.target_std)
            cov_loss = covariance_loss(z)
            loss = mse + args.var_weight * var_loss + args.cov_weight * cov_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema_update(target_encoder, encoder, args.ema_tau)

        if step % args.log_every == 0:
            with torch.no_grad():
                z_std_mean = z.std(dim=0).mean().item()
                cos_sim = F.cosine_similarity(z_pred, z_next_target, dim=-1).mean().item()
                if z_pred_first is not None and z_target_first is not None:
                    cos_sim_k1 = F.cosine_similarity(
                        z_pred_first, z_target_first, dim=-1).mean().item()
                else:
                    cos_sim_k1 = cos_sim
                # mean abs off-diagonal of cov(z) on encoder outputs
                n_b, d_b = z.shape
                z_centered = z - z.mean(dim=0, keepdim=True)
                cov_z = (z_centered.T @ z_centered) / max(n_b - 1, 1)
                abs_cov_z = cov_z.abs()
                n_off = d_b * d_b - d_b
                if n_off > 0:
                    z_offdiag_cov = float(
                        (abs_cov_z.sum() - abs_cov_z.diag().sum()).item() / n_off)
                else:
                    z_offdiag_cov = 0.0
            record = {
                "step": step,
                "loss": loss.item(),
                "mse": mse.item(),
                "var_loss": var_loss.item(),
                "cov_loss": cov_loss.item(),
                "z_std_mean": z_std_mean,
                "cos_sim": cos_sim,
                "cos_sim_k1": cos_sim_k1,
                "cos_sim_kK": cos_sim,
                "z_offdiag_cov": z_offdiag_cov,
                "lr": scheduler.get_last_lr()[0],
            }
            logger.log(record)
            pbar.set_postfix(loss=f"{loss.item():.4f}", z_std=f"{z_std_mean:.3f}")

        if step > 0 and step % args.ckpt_every == 0:
            save_checkpoint(
                Path(args.out).parent / f"jepa_step{step}.pt",
                step=step,
                encoder=encoder,
                target_encoder=target_encoder,
                action_encoder=action_encoder,
                predictor=predictor,
                args=args,
            )

    save_checkpoint(
        Path(args.out),
        step=args.steps,
        encoder=encoder,
        target_encoder=target_encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        args=args,
    )
    print(f"saved final checkpoint to {args.out}")


if __name__ == "__main__":
    main()
