"""JEPA training loop with EMA target encoder and VICReg anti-collapse regularizers.

The predictor is trained with **teacher-forced multi-step** prediction (DINO-WM
convention). Each batch is a window of H+1 consecutive frames; the encoder is
applied to all frames, and the predictor is run independently at each of the H
positions from the *real* encoded frame (no autoregressive chain). The H
single-step losses are averaged.
"""
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


def _flatten_for_stats(z: torch.Tensor) -> torch.Tensor:
    """(*, D) -> (samples, D). Treats every patch in every batch element as one
    sample for variance/covariance computations (I-JEPA / V-JEPA convention)."""
    return z.reshape(-1, z.shape[-1])


def variance_loss(z: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    z = _flatten_for_stats(z)
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(target_std - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    z = _flatten_for_stats(z)
    n, d = z.shape
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / max(n - 1, 1)
    off_diag_sq = cov.pow(2).sum() - cov.diag().pow(2).sum()
    return off_diag_sq / d


def counterfactual_step_loss(
    *,
    s0: torch.Tensor,
    next_states: torch.Tensor,
    encoder: torch.nn.Module,
    target_encoder: torch.nn.Module,
    action_encoder: torch.nn.Module,
    predictor: torch.nn.Module,
) -> dict:
    """Single-step counterfactual JEPA loss.

    For every starting state, the predictor is run on all A action embeddings
    against the same z0 to produce ẑ_a for every a; the target is the
    (stop-grad) target encoder applied to `next_states[:, a]`.

    Shapes:
        s0:           (B, 2, 20, 10)
        next_states:  (B, A, 2, 20, 10)

    Returns {"mse": scalar, "z_pred_all": (B, A, N, D), "n_predictions": int}.
    """
    B, A = next_states.shape[:2]
    state_shape = next_states.shape[2:]

    z0 = encoder(s0)                                                       # (B, N, D)
    _, N, D = z0.shape

    actions_all = torch.arange(A, device=z0.device).repeat(B)              # (B*A,)
    z_repeat = z0.repeat_interleave(A, dim=0)                              # (B*A, N, D)
    a_emb_all = action_encoder(actions_all)                                # (B*A, D)
    z_pred_flat = predictor(z_repeat, a_emb_all)                           # (B*A, N, D)
    z_pred_all = z_pred_flat.view(B, A, N, D)                              # (B, A, N, D)

    s_next_flat = next_states.reshape(B * A, *state_shape)
    with torch.no_grad():
        z_target_all = target_encoder(s_next_flat).view(B, A, N, D)

    mse = F.mse_loss(z_pred_all, z_target_all)
    return {"mse": mse, "z_pred_all": z_pred_all, "n_predictions": B * A}


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
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patch-dim", type=int, default=128,
                        help="Per-patch channel dim D (= encoder's final conv channels).")
    parser.add_argument("--ema-tau", type=float, default=0.99)
    parser.add_argument("--var-weight", type=float, default=1.0)
    parser.add_argument("--cov-weight", type=float, default=0.04)
    parser.add_argument("--target-std", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--run", default=None,
                        help="Run name. Log goes to results/<YYYYMMDD-HHMMSS>[_<run>]/train_log.jsonl.")
    parser.add_argument("--log-file", default=None,
                        help="Explicit training log path. Overrides --run.")
    parser.add_argument("--horizon-h", type=int, default=4,
                        help="Multi-step prediction horizon. Sample H+1-frame windows; "
                             "predictor runs H steps (teacher-forced from real frames by "
                             "default, autoregressive with --autoregressive). H=1 = single-step.")
    parser.add_argument("--autoregressive", action="store_true",
                        help="Train predictor autoregressively: feed its own output forward "
                             "for H steps instead of teacher-forcing from real encoded frames. "
                             "Better at deep rollouts; worse single-step calibration.")
    parser.add_argument("--ar-weight", type=float, default=0.0,
                        help="In teacher-forced mode, add an autoregressive H-step rollout "
                             "loss with this weight on top of the teacher-forced MSE. "
                             "0 = pure teacher-forced; 0.25 = mixed; ignored if --autoregressive.")
    parser.add_argument("--predictor-heads", type=int, default=4,
                        help="Attention heads in the predictor transformer.")
    parser.add_argument("--predictor-depth", type=int, default=2,
                        help="Number of TransformerEncoderLayer blocks in the predictor.")
    parser.add_argument("--predictor-no-residual", action="store_true",
                        help="Disable residual prediction. Default predicts Δz.")
    parser.add_argument("--predictor-film", action="store_true",
                        help="FiLM action conditioning: per-layer (γ,β) modulation of patch "
                             "tokens. Replaces the default extra-token approach.")
    parser.add_argument("--predictor-spatial-film", action="store_true",
                        help="Spatial FiLM: per-patch per-layer (γ,β) computed from action "
                             "fused with each patch's positional embedding. More expressive "
                             "than --predictor-film.")
    parser.add_argument("--predictor-hierarchical-film", action="store_true",
                        help="Hierarchical FiLM: like spatial-film but action context is "
                             "updated each layer by pooling the current sequence state, so "
                             "deeper layers condition on action + prior layer predictions.")
    parser.add_argument("--predictor-cross-attn", action="store_true",
                        help="Cross-attention action conditioning: patches attend to action as "
                             "KV tokens after each self-attention block. Replaces extra-token.")
    parser.add_argument("--counterfactual", action="store_true",
                        help="Train against all NUM_ACTIONS counterfactual next-states per "
                             "starting state. Requires a buffer produced with --counterfactual. "
                             "Always single-step (no horizon).")
    parser.add_argument("--encoder-residual-blocks", type=int, default=0,
                        help="N residual blocks at each conv stage (default 0).")
    parser.add_argument("--encoder-aux-channels", action="store_true",
                        help="Prepend hand-engineered aux channels (heights, holes, bumpiness).")
    parser.add_argument("--encoder-stride-stages", type=int, default=3, choices=[2, 3],
                        help="Stride-2 downsampling stages: 3 (default) = 6 patches, "
                             "2 = 15 patches (finer spatial resolution).")
    parser.add_argument("--encoder-two-scale", action="store_true",
                        help="Concat fine (15) + coarse (6) patch streams → N=21. "
                             "Requires --encoder-stride-stages 2.")
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

    encoder = StateEncoder(
        patch_dim=args.patch_dim,
        residual_blocks=args.encoder_residual_blocks,
        aux_channels=args.encoder_aux_channels,
        stride_stages=args.encoder_stride_stages,
        two_scale=args.encoder_two_scale,
    ).to(device)
    target_encoder = copy.deepcopy(encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad_(False)
    target_encoder.eval()

    action_encoder = ActionEncoder(embed_dim=args.patch_dim).to(device)
    predictor = Predictor(
        patch_dim=args.patch_dim,
        num_patches=encoder.num_patches,
        num_heads=args.predictor_heads,
        depth=args.predictor_depth,
        residual=not args.predictor_no_residual,
        film=args.predictor_film,
        spatial_film=args.predictor_spatial_film,
        hierarchical_film=args.predictor_hierarchical_film,
        cross_attn=args.predictor_cross_attn,
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

    H = args.horizon_h
    state_shape = buf.state_shape

    pbar = tqdm(range(args.steps), desc="train")
    for step in pbar:
        log_now = (step % args.log_every == 0)
        if args.counterfactual:
            mse_tf_val = None
            mse_ar_val = None
            if args.ar_weight > 0:
                # CF + AR: need actions_executed and the chained next-states.
                batch = buf.sample_rollout(args.batch_size, H, rng=rng)
                s0 = torch.from_numpy(batch["s0"]).to(device)
                actions_executed = torch.from_numpy(batch["actions_executed"]).to(device)
                next_states_k = torch.from_numpy(batch["next_states_k"]).to(device)  # (B, K, A, *state)
                # CF loss uses the t=0 counterfactual fan-out (B, A, *state).
                next_states_t0 = next_states_k[:, 0]
            else:
                batch = buf.sample(args.batch_size, rng=rng)
                s0 = torch.from_numpy(batch["s"]).to(device)
                next_states_t0 = torch.from_numpy(batch["next_states"]).to(device)
                actions_executed = None
                next_states_k = None

            cf_out = counterfactual_step_loss(
                s0=s0,
                next_states=next_states_t0,
                encoder=encoder,
                target_encoder=target_encoder,
                action_encoder=action_encoder,
                predictor=predictor,
            )
            mse_cf = cf_out["mse"]
            mse_tf_val = mse_cf.item()
            z_pred_all = cf_out["z_pred_all"]                  # (B, A, N, D)
            z_for_vic = encoder(s0)                            # (B, N, D)

            # For cos_sim logging, pick action 0.
            z_pred_log = z_pred_all[:, 0]                      # (B, N, D)
            with torch.no_grad():
                z_target_log = target_encoder(next_states_t0[:, 0])  # (B, N, D)

            if args.ar_weight > 0:
                # AR rollout along the *executed* action chain. Target at step t
                # is the encoded next_states_k[b, t, actions_executed[b, t]].
                B = s0.shape[0]
                A_dim = next_states_k.shape[2]
                # Gather targets: shape (B, K, *state)
                idx_shape = (B, H, 1) + (1,) * len(state_shape)
                expand_shape = (-1, -1, 1) + tuple(state_shape)
                idx = actions_executed.view(*idx_shape).expand(*expand_shape)
                target_states_chain = next_states_k.gather(2, idx).squeeze(2)
                with torch.no_grad():
                    z_target_chain_flat = target_encoder(
                        target_states_chain.reshape(B * H, *state_shape)
                    )
                    _, N_p, D_p = z_target_chain_flat.shape
                    z_target_chain = z_target_chain_flat.view(B, H, N_p, D_p)

                # AR rollout from encoder(s0).
                z = encoder(s0)                                # (B, N, D)
                z_preds_ar = []
                for t in range(H):
                    a_emb_t = action_encoder(actions_executed[:, t])
                    z = predictor(z, a_emb_t)
                    z_preds_ar.append(z)
                z_pred_ar = torch.stack(z_preds_ar, dim=1)     # (B, H, N, D)
                mse_ar = F.mse_loss(z_pred_ar, z_target_chain)
                mse_ar_val = mse_ar.item()
                mse = mse_cf + args.ar_weight * mse_ar
            else:
                mse = mse_cf

            var_loss = variance_loss(z_for_vic, target_std=args.target_std)
            cov_loss = covariance_loss(z_for_vic)
            loss = mse + args.var_weight * var_loss + args.cov_weight * cov_loss
        else:
            batch = buf.sample_rollout(args.batch_size, H, rng=rng)
            s0 = torch.from_numpy(batch["s0"]).to(device)              # (B, *state)
            actions = torch.from_numpy(batch["actions"]).to(device)    # (B, H)
            s_next_k = torch.from_numpy(batch["s_next_k"]).to(device)  # (B, H, *state)
            B = s0.shape[0]

            # Stack (s0, s_next_k) into H+1 contiguous frames per row.
            frames = torch.cat([s0.unsqueeze(1), s_next_k], dim=1)      # (B, H+1, *state)
            frames_flat = frames.reshape(B * (H + 1), *state_shape)

            # Encode all frames with the online encoder (used for VICReg and,
            # in teacher-forced mode, as predictor inputs at every step).
            z_all_flat = encoder(frames_flat)                           # (B*(H+1), N, D)
            N, D = z_all_flat.shape[1], z_all_flat.shape[2]
            z_all = z_all_flat.view(B, H + 1, N, D)

            # Targets always come from the EMA target encoder at t = 1..H.
            with torch.no_grad():
                z_target = target_encoder(frames[:, 1:].reshape(B * H, *state_shape))
                z_target = z_target.view(B, H, N, D)

            mse_tf_val = None
            mse_ar_val = None
            if args.autoregressive:
                # AR: chain the predictor's own outputs forward for H steps.
                z = z_all[:, 0]                                         # (B, N, D)
                z_preds = []
                for t in range(H):
                    a_emb_t = action_encoder(actions[:, t])             # (B, D)
                    z = predictor(z, a_emb_t)
                    z_preds.append(z)
                z_pred = torch.stack(z_preds, dim=1)                    # (B, H, N, D)
                mse = F.mse_loss(z_pred, z_target)
                mse_ar_val = mse.item()
            else:
                # Teacher-forced: predictor input at each step is the REAL
                # encoded frame, not the previous prediction.
                z_in = z_all[:, :H].reshape(B * H, N, D)
                a_emb = action_encoder(actions.reshape(B * H))
                z_pred = predictor(z_in, a_emb).view(B, H, N, D)
                mse_tf = F.mse_loss(z_pred, z_target)
                mse_tf_val = mse_tf.item()
                if args.ar_weight > 0:
                    # Mixed: also run an autoregressive rollout from z_all[:, 0]
                    # and add its MSE term weighted by --ar-weight.
                    z = z_all[:, 0]
                    z_preds_ar = []
                    for t in range(H):
                        a_emb_t = action_encoder(actions[:, t])
                        z = predictor(z, a_emb_t)
                        z_preds_ar.append(z)
                    z_pred_ar = torch.stack(z_preds_ar, dim=1)
                    mse_ar = F.mse_loss(z_pred_ar, z_target)
                    mse_ar_val = mse_ar.item()
                    mse = mse_tf + args.ar_weight * mse_ar
                else:
                    mse = mse_tf
            var_loss = variance_loss(z_all, target_std=args.target_std)
            cov_loss = covariance_loss(z_all)
            loss = mse + args.var_weight * var_loss + args.cov_weight * cov_loss

            # Per-step cos_sim logging (first/last step in the H window).
            z_pred_log = z_pred[:, 0]
            z_target_log = z_target[:, 0]
            z_pred_last = z_pred[:, -1]
            z_target_last = z_target[:, -1]
            z_for_vic = z_all

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema_update(target_encoder, encoder, args.ema_tau)

        if log_now:
            with torch.no_grad():
                z_flat = _flatten_for_stats(z_for_vic)
                z_std_mean = z_flat.std(dim=0).mean().item()
                cos_sim_k1 = F.cosine_similarity(z_pred_log, z_target_log, dim=-1).mean().item()
                if args.counterfactual:
                    cos_sim_kK = cos_sim_k1
                else:
                    cos_sim_kK = F.cosine_similarity(z_pred_last, z_target_last, dim=-1).mean().item()
                cos_sim = (cos_sim_k1 + cos_sim_kK) / 2.0

                z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
                cov_z = (z_centered.T @ z_centered) / max(z_centered.shape[0] - 1, 1)
                d_b = cov_z.shape[0]
                n_off = d_b * d_b - d_b
                if n_off > 0:
                    z_offdiag_cov = float(
                        (cov_z.abs().sum() - cov_z.abs().diag().sum()).item() / n_off
                    )
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
                "cos_sim_kK": cos_sim_kK,
                "z_offdiag_cov": z_offdiag_cov,
                "lr": scheduler.get_last_lr()[0],
            }
            if mse_tf_val is not None:
                key = "mse_cf" if args.counterfactual else "mse_tf"
                record[key] = mse_tf_val
            if mse_ar_val is not None:
                record["mse_ar"] = mse_ar_val
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

    # ---- post-training multistep rollout accuracy eval ----
    if not args.counterfactual:
        _ACTION_NAMES = ["LEFT", "RIGHT", "ROTATE", "DROP"]
        _eval_horizons = [1, 2, 4, 8, 16]
        _eval_n = 2000
        _eval_rng = np.random.default_rng(args.seed)
        _max_h = max(_eval_horizons)
        _h_set = set(_eval_horizons)

        _batch = buf.sample_rollout(_eval_n, k=_max_h, rng=_eval_rng)
        _s0 = torch.from_numpy(_batch["s0"]).to(device)
        _actions = torch.from_numpy(_batch["actions"]).to(device)
        _s_next_k = torch.from_numpy(_batch["s_next_k"]).to(device)
        _a0_np = _batch["actions"][:, 0]

        _cos_list: list[float] = []
        _mse_list: list[float] = []
        _std_list: list[float] = []
        _pa_cos_k1: dict[str, float] = {}
        _pa_mse_k1: dict[str, float] = {}

        encoder.eval()
        action_encoder.eval()
        predictor.eval()

        with torch.no_grad():
            _z_pred = encoder(_s0)
            for _t in range(_max_h):
                _k = _t + 1
                _z_pred = predictor(_z_pred, action_encoder(_actions[:, _t]))
                if _k not in _h_set:
                    continue
                _z_tgt = encoder(_s_next_k[:, _t])
                _cos_per = F.cosine_similarity(_z_pred, _z_tgt, dim=-1)   # (B, N)
                _mse_per = ((_z_pred - _z_tgt) ** 2).mean(dim=-1)          # (B, N)
                _cos_list.append(_cos_per.mean().item())
                _mse_list.append(_mse_per.mean().item())
                _std_list.append(_z_pred.std(dim=0).mean().item())
                if _k == 1:
                    _cop = _cos_per.cpu().numpy()
                    _mep = _mse_per.cpu().numpy()
                    for _ai, _name in enumerate(_ACTION_NAMES):
                        _idx = np.where(_a0_np == _ai)[0]
                        _pa_cos_k1[_name] = float(_cop[_idx].mean()) if _idx.size > 0 else float("nan")
                        _pa_mse_k1[_name] = float(_mep[_idx].mean()) if _idx.size > 0 else float("nan")

        _eval_result = {
            "horizons": _eval_horizons,
            "cos_sim": _cos_list,
            "mse": _mse_list,
            "z_pred_std": _std_list,
            "per_action_cos_sim_k1": _pa_cos_k1,
            "per_action_mse_k1": _pa_mse_k1,
            "n": _eval_n,
            "buffer": args.buffer,
            "jepa": args.out,
        }
        _eval_path = log_path.parent / "multistep_accuracy.json"
        _eval_path.write_text(json.dumps(_eval_result, indent=2))

        print(f"\n  k  | cos_sim  |    mse")
        for _i, _k in enumerate(_eval_horizons):
            print(f"  {_k:<3} | {_cos_list[_i]:.4f}  | {_mse_list[_i]:.4f}")
        print(f"  DROP mse@1 = {_pa_mse_k1.get('DROP', float('nan')):.4f}")
        print(f"saved multistep eval to {_eval_path}")


if __name__ == "__main__":
    main()
