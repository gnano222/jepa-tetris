"""Set up a board where a line clear is one drop away. Compare planner's choice
against the heuristic's choice and inspect predicted vs actual scores.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from jepa_tetris.data.exploration import best_placement, heuristic_action
from jepa_tetris.env.tetris import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    DROP,
    NUM_ACTIONS,
    NUM_ROTATIONS,
    TetrisEnv,
)
from jepa_tetris.models.action_encoder import ActionEncoder
from jepa_tetris.models.encoder import make_encoder_from_args
from jepa_tetris.models.predictor import Predictor
from jepa_tetris.models.probe import Probe
from jepa_tetris.plan import BFSPlanner
from jepa_tetris.utils.device import get_device


def setup_primed_env(piece: str = "I", n_rows: int = 1, gap_col: int = 5) -> TetrisEnv:
    env = TetrisEnv(seed=0)
    env.reset()
    for r in range(BOARD_HEIGHT - n_rows, BOARD_HEIGHT):
        env.board[r, :] = 1
        env.board[r, gap_col] = 0
    # Force the piece type so we can test with a known piece
    env.piece_name = piece
    env.rotation = 0
    env.piece_row = 0
    env.piece_col = 3
    return env


def compute_real_score(env: TetrisEnv, col: int, rot: int, weights: tuple[float, float, float]) -> tuple[int, int, int, float]:
    """Hard-drop simulate and return (lines, holes, height, weighted_score)."""
    saved = env.board.copy()
    saved_pose = (env.piece_row, env.piece_col, env.rotation)
    if not env._is_valid(env.piece_row, col, rot):
        return -1, -1, -1, -np.inf
    r = env.piece_row
    while env._is_valid(r + 1, col, rot):
        r += 1
    for cr, cc in env._piece_cells(r, col, rot):
        env.board[cr, cc] = 1
    lines = int(env.board.all(axis=1).sum())
    if lines > 0:
        full = np.where(env.board.all(axis=1))[0]
        keep = np.delete(env.board, full, axis=0)
        pad = np.zeros((lines, BOARD_WIDTH), dtype=np.int8)
        env.board = np.vstack([pad, keep])
    holes = env._count_holes()
    height = env._aggregate_height()
    score = weights[0] * lines + weights[1] * holes + weights[2] * height
    env.board[:] = saved
    env.piece_row, env.piece_col, env.rotation = saved_pose
    return lines, holes, height, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lines-w", type=float, default=10.0)
    ap.add_argument("--holes-w", type=float, default=-1.0)
    ap.add_argument("--height-w", type=float, default=-0.3)
    ap.add_argument("--piece", default="I")
    ap.add_argument("--gap-col", type=int, default=5)
    ap.add_argument("--n-rows", type=int, default=2)
    args = ap.parse_args()

    device = get_device()
    ckpt = torch.load(args.jepa, map_location=device, weights_only=False)
    patch_dim = ckpt["args"]["patch_dim"]
    encoder = make_encoder_from_args(ckpt["args"], device=device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    action_encoder = ActionEncoder(embed_dim=patch_dim).to(device)
    action_encoder.load_state_dict(ckpt["action_encoder"])
    action_encoder.eval()
    predictor = Predictor(
        patch_dim=patch_dim,
        num_patches=encoder.num_patches,
        num_heads=ckpt["args"].get("predictor_heads", 4),
        depth=ckpt["args"].get("predictor_depth", 2),
        residual=not ckpt["args"].get("predictor_no_residual", False),
    ).to(device)
    predictor.load_state_dict(ckpt["predictor"])
    predictor.eval()

    probe_ckpt = torch.load(args.probe, map_location=device, weights_only=False)
    probe = Probe(patch_dim=patch_dim, num_targets=3).to(device)
    probe.load_state_dict(probe_ckpt["probe"])
    probe.eval()
    target_mean = probe_ckpt.get("target_mean")
    target_std = probe_ckpt.get("target_std")

    planner = BFSPlanner(
        encoder, action_encoder, predictor, probe,
        depth=args.depth, device=device,
        lines_w=args.lines_w, holes_w=args.holes_w, height_w=args.height_w,
        target_mean=target_mean, target_std=target_std,
    )

    env = setup_primed_env(piece=args.piece, n_rows=args.n_rows, gap_col=args.gap_col)
    print(f"Setup: piece={args.piece}, n_rows={args.n_rows}, gap_col={args.gap_col}")
    print(f"Piece pose: row={env.piece_row}, col={env.piece_col}, rot={env.rotation}")

    # Heuristic recommendation
    h_col, h_rot = best_placement(env)
    h_lines, h_holes, h_height, h_score = compute_real_score(env, h_col, h_rot, (args.lines_w, args.holes_w, args.height_w))
    print(f"Heuristic best: col={h_col}, rot={h_rot} -> real lines={h_lines}, holes={h_holes}, height={h_height}, score={h_score:.3f}")

    # Planner's chosen plan
    obs = env.observe()
    plan = planner.select_plan(obs)
    print(f"Planner plan (depth {args.depth}): {plan}")

    # Real-world simulation of the planner's plan
    sim_env = setup_primed_env(piece=args.piece, n_rows=args.n_rows, gap_col=args.gap_col)
    real_lines = 0
    for a in plan:
        _, info = sim_env.step(a)
        real_lines += int(info["lines_cleared"])
    print(f"Real lines from planner's plan: {real_lines}")

    # Compute predicted vs real score for each first-action choice
    print("\nPredicted vs real score for each first action:")
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        z0 = encoder(obs_t)
        n = planner.sequences.shape[0]
        z = z0.expand(n, -1).contiguous()
        for t in range(args.depth):
            a_idx = planner.sequences[:, t]
            a_emb = planner.action_embs[a_idx]
            z = predictor(z, a_emb)
        feats = probe(z) * planner.target_std + planner.target_mean
        scores = (feats * planner.weights).sum(dim=1)
    # Group by first action and report best score per first action
    for first in range(NUM_ACTIONS):
        mask = (planner.sequences[:, 0] == first)
        if mask.any():
            top = scores[mask].max().item()
            top_idx = scores[mask].argmax().item()
            seq_idx = planner.sequences[mask][top_idx].tolist()
            print(f"  first={first} ({['LEFT','RIGHT','ROTATE','DROP'][first]}): "
                  f"max predicted score={top:.3f}, best seq={seq_idx}")


if __name__ == "__main__":
    main()
