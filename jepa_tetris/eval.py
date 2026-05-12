"""Compare BFS planner against random baseline on lines-per-episode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from jepa_tetris.env.tetris import DROP, NUM_ACTIONS, TetrisEnv
from jepa_tetris.models.action_encoder import ActionEncoder
from jepa_tetris.models.encoder import make_encoder_from_args
from jepa_tetris.models.predictor import Predictor
from jepa_tetris.models.probe import Probe
from jepa_tetris.plan import BFSPlanner, PlacementPlanner, RealDynamicsPlanner
from jepa_tetris.utils.device import get_device
from jepa_tetris.utils.run_paths import run_dir
from jepa_tetris.utils.seed import set_seed


def run_episode(env: TetrisEnv, policy_fn, max_noop_streak: int = 0) -> tuple[int, int]:
    """Run one episode. policy_fn(obs) returns either a single int action or
    a list of actions to commit. If max_noop_streak > 0, force DROP after that
    many consecutive non-DROP actions."""
    obs = env.reset()
    total_lines = 0
    streak = 0
    pending: list[int] = []
    while not env.done:
        if not pending:
            out = policy_fn(obs)
            pending = [out] if isinstance(out, int) else list(out)
        a = pending.pop(0)
        if max_noop_streak > 0 and a != DROP and streak >= max_noop_streak:
            a = DROP
            pending = []  # discard rest of stale plan after forced drop
        if a == DROP:
            streak = 0
        else:
            streak += 1
        obs, info = env.step(a)
        total_lines += int(info["lines_cleared"])
    return total_lines, env.steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run", default=None,
                        help="Run name. Output goes to results/<YYYYMMDD-HHMMSS>[_<run>]/eval.json. "
                             "Ignored if --out is given.")
    parser.add_argument("--out", default=None,
                        help="Explicit eval output path. Overrides --run.")
    parser.add_argument("--lines-w", type=float, default=1.0)
    parser.add_argument("--holes-w", type=float, default=-0.5)
    parser.add_argument("--height-w", type=float, default=-0.1)
    parser.add_argument("--max-noop-streak", type=int, default=0,
                        help="If > 0, force DROP after that many consecutive non-DROP actions.")
    parser.add_argument("--plan-step", type=int, default=1,
                        help="How many actions of the planner's best sequence to execute before replanning.")
    parser.add_argument("--planner", choices=["latent", "real", "placement"], default="latent",
                        help="latent = pure JEPA rollouts; real = depth-K BFS in env; "
                             "placement = enumerate all (col, rot) endpoints (best coverage).")
    args = parser.parse_args()

    if args.out is None:
        out_path = run_dir(args.run) / "eval.json"
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = get_device()
    print(f"using device: {device}")

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
    ).to(device)
    predictor.load_state_dict(ckpt["predictor"])
    predictor.eval()

    probe_ckpt = torch.load(args.probe, map_location=device, weights_only=False)
    probe_depth = probe_ckpt.get("probe_depth", 1)
    probe_hidden = probe_ckpt.get("probe_hidden", 64)
    probe = Probe(patch_dim=patch_dim, num_targets=3,
                  depth=probe_depth, hidden=probe_hidden).to(device)
    probe.load_state_dict(probe_ckpt["probe"])
    probe.eval()
    target_mean = probe_ckpt.get("target_mean")
    target_std = probe_ckpt.get("target_std")

    if args.planner == "latent":
        planner = BFSPlanner(
            encoder, action_encoder, predictor, probe,
            depth=args.depth, device=device,
            lines_w=args.lines_w, holes_w=args.holes_w, height_w=args.height_w,
            target_mean=target_mean, target_std=target_std,
        )
    else:
        # Real-dynamics planner builds per-episode (needs the env)
        planner = None

    rng_for_random = np.random.default_rng(args.seed)

    def planner_policy(obs):
        if args.plan_step <= 1:
            return planner.select_action(obs)
        return planner.select_plan(obs)[: args.plan_step]

    def random_policy(obs):
        return int(rng_for_random.integers(0, NUM_ACTIONS))

    results = {}
    for name, policy in [("random", random_policy), ("planner", planner_policy)]:
        lines_per_ep, steps_per_ep = [], []
        for ep in tqdm(range(args.episodes), desc=name):
            env = TetrisEnv(seed=args.seed + ep, max_steps=args.max_steps)
            if name == "planner" and args.planner == "real":
                local_planner = RealDynamicsPlanner(
                    encoder, probe, env,
                    depth=args.depth, device=device,
                    lines_w=args.lines_w, holes_w=args.holes_w, height_w=args.height_w,
                    target_mean=target_mean, target_std=target_std,
                )
                def episode_policy(obs):
                    if args.plan_step <= 1:
                        return local_planner.select_action(obs)
                    return local_planner.select_plan(obs)[: args.plan_step]
                pol = episode_policy
            elif name == "planner" and args.planner == "placement":
                local_planner = PlacementPlanner(
                    encoder, probe, env, device=device,
                    lines_w=args.lines_w, holes_w=args.holes_w, height_w=args.height_w,
                    target_mean=target_mean, target_std=target_std,
                )
                def episode_policy_p(obs):
                    return local_planner.select_plan(obs)
                pol = episode_policy_p
            else:
                pol = policy
            total_lines, steps = run_episode(env, pol, max_noop_streak=args.max_noop_streak)
            lines_per_ep.append(total_lines)
            steps_per_ep.append(steps)
        results[name] = {
            "lines_mean": float(np.mean(lines_per_ep)),
            "lines_std": float(np.std(lines_per_ep)),
            "steps_mean": float(np.mean(steps_per_ep)),
            "episodes": args.episodes,
        }
        print(
            f"{name}: lines/ep = {results[name]['lines_mean']:.2f} ± {results[name]['lines_std']:.2f}, "
            f"avg episode length = {results[name]['steps_mean']:.1f}"
        )

    if results["random"]["lines_mean"] > 0:
        ratio = results["planner"]["lines_mean"] / results["random"]["lines_mean"]
        results["planner_to_random_ratio"] = ratio
        print(f"planner / random ratio = {ratio:.2f}x  (target >= 1.5x)")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    args_path = out_path.parent / "eval_args.json"
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"saved results to {out_path}")


if __name__ == "__main__":
    main()
