"""Side-by-side comparison of all planners across N episodes.

Runs:
  - random
  - heuristic (epsilon=0)
  - latent (BFS in latent space, depth-K)
  - real (BFS in real env, depth-K, JEPA scores leaves)
  - placement ((col, rot) enumeration, JEPA scores leaves)

Outputs a clean table.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from tqdm import tqdm

from jepa_tetris.data.exploration import MixedExplorationPolicy
from jepa_tetris.env.tetris import DROP, NUM_ACTIONS, TetrisEnv
from jepa_tetris.models.action_encoder import ActionEncoder
from jepa_tetris.models.encoder import StateEncoder
from jepa_tetris.models.predictor import Predictor
from jepa_tetris.models.probe import Probe
from jepa_tetris.plan import BFSPlanner, PlacementPlanner, RealDynamicsPlanner
from jepa_tetris.utils.device import get_device
from jepa_tetris.utils.seed import set_seed


def run_episode(env: TetrisEnv, policy_fn, max_noop_streak: int = 0):
    obs = env.reset()
    total_lines = 0
    streak = 0
    pending: list[int] = []
    while not env.done:
        if not pending:
            out = policy_fn(obs, env)
            pending = [out] if isinstance(out, int) else list(out)
        a = pending.pop(0)
        if max_noop_streak > 0 and a != DROP and streak >= max_noop_streak:
            a = DROP
            pending = []
        if a == DROP:
            streak = 0
        else:
            streak += 1
        obs, info = env.step(a)
        total_lines += int(info["lines_cleared"])
    return total_lines, env.steps


def make_random_policy(rng):
    def policy(obs, env):
        return int(rng.integers(0, NUM_ACTIONS))
    return policy


def make_heuristic_policy(rng):
    state = {"policy": None, "env": None}

    def policy(obs, env):
        if state["env"] is not env:
            state["env"] = env
            state["policy"] = MixedExplorationPolicy(env, rng, epsilon=0.0)
        out = state["policy"]()
        if out == DROP:
            # reset target after DROP for next call
            state["policy"].reset_target()
        return out
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lines-w", type=float, default=10.0)
    ap.add_argument("--holes-w", type=float, default=-1.0)
    ap.add_argument("--height-w", type=float, default=-0.3)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"using device: {device}")

    ckpt = torch.load(args.jepa, map_location=device, weights_only=False)
    latent_dim = ckpt["args"]["latent_dim"]
    encoder = StateEncoder(latent_dim=latent_dim).to(device); encoder.load_state_dict(ckpt["encoder"]); encoder.eval()
    action_encoder = ActionEncoder().to(device); action_encoder.load_state_dict(ckpt["action_encoder"]); action_encoder.eval()
    predictor = Predictor(latent_dim=latent_dim, action_emb_dim=action_encoder.embed_dim).to(device)
    predictor.load_state_dict(ckpt["predictor"]); predictor.eval()

    probe_ckpt = torch.load(args.probe, map_location=device, weights_only=False)
    probe = Probe(latent_dim=latent_dim, num_targets=3).to(device)
    probe.load_state_dict(probe_ckpt["probe"]); probe.eval()
    target_mean = probe_ckpt.get("target_mean")
    target_std = probe_ckpt.get("target_std")

    weights = (args.lines_w, args.holes_w, args.height_w)

    results = {}

    def evaluate(name, make_policy):
        rng = np.random.default_rng(args.seed)
        lines, steps = [], []
        t0 = time.time()
        for ep in tqdm(range(args.episodes), desc=name):
            env = TetrisEnv(seed=args.seed + ep, max_steps=args.max_steps)
            policy = make_policy(env, rng)
            l, s = run_episode(env, policy)
            lines.append(l); steps.append(s)
        dt = time.time() - t0
        results[name] = {
            "lines_mean": float(np.mean(lines)),
            "lines_std": float(np.std(lines)),
            "steps_mean": float(np.mean(steps)),
            "elapsed_s": dt,
        }

    def random_factory(env, rng):
        def policy(obs, env=env):
            return int(rng.integers(0, NUM_ACTIONS))
        return policy

    def heuristic_factory(env, rng):
        explorer = MixedExplorationPolicy(env, rng, epsilon=0.0)
        def policy(obs, env=env):
            a = explorer()
            if a == DROP and not env.done:
                # reset target for next piece
                pass  # will be reset at next call when piece changes
            return a
        # We need to reset target after each DROP. Easier: do it here
        original = explorer.__call__
        def call_with_reset():
            a = original()
            return a
        return policy

    def latent_factory(env, rng):
        planner = BFSPlanner(
            encoder, action_encoder, predictor, probe,
            depth=args.depth, device=device,
            lines_w=args.lines_w, holes_w=args.holes_w, height_w=args.height_w,
            target_mean=target_mean, target_std=target_std,
        )
        def policy(obs, env=env):
            return planner.select_plan(obs)
        return policy

    def real_factory(env, rng):
        planner = RealDynamicsPlanner(
            encoder, probe, env,
            depth=args.depth, device=device,
            lines_w=args.lines_w, holes_w=args.holes_w, height_w=args.height_w,
            target_mean=target_mean, target_std=target_std,
        )
        def policy(obs, env=env):
            return planner.select_plan(obs)
        return policy

    def placement_factory(env, rng):
        planner = PlacementPlanner(
            encoder, probe, env, device=device,
            lines_w=args.lines_w, holes_w=args.holes_w, height_w=args.height_w,
            target_mean=target_mean, target_std=target_std,
        )
        def policy(obs, env=env):
            return planner.select_plan(obs)
        return policy

    # Heuristic — needs target reset after each DROP
    def heuristic_factory_v2(env, rng):
        explorer = MixedExplorationPolicy(env, rng, epsilon=0.0)
        def policy(obs, env=env):
            a = explorer()
            return a
        # Wrap step to reset target after DROP — done implicitly in MixedExplorationPolicy
        # Actually MixedExplorationPolicy doesn't auto-reset; we need to reset manually
        # via a side-effecting wrapper. For pure heuristic, the policy still works
        # because best_placement is computed once per piece (and a new piece spawns
        # after each DROP). Let's reset on each DROP outcome.
        return policy

    print("\n=== Running comparisons ===")
    evaluate("random", random_factory)

    # heuristic that resets after drop
    def heur_factory(env, rng):
        explorer = MixedExplorationPolicy(env, rng, epsilon=0.0)
        prev_piece = [env.piece_name]
        def policy(obs, env=env):
            if env.piece_name != prev_piece[0]:
                explorer.reset_target()
                prev_piece[0] = env.piece_name
            return explorer()
        return policy
    evaluate("heuristic", heur_factory)
    evaluate("latent", latent_factory)
    evaluate("real", real_factory)
    evaluate("placement", placement_factory)

    print("\n=== Summary ===")
    print(f"{'policy':<12} | {'lines/ep':>9} | {'std':>5} | {'len':>6} | {'time_s':>7} | ratio")
    print("-" * 58)
    base = results["random"]["lines_mean"]
    for name, r in results.items():
        ratio = r["lines_mean"] / base if base > 0 else float("inf")
        print(f"{name:<12} | {r['lines_mean']:>9.3f} | {r['lines_std']:>5.2f} | "
              f"{r['steps_mean']:>6.1f} | {r['elapsed_s']:>7.1f} | {ratio:>5.1f}x")


if __name__ == "__main__":
    main()
