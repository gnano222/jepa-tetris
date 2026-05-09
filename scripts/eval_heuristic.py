"""Sanity-check: run the heuristic exploration policy (no JEPA) for N episodes.
This is the upper-bound the JEPA-based planner is trying to match."""
from __future__ import annotations

import argparse

import numpy as np
from tqdm import tqdm

from jepa_tetris.data.exploration import MixedExplorationPolicy
from jepa_tetris.env.tetris import DROP, NUM_ACTIONS, TetrisEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--epsilon", type=float, default=0.0)  # 0.0 = pure heuristic
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    lines_per_ep, steps_per_ep = [], []
    for ep in tqdm(range(args.episodes)):
        env = TetrisEnv(seed=args.seed + ep, max_steps=args.max_steps)
        env.reset()
        rng = np.random.default_rng(args.seed + ep + 1000)
        policy = MixedExplorationPolicy(env, rng, epsilon=args.epsilon)
        total = 0
        while not env.done:
            a = policy()
            _, info = env.step(a)
            total += int(info["lines_cleared"])
            if a == DROP and not env.done:
                policy.reset_target()
        lines_per_ep.append(total)
        steps_per_ep.append(env.steps)

    print(f"heuristic (eps={args.epsilon}): "
          f"lines/ep = {np.mean(lines_per_ep):.2f} ± {np.std(lines_per_ep):.2f}, "
          f"avg episode length = {np.mean(steps_per_ep):.1f}")


if __name__ == "__main__":
    main()
