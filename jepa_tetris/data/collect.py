"""Random-play data collection: produces a replay buffer of (s, a, s', info) triplets.

Optionally primes some episodes with rows that are 1 column shy of full, so that
random play occasionally produces line clears (essential signal for the probe head).

`collect_counterfactual` is the variant for counterfactual training: at each
visited state, all NUM_ACTIONS action branches are evaluated via deepcopy forks
of the env, and all four resulting observations are stored alongside the action
that was actually executed (which advances the real env).
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
from tqdm import tqdm

from jepa_tetris.data.exploration import MixedExplorationPolicy
from jepa_tetris.data.replay_buffer import (
    NUM_ACTIONS,
    STATE_SHAPE,
    CounterfactualReplayBuffer,
    ReplayBuffer,
)
from jepa_tetris.env.pieces import NAME_TO_ID
from jepa_tetris.env.tetris import BOARD_HEIGHT, BOARD_WIDTH, DROP, TetrisEnv


def prime_board(env: TetrisEnv, rng: np.random.Generator, max_rows: int = 4) -> None:
    """Fill bottom N rows with a single-column gap; randomly choose N and gap_col."""
    n_rows = int(rng.integers(1, max_rows + 1))
    gap_col = int(rng.integers(0, BOARD_WIDTH))
    for r in range(BOARD_HEIGHT - n_rows, BOARD_HEIGHT):
        env.board[r, :] = 1
        env.board[r, gap_col] = 0


def collect(
    episodes: int,
    capacity: int,
    seed: int,
    prime_prob: float = 0.0,
    prime_max_rows: int = 4,
    policy: str = "random",
    epsilon: float = 0.3,
) -> ReplayBuffer:
    """Collect (s, a, s', info) triplets.

    policy:
        "random"    : uniform random over 4 actions.
        "heuristic" : best-placement targeting (epsilon=0 means deterministic).
        "mixed"     : with prob epsilon, random; else heuristic toward best placement.
    """
    rng = np.random.default_rng(seed)
    env = TetrisEnv(seed=seed)
    buf = ReplayBuffer(capacity=capacity)

    pbar = tqdm(range(episodes), desc="collecting")
    for _ in pbar:
        s = env.reset()
        if prime_prob > 0 and rng.random() < prime_prob:
            prime_board(env, rng, max_rows=prime_max_rows)
            s = env.observe()

        if policy in ("heuristic", "mixed"):
            eps = epsilon if policy == "mixed" else 0.0
            policy_fn = MixedExplorationPolicy(env, rng, epsilon=eps)
        else:
            policy_fn = None

        while not env.done and buf.size < capacity:
            if policy_fn is None:
                a = int(rng.integers(0, NUM_ACTIONS))
            else:
                a = policy_fn()
            # env.step() mutates piece attrs (especially after DROP, which spawns
            # a new piece). Snapshot the current piece's identity/pose first so
            # the metadata we store describes the piece visible in `s`.
            s_piece_meta = {
                "piece_id": NAME_TO_ID.get(env.piece_name, 0),
                "rotation": int(env.rotation),
                "piece_row": int(env.piece_row),
                "piece_col": int(env.piece_col),
            }
            s_next, info = env.step(a)
            info.update(s_piece_meta)
            buf.add(s, a, s_next, info)
            s = s_next
            if a == DROP and policy_fn is not None and not env.done:
                policy_fn.reset_target()
        pbar.set_postfix(buf=buf.size, lc=int(buf.lines_cleared[: buf.size].sum()))
        if buf.size >= capacity:
            break
    return buf


def collect_counterfactual(
    episodes: int,
    capacity: int,
    seed: int,
    prime_prob: float = 0.0,
    prime_max_rows: int = 4,
    policy: str = "mixed",
    epsilon: float = 0.3,
) -> CounterfactualReplayBuffer:
    """Like `collect`, but at every visited state forks the env four times and
    records the resulting observation for each action. The real env advances
    using the action chosen by the policy."""
    rng = np.random.default_rng(seed)
    env = TetrisEnv(seed=seed)
    buf = CounterfactualReplayBuffer(capacity=capacity)
    next_states_buf = np.zeros((NUM_ACTIONS, *STATE_SHAPE), dtype=np.float32)

    pbar = tqdm(range(episodes), desc="collecting (CF)")
    for _ in pbar:
        s = env.reset()
        if prime_prob > 0 and rng.random() < prime_prob:
            prime_board(env, rng, max_rows=prime_max_rows)
            s = env.observe()

        if policy in ("heuristic", "mixed"):
            eps = epsilon if policy == "mixed" else 0.0
            policy_fn = MixedExplorationPolicy(env, rng, epsilon=eps)
        else:
            policy_fn = None

        while not env.done and buf.size < capacity:
            if policy_fn is None:
                a_executed = int(rng.integers(0, NUM_ACTIONS))
            else:
                a_executed = policy_fn()

            # Snapshot the piece's identity/pose at observation s before any step.
            s_piece_meta = {
                "piece_id": NAME_TO_ID.get(env.piece_name, 0),
                "rotation": int(env.rotation),
                "piece_row": int(env.piece_row),
                "piece_col": int(env.piece_col),
            }

            # Fork four times, apply each action on a fork; the real env is
            # untouched until we step it below with `a_executed`.
            for a in range(NUM_ACTIONS):
                fork = copy.deepcopy(env)
                next_states_buf[a] = fork.step(a)[0]

            s_next, info = env.step(a_executed)
            info.update(s_piece_meta)
            buf.add(s, next_states=next_states_buf, a_executed=a_executed, info=info)
            s = s_next
            if a_executed == DROP and policy_fn is not None and not env.done:
                policy_fn.reset_target()
        pbar.set_postfix(buf=buf.size, lc=int(buf.lines_cleared[: buf.size].sum()))
        if buf.size >= capacity:
            break
    return buf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--out", type=str, default="data/buffer.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--capacity", type=int, default=500_000)
    parser.add_argument("--prime-prob", type=float, default=0.0,
                        help="Probability of priming an episode with rows almost full.")
    parser.add_argument("--prime-max-rows", type=int, default=4)
    parser.add_argument("--policy", choices=["random", "heuristic", "mixed"], default="random")
    parser.add_argument("--epsilon", type=float, default=0.3,
                        help="Random-action probability for mixed policy.")
    parser.add_argument("--counterfactual", action="store_true",
                        help="Store all 4 action branches per row (CounterfactualReplayBuffer).")
    args = parser.parse_args()

    if args.counterfactual:
        buf = collect_counterfactual(
            args.episodes, args.capacity, args.seed,
            prime_prob=args.prime_prob, prime_max_rows=args.prime_max_rows,
            policy=args.policy, epsilon=args.epsilon,
        )
    else:
        buf = collect(
            args.episodes, args.capacity, args.seed,
            prime_prob=args.prime_prob, prime_max_rows=args.prime_max_rows,
            policy=args.policy, epsilon=args.epsilon,
        )
    buf.save(args.out)
    n_clears = int(buf.lines_cleared[: buf.size].sum())
    label = "counterfactual rows" if args.counterfactual else "triplets"
    print(f"saved {buf.size} {label} to {args.out} ({n_clears} total line-clears)")


if __name__ == "__main__":
    main()
