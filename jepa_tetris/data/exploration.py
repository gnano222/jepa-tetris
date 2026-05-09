"""Heuristic-guided exploration: picks placements that maximize line clears
and minimize holes. Used to seed the replay buffer with line-clear examples
that pure random play almost never produces.

The exploration policy alternates between: (a) random action with probability
`epsilon`, (b) action that moves the current piece toward a pre-computed
'best placement' for the current piece. The best placement is recomputed
after each DROP (when a new piece spawns).
"""
from __future__ import annotations

import numpy as np

from jepa_tetris.env.tetris import (
    BOARD_WIDTH,
    DROP,
    LEFT,
    NUM_ACTIONS,
    NUM_ROTATIONS,
    RIGHT,
    ROTATE,
    TetrisEnv,
)


def best_placement(env: TetrisEnv) -> tuple[int, int]:
    """Return (col, rotation) that maximizes a Tetris heuristic for the current piece.

    Heuristic: 10 * lines_cleared - holes - 0.3 * aggregate_height.
    Brute-force enumerates all (rotation, col) pairs that have a valid spawn position,
    simulates a hard drop, scores the resulting board, restores state.
    """
    if env.done:
        return env.piece_col, env.rotation

    best_score = -float("inf")
    best = (env.piece_col, env.rotation)
    saved_board = env.board.copy()
    saved_pos = (env.piece_row, env.piece_col, env.rotation)

    for rot in range(NUM_ROTATIONS):
        for col in range(BOARD_WIDTH):
            if not env._is_valid(env.piece_row, col, rot):
                continue
            r = env.piece_row
            while env._is_valid(r + 1, col, rot):
                r += 1
            for cr, cc in env._piece_cells(r, col, rot):
                env.board[cr, cc] = 1
            lines = int(env.board.all(axis=1).sum())
            holes = env._count_holes()
            height = env._aggregate_height()
            env.board[:] = saved_board
            score = 10.0 * lines - 1.0 * holes - 0.3 * height
            if score > best_score:
                best_score = score
                best = (col, rot)

    env.board[:] = saved_board
    env.piece_row, env.piece_col, env.rotation = saved_pos
    return best


def heuristic_action(env: TetrisEnv, target_col: int, target_rot: int) -> int:
    """Pick a single action that moves toward (target_col, target_rot)."""
    if env.rotation != target_rot:
        return ROTATE
    if env.piece_col < target_col:
        return RIGHT
    if env.piece_col > target_col:
        return LEFT
    return DROP


class MixedExplorationPolicy:
    """Mixed random / heuristic policy. Recomputes target placement after each DROP."""

    def __init__(self, env: TetrisEnv, rng: np.random.Generator, epsilon: float = 0.3):
        self.env = env
        self.rng = rng
        self.epsilon = epsilon
        self.target_col, self.target_rot = best_placement(env)

    def reset_target(self) -> None:
        self.target_col, self.target_rot = best_placement(self.env)

    def __call__(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, NUM_ACTIONS))
        return heuristic_action(self.env, self.target_col, self.target_rot)
