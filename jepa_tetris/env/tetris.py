"""Simplified Tetris environment with hard-drop semantics.

Pieces do not auto-fall; the agent issues LEFT/RIGHT/ROTATE freely while the
piece sits at its spawn row, and DROP locks the piece at the lowest valid row
in the current column. A new piece spawns immediately after.
"""
from __future__ import annotations

import numpy as np

from .pieces import NAME_TO_ID, NUM_ROTATIONS, PIECES, PIECE_NAMES

BOARD_HEIGHT = 20
BOARD_WIDTH = 10
SPAWN_ROW = 0
SPAWN_COL = 3

LEFT, RIGHT, ROTATE, DROP = 0, 1, 2, 3
NUM_ACTIONS = 4
ACTION_NAMES = ("LEFT", "RIGHT", "ROTATE", "DROP")


class TetrisEnv:
    def __init__(self, seed: int | None = None, max_steps: int = 500):
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int8)
        self.piece_name = "I"
        self.rotation = 0
        self.piece_row = SPAWN_ROW
        self.piece_col = SPAWN_COL
        self.steps = 0
        self.done = False
        self.reset()

    def reset(self) -> np.ndarray:
        self.board.fill(0)
        self.steps = 0
        self.done = False
        self._spawn_piece()
        return self.observe()

    def _spawn_piece(self) -> None:
        idx = int(self.rng.integers(0, len(PIECE_NAMES)))
        self.piece_name = PIECE_NAMES[idx]
        self.rotation = 0
        self.piece_row = SPAWN_ROW
        self.piece_col = SPAWN_COL
        if not self._is_valid(self.piece_row, self.piece_col, self.rotation):
            self.done = True

    def _piece_cells(self, row: int, col: int, rotation: int) -> list[tuple[int, int]]:
        offsets = PIECES[self.piece_name][rotation]
        return [(row + dr, col + dc) for dr, dc in offsets]

    def _is_valid(self, row: int, col: int, rotation: int) -> bool:
        for r, c in self._piece_cells(row, col, rotation):
            if r < 0 or r >= BOARD_HEIGHT or c < 0 or c >= BOARD_WIDTH:
                return False
            if self.board[r, c]:
                return False
        return True

    def step(self, action: int) -> tuple[np.ndarray, dict]:
        if self.done:
            return self.observe(), self._info(0)

        lines_cleared = 0
        if action == LEFT:
            if self._is_valid(self.piece_row, self.piece_col - 1, self.rotation):
                self.piece_col -= 1
        elif action == RIGHT:
            if self._is_valid(self.piece_row, self.piece_col + 1, self.rotation):
                self.piece_col += 1
        elif action == ROTATE:
            new_rot = (self.rotation + 1) % NUM_ROTATIONS
            if self._is_valid(self.piece_row, self.piece_col, new_rot):
                self.rotation = new_rot
        elif action == DROP:
            r = self.piece_row
            while self._is_valid(r + 1, self.piece_col, self.rotation):
                r += 1
            self.piece_row = r
            for cr, cc in self._piece_cells(self.piece_row, self.piece_col, self.rotation):
                self.board[cr, cc] = 1
            lines_cleared = self._clear_lines()
            self._spawn_piece()
        else:
            raise ValueError(f"invalid action: {action}")

        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True
        return self.observe(), self._info(lines_cleared)

    def _clear_lines(self) -> int:
        full_rows = np.where(self.board.all(axis=1))[0]
        n = len(full_rows)
        if n == 0:
            return 0
        keep = np.delete(self.board, full_rows, axis=0)
        pad = np.zeros((n, BOARD_WIDTH), dtype=np.int8)
        self.board = np.vstack([pad, keep])
        return int(n)

    def observe(self) -> np.ndarray:
        obs = np.zeros((2, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
        obs[0] = self.board.astype(np.float32)
        if not self.done:
            for r, c in self._piece_cells(self.piece_row, self.piece_col, self.rotation):
                if 0 <= r < BOARD_HEIGHT and 0 <= c < BOARD_WIDTH:
                    obs[1, r, c] = 1.0
        return obs

    def _info(self, lines_cleared: int) -> dict:
        return {
            "lines_cleared": lines_cleared,
            "holes": self._count_holes(),
            "aggregate_height": self._aggregate_height(),
            "done": self.done,
            "steps": self.steps,
            "piece_id": NAME_TO_ID.get(self.piece_name, 0),
            "rotation": int(self.rotation),
            "piece_row": int(self.piece_row),
            "piece_col": int(self.piece_col),
        }

    def _count_holes(self) -> int:
        holes = 0
        for c in range(BOARD_WIDTH):
            col = self.board[:, c]
            filled = np.where(col)[0]
            if len(filled) == 0:
                continue
            top = int(filled[0])
            holes += int((col[top:] == 0).sum())
        return holes

    def _aggregate_height(self) -> int:
        total = 0
        for c in range(BOARD_WIDTH):
            col = self.board[:, c]
            filled = np.where(col)[0]
            if len(filled) > 0:
                total += BOARD_HEIGHT - int(filled[0])
        return total
