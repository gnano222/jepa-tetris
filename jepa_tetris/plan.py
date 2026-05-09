"""Brute-force BFS planners.

Two variants:
- BFSPlanner: rolls out action sequences in *latent* space via the JEPA predictor.
  Pure JEPA approach but suffers from compounding multi-step prediction error.
- RealDynamicsPlanner: rolls out action sequences in the *real env* (cheap copy
  of board state), then uses the JEPA encoder + probe only to score leaves.
  Hybrid approach — isolates whether the probe is useful given accurate dynamics.

Both planners optionally filter to action sequences that contain at least one
DROP, preventing the degenerate "stall" exploit where the agent maximizes score
by never advancing the game.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import torch

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


class BFSPlanner:
    """Enumerate all action sequences of length `depth`, roll out predicted latents,
    score each leaf with the probe head, and return the first action of the best
    sequence.

    Score: w_lines * lines + w_holes * holes + w_height * aggregate_height.
    Default weights: maximize lines, penalize holes and stack height.
    """

    def __init__(
        self,
        encoder,
        action_encoder,
        predictor,
        probe,
        depth: int = 4,
        device: str | torch.device = "cpu",
        lines_w: float = 1.0,
        holes_w: float = -0.5,
        height_w: float = -0.1,
        target_mean: np.ndarray | None = None,
        target_std: np.ndarray | None = None,
        require_drop: bool = True,
    ):
        self.encoder = encoder
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.probe = probe
        self.depth = depth
        self.device = torch.device(device)
        self.weights = torch.tensor([lines_w, holes_w, height_w], device=self.device)
        if target_mean is None:
            target_mean = np.zeros(3, dtype=np.float32)
        if target_std is None:
            target_std = np.ones(3, dtype=np.float32)
        self.target_mean = torch.from_numpy(np.asarray(target_mean, dtype=np.float32)).to(self.device)
        self.target_std = torch.from_numpy(np.asarray(target_std, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            actions = torch.arange(NUM_ACTIONS, device=self.device)
            self.action_embs = action_encoder(actions)
        seqs = list(product(range(NUM_ACTIONS), repeat=depth))
        if require_drop:
            seqs = [s for s in seqs if DROP in s]
        self.sequences = torch.tensor(seqs, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> int:
        return self.select_plan(obs)[0]

    @torch.no_grad()
    def select_plan(self, obs: np.ndarray) -> list[int]:
        """Return the full best action sequence (length == depth)."""
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        z = self.encoder(obs_t)
        n = self.sequences.shape[0]
        z = z.expand(n, -1).contiguous()
        for t in range(self.depth):
            a_idx = self.sequences[:, t]
            a_emb = self.action_embs[a_idx]
            z = self.predictor(z, a_emb)
        # Probe predicts normalized features; unnormalize before scoring.
        features = self.probe(z) * self.target_std + self.target_mean
        scores = (features * self.weights).sum(dim=1)
        best = int(scores.argmax().item())
        return [int(x) for x in self.sequences[best].tolist()]


def _snapshot_env(env: TetrisEnv) -> tuple:
    return (
        env.board.copy(),
        env.piece_name,
        env.rotation,
        env.piece_row,
        env.piece_col,
        env.steps,
        env.done,
        env.rng.bit_generator.state,
    )


def _restore_env(env: TetrisEnv, snap: tuple) -> None:
    board, name, rot, prow, pcol, steps, done, rng_state = snap
    env.board[:] = board
    env.piece_name = name
    env.rotation = rot
    env.piece_row = prow
    env.piece_col = pcol
    env.steps = steps
    env.done = done
    env.rng.bit_generator.state = rng_state


class RealDynamicsPlanner:
    """BFS over action sequences using the actual env for transitions.
    Scores leaves by encoding the post-rollout observation through the JEPA
    encoder and probe. Isolates whether the probe's representation is useful
    given accurate dynamics (no predictor compounding error).
    """

    def __init__(
        self,
        encoder,
        probe,
        env: TetrisEnv,
        depth: int = 4,
        device: str | torch.device = "cpu",
        lines_w: float = 1.0,
        holes_w: float = -0.5,
        height_w: float = -0.1,
        target_mean: np.ndarray | None = None,
        target_std: np.ndarray | None = None,
        require_drop: bool = True,
    ):
        self.encoder = encoder
        self.probe = probe
        self.env = env
        self.depth = depth
        self.device = torch.device(device)
        self.weights = torch.tensor([lines_w, holes_w, height_w], device=self.device)
        if target_mean is None:
            target_mean = np.zeros(3, dtype=np.float32)
        if target_std is None:
            target_std = np.ones(3, dtype=np.float32)
        self.target_mean = torch.from_numpy(np.asarray(target_mean, dtype=np.float32)).to(self.device)
        self.target_std = torch.from_numpy(np.asarray(target_std, dtype=np.float32)).to(self.device)
        seqs = list(product(range(NUM_ACTIONS), repeat=depth))
        if require_drop:
            seqs = [s for s in seqs if DROP in s]
        self.sequences = seqs

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> int:
        return self.select_plan(obs)[0]

    @torch.no_grad()
    def select_plan(self, obs: np.ndarray) -> list[int]:
        snap = _snapshot_env(self.env)
        leaves = []
        cumulative_lines = []
        for seq in self.sequences:
            _restore_env(self.env, snap)
            cl = 0
            for a in seq:
                obs_after, info = self.env.step(int(a))
                cl += int(info["lines_cleared"])
                if self.env.done:
                    break
            leaves.append(self.env.observe())
            cumulative_lines.append(cl)
        _restore_env(self.env, snap)

        leaves_arr = np.stack(leaves, axis=0).astype(np.float32)
        leaves_t = torch.from_numpy(leaves_arr).to(self.device)
        z = self.encoder(leaves_t)
        features = self.probe(z) * self.target_std + self.target_mean
        # Replace probe-predicted lines with TRUE cumulative lines from rollout
        # (since the rollout gave us the real number, no need to predict it).
        cumulative_t = torch.tensor(cumulative_lines, dtype=torch.float32, device=self.device)
        features = torch.stack([cumulative_t, features[:, 1], features[:, 2]], dim=1)
        scores = (features * self.weights).sum(dim=1)
        best = int(scores.argmax().item())
        return [int(x) for x in self.sequences[best]]


class PlacementPlanner:
    """Enumerate all (col, rotation) endpoints for the current piece, simulate
    each in the real env, score with the JEPA encoder + probe (using true
    cumulative lines from the simulation). Removes the depth-K reachability
    limitation of BFS-over-actions.
    """

    def __init__(
        self,
        encoder,
        probe,
        env: TetrisEnv,
        device: str | torch.device = "cpu",
        lines_w: float = 1.0,
        holes_w: float = -0.5,
        height_w: float = -0.1,
        target_mean: np.ndarray | None = None,
        target_std: np.ndarray | None = None,
        max_path_steps: int = 16,
    ):
        self.encoder = encoder
        self.probe = probe
        self.env = env
        self.device = torch.device(device)
        self.weights = torch.tensor([lines_w, holes_w, height_w], device=self.device)
        if target_mean is None:
            target_mean = np.zeros(3, dtype=np.float32)
        if target_std is None:
            target_std = np.ones(3, dtype=np.float32)
        self.target_mean = torch.from_numpy(np.asarray(target_mean, dtype=np.float32)).to(self.device)
        self.target_std = torch.from_numpy(np.asarray(target_std, dtype=np.float32)).to(self.device)
        self.max_path_steps = max_path_steps

    def _path_to_target(self, snap, target_col: int, target_rot: int) -> tuple[list[int], int, np.ndarray] | None:
        """Restore env to snap, drive piece to (target_col, target_rot), then drop.
        Return (action_sequence, cumulative_lines, leaf_obs) or None if unreachable."""
        _restore_env(self.env, snap)
        plan: list[int] = []
        cumulative_lines = 0
        for _ in range(self.max_path_steps):
            if self.env.done:
                return None
            if self.env.rotation == target_rot and self.env.piece_col == target_col:
                _, info = self.env.step(DROP)
                plan.append(DROP)
                cumulative_lines += int(info["lines_cleared"])
                return plan, cumulative_lines, self.env.observe()
            if self.env.rotation != target_rot:
                a = ROTATE
            elif self.env.piece_col < target_col:
                a = RIGHT
            elif self.env.piece_col > target_col:
                a = LEFT
            else:
                a = DROP  # shouldn't reach
            prev_rot = self.env.rotation
            prev_col = self.env.piece_col
            self.env.step(a)
            plan.append(a)
            # If the action had no effect (invalid move), abort.
            if a == ROTATE and self.env.rotation == prev_rot:
                return None
            if a in (LEFT, RIGHT) and self.env.piece_col == prev_col:
                return None
        return None

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> int:
        return self.select_plan(obs)[0]

    @torch.no_grad()
    def select_plan(self, obs: np.ndarray) -> list[int]:
        snap = _snapshot_env(self.env)
        leaves: list[np.ndarray] = []
        plans: list[list[int]] = []
        cumulative_lines: list[int] = []
        for col in range(BOARD_WIDTH):
            for rot in range(NUM_ROTATIONS):
                result = self._path_to_target(snap, col, rot)
                if result is None:
                    continue
                p, cl, leaf = result
                plans.append(p)
                cumulative_lines.append(cl)
                leaves.append(leaf)
        _restore_env(self.env, snap)

        if not plans:
            return [DROP]  # no reachable placement — just drop in place

        leaves_arr = np.stack(leaves, axis=0).astype(np.float32)
        leaves_t = torch.from_numpy(leaves_arr).to(self.device)
        z = self.encoder(leaves_t)
        features = self.probe(z) * self.target_std + self.target_mean
        cumulative_t = torch.tensor(cumulative_lines, dtype=torch.float32, device=self.device)
        features = torch.stack([cumulative_t, features[:, 1], features[:, 2]], dim=1)
        scores = (features * self.weights).sum(dim=1)
        best = int(scores.argmax().item())
        return plans[best]
