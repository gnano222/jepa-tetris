"""Numpy-backed ring replay buffers.

`ReplayBuffer` stores classic (s, a, s', info) triplets — one action per row.
`CounterfactualReplayBuffer` stores (s, next_states[A], a_executed, info) rows,
where `next_states[i]` is the observation after applying action i to the same
starting state s. The counterfactual buffer is the data substrate for
training the predictor with direct contrast across actions.

The two buffers serialize to disjoint `.npz` schemas (distinguished by the
presence of the `next_states` key), so loading the wrong format raises rather
than silently misreading.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

STATE_SHAPE = (2, 20, 10)
NUM_ACTIONS = 4

# Optional info-dict keys that, when present, populate the piece-metadata arrays.
# Buffers loaded from older `.npz` files lack these and report has_piece_meta=False.
_PIECE_META_KEYS = ("piece_id", "rotation", "piece_row", "piece_col")


class ReplayBuffer:
    def __init__(self, capacity: int, state_shape: tuple[int, ...] = STATE_SHAPE):
        self.capacity = capacity
        self.state_shape = state_shape
        self.s = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.s_next = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.lines_cleared = np.zeros(capacity, dtype=np.float32)
        self.holes = np.zeros(capacity, dtype=np.float32)
        self.aggregate_height = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.bool_)
        # Piece metadata (added in v2). Old buffers load with these zero-filled
        # and has_piece_meta=False so consumers can branch on availability.
        self.piece_id = np.zeros(capacity, dtype=np.int8)
        self.rotation = np.zeros(capacity, dtype=np.int8)
        self.piece_row = np.zeros(capacity, dtype=np.int8)
        self.piece_col = np.zeros(capacity, dtype=np.int8)
        self.has_piece_meta = True
        self.size = 0
        self.idx = 0

    def add(self, s: np.ndarray, a: int, s_next: np.ndarray, info: dict) -> None:
        i = self.idx
        self.s[i] = s
        self.a[i] = a
        self.s_next[i] = s_next
        self.lines_cleared[i] = info["lines_cleared"]
        self.holes[i] = info["holes"]
        self.aggregate_height[i] = info["aggregate_height"]
        self.done[i] = info["done"]
        # Piece-metadata keys are optional so legacy callers (e.g. unit tests
        # that build minimal info dicts) keep working unchanged.
        self.piece_id[i] = info.get("piece_id", 0)
        self.rotation[i] = info.get("rotation", 0)
        self.piece_row[i] = info.get("piece_row", 0)
        self.piece_col[i] = info.get("piece_col", 0)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator | None = None) -> dict:
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            "s": self.s[idx],
            "a": self.a[idx],
            "s_next": self.s_next[idx],
            "lines_cleared": self.lines_cleared[idx],
            "holes": self.holes[idx],
            "aggregate_height": self.aggregate_height[idx],
            "done": self.done[idx],
            "piece_id": self.piece_id[idx],
            "rotation": self.rotation[idx],
            "piece_row": self.piece_row[idx],
            "piece_col": self.piece_col[idx],
            "indices": idx,
        }

    def sample_rollout(self, batch_size: int, k: int, rng: np.random.Generator | None = None) -> dict:
        """Sample K-step rollouts from contiguous within-episode triplets.

        Returns:
            s0:        (B, *state_shape)
            actions:   (B, K)
            s_next_k:  (B, K, *state_shape)  — the actual s_next at each rollout step
        """
        if rng is None:
            rng = np.random.default_rng()
        if k < 1:
            raise ValueError("k must be >= 1")
        max_start = self.size - k
        if max_start <= 0:
            raise ValueError(f"buffer too small ({self.size}) for k={k}")
        # Oversample candidate starts and filter out those crossing episode boundaries.
        # An episode boundary at index j means buf.done[j] is True and buf.s[j+1]
        # is from a new episode. Sequences crossing this boundary are invalid.
        # For a K-step rollout starting at `s`, we need done[s..s+k-2] all False.
        valid: list[int] = []
        attempts = 0
        while len(valid) < batch_size and attempts < 8:
            attempts += 1
            cand = rng.integers(0, max_start, size=batch_size * 4)
            for s in cand:
                if k == 1 or not self.done[s : s + k - 1].any():
                    valid.append(int(s))
                    if len(valid) >= batch_size:
                        break
        if len(valid) < batch_size:
            raise RuntimeError(f"could not find {batch_size} valid rollouts of length {k}")
        starts = np.array(valid[:batch_size])
        s0 = self.s[starts]
        actions = np.stack([self.a[starts + i] for i in range(k)], axis=1)
        s_next_k = np.stack([self.s_next[starts + i] for i in range(k)], axis=1)
        return {"s0": s0, "actions": actions, "s_next_k": s_next_k, "starts": starts}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            s=self.s[: self.size],
            a=self.a[: self.size],
            s_next=self.s_next[: self.size],
            lines_cleared=self.lines_cleared[: self.size],
            holes=self.holes[: self.size],
            aggregate_height=self.aggregate_height[: self.size],
            done=self.done[: self.size],
            piece_id=self.piece_id[: self.size],
            rotation=self.rotation[: self.size],
            piece_row=self.piece_row[: self.size],
            piece_col=self.piece_col[: self.size],
        )

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        data = np.load(path)
        size = int(data["a"].shape[0])
        buf = cls(capacity=max(size, 1))
        buf.s[:size] = data["s"]
        buf.a[:size] = data["a"]
        buf.s_next[:size] = data["s_next"]
        buf.lines_cleared[:size] = data["lines_cleared"]
        buf.holes[:size] = data["holes"]
        buf.aggregate_height[:size] = data["aggregate_height"]
        buf.done[:size] = data["done"]
        # Backwards compatibility: pre-v2 buffers don't carry piece metadata.
        # Detect by inspecting the npz file's array names; absence -> zero-fill
        # the piece arrays and flag the buffer as lacking metadata.
        present = set(data.files)
        if all(k in present for k in _PIECE_META_KEYS):
            buf.piece_id[:size] = data["piece_id"]
            buf.rotation[:size] = data["rotation"]
            buf.piece_row[:size] = data["piece_row"]
            buf.piece_col[:size] = data["piece_col"]
            buf.has_piece_meta = True
        else:
            buf.has_piece_meta = False
        buf.size = size
        buf.idx = size % buf.capacity
        return buf


class CounterfactualReplayBuffer:
    """Replay buffer storing all `NUM_ACTIONS` counterfactual next-states per row.

    Each row holds:
        s                : starting observation
        next_states[a]   : observation after applying action a to a fork of the env
        a_executed       : action the policy actually took (chain continuation)
        info fields      : the info dict from the executed action (lines_cleared,
                           holes, aggregate_height, done, plus piece metadata)

    The serialized `.npz` carries `next_states` as a key — its presence is how
    the loader distinguishes this format from the single-action `ReplayBuffer`.
    """

    def __init__(
        self,
        capacity: int,
        state_shape: tuple[int, ...] = STATE_SHAPE,
        num_actions: int = NUM_ACTIONS,
    ):
        self.capacity = capacity
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.s = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, num_actions, *state_shape), dtype=np.float32)
        self.a_executed = np.zeros(capacity, dtype=np.int64)
        self.lines_cleared = np.zeros(capacity, dtype=np.float32)
        self.holes = np.zeros(capacity, dtype=np.float32)
        self.aggregate_height = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.bool_)
        self.piece_id = np.zeros(capacity, dtype=np.int8)
        self.rotation = np.zeros(capacity, dtype=np.int8)
        self.piece_row = np.zeros(capacity, dtype=np.int8)
        self.piece_col = np.zeros(capacity, dtype=np.int8)
        self.size = 0
        self.idx = 0

    def add(
        self,
        s: np.ndarray,
        *,
        next_states: np.ndarray,
        a_executed: int,
        info: dict,
    ) -> None:
        if next_states.shape[0] != self.num_actions:
            raise ValueError(
                f"expected next_states.shape[0]={self.num_actions}, got {next_states.shape[0]}"
            )
        if next_states.shape[1:] != self.state_shape:
            raise ValueError(
                f"expected next_states[i].shape={self.state_shape}, got {next_states.shape[1:]}"
            )
        i = self.idx
        self.s[i] = s
        self.next_states[i] = next_states
        self.a_executed[i] = int(a_executed)
        self.lines_cleared[i] = info["lines_cleared"]
        self.holes[i] = info["holes"]
        self.aggregate_height[i] = info["aggregate_height"]
        self.done[i] = info["done"]
        self.piece_id[i] = info.get("piece_id", 0)
        self.rotation[i] = info.get("rotation", 0)
        self.piece_row[i] = info.get("piece_row", 0)
        self.piece_col[i] = info.get("piece_col", 0)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator | None = None) -> dict:
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            "s": self.s[idx],
            "next_states": self.next_states[idx],
            "a_executed": self.a_executed[idx],
            "lines_cleared": self.lines_cleared[idx],
            "holes": self.holes[idx],
            "aggregate_height": self.aggregate_height[idx],
            "done": self.done[idx],
            "piece_id": self.piece_id[idx],
            "rotation": self.rotation[idx],
            "piece_row": self.piece_row[idx],
            "piece_col": self.piece_col[idx],
            "indices": idx,
        }

    def sample_rollout(self, batch_size: int, k: int, rng: np.random.Generator | None = None) -> dict:
        """Sample K-step rollouts. The chain follows `a_executed` at each step;
        `next_states_k[b, t]` carries all NUM_ACTIONS counterfactuals at step t."""
        if rng is None:
            rng = np.random.default_rng()
        if k < 1:
            raise ValueError("k must be >= 1")
        max_start = self.size - k
        if max_start <= 0:
            raise ValueError(f"buffer too small ({self.size}) for k={k}")
        valid: list[int] = []
        attempts = 0
        while len(valid) < batch_size and attempts < 8:
            attempts += 1
            cand = rng.integers(0, max_start, size=batch_size * 4)
            for s in cand:
                if k == 1 or not self.done[s : s + k - 1].any():
                    valid.append(int(s))
                    if len(valid) >= batch_size:
                        break
        if len(valid) < batch_size:
            raise RuntimeError(f"could not find {batch_size} valid rollouts of length {k}")
        starts = np.array(valid[:batch_size])
        s0 = self.s[starts]
        actions_executed = np.stack([self.a_executed[starts + i] for i in range(k)], axis=1)
        next_states_k = np.stack([self.next_states[starts + i] for i in range(k)], axis=1)
        return {
            "s0": s0,
            "actions_executed": actions_executed,
            "next_states_k": next_states_k,
            "starts": starts,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            s=self.s[: self.size],
            next_states=self.next_states[: self.size],
            a_executed=self.a_executed[: self.size],
            lines_cleared=self.lines_cleared[: self.size],
            holes=self.holes[: self.size],
            aggregate_height=self.aggregate_height[: self.size],
            done=self.done[: self.size],
            piece_id=self.piece_id[: self.size],
            rotation=self.rotation[: self.size],
            piece_row=self.piece_row[: self.size],
            piece_col=self.piece_col[: self.size],
        )

    @classmethod
    def load(cls, path: str | Path) -> "CounterfactualReplayBuffer":
        data = np.load(path)
        if "next_states" not in data.files:
            raise ValueError(
                f"{path} does not look like a counterfactual buffer "
                f"(missing 'next_states' key — got {sorted(data.files)})"
            )
        size = int(data["a_executed"].shape[0])
        num_actions = int(data["next_states"].shape[1])
        buf = cls(capacity=max(size, 1), num_actions=num_actions)
        buf.s[:size] = data["s"]
        buf.next_states[:size] = data["next_states"]
        buf.a_executed[:size] = data["a_executed"]
        buf.lines_cleared[:size] = data["lines_cleared"]
        buf.holes[:size] = data["holes"]
        buf.aggregate_height[:size] = data["aggregate_height"]
        buf.done[:size] = data["done"]
        present = set(data.files)
        if all(k in present for k in _PIECE_META_KEYS):
            buf.piece_id[:size] = data["piece_id"]
            buf.rotation[:size] = data["rotation"]
            buf.piece_row[:size] = data["piece_row"]
            buf.piece_col[:size] = data["piece_col"]
        buf.size = size
        buf.idx = size % buf.capacity
        return buf
