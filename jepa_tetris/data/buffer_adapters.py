"""Buffer-type-agnostic adapters.

Consumers that want to work uniformly with both `ReplayBuffer` (single-action)
and `CounterfactualReplayBuffer` (per-row 4-action fanout) call these adapters
to get a consistent ``s, a, s_next`` view back. The decoder training script
uses them so a single code path handles either training mode.

Why a CF row's "s_next" is the executed branch: at training step i the agent
took action `a_executed[i]`, and the simulator produced `next_states[a_executed]`.
That is the on-policy transition. The other three branches are off-policy
counterfactuals — useful for the predictor (they're what `train.py --counterfactual`
contrasts over) but not the right "actual next state" for a single-action view.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from jepa_tetris.data.replay_buffer import (
    CounterfactualReplayBuffer,
    ReplayBuffer,
)


def load_buffer(path: str | Path):
    """Load either buffer type by inspecting the npz schema.

    Counterfactual buffers carry a `next_states` key; legacy single-action
    buffers carry `s_next`.
    """
    data = np.load(path)
    if "next_states" in data.files:
        return CounterfactualReplayBuffer.load(path)
    return ReplayBuffer.load(path)


def sample_normalized(
    buf: ReplayBuffer | CounterfactualReplayBuffer,
    n: int,
    rng: np.random.Generator,
) -> dict:
    """Return a (s, a, s_next, info) view from either buffer type.

    For ``CounterfactualReplayBuffer``, `s_next` is the on-policy branch —
    ``next_states[a_executed]``. The other 3 branches are dropped here; consumers
    that need them should call ``buf.sample`` directly.
    """
    if isinstance(buf, CounterfactualReplayBuffer):
        batch = buf.sample(n, rng=rng)
        a = batch["a_executed"]                              # (n,)
        idx = a.astype(np.int64)[:, None, None, None, None]  # (n, 1, 1, 1, 1)
        s_next = np.take_along_axis(batch["next_states"], idx, axis=1).squeeze(1)
        return {
            "s": batch["s"],
            "a": a,
            "s_next": s_next,
            "lines_cleared": batch["lines_cleared"],
            "holes": batch["holes"],
            "aggregate_height": batch["aggregate_height"],
            "done": batch["done"],
            "indices": batch["indices"],
        }
    return buf.sample(n, rng=rng)


def cf_to_single_action_buffer(cf: CounterfactualReplayBuffer) -> ReplayBuffer:
    """Derive a single-action `ReplayBuffer` from a `CounterfactualReplayBuffer`.

    Used to feed the same exact transitions into both training paths in the
    CF-vs-single comparison: train the CF model on `cf.npz`, then convert and
    train the single-action baseline on the derived `.npz` so the only
    difference between runs is the loss function (counterfactual vs the
    single-action MSE on the executed branch).
    """
    n = cf.size
    rb = ReplayBuffer(capacity=max(n, 1))
    rng_actions = cf.a_executed[:n].astype(np.int64)
    idx = rng_actions[:, None, None, None, None]                              # (n, 1, 1, 1, 1)
    s_next = np.take_along_axis(cf.next_states[:n], idx, axis=1).squeeze(1)   # (n, *state_shape)
    rb.s[:n] = cf.s[:n]
    rb.a[:n] = rng_actions
    rb.s_next[:n] = s_next
    rb.lines_cleared[:n] = cf.lines_cleared[:n]
    rb.holes[:n] = cf.holes[:n]
    rb.aggregate_height[:n] = cf.aggregate_height[:n]
    rb.done[:n] = cf.done[:n]
    rb.piece_id[:n] = cf.piece_id[:n]
    rb.rotation[:n] = cf.rotation[:n]
    rb.piece_row[:n] = cf.piece_row[:n]
    rb.piece_col[:n] = cf.piece_col[:n]
    rb.size = n
    rb.idx = n % rb.capacity
    rb.has_piece_meta = True
    return rb


def sample_rollout_normalized(
    buf: ReplayBuffer | CounterfactualReplayBuffer,
    n: int,
    k: int,
    rng: np.random.Generator,
) -> dict:
    """Return a (s0, actions, s_next_k) rollout view from either buffer type.

    For ``CounterfactualReplayBuffer``, the chain follows ``a_executed`` at each
    step and `s_next_k[b, t]` is the on-policy executed branch at step t.
    """
    if isinstance(buf, CounterfactualReplayBuffer):
        roll = buf.sample_rollout(n, k, rng=rng)
        actions = roll["actions_executed"]                                       # (n, k)
        b_idx = np.arange(actions.shape[0])[:, None]
        k_idx = np.arange(actions.shape[1])[None, :]
        s_next_k = roll["next_states_k"][b_idx, k_idx, actions]                  # (n, k, *)
        return {
            "s0": roll["s0"],
            "actions": actions,
            "s_next_k": s_next_k,
            "starts": roll["starts"],
        }
    return buf.sample_rollout(n, k, rng=rng)
