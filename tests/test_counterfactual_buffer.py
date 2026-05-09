"""Tests for CounterfactualReplayBuffer (per-row 4-action fanout)."""
from __future__ import annotations

import numpy as np

from jepa_tetris.data.replay_buffer import (
    NUM_ACTIONS,
    CounterfactualReplayBuffer,
)


def _info(lc=0, h=0, ah=0, done=False):
    return {"lines_cleared": lc, "holes": h, "aggregate_height": ah, "done": done}


def _state(value: int = 0) -> np.ndarray:
    s = np.zeros((2, 20, 10), dtype=np.float32)
    s[0, 0, 0] = float(value)
    return s


def test_add_increments_size_and_stores_per_action_next_states():
    buf = CounterfactualReplayBuffer(capacity=4)
    s = _state(0)
    next_states = np.stack([_state(i + 1) for i in range(NUM_ACTIONS)])
    buf.add(s, next_states=next_states, a_executed=2, info=_info())
    assert buf.size == 1
    np.testing.assert_array_equal(buf.s[0], s)
    np.testing.assert_array_equal(buf.next_states[0], next_states)
    assert buf.a_executed[0] == 2


def test_add_rejects_wrong_shape_next_states():
    import pytest
    buf = CounterfactualReplayBuffer(capacity=4)
    s = _state()
    bad = np.zeros((NUM_ACTIONS - 1, 2, 20, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        buf.add(s, next_states=bad, a_executed=0, info=_info())


def test_sample_returns_correct_shapes():
    buf = CounterfactualReplayBuffer(capacity=20)
    next_states = np.stack([_state(i + 1) for i in range(NUM_ACTIONS)])
    for k in range(10):
        buf.add(_state(k), next_states=next_states, a_executed=k % NUM_ACTIONS, info=_info(lc=k))
    batch = buf.sample(8)
    assert batch["s"].shape == (8, 2, 20, 10)
    assert batch["next_states"].shape == (8, NUM_ACTIONS, 2, 20, 10)
    assert batch["a_executed"].shape == (8,)
    assert batch["lines_cleared"].shape == (8,)


def test_save_load_roundtrip(tmp_path):
    buf = CounterfactualReplayBuffer(capacity=10)
    next_states = np.stack([_state(i + 7) for i in range(NUM_ACTIONS)])
    for i in range(5):
        buf.add(_state(i), next_states=next_states, a_executed=i % NUM_ACTIONS,
                info=_info(lc=i, h=i + 1, ah=i + 2))
    path = tmp_path / "cf_buf.npz"
    buf.save(str(path))
    loaded = CounterfactualReplayBuffer.load(str(path))
    assert loaded.size == 5
    np.testing.assert_array_equal(loaded.s[:5], buf.s[:5])
    np.testing.assert_array_equal(loaded.next_states[:5], buf.next_states[:5])
    np.testing.assert_array_equal(loaded.a_executed[:5], buf.a_executed[:5])


def test_load_rejects_legacy_single_action_buffer(tmp_path):
    """Old ReplayBuffer .npz files lack the per-action next_states key. Loading
    one as a counterfactual buffer must fail loudly so the two formats don't
    silently mix."""
    import pytest

    from jepa_tetris.data.replay_buffer import ReplayBuffer
    legacy = ReplayBuffer(capacity=5)
    s = _state()
    legacy.add(s, 0, s, _info())
    legacy_path = tmp_path / "legacy.npz"
    legacy.save(str(legacy_path))
    with pytest.raises((KeyError, ValueError)):
        CounterfactualReplayBuffer.load(str(legacy_path))


def test_sample_rollout_returns_correct_shapes():
    buf = CounterfactualReplayBuffer(capacity=200)
    next_states = np.stack([_state(i) for i in range(NUM_ACTIONS)])
    for i in range(50):
        buf.add(_state(i), next_states=next_states, a_executed=i % NUM_ACTIONS,
                info=_info(done=(i == 49)))
    rng = np.random.default_rng(0)
    batch = buf.sample_rollout(8, k=4, rng=rng)
    assert batch["s0"].shape == (8, 2, 20, 10)
    assert batch["next_states_k"].shape == (8, 4, NUM_ACTIONS, 2, 20, 10)
    assert batch["actions_executed"].shape == (8, 4)


def test_sample_rollout_chain_uses_actions_executed():
    """Within a K-step rollout, the chain follows the action that was actually
    executed at each step. Verify next_states_k aligns with that path."""
    buf = CounterfactualReplayBuffer(capacity=200)
    next_states = np.stack([_state(100 + i) for i in range(NUM_ACTIONS)])
    for i in range(20):
        buf.add(_state(i), next_states=next_states,
                a_executed=i % NUM_ACTIONS, info=_info(done=(i == 19)))
    rng = np.random.default_rng(0)
    batch = buf.sample_rollout(4, k=3, rng=rng)
    starts = batch["starts"]
    for b in range(4):
        for t in range(3):
            assert batch["actions_executed"][b, t] == buf.a_executed[starts[b] + t]
            np.testing.assert_array_equal(
                batch["next_states_k"][b, t], buf.next_states[starts[b] + t],
            )
