"""Tests for buffer-type-agnostic sampling adapters.

Used by the decoder training script (and any future consumer that wants to
work uniformly across single-action and counterfactual replay buffers).
"""
from __future__ import annotations

import numpy as np

from jepa_tetris.data.buffer_adapters import (
    load_buffer,
    sample_normalized,
    sample_rollout_normalized,
)
from jepa_tetris.data.replay_buffer import (
    NUM_ACTIONS,
    CounterfactualReplayBuffer,
    ReplayBuffer,
)


def _info(lc=0, h=0, ah=0, done=False):
    return {"lines_cleared": lc, "holes": h, "aggregate_height": ah, "done": done}


def _state(value: int = 0) -> np.ndarray:
    s = np.zeros((2, 20, 10), dtype=np.float32)
    s[0, 0, 0] = float(value)
    return s


def _build_legacy_buffer(n: int = 30) -> ReplayBuffer:
    buf = ReplayBuffer(capacity=n)
    for i in range(n):
        buf.add(_state(i), i % NUM_ACTIONS, _state(i + 100), _info(done=(i == n - 1)))
    return buf


def _build_cf_buffer(n: int = 30) -> CounterfactualReplayBuffer:
    buf = CounterfactualReplayBuffer(capacity=n)
    for i in range(n):
        # Encode the executed-action's next-state with a distinctive marker so we
        # can verify that the adapter pulls the right slice from next_states.
        next_states = np.stack([_state(i * 10 + a) for a in range(NUM_ACTIONS)])
        buf.add(_state(i), next_states=next_states, a_executed=i % NUM_ACTIONS,
                info=_info(done=(i == n - 1)))
    return buf


def test_sample_normalized_shape_for_legacy_buffer():
    buf = _build_legacy_buffer()
    rng = np.random.default_rng(0)
    out = sample_normalized(buf, n=8, rng=rng)
    assert out["s"].shape == (8, 2, 20, 10)
    assert out["a"].shape == (8,)
    assert out["s_next"].shape == (8, 2, 20, 10)


def test_sample_normalized_shape_for_cf_buffer():
    buf = _build_cf_buffer()
    rng = np.random.default_rng(0)
    out = sample_normalized(buf, n=8, rng=rng)
    assert out["s"].shape == (8, 2, 20, 10)
    assert out["a"].shape == (8,)
    assert out["s_next"].shape == (8, 2, 20, 10)


def test_sample_normalized_cf_picks_executed_branch_as_s_next():
    """For a CF row with a_executed=k, s_next must equal next_states[k] —
    not next_states[0] or some other branch."""
    buf = _build_cf_buffer()
    rng = np.random.default_rng(0)
    out = sample_normalized(buf, n=16, rng=rng)
    for i in range(16):
        a = int(out["a"][i])
        # The executed branch's value-marker for row j is j*10 + a (encoded above).
        np.testing.assert_array_equal(
            out["s_next"][i],
            np.take(buf.next_states, out["indices"][i], axis=0)[a],
        )


def test_sample_rollout_normalized_for_legacy_buffer():
    buf = _build_legacy_buffer()
    rng = np.random.default_rng(0)
    out = sample_rollout_normalized(buf, n=4, k=3, rng=rng)
    assert out["s0"].shape == (4, 2, 20, 10)
    assert out["actions"].shape == (4, 3)
    assert out["s_next_k"].shape == (4, 3, 2, 20, 10)


def test_sample_rollout_normalized_for_cf_buffer_chains_executed_action():
    """For each step t, s_next_k[b, t] must equal next_states[start+t, a_executed[start+t]]."""
    buf = _build_cf_buffer()
    rng = np.random.default_rng(0)
    out = sample_rollout_normalized(buf, n=4, k=3, rng=rng)
    assert out["s0"].shape == (4, 2, 20, 10)
    assert out["actions"].shape == (4, 3)
    assert out["s_next_k"].shape == (4, 3, 2, 20, 10)
    starts = out["starts"]
    for b in range(4):
        for t in range(3):
            a_t = int(out["actions"][b, t])
            np.testing.assert_array_equal(
                out["s_next_k"][b, t],
                buf.next_states[starts[b] + t, a_t],
            )


def test_load_buffer_dispatches_by_schema(tmp_path):
    legacy = _build_legacy_buffer(n=10)
    cf = _build_cf_buffer(n=10)
    legacy_path = tmp_path / "legacy.npz"
    cf_path = tmp_path / "cf.npz"
    legacy.save(str(legacy_path))
    cf.save(str(cf_path))
    assert isinstance(load_buffer(str(legacy_path)), ReplayBuffer)
    assert isinstance(load_buffer(str(cf_path)), CounterfactualReplayBuffer)
