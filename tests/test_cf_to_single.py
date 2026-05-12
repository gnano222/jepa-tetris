"""Tests for the CF buffer -> single-action buffer conversion utility.

The compute-parity comparison wants to train a single-action JEPA on the
exact same transitions as a CF JEPA. Easiest way: derive a `ReplayBuffer`
file from a `CounterfactualReplayBuffer` file by extracting the on-policy
executed branch (`next_states[a_executed]`) per row.
"""
from __future__ import annotations

import numpy as np

from jepa_tetris.data.replay_buffer import (
    NUM_ACTIONS,
    CounterfactualReplayBuffer,
    ReplayBuffer,
)
from jepa_tetris.data.buffer_adapters import cf_to_single_action_buffer


def _info(lc=0, h=0, ah=0, done=False, **piece):
    return {"lines_cleared": lc, "holes": h, "aggregate_height": ah, "done": done, **piece}


def _state(value: int) -> np.ndarray:
    s = np.zeros((2, 20, 10), dtype=np.float32)
    s[0, 0, 0] = float(value)
    return s


def _build_cf_buffer(n: int = 20) -> CounterfactualReplayBuffer:
    buf = CounterfactualReplayBuffer(capacity=n)
    for i in range(n):
        next_states = np.stack([_state(i * 10 + a) for a in range(NUM_ACTIONS)])
        buf.add(
            _state(i),
            next_states=next_states,
            a_executed=i % NUM_ACTIONS,
            info=_info(lc=float(i % 3), h=float(i), ah=float(i + 1), done=(i == n - 1),
                       piece_id=i % 7, rotation=i % 4, piece_row=i % 20, piece_col=i % 10),
        )
    return buf


def test_round_trip_size_and_shapes():
    cf = _build_cf_buffer(n=20)
    rb = cf_to_single_action_buffer(cf)
    assert rb.size == cf.size
    assert rb.s.shape[1:] == (2, 20, 10)
    assert rb.s_next.shape[1:] == (2, 20, 10)


def test_s_next_uses_executed_branch():
    cf = _build_cf_buffer(n=12)
    rb = cf_to_single_action_buffer(cf)
    for i in range(rb.size):
        a = int(cf.a_executed[i])
        np.testing.assert_array_equal(rb.s[i], cf.s[i])
        np.testing.assert_array_equal(rb.s_next[i], cf.next_states[i, a])
        assert rb.a[i] == a


def test_info_fields_preserved():
    cf = _build_cf_buffer(n=8)
    rb = cf_to_single_action_buffer(cf)
    np.testing.assert_array_equal(rb.lines_cleared[: rb.size], cf.lines_cleared[: cf.size])
    np.testing.assert_array_equal(rb.holes[: rb.size], cf.holes[: cf.size])
    np.testing.assert_array_equal(rb.aggregate_height[: rb.size], cf.aggregate_height[: cf.size])
    np.testing.assert_array_equal(rb.done[: rb.size], cf.done[: cf.size])
    np.testing.assert_array_equal(rb.piece_id[: rb.size], cf.piece_id[: cf.size])
    np.testing.assert_array_equal(rb.rotation[: rb.size], cf.rotation[: cf.size])
    np.testing.assert_array_equal(rb.piece_row[: rb.size], cf.piece_row[: cf.size])
    np.testing.assert_array_equal(rb.piece_col[: rb.size], cf.piece_col[: cf.size])


def test_save_and_reload_via_replay_buffer(tmp_path):
    cf = _build_cf_buffer(n=15)
    rb = cf_to_single_action_buffer(cf)
    path = tmp_path / "single.npz"
    rb.save(str(path))
    rb2 = ReplayBuffer.load(str(path))
    assert rb2.size == cf.size
    np.testing.assert_array_equal(rb2.s[: rb2.size], rb.s[: rb.size])
    np.testing.assert_array_equal(rb2.s_next[: rb2.size], rb.s_next[: rb.size])
    np.testing.assert_array_equal(rb2.a[: rb2.size], rb.a[: rb.size])
