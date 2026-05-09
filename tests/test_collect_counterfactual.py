"""Tests for counterfactual data collection."""
from __future__ import annotations

import numpy as np

from jepa_tetris.data.collect import collect_counterfactual
from jepa_tetris.data.replay_buffer import NUM_ACTIONS, CounterfactualReplayBuffer


def test_collect_counterfactual_returns_filled_buffer():
    buf = collect_counterfactual(
        episodes=2, capacity=200, seed=0, policy="random",
    )
    assert isinstance(buf, CounterfactualReplayBuffer)
    assert buf.size > 0
    assert buf.next_states[: buf.size].shape == (buf.size, NUM_ACTIONS, 2, 20, 10)


def test_collect_counterfactual_executed_branch_matches_next_row():
    """If action a was executed at step i (within an episode), the env's
    next-observation at step i+1 must equal next_states[i, a]. This is the
    correctness check that the deepcopy-fork branch agrees with the real
    forward stepping."""
    buf = collect_counterfactual(
        episodes=4, capacity=200, seed=1, policy="mixed", epsilon=0.5,
    )
    checks = 0
    for i in range(buf.size - 1):
        if buf.done[i]:
            continue
        a = int(buf.a_executed[i])
        np.testing.assert_array_equal(buf.s[i + 1], buf.next_states[i, a])
        checks += 1
    assert checks > 10  # ensure we actually exercised the assertion


def test_collect_counterfactual_drop_branch_changes_observation():
    buf = collect_counterfactual(
        episodes=3, capacity=200, seed=2, policy="random",
    )
    # DROP locks the piece — the DROP branch's observation must differ from s
    # for at least one row (counts where DROP is non-trivially executable).
    differing = 0
    for i in range(buf.size):
        if not np.array_equal(buf.next_states[i, 3], buf.s[i]):  # DROP = 3
            differing += 1
    assert differing > 0
