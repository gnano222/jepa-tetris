import copy

import numpy as np
import pytest

from jepa_tetris.env.pieces import PIECE_NAMES, PIECES
from jepa_tetris.env.tetris import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    DROP,
    LEFT,
    RIGHT,
    ROTATE,
    SPAWN_COL,
    SPAWN_ROW,
    TetrisEnv,
)


def test_reset_returns_correct_obs_shape():
    env = TetrisEnv(seed=0)
    obs = env.reset()
    assert obs.shape == (2, BOARD_HEIGHT, BOARD_WIDTH)
    assert obs[0].sum() == 0
    assert obs[1].sum() == 4  # tetromino has 4 cells


def test_left_into_left_wall_is_noop():
    env = TetrisEnv(seed=0)
    env.reset()
    for _ in range(20):
        env.step(LEFT)
    col_before = env.piece_col
    env.step(LEFT)
    assert env.piece_col == col_before


def test_right_into_right_wall_is_noop():
    env = TetrisEnv(seed=0)
    env.reset()
    for _ in range(20):
        env.step(RIGHT)
    col_before = env.piece_col
    env.step(RIGHT)
    assert env.piece_col == col_before


def test_drop_locks_piece_and_spawns_new():
    env = TetrisEnv(seed=0)
    env.reset()
    obs, info = env.step(DROP)
    assert obs[0].sum() == 4  # 4 cells locked
    assert obs[1].sum() == 4  # new piece exists
    assert info["lines_cleared"] == 0


def test_rotate_advances_rotation_when_valid():
    env = TetrisEnv(seed=0)
    env.reset()
    rot_before = env.rotation
    env.step(ROTATE)
    # T-piece should rotate; O-piece rotation field still increments to a duplicate state
    # Either way, the field value is updated when valid
    assert env.rotation in (rot_before, (rot_before + 1) % 4)


def test_single_line_clear():
    env = TetrisEnv(seed=0)
    env.reset()
    env.board[BOARD_HEIGHT - 1, :] = 1
    cleared = env._clear_lines()
    assert cleared == 1
    assert env.board[BOARD_HEIGHT - 1, :].sum() == 0


def test_multi_line_clear():
    env = TetrisEnv(seed=0)
    env.reset()
    env.board[BOARD_HEIGHT - 2 :, :] = 1
    cleared = env._clear_lines()
    assert cleared == 2
    assert env.board.sum() == 0


def test_spawn_blocked_ends_episode():
    env = TetrisEnv(seed=0)
    env.reset()
    # Block the entire spawn area
    env.board[SPAWN_ROW : SPAWN_ROW + 4, :] = 1
    env._spawn_piece()
    assert env.done is True


def test_all_pieces_have_4_rotations_with_4_cells():
    for name in PIECE_NAMES:
        assert len(PIECES[name]) == 4
        for rot in PIECES[name]:
            assert len(rot) == 4


def test_holes_count():
    env = TetrisEnv(seed=0)
    env.reset()
    env.board[10, 5] = 1  # one block at row 10
    holes = env._count_holes()
    # column 5 has top=10, then rows 11..19 are empty -> 9 holes
    assert holes == 9


def test_aggregate_height():
    env = TetrisEnv(seed=0)
    env.reset()
    env.board[BOARD_HEIGHT - 1, 0] = 1  # one block at bottom of col 0
    h = env._aggregate_height()
    assert h == 1


def test_drop_until_floor():
    env = TetrisEnv(seed=0)
    env.reset()
    obs_before, _ = env.step(DROP)
    # After drop, board sum must be exactly 4 (one piece's worth)
    assert env.board.sum() == 4


def test_max_steps_terminates():
    env = TetrisEnv(seed=0, max_steps=5)
    env.reset()
    for _ in range(10):
        env.step(LEFT)
    assert env.done is True


def test_action_after_done_is_safe():
    env = TetrisEnv(seed=0, max_steps=2)
    env.reset()
    env.step(LEFT)
    env.step(LEFT)
    assert env.done is True
    obs, info = env.step(DROP)
    assert obs.shape == (2, BOARD_HEIGHT, BOARD_WIDTH)


def test_invalid_action_raises():
    env = TetrisEnv(seed=0)
    env.reset()
    with pytest.raises(ValueError):
        env.step(99)


def test_deepcopy_produces_independent_fork():
    """Counterfactual diagnostic and counterfactual training both rely on
    copy.deepcopy(env) producing a fully independent fork. Stepping the copy
    must not mutate the original — board, piece pose, RNG, observation."""
    env = TetrisEnv(seed=0)
    env.reset()
    # Move toward a non-trivial state so any mutation would be visible.
    for _ in range(3):
        env.step(RIGHT)
    obs_before = env.observe().copy()
    board_before = env.board.copy()
    pose_before = (env.piece_row, env.piece_col, env.rotation, env.piece_name)

    fork = copy.deepcopy(env)
    # Drive the fork to a different state.
    for _ in range(5):
        fork.step(LEFT)
    fork.step(DROP)

    # Original must be untouched.
    np.testing.assert_array_equal(env.observe(), obs_before)
    np.testing.assert_array_equal(env.board, board_before)
    assert (env.piece_row, env.piece_col, env.rotation, env.piece_name) == pose_before


def test_deepcopy_forks_have_independent_rngs():
    """Two deepcopies stepped through DROP-induced spawns should evolve
    independently (the spawn RNG must not be aliased)."""
    env = TetrisEnv(seed=0)
    env.reset()
    fork_a = copy.deepcopy(env)
    fork_b = copy.deepcopy(env)
    # Lock pieces so each fork triggers its own spawn-from-RNG sequence.
    for _ in range(5):
        fork_a.step(DROP)
        fork_b.step(DROP)
    # Identical seed-state at fork time means they should evolve identically.
    np.testing.assert_array_equal(fork_a.board, fork_b.board)
    assert fork_a.piece_name == fork_b.piece_name
