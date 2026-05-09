import numpy as np

from jepa_tetris.data.exploration import MixedExplorationPolicy, best_placement
from jepa_tetris.env.tetris import BOARD_HEIGHT, BOARD_WIDTH, NUM_ACTIONS, TetrisEnv


def test_best_placement_returns_valid_pose():
    env = TetrisEnv(seed=0)
    env.reset()
    col, rot = best_placement(env)
    assert 0 <= col < BOARD_WIDTH
    assert 0 <= rot < 4
    # Restoring state: env should still be valid
    assert env._is_valid(env.piece_row, env.piece_col, env.rotation)


def test_best_placement_does_not_mutate_board():
    env = TetrisEnv(seed=0)
    env.reset()
    board_before = env.board.copy()
    pose_before = (env.piece_row, env.piece_col, env.rotation)
    best_placement(env)
    np.testing.assert_array_equal(env.board, board_before)
    assert (env.piece_row, env.piece_col, env.rotation) == pose_before


def test_best_placement_prefers_line_clear():
    env = TetrisEnv(seed=42, max_steps=10)
    env.reset()
    # Set up a board with bottom row missing only column 5
    env.board[BOARD_HEIGHT - 1, :] = 1
    env.board[BOARD_HEIGHT - 1, 5] = 0
    # Force the current piece to be I (vertical clears 1 row at col 5 for sure)
    env.piece_name = "I"
    env.rotation = 0
    env.piece_row = 0
    env.piece_col = 0
    col, rot = best_placement(env)
    # Best placement should drop the I in column 5 to clear the line
    # (with vertical I = rotation 1, placed so its column is 5)
    assert col == 3 or col == 4 or col == 5  # I-piece bounding box can put column 2 of 4-wide at col 5
    # Stronger check: line clears with this placement
    saved = env.board.copy()
    r = env.piece_row
    while env._is_valid(r + 1, col, rot):
        r += 1
    for cr, cc in env._piece_cells(r, col, rot):
        env.board[cr, cc] = 1
    lines = int(env.board.all(axis=1).sum())
    env.board[:] = saved
    assert lines >= 1


def test_mixed_policy_returns_valid_action():
    env = TetrisEnv(seed=0)
    env.reset()
    rng = np.random.default_rng(0)
    policy = MixedExplorationPolicy(env, rng, epsilon=0.3)
    a = policy()
    assert 0 <= a < NUM_ACTIONS


def test_mixed_policy_with_zero_epsilon_is_deterministic():
    env = TetrisEnv(seed=0)
    env.reset()
    rng = np.random.default_rng(0)
    policy = MixedExplorationPolicy(env, rng, epsilon=0.0)
    # With epsilon=0, two calls to policy() should give the same action
    # (since target placement is fixed and current pose is fixed)
    a1 = policy()
    a2 = policy()
    assert a1 == a2
