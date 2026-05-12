import numpy as np

from jepa_tetris.env.tetris import NUM_ACTIONS, TetrisEnv
from jepa_tetris.models.action_encoder import ActionEncoder
from jepa_tetris.models.encoder import StateEncoder
from jepa_tetris.models.predictor import Predictor
from jepa_tetris.models.probe import Probe
from jepa_tetris.plan import BFSPlanner, PlacementPlanner, RealDynamicsPlanner


PATCH_DIM = 64


def _make_planner(depth=2):
    encoder = StateEncoder(patch_dim=PATCH_DIM)
    action_encoder = ActionEncoder(embed_dim=PATCH_DIM)
    predictor = Predictor(patch_dim=PATCH_DIM, num_patches=encoder.num_patches)
    probe = Probe(patch_dim=PATCH_DIM, num_targets=3)
    return BFSPlanner(encoder, action_encoder, predictor, probe, depth=depth, device="cpu")


def test_planner_returns_valid_action():
    planner = _make_planner(depth=2)
    env = TetrisEnv(seed=0)
    obs = env.reset()
    a = planner.select_action(obs)
    assert 0 <= a < NUM_ACTIONS


def test_planner_handles_depth_4():
    planner = _make_planner(depth=4)
    env = TetrisEnv(seed=0)
    obs = env.reset()
    a = planner.select_action(obs)
    assert 0 <= a < NUM_ACTIONS
    assert planner.sequences.shape == (175, 4)


def test_planner_runs_episode_to_completion():
    planner = _make_planner(depth=2)
    env = TetrisEnv(seed=0, max_steps=20)
    obs = env.reset()
    while not env.done:
        a = planner.select_action(obs)
        obs, _ = env.step(a)
    assert env.done is True


def test_real_dynamics_planner_runs():
    encoder = StateEncoder(patch_dim=PATCH_DIM)
    probe = Probe(patch_dim=PATCH_DIM, num_targets=3)
    env = TetrisEnv(seed=0, max_steps=10)
    env.reset()
    planner = RealDynamicsPlanner(encoder, probe, env, depth=2, device="cpu")
    obs = env.observe()
    a = planner.select_action(obs)
    assert 0 <= a < NUM_ACTIONS
    assert env.steps == 0


def test_placement_planner_runs():
    encoder = StateEncoder(patch_dim=PATCH_DIM)
    probe = Probe(patch_dim=PATCH_DIM, num_targets=3)
    env = TetrisEnv(seed=0, max_steps=20)
    env.reset()
    planner = PlacementPlanner(encoder, probe, env, device="cpu")
    obs = env.observe()
    plan = planner.select_plan(obs)
    assert len(plan) > 0
    from jepa_tetris.env.tetris import DROP
    assert plan[-1] == DROP


def test_placement_planner_does_not_mutate_env():
    encoder = StateEncoder(patch_dim=PATCH_DIM)
    probe = Probe(patch_dim=PATCH_DIM, num_targets=3)
    env = TetrisEnv(seed=0, max_steps=20)
    env.reset()
    board_before = env.board.copy()
    pose_before = (env.piece_row, env.piece_col, env.rotation)
    steps_before = env.steps
    planner = PlacementPlanner(encoder, probe, env, device="cpu")
    planner.select_plan(env.observe())
    np.testing.assert_array_equal(env.board, board_before)
    assert (env.piece_row, env.piece_col, env.rotation) == pose_before
    assert env.steps == steps_before
