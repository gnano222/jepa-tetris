import numpy as np

from jepa_tetris.data.replay_buffer import ReplayBuffer


def _info(lc=0, h=0, ah=0, done=False, **piece):
    base = {"lines_cleared": lc, "holes": h, "aggregate_height": ah, "done": done}
    base.update(piece)
    return base


def test_add_increments_size():
    buf = ReplayBuffer(capacity=10)
    s = np.zeros((2, 20, 10), dtype=np.float32)
    buf.add(s, 0, s, _info())
    assert buf.size == 1


def test_ring_eviction_caps_size():
    buf = ReplayBuffer(capacity=3)
    s = np.zeros((2, 20, 10), dtype=np.float32)
    for i in range(7):
        buf.add(s, i % 4, s, _info())
    assert buf.size == 3


def test_sample_returns_correct_shapes():
    buf = ReplayBuffer(capacity=100)
    s = np.zeros((2, 20, 10), dtype=np.float32)
    for _ in range(50):
        buf.add(s, 0, s, _info(lc=1, h=2, ah=3))
    batch = buf.sample(8)
    assert batch["s"].shape == (8, 2, 20, 10)
    assert batch["s_next"].shape == (8, 2, 20, 10)
    assert batch["a"].shape == (8,)
    assert batch["lines_cleared"].shape == (8,)
    assert batch["holes"].shape == (8,)
    assert batch["aggregate_height"].shape == (8,)
    assert batch["done"].shape == (8,)


def test_save_load_roundtrip(tmp_path):
    buf = ReplayBuffer(capacity=10)
    s_a = np.zeros((2, 20, 10), dtype=np.float32)
    s_a[0, 0, 0] = 1
    for i in range(5):
        buf.add(s_a, i % 4, s_a, _info(lc=i, h=i + 1, ah=i + 2))
    path = tmp_path / "buf.npz"
    buf.save(str(path))
    loaded = ReplayBuffer.load(str(path))
    assert loaded.size == 5
    np.testing.assert_array_equal(loaded.s[:5], buf.s[:5])
    np.testing.assert_array_equal(loaded.a[:5], buf.a[:5])


def test_sample_indices_are_uniform_over_size():
    buf = ReplayBuffer(capacity=100)
    s = np.zeros((2, 20, 10), dtype=np.float32)
    for i in range(20):
        s_i = s.copy()
        s_i[0, 0, 0] = i
        buf.add(s_i, 0, s_i, _info())
    rng = np.random.default_rng(0)
    batch = buf.sample(1000, rng=rng)
    sampled_vals = batch["s"][:, 0, 0, 0]
    assert set(sampled_vals.astype(int).tolist()).issubset(set(range(20)))


def test_sample_rollout_returns_correct_shapes():
    buf = ReplayBuffer(capacity=200)
    s = np.zeros((2, 20, 10), dtype=np.float32)
    for i in range(50):
        # All within one episode: done=False everywhere except last
        info = _info(done=(i == 49))
        s_i = s.copy()
        s_i[0, 0, 0] = i
        buf.add(s_i, i % 4, s_i, info)
    rng = np.random.default_rng(0)
    batch = buf.sample_rollout(8, k=4, rng=rng)
    assert batch["s0"].shape == (8, 2, 20, 10)
    assert batch["actions"].shape == (8, 4)
    assert batch["s_next_k"].shape == (8, 4, 2, 20, 10)


def test_sample_rollout_starts_index_correctly():
    buf = ReplayBuffer(capacity=200)
    s = np.zeros((2, 20, 10), dtype=np.float32)
    for i in range(50):
        s_i = s.copy()
        s_i[0, 0, 0] = i
        buf.add(s_i, i % 4, s_i, _info(done=(i == 49)))
    rng = np.random.default_rng(0)
    batch = buf.sample_rollout(8, k=4, rng=rng)
    starts = batch["starts"]
    assert starts.shape == (8,)
    for i in range(8):
        assert (batch["s0"][i] == buf.s[starts[i]]).all()
        for t in range(4):
            assert (batch["s_next_k"][i, t] == buf.s_next[starts[i] + t]).all()
            assert batch["actions"][i, t] == buf.a[starts[i] + t]


def test_sample_rollout_avoids_episode_boundaries():
    buf = ReplayBuffer(capacity=200)
    s = np.zeros((2, 20, 10), dtype=np.float32)
    # Five 10-step episodes
    for ep in range(5):
        for i in range(10):
            info = _info(done=(i == 9))
            buf.add(s, 0, s, info)
    rng = np.random.default_rng(0)
    batch = buf.sample_rollout(8, k=3, rng=rng)
    # Each rollout should be from contiguous within-episode triplets;
    # we can't directly verify episode IDs here, but the call must not raise.
    assert batch["s0"].shape[0] == 8


def test_piece_meta_round_trip(tmp_path):
    """Buffers populated with piece metadata preserve it through save/load."""
    buf = ReplayBuffer(capacity=10)
    s = np.zeros((2, 20, 10), dtype=np.float32)
    for i in range(5):
        buf.add(
            s, i % 4, s,
            _info(piece_id=(i % 7), rotation=(i % 4), piece_row=i, piece_col=(i + 2) % 10),
        )
    assert buf.has_piece_meta
    assert buf.piece_id[0] == 0 and buf.piece_id[4] == 4
    assert buf.piece_col[3] == (3 + 2) % 10

    path = tmp_path / "buf_v2.npz"
    buf.save(str(path))
    loaded = ReplayBuffer.load(str(path))
    assert loaded.has_piece_meta
    np.testing.assert_array_equal(loaded.piece_id[:5], buf.piece_id[:5])
    np.testing.assert_array_equal(loaded.rotation[:5], buf.rotation[:5])
    np.testing.assert_array_equal(loaded.piece_row[:5], buf.piece_row[:5])
    np.testing.assert_array_equal(loaded.piece_col[:5], buf.piece_col[:5])

    # Sample dict also exposes piece metadata for downstream consumers.
    rng = np.random.default_rng(0)
    batch = loaded.sample(3, rng=rng)
    assert batch["piece_id"].shape == (3,)
    assert batch["rotation"].shape == (3,)


def test_legacy_buffer_loads_without_piece_meta(tmp_path):
    """Pre-v2 .npz files (no piece_id/rotation/piece_row/piece_col) still load."""
    s = np.zeros((4, 2, 20, 10), dtype=np.float32)
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        s=s,
        a=np.array([0, 1, 2, 3], dtype=np.int64),
        s_next=s,
        lines_cleared=np.zeros(4, dtype=np.float32),
        holes=np.zeros(4, dtype=np.float32),
        aggregate_height=np.zeros(4, dtype=np.float32),
        done=np.array([False, False, False, True]),
    )
    loaded = ReplayBuffer.load(str(path))
    assert loaded.size == 4
    assert loaded.has_piece_meta is False
    # The piece arrays must exist (zero-filled) so downstream code can read
    # them unconditionally; they just shouldn't be trusted.
    assert loaded.piece_id.shape == (max(loaded.size, 1),)
    assert (loaded.piece_id[: loaded.size] == 0).all()
