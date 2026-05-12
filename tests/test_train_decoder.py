"""Smoke test for the decoder training inner loop with multi-distribution mixing.

Builds a tiny replay buffer + tiny JEPA stack from scratch (no checkpoint on
disk), then runs ~50 training steps with `predictor_mix=0.5`. Asserts that
both the encoder-distribution and predictor-distribution BCE terms decrease
between the first and last logged step.

This is a sanity check on the training procedure itself; full quality is
verified manually via scripts/visualize_predictions.py after a real run.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from jepa_tetris.data.replay_buffer import ReplayBuffer
from jepa_tetris.env.tetris import NUM_ACTIONS, TetrisEnv
from jepa_tetris.models.action_encoder import ActionEncoder
from jepa_tetris.models.decoder import StateDecoder
from jepa_tetris.models.encoder import StateEncoder
from jepa_tetris.models.predictor import Predictor


def _build_tiny_buffer(n_episodes: int = 4, seed: int = 0) -> ReplayBuffer:
    env = TetrisEnv(seed=seed, max_steps=40)
    rng = np.random.default_rng(seed)
    buf = ReplayBuffer(capacity=512)
    for _ in range(n_episodes):
        s = env.reset()
        while not env.done and buf.size < buf.capacity:
            a = int(rng.integers(0, NUM_ACTIONS))
            s_next, info = env.step(a)
            buf.add(s, a, s_next, info)
            s = s_next
    return buf


def test_train_decoder_loop_reduces_both_losses():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = torch.device("cpu")

    buf = _build_tiny_buffer(n_episodes=4, seed=0)
    assert buf.size >= 32, f"tiny buffer too small to train: {buf.size}"

    patch_dim = 32
    encoder = StateEncoder(patch_dim=patch_dim).to(device).eval()
    action_encoder = ActionEncoder(embed_dim=patch_dim).to(device).eval()
    predictor = Predictor(patch_dim=patch_dim, num_patches=encoder.num_patches).to(device).eval()
    decoder = StateDecoder(patch_dim=patch_dim).to(device)

    for m in (encoder, action_encoder, predictor):
        for p in m.parameters():
            p.requires_grad_(False)

    optimizer = AdamW(decoder.parameters(), lr=1e-3)
    n_total = 16
    n_pred = 8
    n_enc = n_total - n_pred
    rollout_k = 3

    enc_first = enc_last = pred_first = pred_last = None

    for step in range(50):
        # Encoder branch.
        batch = buf.sample(n_enc, rng=rng)
        s = torch.from_numpy(batch["s"]).to(device)
        with torch.no_grad():
            z_enc = encoder(s)
        bce_enc = F.binary_cross_entropy_with_logits(decoder(z_enc), s)

        # Predictor branch at random depth in [1, rollout_k].
        roll = buf.sample_rollout(n_pred, k=rollout_k, rng=rng)
        depth = int(rng.integers(1, rollout_k + 1))
        s0 = torch.from_numpy(roll["s0"]).to(device)
        actions = torch.from_numpy(roll["actions"][:, :depth]).to(device)
        target = torch.from_numpy(roll["s_next_k"][:, depth - 1]).to(device)
        with torch.no_grad():
            z = encoder(s0)
            for t in range(depth):
                z = predictor(z, action_encoder(actions[:, t]))
        bce_pred = F.binary_cross_entropy_with_logits(decoder(z), target)

        loss = (bce_enc + bce_pred) / 2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 0:
            enc_first = bce_enc.item()
            pred_first = bce_pred.item()
        if step == 49:
            enc_last = bce_enc.item()
            pred_last = bce_pred.item()

    assert enc_last < enc_first, f"encoder BCE didn't drop: {enc_first:.4f} -> {enc_last:.4f}"
    assert pred_last < pred_first, f"predictor BCE didn't drop: {pred_first:.4f} -> {pred_last:.4f}"
