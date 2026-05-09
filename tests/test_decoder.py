import numpy as np
import torch
import torch.nn.functional as F

from jepa_tetris.env.tetris import TetrisEnv
from jepa_tetris.models.decoder import StateDecoder
from jepa_tetris.models.encoder import StateEncoder


def test_decoder_output_shape():
    dec = StateDecoder(latent_dim=64)
    z = torch.randn(4, 64)
    out = dec(z)
    assert out.shape == (4, 2, 20, 10)


def test_decoder_latent_dim_configurable():
    dec = StateDecoder(latent_dim=32)
    z = torch.randn(2, 32)
    out = dec(z)
    assert out.shape == (2, 2, 20, 10)


def test_decoder_gradients_flow():
    dec = StateDecoder(latent_dim=64)
    z = torch.randn(2, 64)
    out = dec(z)
    out.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in dec.parameters())


def test_encoder_decoder_round_trip_overfits_one_state():
    """A single state is easy to memorize: train enc+dec jointly for ~200 steps and
    expect near-perfect reconstruction. This is a sanity check that the architectures
    compose correctly, not a check on the post-hoc probe procedure."""
    torch.manual_seed(0)
    env = TetrisEnv(seed=0)
    s = env.reset()
    # Step a few times so the board has both a piece and locked cells.
    for a in (0, 0, 3, 1, 1, 3):
        s, _ = env.step(a)
    s_t = torch.from_numpy(s).unsqueeze(0).repeat(8, 1, 1, 1)

    enc = StateEncoder(latent_dim=64)
    dec = StateDecoder(latent_dim=64)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=3e-3)
    for _ in range(200):
        opt.zero_grad()
        z = enc(s_t)
        logits = dec(z)
        loss = F.binary_cross_entropy_with_logits(logits, s_t)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = (torch.sigmoid(dec(enc(s_t[:1]))) > 0.5).float()
    acc = (pred == s_t[:1]).float().mean().item()
    assert acc >= 0.95, f"round-trip binary accuracy too low: {acc:.3f}"
