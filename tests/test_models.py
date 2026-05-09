import torch

from jepa_tetris.models.action_encoder import ActionEncoder
from jepa_tetris.models.encoder import StateEncoder
from jepa_tetris.models.predictor import Predictor
from jepa_tetris.models.probe import Probe


def test_encoder_output_shape():
    enc = StateEncoder(latent_dim=64)
    x = torch.randn(4, 2, 20, 10)
    z = enc(x)
    assert z.shape == (4, 64)


def test_encoder_latent_dim_configurable():
    enc = StateEncoder(latent_dim=32)
    x = torch.randn(2, 2, 20, 10)
    z = enc(x)
    assert z.shape == (2, 32)


def test_action_encoder_shape():
    ae = ActionEncoder(num_actions=4, embed_dim=16)
    a = torch.tensor([0, 1, 2, 3])
    e = ae(a)
    assert e.shape == (4, 16)


def test_predictor_shape():
    pred = Predictor(latent_dim=64, action_emb_dim=16)
    z = torch.randn(8, 64)
    a = torch.randn(8, 16)
    out = pred(z, a)
    assert out.shape == (8, 64)


def test_probe_shape():
    probe = Probe(latent_dim=64, num_targets=3)
    z = torch.randn(4, 64)
    out = probe(z)
    assert out.shape == (4, 3)


def test_encoder_gradients_flow():
    enc = StateEncoder(latent_dim=64)
    x = torch.randn(2, 2, 20, 10)
    z = enc(x)
    z.sum().backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
    assert has_grad


def test_full_jepa_pipeline_runs():
    enc = StateEncoder(latent_dim=64)
    ae = ActionEncoder()
    pred = Predictor(latent_dim=64)
    s = torch.randn(4, 2, 20, 10)
    s_next = torch.randn(4, 2, 20, 10)
    a = torch.tensor([0, 1, 2, 3])
    z = enc(s)
    a_emb = ae(a)
    z_pred = pred(z, a_emb)
    z_next = enc(s_next)
    loss = ((z_pred - z_next.detach()) ** 2).mean()
    loss.backward()
    assert torch.isfinite(loss)
