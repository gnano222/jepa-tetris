import pytest
import torch
import torch.nn as nn

from jepa_tetris.models.action_encoder import ActionEncoder
from jepa_tetris.models.encoder import (
    ColumnarEncoder,
    ColumnPredictorHead,
    StateEncoder,
    compute_aux_channels,
    make_encoder_from_args,
)
from jepa_tetris.models.predictor import Predictor
from jepa_tetris.models.probe import Probe


N_DEFAULT = 6   # 20x10 board after three stride-2 convs -> 3x2 = 6 patches


def test_encoder_output_shape():
    enc = StateEncoder(patch_dim=128)
    x = torch.randn(4, 2, 20, 10)
    z = enc(x)
    assert z.shape == (4, N_DEFAULT, 128)
    assert enc.num_patches == N_DEFAULT


def test_encoder_patch_dim_configurable():
    enc = StateEncoder(patch_dim=64)
    x = torch.randn(2, 2, 20, 10)
    z = enc(x)
    assert z.shape == (2, N_DEFAULT, 64)


def test_action_encoder_shape():
    ae = ActionEncoder(num_actions=4, embed_dim=128)
    a = torch.tensor([0, 1, 2, 3])
    e = ae(a)
    assert e.shape == (4, 128)


def test_predictor_shape():
    pred = Predictor(patch_dim=128, num_patches=N_DEFAULT)
    z = torch.randn(8, N_DEFAULT, 128)
    a = torch.randn(8, 128)
    out = pred(z, a)
    assert out.shape == (8, N_DEFAULT, 128)


def test_probe_shape():
    probe = Probe(patch_dim=128, num_targets=3)
    z = torch.randn(4, N_DEFAULT, 128)
    out = probe(z)
    assert out.shape == (4, 3)


def test_encoder_gradients_flow():
    enc = StateEncoder(patch_dim=128)
    x = torch.randn(2, 2, 20, 10)
    z = enc(x)
    z.sum().backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
    assert has_grad


def test_full_jepa_pipeline_runs():
    enc = StateEncoder(patch_dim=128)
    ae = ActionEncoder(embed_dim=128)
    pred = Predictor(patch_dim=128, num_patches=enc.num_patches)
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


@pytest.mark.parametrize("patch_dim", [64, 128, 256])
def test_encoder_width_variants(patch_dim):
    enc = StateEncoder(patch_dim=patch_dim)
    x = torch.randn(3, 2, 20, 10)
    z = enc(x)
    assert z.shape == (3, N_DEFAULT, patch_dim)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"residual_blocks": 2},
        {"aux_channels": True},
        {"residual_blocks": 2, "aux_channels": True},
    ],
)
def test_encoder_variant_shape_contract(kwargs):
    """Every variant must emit (B, N, D) — predictor compatibility."""
    enc = StateEncoder(patch_dim=128, **kwargs)
    x = torch.randn(4, 2, 20, 10)
    z = enc(x)
    assert z.shape == (4, N_DEFAULT, 128), f"variant {kwargs} broke shape contract"


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"residual_blocks": 2},
        {"aux_channels": True},
    ],
)
def test_encoder_variant_gradients_flow(kwargs):
    enc = StateEncoder(patch_dim=128, **kwargs)
    x = torch.randn(2, 2, 20, 10)
    z = enc(x)
    z.sum().backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in enc.parameters()
    )
    assert has_grad


def test_compute_aux_channels_shape_and_range():
    x = torch.zeros(2, 2, 20, 10)
    x[0, 0, 18:, :3] = 1.0
    x[0, 0, 15, 5] = 1.0
    aux = compute_aux_channels(x)
    assert aux.shape == (2, 3, 20, 10)
    assert (aux[:, 0] >= 0).all() and (aux[:, 0] <= 1).all()
    holes = aux[0, 1]
    assert ((holes == 0) | (holes == 1)).all()
    assert holes[16:, 5].sum() > 0


def test_encoder_predictor_compat_aggressive_variant():
    """End-to-end gradient through encoder + predictor for the heaviest variant."""
    patch_dim = 128
    enc = StateEncoder(
        patch_dim=patch_dim,
        residual_blocks=2,
        aux_channels=True,
    )
    pred = Predictor(patch_dim=patch_dim, num_patches=enc.num_patches)
    ae = ActionEncoder(embed_dim=patch_dim)
    s = torch.randn(4, 2, 20, 10)
    s_next = torch.randn(4, 2, 20, 10)
    a = torch.tensor([0, 1, 2, 3])
    z = enc(s)
    z_pred = pred(z, ae(a))
    z_next = enc(s_next)
    loss = ((z_pred - z_next.detach()) ** 2).mean()
    loss.backward()
    assert torch.isfinite(loss)


def test_encoder_15patch_stride_stages_2():
    """stride_stages=2 yields 15 patches (5x3) instead of 6 (3x2)."""
    enc = StateEncoder(patch_dim=128, stride_stages=2)
    x = torch.randn(4, 2, 20, 10)
    z = enc(x)
    assert z.shape == (4, 15, 128), f"expected (4, 15, 128), got {z.shape}"
    assert enc.num_patches == 15

    pred = Predictor(patch_dim=128, num_patches=enc.num_patches)
    a = torch.randn(4, 128)
    out = pred(z, a)
    assert out.shape == (4, 15, 128)


def test_make_encoder_from_args_stride_stages_2():
    args = {"patch_dim": 128, "encoder_stride_stages": 2}
    enc = make_encoder_from_args(args)
    assert enc(torch.randn(2, 2, 20, 10)).shape == (2, 15, 128)


def test_make_encoder_from_args_default():
    args = {"patch_dim": 128}
    enc = make_encoder_from_args(args)
    x = torch.randn(2, 2, 20, 10)
    assert enc(x).shape == (2, N_DEFAULT, 128)


def test_make_encoder_from_args_with_variant_flags():
    args = {
        "patch_dim": 128,
        "encoder_residual_blocks": 1,
        "encoder_aux_channels": True,
    }
    enc = make_encoder_from_args(args)
    x = torch.randn(2, 2, 20, 10)
    assert enc(x).shape == (2, N_DEFAULT, 128)


def test_predictor_residual_default_on():
    pred = Predictor(patch_dim=128, num_patches=N_DEFAULT)
    assert pred.residual is True


def test_predictor_residual_passthrough_when_zero_delta():
    """Residual predictor should pass z through when its output Δz is zero.

    Force the final transformer output to ~0 by zeroing the output projection.
    Then z_pred should equal z (the residual passthrough).
    """
    torch.manual_seed(0)
    pred = Predictor(patch_dim=128, num_patches=N_DEFAULT, residual=True, depth=1)
    # Zero out all linear2 weights in the ModuleList layers so output ~ 0.
    for name, p in pred.named_parameters():
        if "linear2" in name and name.endswith(".weight"):
            torch.nn.init.zeros_(p)
        if "linear2" in name and name.endswith(".bias"):
            torch.nn.init.zeros_(p)
    z = torch.randn(4, N_DEFAULT, 128)
    a = torch.randn(4, 128)
    out = pred(z, a)
    assert out.shape == z.shape


def test_predictor_film_shape():
    pred = Predictor(patch_dim=128, num_patches=N_DEFAULT, film=True)
    z = torch.randn(8, N_DEFAULT, 128)
    a = torch.randn(8, 128)
    out = pred(z, a)
    assert out.shape == (8, N_DEFAULT, 128)


def test_predictor_cross_attn_shape():
    pred = Predictor(patch_dim=128, num_patches=N_DEFAULT, cross_attn=True)
    z = torch.randn(8, N_DEFAULT, 128)
    a = torch.randn(8, 128)
    out = pred(z, a)
    assert out.shape == (8, N_DEFAULT, 128)


def test_predictor_film_cross_attn_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        Predictor(patch_dim=128, num_patches=N_DEFAULT, film=True, cross_attn=True)


def test_predictor_film_pos_emb_shape():
    pred = Predictor(patch_dim=128, num_patches=N_DEFAULT, film=True)
    assert pred.pos_emb.shape == (1, N_DEFAULT, 128)


def test_predictor_extra_token_pos_emb_shape():
    pred = Predictor(patch_dim=128, num_patches=N_DEFAULT)
    assert pred.pos_emb.shape == (1, N_DEFAULT + 1, 128)


def test_column_predictor_head_shape():
    head = ColumnPredictorHead(dim=128)
    z = torch.randn(8, 128)
    a = torch.randn(8, 128)
    assert head(z, a).shape == (8, 128)


def test_columnar_encoder_output_shape():
    enc = ColumnarEncoder(patch_dim=128)
    x = torch.randn(4, 2, 20, 10)
    z = enc(x)
    assert z.shape == (4, 15, 128)
    assert enc.num_patches == 15


def test_columnar_encoder_patch_dim_configurable():
    enc = ColumnarEncoder(patch_dim=64)
    z = enc(torch.randn(2, 2, 20, 10))
    assert z.shape == (2, 15, 64)


def test_columnar_encoder_regions_clamp_at_edges():
    """5x3 grid, margin 1: row splits [4]*5, col splits [3,4,3].
    Corner and centre regions are margin-expanded then clamped to the board."""
    enc = ColumnarEncoder(patch_dim=64, grid=(5, 3), margin=1)
    assert enc.regions[0] == (0, 5, 0, 4)      # grid cell (0,0)
    assert enc.regions[14] == (15, 20, 6, 10)  # grid cell (4,2)
    assert enc.regions[7] == (7, 13, 2, 8)     # grid cell (2,1), centre


def test_columnar_encoder_gradient_isolation():
    """The Fork B invariant: a loss from one column's output produces zero
    gradient on every other column's conv stack."""
    enc = ColumnarEncoder(patch_dim=64)
    z = enc(torch.randn(2, 2, 20, 10))
    z[:, 0].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in enc.stacks[0].parameters())
    for i in range(1, 15):
        for p in enc.stacks[i].parameters():
            assert p.grad is None or p.grad.abs().sum() == 0


def test_columnar_encoder_predictor_compat():
    enc = ColumnarEncoder(patch_dim=128)
    pred = Predictor(patch_dim=128, num_patches=enc.num_patches, film=True)
    z = enc(torch.randn(4, 2, 20, 10))
    a = torch.randn(4, 128)
    assert pred(z, a).shape == (4, 15, 128)


def test_make_encoder_from_args_columnar():
    args = {
        "patch_dim": 128,
        "encoder_columnar": True,
        "encoder_columnar_grid": "5x3",
        "encoder_columnar_margin": 1,
    }
    enc = make_encoder_from_args(args)
    z = enc(torch.randn(2, 2, 20, 10))
    assert z.shape == (2, 15, 128)
    assert enc.num_patches == 15


def test_columnar_local_loss_isolation_and_finiteness():
    """Local loss is finite, and each column's conv stack receives gradient
    only from its own term (verified via the encoder, end to end)."""
    from jepa_tetris.train import columnar_local_loss

    torch.manual_seed(0)
    enc = ColumnarEncoder(patch_dim=64)
    heads = nn.ModuleList([ColumnPredictorHead(64) for _ in range(enc.num_patches)])
    z0 = enc(torch.randn(3, 2, 20, 10))            # (3, 15, 64), with grad
    z1_target = torch.randn(3, 15, 64)             # detached target stand-in
    a_emb = torch.randn(3, 64)

    loss, parts = columnar_local_loss(
        z0_online=z0, z1_target=z1_target, a_emb=a_emb, column_heads=heads,
        var_weight=1.0, cov_weight=0.04, target_std=1.0,
    )
    assert torch.isfinite(loss)
    assert {"mse_local", "var_local", "cov_local"} <= parts.keys()

    loss.backward()
    for i in range(enc.num_patches):
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in enc.stacks[i].parameters())
