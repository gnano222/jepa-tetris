"""Streamlit interactive viewer for the JEPA Tetris decoder.

Three modes:
  * Live env       — step a Tetris env interactively, see the decoder's
                     prediction for each candidate next action.
  * Buffer scrubber— jump to any (s, a, s_next) triplet in a replay buffer,
                     compare encoder/predictor reconstructions side by side,
                     and roll out the predictor for K steps from there.
  * Latent space   — UMAP projection of N buffer latents with click-to-decode;
                     the nearest-neighbor real board is shown alongside.

Launch:
    streamlit run scripts/decoder_explorer.py
Then point the sidebar inputs at your JEPA + decoder checkpoints and a buffer.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F

from jepa_tetris.data.replay_buffer import ReplayBuffer
from jepa_tetris.env.pieces import PIECE_NAMES
from jepa_tetris.env.tetris import (
    ACTION_NAMES,
    DROP,
    LEFT,
    NUM_ACTIONS,
    RIGHT,
    ROTATE,
    TetrisEnv,
)
from jepa_tetris.utils.checkpoint import load_decoder, load_jepa
from jepa_tetris.utils.device import get_device
from jepa_tetris.viz.render import render_board

# streamlit-plotly-events is optional; without it we fall back to a numeric
# "select index" input instead of clicking points on the scatter.
try:
    from streamlit_plotly_events import plotly_events  # type: ignore
    HAVE_PLOTLY_EVENTS = True
except ImportError:
    HAVE_PLOTLY_EVENTS = False

# UMAP is optional too — without it the latent-space tab will explain how to
# install. The other two modes are unaffected.
try:
    import umap  # type: ignore
    HAVE_UMAP = True
except ImportError:
    HAVE_UMAP = False


st.set_page_config(page_title="JEPA Tetris decoder", layout="wide")


# =============================================================================
# Caching: heavy resources (models, UMAP) survive Streamlit's per-interaction
# script reruns thanks to @st.cache_resource / @st.cache_data.
# =============================================================================

@st.cache_resource(show_spinner="Loading JEPA + decoder checkpoints…")
def _load_models(jepa_path: str, decoder_path: str, device_str: str):
    device = torch.device(device_str)
    bundle = load_jepa(jepa_path, device)
    decoder = load_decoder(decoder_path, bundle.patch_dim, device)
    return bundle, decoder, device


@st.cache_resource(show_spinner="Loading replay buffer…")
def _load_buffer(buffer_path: str) -> ReplayBuffer:
    return ReplayBuffer.load(buffer_path)


@st.cache_data(show_spinner="Encoding samples + running UMAP…", max_entries=4)
def _compute_umap(buffer_path: str, jepa_path: str, decoder_path: str,
                  n_samples: int, seed: int):
    """Sample N buffer states, encode with frozen JEPA, project with UMAP.

    `decoder_path` participates in the cache key only — we don't actually need
    the decoder here, but the underlying _load_models cache is keyed on it.
    """
    if not HAVE_UMAP:
        raise RuntimeError("umap-learn is not installed")
    buf = _load_buffer(buffer_path)
    bundle, _, device = _load_models(jepa_path, decoder_path, str(get_device()))
    rng = np.random.default_rng(seed)
    n = min(n_samples, buf.size)
    idx = rng.choice(buf.size, size=n, replace=False)
    s = torch.from_numpy(buf.s[idx]).to(device)
    with torch.no_grad():
        z = bundle.encoder(s).cpu().numpy()
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed)
    coords = reducer.fit_transform(z)
    return {
        "coords": coords,
        "latents": z,
        "indices": idx.astype(np.int64),
        "piece_id": buf.piece_id[idx].astype(np.int64),
        "holes": buf.holes[idx],
        "height": buf.aggregate_height[idx],
        "lines": buf.lines_cleared[idx],
        "has_piece_meta": bool(buf.has_piece_meta),
    }


# =============================================================================
# Rendering helpers
# =============================================================================

def _decode(decoder, z: torch.Tensor) -> np.ndarray:
    """Single-latent decode → (2, 20, 10) probability grid."""
    with torch.no_grad():
        logits = decoder(z)
    return torch.sigmoid(logits).squeeze(0).cpu().numpy()


def _board_fig(grid: np.ndarray, title: str | None = None) -> plt.Figure:
    """Render a (2,20,10) board to a small matplotlib figure for st.pyplot."""
    fig, ax = plt.subplots(figsize=(2.0, 3.6))
    render_board(grid, ax, title=title)
    fig.tight_layout(pad=0.5)
    return fig


def _piece_caption(piece_id: int, rotation: int, row: int, col: int) -> str:
    name = PIECE_NAMES[int(piece_id)] if 0 <= int(piece_id) < len(PIECE_NAMES) else "?"
    return f"piece **{name}** · rot {int(rotation)} · @ (r{int(row)}, c{int(col)})"


def _binary_acc(prob: np.ndarray, truth: np.ndarray) -> float:
    return float(((prob > 0.5).astype(np.float32) == truth).mean())


# =============================================================================
# Sidebar
# =============================================================================

st.sidebar.header("Checkpoints")
default_jepa = "checkpoints/jepa_cf.pt"
default_dec = "checkpoints/decoder_cf.pt"
default_buf = "data/cf_buffer.npz"

jepa_path = st.sidebar.text_input("JEPA checkpoint", default_jepa)
decoder_path = st.sidebar.text_input("Decoder checkpoint", default_dec)
buffer_path = st.sidebar.text_input("Replay buffer", default_buf)
seed = st.sidebar.number_input("Seed", value=0, step=1)

st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "Mode",
    options=["Live env", "Buffer scrubber", "Latent space"],
    index=0,
)

# Validate paths early so a missing checkpoint shows a clear message instead
# of a stack trace from torch.load.
missing = [p for p in (jepa_path, decoder_path) if not Path(p).exists()]
if missing:
    st.error(
        "Missing file(s): " + ", ".join(missing)
        + "\n\nTrain a JEPA + decoder first (see README), or update the paths in the sidebar."
    )
    st.stop()

bundle, decoder, device = _load_models(jepa_path, decoder_path, str(get_device()))
encoder = bundle.encoder
target_encoder = bundle.target_encoder
action_encoder = bundle.action_encoder
predictor = bundle.predictor


# =============================================================================
# Mode 1: Live env
# =============================================================================

def _mode_live_env():
    st.subheader("Live env stepper")
    st.caption(
        "Step a Tetris env one action at a time. The 4-up panel below the "
        "current board shows what the decoder *predicts* the board will look "
        "like under each candidate next action."
    )

    if "live_env" not in st.session_state or st.button("Reset env", key="reset_env"):
        st.session_state.live_env = TetrisEnv(seed=int(seed), max_steps=500)
        st.session_state.live_obs = st.session_state.live_env.observe()
        st.session_state.live_history = []  # list of (action, obs)

    env: TetrisEnv = st.session_state.live_env

    # Action buttons. Streamlit triggers a rerun on click, so we mutate
    # session_state and let the next pass redraw.
    cols = st.columns(NUM_ACTIONS)
    button_labels = {LEFT: "◀ LEFT", RIGHT: "RIGHT ▶", ROTATE: "↻ ROTATE", DROP: "▼ DROP"}
    for a in range(NUM_ACTIONS):
        if cols[a].button(button_labels[a], key=f"act_{a}", disabled=env.done):
            obs, _info = env.step(a)
            st.session_state.live_obs = obs
            st.session_state.live_history.append((a, obs))

    obs = st.session_state.live_obs

    if env.done:
        st.warning("Episode terminated. Click *Reset env* to start a new one.")

    piece_id = PIECE_NAMES.index(env.piece_name) if env.piece_name in PIECE_NAMES else 0
    st.markdown(_piece_caption(piece_id, env.rotation, env.piece_row, env.piece_col))

    s_t = torch.from_numpy(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        z_t = encoder(s_t)
        recon = _decode(decoder, z_t)
    recon_acc = _binary_acc(recon, obs)

    left, right = st.columns(2)
    with left:
        st.markdown("**Actual obs `s_t`**")
        st.pyplot(_board_fig(obs))
    with right:
        st.markdown(f"**`decoder(encoder(s_t))`**  · binary acc {recon_acc:.3f}")
        st.pyplot(_board_fig(recon))

    st.markdown("---")
    st.markdown("**Per-action prediction:** `decoder(predictor(z_t, a))`")
    pred_cols = st.columns(NUM_ACTIONS)
    for a in range(NUM_ACTIONS):
        with torch.no_grad():
            a_emb = action_encoder(torch.tensor([a], device=device))
            z_pred = predictor(z_t, a_emb)
            pred_grid = _decode(decoder, z_pred)
        with pred_cols[a]:
            st.markdown(f"**{ACTION_NAMES[a]}**")
            st.pyplot(_board_fig(pred_grid))

    if st.session_state.live_history:
        st.markdown("---")
        st.caption(f"Steps so far: {len(st.session_state.live_history)} · "
                   f"last action: {ACTION_NAMES[st.session_state.live_history[-1][0]]}")


# =============================================================================
# Mode 2: Buffer scrubber
# =============================================================================

def _mode_buffer():
    st.subheader("Buffer scrubber")
    if not Path(buffer_path).exists():
        st.error(f"Buffer not found: {buffer_path}")
        return

    buf = _load_buffer(buffer_path)
    st.caption(f"Loaded {buf.size:,} triplets · "
               f"piece metadata available: **{buf.has_piece_meta}**")

    idx = st.slider("Buffer index", min_value=0, max_value=max(buf.size - 1, 0), value=0)
    s = buf.s[idx]
    s_next = buf.s_next[idx]
    a = int(buf.a[idx])

    info_bits = [f"action: **{ACTION_NAMES[a]}**",
                 f"lines: {int(buf.lines_cleared[idx])}",
                 f"holes: {int(buf.holes[idx])}",
                 f"height: {int(buf.aggregate_height[idx])}"]
    if buf.has_piece_meta:
        info_bits.append(_piece_caption(
            int(buf.piece_id[idx]), int(buf.rotation[idx]),
            int(buf.piece_row[idx]), int(buf.piece_col[idx]),
        ))
    st.markdown(" · ".join(info_bits))

    s_t = torch.from_numpy(s).unsqueeze(0).to(device)
    s_next_t = torch.from_numpy(s_next).unsqueeze(0).to(device)
    a_t = torch.tensor([a], device=device)

    with torch.no_grad():
        z_s = encoder(s_t)
        z_next = target_encoder(s_next_t)
        a_emb = action_encoder(a_t)
        z_pred = predictor(z_s, a_emb)
        recon_s = _decode(decoder, z_s)
        recon_next = _decode(decoder, z_next)
        pred_next = _decode(decoder, z_pred)
        cos = float(F.cosine_similarity(z_pred, z_next, dim=-1).item())
        l2 = float((z_pred - z_next).norm(dim=-1).item())

    cols = st.columns(5)
    with cols[0]:
        st.markdown("**`s_t`**")
        st.pyplot(_board_fig(s))
    with cols[1]:
        st.markdown(f"**dec(enc(s_t))**  ·  acc {_binary_acc(recon_s, s):.3f}")
        st.pyplot(_board_fig(recon_s))
    with cols[2]:
        st.markdown(f"**dec(pred(z_t, a))**  ·  acc {_binary_acc(pred_next, s_next):.3f}")
        st.pyplot(_board_fig(pred_next))
    with cols[3]:
        st.markdown(f"**dec(enc(s_t+1))**  ·  acc {_binary_acc(recon_next, s_next):.3f}")
        st.pyplot(_board_fig(recon_next))
    with cols[4]:
        st.markdown("**`s_t+1`**")
        st.pyplot(_board_fig(s_next))

    st.caption(f"latent metrics — cos(ẑ, z*) = {cos:.4f}  ·  ‖ẑ − z*‖ = {l2:.4f}")

    # ----------------------------- rollout -----------------------------------
    st.markdown("---")
    st.markdown("**Autoregressive rollout from this state**")
    horizon = st.slider("Horizon H", 1, 16, value=4)
    action_mode = st.radio(
        "Actions",
        options=["Use buffer's actual actions", "Pick per step"],
        horizontal=True,
    )

    if action_mode == "Pick per step":
        chosen: list[int] = []
        cols_a = st.columns(horizon)
        for t in range(horizon):
            with cols_a[t]:
                pick = st.selectbox(f"a_{t+1}", ACTION_NAMES, key=f"roll_a_{t}")
                chosen.append(ACTION_NAMES.index(pick))
        actions_seq = chosen
    else:
        actions_seq = [int(buf.a[min(idx + t, buf.size - 1)]) for t in range(horizon)]

    if st.button("Run rollout", key="run_rollout"):
        # Step the actual env starting from the buffer state. We can't restore
        # the exact env (board is in s, but the spawned-piece order isn't), so
        # this is approximate: we initialize a fresh env and overwrite its
        # board+piece from the buffer, then step.
        env = TetrisEnv(seed=int(seed))
        env.board = (s[0] > 0.5).astype(np.int8)
        if buf.has_piece_meta:
            pid = int(buf.piece_id[idx])
            env.piece_name = PIECE_NAMES[pid]
            env.rotation = int(buf.rotation[idx])
            env.piece_row = int(buf.piece_row[idx])
            env.piece_col = int(buf.piece_col[idx])
        env.done = False
        actuals = [env.observe()]
        for a_t_ in actions_seq:
            obs_next, _info = env.step(int(a_t_))
            actuals.append(obs_next)
            if env.done:
                while len(actuals) < horizon + 1:
                    actuals.append(obs_next)
                break

        with torch.no_grad():
            z_curr = encoder(torch.from_numpy(actuals[0]).unsqueeze(0).to(device))
            decoded = [_decode(decoder, z_curr)]
            cosines = []
            for t, a_t_ in enumerate(actions_seq):
                a_emb = action_encoder(torch.tensor([int(a_t_)], device=device))
                z_curr = predictor(z_curr, a_emb)
                decoded.append(_decode(decoder, z_curr))
                z_tgt = target_encoder(torch.from_numpy(actuals[t + 1]).unsqueeze(0).to(device))
                cosines.append(float(F.cosine_similarity(z_curr, z_tgt, dim=-1).item()))

        # Two rows of `horizon+1` cells: top = actual, bottom = decoded prediction.
        st.markdown("Top row = actual env · Bottom row = decoded prediction")
        for label, frames in (("actual", actuals), ("predicted", decoded)):
            cols_r = st.columns(horizon + 1)
            for t in range(horizon + 1):
                with cols_r[t]:
                    title = "t=0" if t == 0 else f"t={t} ← {ACTION_NAMES[actions_seq[t-1]]}"
                    st.pyplot(_board_fig(frames[t], title=title))
        if cosines:
            st.caption("per-step cos(ẑ, z*): " + " → ".join(f"{c:.3f}" for c in cosines))


# =============================================================================
# Mode 3: Latent space
# =============================================================================

def _mode_latent():
    st.subheader("Latent space (UMAP)")
    if not HAVE_UMAP:
        st.error(
            "`umap-learn` is not installed. Install with:\n\n"
            "    pip install umap-learn\n\n"
            "Then re-run the app."
        )
        return
    if not Path(buffer_path).exists():
        st.error(f"Buffer not found: {buffer_path}")
        return

    n_samples = st.slider("Samples to project", 500, 10_000, value=2_000, step=500)
    color_by = st.selectbox(
        "Color by",
        options=["aggregate_height", "holes", "lines_cleared", "piece_id"],
        index=0,
    )
    if st.button("Recompute UMAP", key="recompute_umap"):
        _compute_umap.clear()

    proj = _compute_umap(buffer_path, jepa_path, decoder_path, n_samples, int(seed))

    color_values = {
        "aggregate_height": proj["height"],
        "holes": proj["holes"],
        "lines_cleared": proj["lines"],
        "piece_id": proj["piece_id"],
    }[color_by]
    if color_by == "piece_id" and not proj["has_piece_meta"]:
        st.warning("Buffer lacks piece metadata; piece_id will be all zeros. "
                   "Re-collect with the v2 collector to populate it.")

    fig = go.Figure(
        data=go.Scattergl(
            x=proj["coords"][:, 0],
            y=proj["coords"][:, 1],
            mode="markers",
            marker=dict(
                size=4,
                color=color_values,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=color_by, len=0.5),
            ),
            text=[f"buf[{i}]" for i in proj["indices"]],
            hovertemplate="%{text}<br>%{marker.color:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
    )

    selected_local: int | None = None
    if HAVE_PLOTLY_EVENTS:
        clicks = plotly_events(fig, click_event=True, select_event=False, override_height=520)
        if clicks:
            selected_local = clicks[0]["pointIndex"]
    else:
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "Install `streamlit-plotly-events` (`pip install streamlit-plotly-events`) "
            "to click points directly. Until then, pick an index manually:"
        )
        selected_local = st.number_input(
            "Sample index (0..N-1)", min_value=0, max_value=len(proj["indices"]) - 1, value=0
        )

    if selected_local is None:
        st.caption("Click a point to decode its latent.")
        return

    buf_idx = int(proj["indices"][int(selected_local)])
    buf = _load_buffer(buffer_path)
    real = buf.s[buf_idx]
    z = torch.from_numpy(proj["latents"][int(selected_local)]).unsqueeze(0).to(device)
    decoded_z = _decode(decoder, z)

    # Nearest neighbor in latent space (cosine distance) within the projected sample.
    z_all = proj["latents"]
    z_q = z_all[int(selected_local)]
    sims = z_all @ z_q / (np.linalg.norm(z_all, axis=1) * np.linalg.norm(z_q) + 1e-9)
    sims[int(selected_local)] = -np.inf
    nn_local = int(np.argmax(sims))
    nn_buf_idx = int(proj["indices"][nn_local])
    nn_real = buf.s[nn_buf_idx]

    cols = st.columns(3)
    with cols[0]:
        st.markdown(f"**Decoded latent**  ·  buf[{buf_idx}]")
        st.pyplot(_board_fig(decoded_z))
    with cols[1]:
        st.markdown(f"**Real board**  ·  buf[{buf_idx}]")
        meta = ""
        if buf.has_piece_meta:
            meta = "\n\n" + _piece_caption(int(buf.piece_id[buf_idx]),
                                           int(buf.rotation[buf_idx]),
                                           int(buf.piece_row[buf_idx]),
                                           int(buf.piece_col[buf_idx]))
        st.markdown(meta or "_no piece metadata_")
        st.pyplot(_board_fig(real))
    with cols[2]:
        st.markdown(f"**Nearest-neighbor real**  ·  buf[{nn_buf_idx}]  ·  cos {sims[nn_local]:.3f}")
        st.pyplot(_board_fig(nn_real))


# =============================================================================
# Dispatch
# =============================================================================

if mode == "Live env":
    _mode_live_env()
elif mode == "Buffer scrubber":
    _mode_buffer()
else:
    _mode_latent()

st.sidebar.markdown("---")
st.sidebar.caption(f"device: `{device}`  ·  patch_dim: {bundle.patch_dim}  ·  N: {bundle.num_patches}")
