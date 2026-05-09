"""Render Tetris boards and JEPA predictions with matplotlib.

Boards are (2, 20, 10) arrays where channel 0 is the locked board (gray) and
channel 1 is the falling piece (orange). Channels may be hard {0, 1} (real
states) or soft probabilities in [0, 1] (decoder output, post-sigmoid); for
soft inputs, alpha tracks the value so uncertain cells fade out.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from jepa_tetris.env.tetris import ACTION_NAMES, BOARD_HEIGHT, BOARD_WIDTH

BOARD_RGB = (0.55, 0.55, 0.60)   # locked cells
PIECE_RGB = (0.95, 0.55, 0.10)   # falling piece


def _grid_to_rgba(grid: np.ndarray) -> np.ndarray:
    """(2, H, W) probabilities/binary -> (H, W, 4) RGBA with alpha = max(channel)."""
    if grid.ndim != 3 or grid.shape[0] != 2:
        raise ValueError(f"expected (2, H, W), got {grid.shape}")
    board = np.clip(grid[0], 0.0, 1.0)
    piece = np.clip(grid[1], 0.0, 1.0)
    h, w = board.shape
    rgba = np.ones((h, w, 4), dtype=np.float32)
    # Piece on top of board: blend piece color where it's stronger.
    use_piece = piece >= board
    for k in range(3):
        rgba[..., k] = np.where(use_piece, PIECE_RGB[k], BOARD_RGB[k])
    rgba[..., 3] = np.maximum(board, piece)
    return rgba


def render_board(grid: np.ndarray, ax: plt.Axes, title: str | None = None) -> None:
    rgba = _grid_to_rgba(grid)
    ax.imshow(rgba, interpolation="nearest", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    # Light grid overlay so individual cells are readable.
    ax.set_xticks(np.arange(-0.5, BOARD_WIDTH, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, BOARD_HEIGHT, 1), minor=True)
    ax.grid(which="minor", color="#ddd", linewidth=0.4)
    if title:
        ax.set_title(title, fontsize=10)


def _format_metrics(metrics: dict | None, sep: str = "  ") -> str:
    if not metrics:
        return ""
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.3f}")
        else:
            parts.append(f"{k}={v}")
    return sep.join(parts)


def render_compare(
    s_t: np.ndarray,
    s_t1: np.ndarray,
    s_t1_pred: np.ndarray,
    action: int,
    metrics: dict | None = None,
    savepath: str | Path | None = None,
) -> plt.Figure:
    """1x3 panel: original | actual next | predicted next."""
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 4.2))
    action_name = ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else str(action)
    render_board(s_t, axes[0], title=f"s_t  (action: {action_name})")
    render_board(s_t1, axes[1], title="s_{t+1}  actual")
    render_board(s_t1_pred, axes[2], title="ŝ_{t+1}  predicted")
    suptitle = _format_metrics(metrics)
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, y=0.02, va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=120)
    return fig


def render_rollout(
    actions: Sequence[int],
    actual_states: Sequence[np.ndarray],
    predicted_grids: Sequence[np.ndarray],
    metrics_per_step: Sequence[dict] | None = None,
    savepath: str | Path | None = None,
    to_gif: bool = False,
) -> plt.Figure | animation.FuncAnimation:
    """2 x (k+1) strip: top = actual env rollout, bottom = decoded predicted rollout.

    `actual_states[0]` is s_0 (also where the prediction starts); subsequent
    entries are s_1..s_k. `predicted_grids` should have the same length: the
    first is the decoded reconstruction of s_0 (z_0 -> decode), and entries
    1..k are decoded predicted latents under `actions[0..k-1]`.
    """
    if len(actual_states) != len(predicted_grids):
        raise ValueError("actual_states and predicted_grids must have equal length")
    k_plus_1 = len(actual_states)
    if k_plus_1 < 2:
        raise ValueError("need at least one prediction step")

    if to_gif:
        return _rollout_gif(
            actions=actions,
            actual_states=actual_states,
            predicted_grids=predicted_grids,
            metrics_per_step=metrics_per_step,
            savepath=savepath,
        )

    fig, axes = plt.subplots(2, k_plus_1, figsize=(1.8 * k_plus_1 + 0.5, 6.5))
    if k_plus_1 == 1:
        axes = axes.reshape(2, 1)
    for t in range(k_plus_1):
        action_label = ""
        if t > 0 and t - 1 < len(actions):
            action_label = f"\n← {ACTION_NAMES[actions[t - 1]]}"
        top_title = ("s_0" if t == 0 else f"s_{t}") + action_label
        bot_title = "ŝ_0  (decoded)" if t == 0 else f"ŝ_{t}  (predicted)"
        render_board(actual_states[t], axes[0, t], title=top_title)
        render_board(predicted_grids[t], axes[1, t], title=bot_title)
        if metrics_per_step and t < len(metrics_per_step):
            m = _format_metrics(metrics_per_step[t], sep="\n")
            if m:
                axes[1, t].set_xlabel(m, fontsize=7)
    axes[0, 0].set_ylabel("actual", fontsize=11)
    axes[1, 0].set_ylabel("predicted", fontsize=11)
    fig.tight_layout()
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=120)
    return fig


def _rollout_gif(
    *,
    actions: Sequence[int],
    actual_states: Sequence[np.ndarray],
    predicted_grids: Sequence[np.ndarray],
    metrics_per_step: Sequence[dict] | None,
    savepath: str | Path | None,
) -> animation.FuncAnimation:
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 4.5))

    def draw(t: int) -> Iterable:
        for ax in axes:
            ax.clear()
        action_label = ""
        if t > 0 and t - 1 < len(actions):
            action_label = f"  (action: {ACTION_NAMES[actions[t - 1]]})"
        render_board(actual_states[t], axes[0], title=f"actual  t={t}{action_label}")
        bot_title = "predicted  t=0  (decoded)" if t == 0 else f"predicted  t={t}"
        render_board(predicted_grids[t], axes[1], title=bot_title)
        if metrics_per_step and t < len(metrics_per_step):
            m = _format_metrics(metrics_per_step[t])
            if m:
                axes[1].set_xlabel(m, fontsize=8)
        fig.tight_layout()
        return list(axes)

    anim = animation.FuncAnimation(
        fig, draw, frames=len(actual_states), interval=600, blit=False
    )
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        anim.save(savepath, writer=animation.PillowWriter(fps=2))
        plt.close(fig)
    return anim
