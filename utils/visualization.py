# ============================================================================
# Visualization Utilities
# ============================================================================
"""
Plotting helpers for training diagnostics, parity plots, and
microstructure visualization.

All functions produce publication-quality figures using Matplotlib with
consistent styling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


# ── Global style ────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.figsize": (6, 5),
    "figure.dpi": 150,
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def plot_loss_curve(
    train_losses: Sequence[float],
    val_losses: Sequence[float] | None = None,
    title: str = "Training Convergence",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot training (and optionally validation) loss curves.

    Parameters
    ----------
    train_losses : Sequence[float]
        Per-epoch training loss values.
    val_losses : Sequence[float], optional
        Per-epoch validation loss values.
    title : str
        Figure title.
    save_path : str or Path, optional
        If provided, saves the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots()
    epochs = np.arange(1, len(train_losses) + 1)

    ax.semilogy(epochs, train_losses, label="Train", color="#2563eb")
    if val_losses is not None:
        ax.semilogy(epochs, val_losses, label="Validation", color="#dc2626")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = r"$E_{\mathrm{eff}}$ [GPa]",
    title: str = "Surrogate Prediction Parity",
    r2: float | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Parity plot: predicted vs. ground-truth property values.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values.
    y_pred : np.ndarray
        Model predictions.
    label : str
        Axis label for the physical quantity.
    title : str
        Figure title.
    r2 : float, optional
        R² value to annotate on the plot.
    save_path : str or Path, optional
        If provided, saves the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots()

    ax.scatter(y_true, y_pred, s=8, alpha=0.5, color="#6366f1", edgecolors="none")

    # Perfect-prediction line
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "--", color="#94a3b8", linewidth=1, label="y = x")

    ax.set_xlabel(f"True {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")

    if r2 is not None:
        ax.text(
            0.05, 0.92, f"$R^2 = {r2:.3f}$",
            transform=ax.transAxes, fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def show_microstructure(
    image: np.ndarray,
    title: str = "Microstructure",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Display a single microstructure image.

    Parameters
    ----------
    image : np.ndarray
        2D array of shape ``(H, W)``.
    title : str
        Figure title.
    save_path : str or Path, optional
        If provided, saves the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray", interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def show_microstructure_grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str] | None = None,
    ncols: int = 4,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Display a grid of microstructure images.

    Parameters
    ----------
    images : Sequence[np.ndarray]
        List of 2D arrays.
    titles : Sequence[str], optional
        Per-image titles.
    ncols : int
        Number of columns.
    save_path : str or Path, optional
        If provided, saves the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)

    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i < n:
            ax.imshow(images[i], cmap="gray", interpolation="nearest")
            if titles:
                ax.set_title(titles[i], fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
