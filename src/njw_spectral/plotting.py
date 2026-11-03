"""Figures for the stage-1 experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

COLORS = plt.get_cmap("tab10").colors
POINT_SIZE = 9


def align_labels_for_display(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Relabel y_pred to best match y_true (Hungarian matching).

    Display only: cluster ids carry no meaning, and ARI is unchanged by this.
    """
    true_ids, pred_ids = np.unique(y_true), np.unique(y_pred)
    overlap = np.array([[np.sum((y_pred == p) & (y_true == t)) for t in true_ids] for p in pred_ids])
    rows, cols = linear_sum_assignment(-overlap)
    mapping = {pred_ids[r]: true_ids[c] for r, c in zip(rows, cols)}
    unused = iter(sorted(set(range(len(pred_ids) + len(true_ids))) - set(true_ids)))
    return np.array([mapping[p] if p in mapping else next(unused) for p in y_pred])


def _scatter(ax, pts, labels, title):
    for lab in np.unique(labels):
        m = labels == lab
        ax.scatter(pts[m, 0], pts[m, 1], s=POINT_SIZE, color=COLORS[int(lab) % 10], label=f"cluster {lab}")
    ax.set_title(title, fontsize=10)


def _data_limits(pts, pad=0.08):
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = hi - lo
    return (lo[0] - pad * span[0], hi[0] + pad * span[0]), (lo[1] - pad * span[1], hi[1] + pad * span[1])


def save_figure(fig, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_method_comparison(rows: list[dict], stem: Path) -> list[Path]:
    """One row per dataset: ground truth | K-means | NJW, shared axes per row.

    Each dict needs: title, points, y_true, kmeans_labels, njw_labels, sigma.
    """
    fig, axes = plt.subplots(len(rows), 3, figsize=(11, 3.6 * len(rows)))
    axes = np.atleast_2d(axes)
    for ax_row, r in zip(axes, rows):
        pts, y = r["points"], r["y_true"]
        km = align_labels_for_display(y, r["kmeans_labels"])
        nj = align_labels_for_display(y, r["njw_labels"])
        panels = [
            (y, f"{r['title']}: ground truth"),
            (km, f"K-means on raw coords (ARI = {adjusted_rand_score(y, km):.3f})"),
            (nj, f"NJW, sigma = {r['sigma']:.3g} (ARI = {adjusted_rand_score(y, nj):.3f})"),
        ]
        xlim, ylim = _data_limits(pts)
        for ax, (labels, title) in zip(ax_row, panels):
            _scatter(ax, pts, labels, title)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal")
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
        ax_row[0].legend(fontsize=7, loc="best", markerscale=1.5)
    fig.suptitle(
        "Ground truth vs. K-means vs. NJW spectral clustering\n"
        "(predicted cluster ids re-mapped to ground-truth colours for display only)",
        fontsize=11,
    )
    fig.tight_layout()
    return save_figure(fig, stem)


def plot_circles_pipeline(points, y_true, result, stem: Path) -> list[Path]:
    """Raw points -> rows of X -> rows of Y -> final NJW clustering (k = 2)."""
    if result.k != 2:
        raise ValueError("pipeline figure draws a 2-D embedding and needs k = 2")
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    lam = ", ".join(f"{v:.4f}" for v in result.eigenvalues)
    gap = f", lambda3 = {result.next_eigenvalue:.4f}" if result.next_eigenvalue is not None else ""

    xlim, ylim = _data_limits(points)
    _scatter(axes[0], points, y_true, "1. Raw points (colour = ground truth)")
    axes[0].set(xlim=xlim, ylim=ylim, xlabel="x1", ylabel="x2")

    _scatter(axes[1], result.X, y_true, f"2. Rows of X, no row normalization\n(lambda1, lambda2 = {lam}{gap})")
    xl, yl = _data_limits(result.X, pad=0.15)
    axes[1].set(xlim=xl, ylim=yl, xlabel="eigenvector 1 (X[:, 0])", ylabel="eigenvector 2 (X[:, 1])")

    theta = np.linspace(0, 2 * np.pi, 400)
    axes[2].plot(np.cos(theta), np.sin(theta), color="0.75", lw=0.8, ls="--", label="unit circle")
    _scatter(axes[2], result.Y, y_true, "3. Rows of Y = row-normalized X\n(colour = ground truth)")
    axes[2].set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), xlabel="Y[:, 0]", ylabel="Y[:, 1]")

    njw = align_labels_for_display(y_true, result.labels)
    _scatter(axes[3], points, njw, f"4. NJW clustering on raw points\n(ARI = {adjusted_rand_score(y_true, njw):.3f})")
    axes[3].set(xlim=xlim, ylim=ylim, xlabel="x1", ylabel="x2")

    for ax in axes:
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="best", markerscale=1.5)
    fig.suptitle(
        f"NJW pipeline on concentric circles (sigma = {result.sigma:.3g}, k = 2). "
        "Eigenvector signs/basis are arbitrary; panel 4 colours re-mapped to ground truth for display.",
        fontsize=10,
    )
    fig.tight_layout()
    return save_figure(fig, stem)
