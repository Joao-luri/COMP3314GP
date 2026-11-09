"""Figures and numbers for the v2 slide deck.

Everything here is recomputed with our own NJW implementation (src/njw_spectral)
or read from results/tables. Images go to slides/assets/gen/; point coordinates
that the deck draws as native PowerPoint shapes (for Morph animation) are
returned in a dict.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, TwoSlopeNorm
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from njw_spectral.algorithm import (  # noqa: E402
    KMeansConfig,
    gaussian_affinity,
    njw_from_affinity,
    njw_spectral_clustering,
    normalized_affinity,
    sigma_knn_median,
)
from njw_spectral.baselines import connected_components_for_k  # noqa: E402
from njw_spectral.datasets import (  # noqa: E402
    BRIDGE_MOONS_SPEC,
    make_dataset,
    make_moons_with_bridge,
    make_three_circles,
)
from njw_spectral.paper_like_data import PAPER_LIKE_BUILDERS  # noqa: E402

GEN = ROOT / "slides" / "assets" / "gen"
TABLES = ROOT / "results" / "tables"

NAVY = "#152C43"
TEAL = "#087F8C"
SLATE = "#526273"
ORANGE = "#E07A1F"
GREY = "#9AA6B2"
PURPLE = "#7A5195"
CLUSTER_COLORS = [TEAL, ORANGE, PURPLE, NAVY, "#C44E52", "#8C6D31", "#4C956C", "#5B8DB8"]

HEAT = LinearSegmentedColormap.from_list("heat", ["#FFFFFF", "#A8D8DC", TEAL, NAVY])
DIVERGE = LinearSegmentedColormap.from_list("div", [ORANGE, "#FFFFFF", TEAL])

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": SLATE,
        "axes.labelcolor": SLATE,
        "axes.titlecolor": NAVY,
        "axes.titlesize": 12,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "mathtext.fontset": "dejavusans",
    }
)

TOY_SIGMA = 0.16
TOY_SPEC = {"n_in": 16, "n_out": 36, "r_in": 0.3, "jitter": 0.04, "seed": 2}
SWEEP_RATIOS = (1.0, 1.4, 2.0, 4.0)


# ------------------------------------------------------------------ helpers


def _save(fig, name: str, **kw) -> Path:
    GEN.mkdir(parents=True, exist_ok=True)
    path = GEN / f"{name}.png"
    fig.savefig(path, dpi=kw.pop("dpi", 220), transparent=kw.pop("transparent", False), **kw)
    plt.close(fig)
    return path


def fmt2(a: float) -> str:
    return f"{0.0 if abs(a) < 0.005 else a:.2f}"


def match_labels(y_true, y_pred) -> np.ndarray:
    """Re-map predicted ids to ground-truth ids (display only; ARI is unaffected)."""
    t_ids, p_ids = np.unique(y_true[y_true >= 0]), np.unique(y_pred)
    cost = np.zeros((p_ids.size, t_ids.size))
    for a, p in enumerate(p_ids):
        for b, t in enumerate(t_ids):
            cost[a, b] = -np.sum((y_pred == p) & (y_true == t))
    rows, cols = linear_sum_assignment(cost)
    mapping = {p_ids[r]: t_ids[c] for r, c in zip(rows, cols)}
    extra = iter(range(int(t_ids.max()) + 1, int(t_ids.max()) + 1 + p_ids.size))
    return np.array([mapping[p] if p in mapping else next(extra) for p in y_pred])


def fix_signs(X: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    """Eigenvector signs are arbitrary; fix them so frames line up (display only)."""
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        s = col.sum() if j == 0 else col[ref_mask].mean()
        if s < 0:
            X[:, j] = -col
    return X


def heatmap_png(M: np.ndarray, name: str, norm=None, cmap=HEAT, px: int = 900) -> Path:
    """Square heatmap filling the whole image, so rows can be aligned with shapes."""
    fig = plt.figure(figsize=(px / 200, px / 200))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(M, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
    ax.set_axis_off()
    return _save(fig, name, dpi=200)


def column_heat_png(X: np.ndarray, name: str) -> Path:
    vmax = np.abs(X).max()
    fig = plt.figure(figsize=(0.4 * X.shape[1], 4.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(X, cmap=DIVERGE, norm=TwoSlopeNorm(0, -vmax, vmax), interpolation="nearest", aspect="auto")
    for j in range(1, X.shape[1]):
        ax.axvline(j - 0.5, color="white", lw=3)
    ax.set_axis_off()
    return _save(fig, name, dpi=200)


def degree_strip_png(d: np.ndarray, name: str) -> Path:
    fig = plt.figure(figsize=(1.0, 4.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.barh(np.arange(d.size), d, height=0.8, color=SLATE)
    ax.set_ylim(d.size - 0.5, -0.5)
    ax.set_xlim(0, d.max() * 1.05)
    ax.set_axis_off()
    return _save(fig, name, dpi=200, transparent=True)


def math_png(tex: str, name: str, size: float = 20, color: str = NAVY) -> tuple[Path, float, float]:
    """Render mathtext at `size` pt; returns path and natural size in inches."""
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex, fontsize=size, color=color)
    path = _save(fig, name, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02)
    from PIL import Image

    w, h = Image.open(path).size
    return path, w / 300, h / 300


def scatter_axes(ax, lim=None):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if lim is not None:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)


# ------------------------------------------------------------------ toy example


def toy_rings():
    """Two rings, inner ring first so A and L show their blocks in row order."""
    s = TOY_SPEC
    rng = np.random.default_rng(s["seed"])
    pts, lab = [], []
    for c, (m, r) in enumerate([(s["n_in"], s["r_in"]), (s["n_out"], 1.0)]):
        t = np.linspace(0, 2 * np.pi, m, endpoint=False) + rng.uniform(0, 2 * np.pi)
        t = t + rng.normal(0, 0.6 * np.pi / m, m)
        rr = r + rng.normal(0, s["jitter"], m)
        pts.append(np.c_[rr * np.cos(t), rr * np.sin(t)])
        lab += [c] * m
    return np.vstack(pts), np.array(lab)


def toy_data() -> dict:
    P, y = toy_rings()
    n = len(y)
    A = gaussian_affinity(P, TOY_SIGMA)
    d, L = normalized_affinity(A)
    vals, vecs = eigh(L)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    X = fix_signs(vecs[:, :2], y == 0)
    norms = np.linalg.norm(X, axis=1)
    Y = X / norms[:, None]
    res = njw_from_affinity(A, 2)
    labels = match_labels(y, res.labels)

    # The highlighted point i in the affinity slide: one on each ring.
    hi_outer = int(np.flatnonzero(y == 1)[np.argmax(P[y == 1, 0] + 0.3 * P[y == 1, 1])])
    hi_inner = int(np.flatnonzero(y == 0)[np.argmax(P[y == 0, 0])])

    norm_A = PowerNorm(0.6, 0, 1)
    out = {
        "P": P,
        "y": y,
        "n": n,
        "sigma": TOY_SIGMA,
        "A": A,
        "d": d,
        "L": L,
        "evals": vals[:8],
        "X": X,
        "Y": Y,
        "x_scale": 0.85 / norms.max(),
        "labels": labels,
        "ari": adjusted_rand_score(y, res.labels),
        "hi": [hi_outer, hi_inner],
        "img_A": heatmap_png(A, "toy_A", norm_A),
        "img_L": heatmap_png(L, "toy_L", PowerNorm(0.6, 0, L.max())),
        "img_d": degree_strip_png(d, "toy_d"),
        "img_X": column_heat_png(X, "toy_X"),
    }
    out["img_eig"] = eigen_bars_png(vals[:8], 2, "toy_eig")
    return out


def eigen_bars_png(vals: np.ndarray, k: int, name: str) -> Path:
    vals = vals[:6]
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    idx = np.arange(1, vals.size + 1)
    colors = [TEAL if i <= k else GREY for i in idx]
    ax.bar(idx, vals, color=colors, width=0.62)
    ax.set_ylim(0.9, 1.006)
    ax.set_xticks(idx, [f"$\\lambda_{{{i}}}$" for i in idx], fontsize=12)
    ax.set_yticks([0.9, 0.95, 1.0])
    ax.set_ylabel("eigenvalue of L")
    x = k + 0.5
    ax.annotate("", xy=(x, vals[k]), xytext=(x, vals[k - 1]),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.8, shrinkA=0, shrinkB=0))
    ax.annotate(f"gap $\\delta$ = {vals[k - 1] - vals[k]:.3f}", xy=(x + 0.05, (vals[k] + vals[k - 1]) / 2),
                xytext=(x + 1.3, 0.996), color=ORANGE, fontsize=11, va="center",
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
    fig.tight_layout()
    return _save(fig, name)


def ideal_case_png(toy: dict) -> Path:
    """Cross-ring weights removed: block-diagonal L, its eigenvectors, rows of X and Y."""
    y = toy["y"]
    A_hat = toy["A"] * (y[:, None] == y[None, :])
    _, L_hat = normalized_affinity(A_hat)
    vals, vecs = eigh(L_hat)
    X = vecs[:, np.argsort(vals)[::-1][:2]]
    X = fix_signs(X, y == 0)
    Yh = X / np.linalg.norm(X, axis=1)[:, None]

    fig = plt.figure(figsize=(11.6, 3.6))
    ax1 = fig.add_axes([0.0, 0.08, 0.27, 0.8])
    ax1.imshow(L_hat, cmap=HEAT, norm=PowerNorm(0.6, 0, L_hat.max()), interpolation="nearest")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("$\\hat{L}$ is block diagonal", fontsize=13)
    ax2 = fig.add_axes([0.31, 0.08, 0.07, 0.8])
    vmax = np.abs(X).max()
    ax2.imshow(X, cmap=DIVERGE, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto", interpolation="nearest")
    ax2.axvline(0.5, color="white", lw=3)
    ax2.set_xticks([0, 1], ["$v_1$", "$v_2$"], fontsize=12)
    ax2.set_yticks([])
    ax2.set_title("$\\hat{X}$", fontsize=13)
    ax3 = fig.add_axes([0.45, 0.04, 0.26, 0.86])
    xs = X * (0.85 / np.linalg.norm(X, axis=1).max())
    for c in (0, 1):
        m = y == c
        ax3.plot([0, xs[m, 0].mean() * 1.6], [0, xs[m, 1].mean() * 1.6], color=CLUSTER_COLORS[c], lw=1, ls=":")
        ax3.scatter(xs[m, 0], xs[m, 1], s=22, color=CLUSTER_COLORS[c], zorder=3)
    ax3.axhline(0, color=GREY, lw=0.8)
    ax3.axvline(0, color=GREY, lw=0.8)
    scatter_axes(ax3, 1.1)
    ax3.set_title("rows of $\\hat{X}$: one ray per cluster", fontsize=12)
    ax4 = fig.add_axes([0.74, 0.04, 0.26, 0.86])
    t = np.linspace(0, 2 * np.pi, 200)
    ax4.plot(np.cos(t), np.sin(t), color=GREY, lw=0.8)
    ax4.axhline(0, color=GREY, lw=0.8)
    ax4.axvline(0, color=GREY, lw=0.8)
    for c in (0, 1):
        m = y == c
        ax4.scatter(Yh[m, 0], Yh[m, 1], s=90, color=CLUSTER_COLORS[c], zorder=3)
    a0, a1 = (np.degrees(np.arctan2(Yh[y == c, 1].mean(), Yh[y == c, 0].mean())) for c in (0, 1))
    th = np.radians(np.linspace(min(a0, a1), max(a0, a1), 50))
    ax4.plot(0.28 * np.cos(th), 0.28 * np.sin(th), color=NAVY, lw=1.2)
    ax4.text(0.33, 0.0, "90°", color=NAVY, fontsize=12, va="center")
    scatter_axes(ax4, 1.15)
    ax4.set_title("rows of $\\hat{Y}$: orthogonal points", fontsize=12)
    return _save(fig, "ideal_case")


def perturbation_png(toy: dict) -> tuple[Path, float]:
    """A = A_hat + E at a slightly larger sigma, so E is visible."""
    sigma = 0.24
    y = toy["y"]
    A = gaussian_affinity(toy["P"], sigma)
    same = y[:, None] == y[None, :]
    A_hat, E = A * same, A * ~same
    scale = 1.0 / max(E.max(), 1e-12)
    norm = PowerNorm(0.6, 0, 1)
    fig = plt.figure(figsize=(9.6, 3.0))
    boxes = [0.0, 0.36, 0.72]
    mats = [A, A_hat, E * scale]
    titles = ["$A$", "$\\hat{A}$  (within clusters)", f"$E$  (across, ×{scale:.0f})"]
    for x0, M, ttl in zip(boxes, mats, titles):
        ax = fig.add_axes([x0, 0.02, 0.28, 0.84])
        ax.imshow(M, cmap=HEAT, norm=norm, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(ttl, fontsize=13)
    fig.text(0.315, 0.44, "=", fontsize=30, color=NAVY, ha="center", va="center")
    fig.text(0.675, 0.44, "+", fontsize=30, color=NAVY, ha="center", va="center")
    return _save(fig, "perturbation"), sigma


# ------------------------------------------------------------------ experiments


def kmeans_motivation_png() -> tuple[Path, dict]:
    P, y, k = make_dataset("circles", 0)
    km = KMeansConfig().build(k).fit(P)
    s0 = sigma_knn_median(P)
    nj = njw_spectral_clustering(P, k, s0)
    ari_km = adjusted_rand_score(y, km.labels_)
    ari_nj = adjusted_rand_score(y, nj.labels)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))
    c = km.cluster_centers_
    lab = match_labels(y, km.labels_)
    ax = axes[0]
    for j in range(k):
        ax.scatter(P[lab == j, 0], P[lab == j, 1], s=9, color=CLUSTER_COLORS[j])
    # Perpendicular bisector of the two centroids = K-means decision boundary.
    mid, dvec = c.mean(0), c[1] - c[0]
    nvec = np.array([-dvec[1], dvec[0]]) / np.linalg.norm(dvec)
    seg = np.array([mid - 2 * nvec, mid + 2 * nvec])
    ax.plot(seg[:, 0], seg[:, 1], color=NAVY, lw=1.6, ls="--")
    ax.scatter(c[:, 0], c[:, 1], marker="X", s=160, color=NAVY, edgecolor="white", lw=1.2, zorder=4)
    scatter_axes(ax, 1.25)
    ax.set_title(f"K-means on raw points   ARI = {ari_km:.3f}")
    ax = axes[1]
    lab = match_labels(y, nj.labels)
    for j in range(k):
        ax.scatter(P[lab == j, 0], P[lab == j, 1], s=9, color=CLUSTER_COLORS[j])
    scatter_axes(ax, 1.25)
    ax.set_title(f"NJW, same points and k   ARI = {ari_nj:.3f}")
    fig.tight_layout()
    return _save(fig, "motivation"), {"ari_km": ari_km, "ari_njw": ari_nj}


def stage1_grid_png() -> tuple[Path, dict]:
    names = [("circles", "Concentric circles"), ("moons", "Two moons"), ("blobs", "Gaussian blobs")]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4))
    nums = {}
    for j, (name, title) in enumerate(names):
        P, y, k = make_dataset(name, 0)
        km = KMeansConfig().build(k).fit_predict(P)
        s0 = sigma_knn_median(P)
        nj = njw_spectral_clustering(P, k, s0)
        nums[name] = {"km": adjusted_rand_score(y, km), "njw": adjusted_rand_score(y, nj.labels),
                      "sigma": s0, "n": len(y)}
        for i, (lab, meth, a) in enumerate([(km, "K-means", nums[name]["km"]),
                                            (nj.labels, "NJW", nums[name]["njw"])]):
            ax = axes[i, j]
            lab = match_labels(y, lab)
            for c in range(k):
                ax.scatter(P[lab == c, 0], P[lab == c, 1], s=7, color=CLUSTER_COLORS[c])
            scatter_axes(ax)
            ax.set_aspect("equal", adjustable="datalim")
            good = a > 0.99
            ax.set_title(f"{meth}   ARI = {fmt2(a)}",
                         color=TEAL if good else ORANGE, fontsize=13, fontweight="bold")
    fig.tight_layout(h_pad=2.0, rect=(0, 0, 1, 0.95))
    for j, (_, title) in enumerate(names):
        box = axes[0, j].get_position()
        fig.text((box.x0 + box.x1) / 2, 0.975, title, ha="center", va="center", fontsize=14, color=NAVY)
    return _save(fig, "stage1_grid"), nums


def _read_csv(name: str) -> list[dict]:
    with open(TABLES / name) as f:
        return list(csv.DictReader(f))


def paper_like_png() -> tuple[Path, dict]:
    rows = _read_csv("paper_like_results.csv")
    get = {(r["dataset"], int(r["k"]), r["method"]): r for r in rows}
    panels = [
        ("twocircles", 2, "two circles"),
        ("lineandballs", 3, "line and balls"),
        ("fourclouds", 2, "four clouds"),
        ("squiggles", 4, "squiggles"),
        ("nips", 8, "NIPS strokes"),
        ("threecircles_joined", 3, "joined rings"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.8))
    nums = {}
    for ax, (name, k, title) in zip(axes.ravel(), panels):
        P, y, k_, _ = PAPER_LIKE_BUILDERS[name](0)
        sigma = float(get[(name, k, "njw")]["sigma"])
        res = njw_spectral_clustering(P, k, sigma)
        core = y >= 0
        a = adjusted_rand_score(y[core], res.labels[core])
        a_km = float(get[(name, k, "kmeans")]["ARI"])
        nums[name] = {"njw": a, "km": a_km, "sigma": sigma}
        lab = match_labels(y, res.labels)
        for c in np.unique(lab):
            m = (lab == c) & core
            ax.scatter(P[m, 0], P[m, 1], s=6, color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)])
        if not core.all():
            ax.scatter(P[~core, 0], P[~core, 1], s=12, marker="x", color=NAVY, lw=0.8)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        good = a > 0.99
        ax.set_title(f"{title}, k={k}\nNJW {fmt2(a)}   K-means {fmt2(a_km)}",
                     color=TEAL if good else ORANGE, fontsize=12)
    fig.tight_layout()
    return _save(fig, "paper_like"), nums


def sigma_sweep() -> dict:
    """Circles from stage 1 at several sigma ratios: A heatmaps and Y rows per frame."""
    P, y, k = make_dataset("circles", 0)
    # make_circles labels the outer ring 0; show the inner ring first, each ring by angle.
    order = np.lexsort((np.arctan2(P[:, 1], P[:, 0]), -y))
    P, y = P[order], 1 - y[order]
    s0 = sigma_knn_median(P)
    frames = []
    for r in SWEEP_RATIOS:
        res = njw_spectral_clustering(P, k, s0 * r)
        Y = fix_signs(res.Y, y == 0)
        frames.append(
            {
                "r": r,
                "sigma": s0 * r,
                "ari": adjusted_rand_score(y, res.labels),
                "Y": Y,
                "img_A": heatmap_png(res.affinity, f"sweep_A_{r:g}", PowerNorm(0.5, 0, 1), px=800),
            }
        )
    curve = {}
    for row in _read_csv("sigma_sensitivity.csv"):
        if row["status"] == "ok":
            curve.setdefault(row["dataset"], []).append((float(row["ratio_r"]), float(row["ARI"])))
    # Keep only a subsample of points as native shapes; Y collapses anyway.
    rng = np.random.default_rng(0)
    keep = np.sort(rng.choice(len(y), 160, replace=False))
    return {"frames": frames, "y": y, "keep": keep, "curve": curve, "sigma0": s0, "n": len(y)}


def wrong_k_png() -> tuple[Path, dict]:
    P, y, _ = make_three_circles(0)
    s0 = sigma_knn_median(P)
    runs = [
        (None, None, "3 rings (truth)"),
        (2, s0, "NJW, k = 2, σ₀"),
        (3, s0, "NJW, k = 3, σ₀"),
        (3, 0.7 * s0, "NJW, k = 3, 0.7σ₀"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.8))
    nums = {"sigma0": s0}
    merged = np.where(y == 0, 0, 1)
    for ax, (k, s, title) in zip(axes, runs):
        if k is None:
            lab, sub = y, ""
        else:
            res = njw_spectral_clustering(P, k, s)
            a = adjusted_rand_score(y, res.labels)
            nums[(k, round(s / s0, 2))] = a
            lab = match_labels(y if k == 3 else merged, res.labels)
            sub = f"ARI vs 3 rings = {a:.2f}"
            if k == 2:
                nums["k2_vs_merge"] = adjusted_rand_score(merged, res.labels)
                sub += f"\nARI vs inner | rest = {nums['k2_vs_merge']:.2f}"
        for c in np.unique(lab):
            ax.scatter(P[lab == c, 0], P[lab == c, 1], s=5, color=CLUSTER_COLORS[c])
        scatter_axes(ax, 1.12)
        ax.set_title(title + ("\n" + sub if sub else ""), fontsize=11.5, va="bottom")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, "wrong_k"), nums


def bridge_png() -> tuple[Path, dict]:
    P, y_eval, is_bridge, k = make_moons_with_bridge(0, n_bridge=BRIDGE_MOONS_SPEC["n_bridge_strong"])
    s0 = sigma_knn_median(P)
    nj = njw_spectral_clustering(P, k, s0)
    cc_labels, _, _, _ = connected_components_for_k(P, k)
    core = ~is_bridge
    nums = {
        "njw": adjusted_rand_score(y_eval[core], nj.labels[core]),
        "cc": adjusted_rand_score(y_eval[core], cc_labels[core]),
        "n_bridge": int(is_bridge.sum()),
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.9))
    for ax, lab, title, a in [
        (axes[0], nj.labels, "NJW", nums["njw"]),
        (axes[1], cc_labels, "Connected components", nums["cc"]),
    ]:
        lab = match_labels(np.where(core, y_eval, -1), lab)
        for c in np.unique(lab):
            m = (lab == c) & core
            ax.scatter(P[m, 0], P[m, 1], s=5, color=CLUSTER_COLORS[c])
        ax.scatter(P[is_bridge, 0], P[is_bridge, 1], s=10, marker="x", color=NAVY, lw=0.8)
        scatter_axes(ax)
        ax.set_title(f"{title}   ARI = {a:.2f}", color=TEAL if a > 0.99 else ORANGE,
                     fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "bridge"), nums


def sigma_curves_png() -> Path:
    curve = {}
    for row in _read_csv("sigma_sensitivity.csv"):
        curve.setdefault(row["dataset"], []).append(
            (float(row["ratio_r"]), float(row["ARI"]) if row["status"] == "ok" else np.nan, row["status"]))
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.0), sharey=True)
    for ax, (name, title) in zip(axes, [("circles", "Concentric circles"), ("moons", "Two moons"),
                                        ("blobs", "Gaussian blobs")]):
        r, a, st = zip(*curve[name])
        r, a = np.array(r), np.array(a)
        ax.plot(r, a, "-o", color=TEAL, ms=5)
        bad = [x for x, s in zip(r, st) if s != "ok"]
        if bad:
            ax.scatter(bad, [0.0] * len(bad), marker="x", color=ORANGE, s=50, zorder=4)
            ax.text(bad[0] * 1.1, 0.08, "row of X ≈ 0", color=ORANGE, fontsize=9)
        ax.axvline(1.0, color=GREY, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xticks([0.25, 0.5, 1, 2, 4, 8], ["0.25", "0.5", "1", "2", "4", "8"])
        ax.minorticks_off()
        ax.set_ylim(-0.08, 1.08)
        ax.set_title(title)
        ax.set_xlabel("r = σ / σ₀")
    axes[0].set_ylabel("NJW ARI")
    fig.tight_layout()
    return _save(fig, "sigma_curves")


# ------------------------------------------------------------------ v3 deck (moons + six points)

SIX_POINTS = np.array([[0.0, 0.0], [0.8, 0.3], [0.3, 0.9], [2.4, 0.5], [3.2, 0.2], [2.9, 1.0]])
SIX_SIGMA = 0.7
CROSS_FACTORS = (0.0, 1.0, 10.0)


def _procrustes(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Rotate X onto ref (display only; rotations keep lengths and angles)."""
    u, _, vt = np.linalg.svd(X.T @ ref)
    return X @ (u @ vt)


def _six_njw(A: np.ndarray):
    d, L = normalized_affinity(A)
    vals, vecs = eigh(L)
    order = np.argsort(vals)[::-1]
    return d, L, vals[order], vecs[:, order[:2]]


def six_points() -> dict:
    """The six-point worked example: every matrix of NJW with readable numbers."""
    P = SIX_POINTS
    group = np.array([0, 0, 0, 1, 1, 1])
    A = gaussian_affinity(P, SIX_SIGMA)
    d, L, vals, X = _six_njw(A)
    X = fix_signs(X, group == 0)
    norms = np.linalg.norm(X, axis=1)
    Y = X / norms[:, None]
    res = njw_from_affinity(A, 2)
    out = {"P": P, "group": group, "sigma": SIX_SIGMA, "A": A, "d": d, "L": L, "evals": vals,
           "X": X, "norms": norms, "Y": Y, "labels": match_labels(group, res.labels),
           "dist12": float(np.linalg.norm(P[0] - P[1]))}
    cross = group[:, None] != group[None, :]
    frames = []
    for f in CROSS_FACTORS:
        Af = np.where(cross, A * f, A)
        if f == 0:
            # Eigenvalue 1 is repeated, so the solver's basis is arbitrary; build it explicitly.
            df = Af.sum(1)
            Xf = np.zeros((6, 2))
            for c in (0, 1):
                m = group == c
                Xf[m, c] = np.sqrt(df[m]) / np.linalg.norm(np.sqrt(df[m]))
            _, _, vf, _ = _six_njw(Af)
        else:
            _, _, vf, Xf = _six_njw(Af)
        Xf = _procrustes(Xf, X)
        frames.append({"factor": f, "A": Af, "evals": vf, "Y": Xf / np.linalg.norm(Xf, axis=1)[:, None]})
    out["cross_frames"] = frames
    return out


def compare_png(name: str, mode: str, fname: str) -> tuple[Path, dict]:
    """Two panels on a stage-1 dataset. mode 'truth_km': expected vs K-means; 'km_njw': K-means vs NJW."""
    P, y, k = make_dataset(name, 0)
    km = KMeansConfig().build(k).fit(P)
    nj = njw_spectral_clustering(P, k, sigma_knn_median(P))
    nums = {"km": adjusted_rand_score(y, km.labels_), "njw": adjusted_rand_score(y, nj.labels)}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    panels = ([(y, "Expected groups", None), (km.labels_, f"K-means   ARI = {fmt2(nums['km'])}", km)]
              if mode == "truth_km" else
              [(km.labels_, f"K-means   ARI = {fmt2(nums['km'])}", km), (nj.labels, f"NJW   ARI = {fmt2(nums['njw'])}", None)])
    for ax, (lab, title, model) in zip(axes, panels):
        lab = match_labels(y, lab)
        for c in range(k):
            ax.scatter(P[lab == c, 0], P[lab == c, 1], s=9, color=CLUSTER_COLORS[c])
        if model is not None:
            c = model.cluster_centers_
            mid, dvec = c.mean(0), c[1] - c[0]
            nvec = np.array([-dvec[1], dvec[0]]) / np.linalg.norm(dvec)
            seg = np.array([mid - 1.2 * nvec, mid + 1.2 * nvec])
            ax.plot(seg[:, 0], seg[:, 1], color=NAVY, lw=1.6, ls="--")
            ax.scatter(c[:, 0], c[:, 1], marker="X", s=150, color=NAVY, edgecolor="white", lw=1.2, zorder=4)
        scatter_axes(ax)
        ax.set_xlim(-1.35, 2.35)
        ax.set_ylim(-0.9, 1.4)
        ax.set_title(title, fontsize=13)
    fig.tight_layout()
    return _save(fig, fname), nums


def moons_overview_png() -> Path:
    """Raw points -> similarity graph -> rows of Y -> labels, on the stage-1 moons."""
    P, y, k = make_dataset("moons", 0)
    s0 = sigma_knn_median(P)
    res = njw_spectral_clustering(P, k, s0)
    lab = match_labels(y, res.labels)
    Y = fix_signs(res.Y, lab == 0)
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.0))
    ax = axes[0]
    ax.scatter(P[:, 0], P[:, 1], s=5, color=GREY)
    ax.set_title("1  Points in the plane")
    ax = axes[1]
    i, j = np.nonzero(np.triu(res.affinity > 0.25, 1))
    from matplotlib.collections import LineCollection

    ax.add_collection(LineCollection(np.stack([P[i], P[j]], 1), colors=TEAL, linewidths=0.6, alpha=0.6))
    ax.scatter(P[:, 0], P[:, 1], s=4, color=NAVY, zorder=3)
    ax.set_title("2  Similarity graph A")
    for a in axes[:2]:
        scatter_axes(a)
        a.set_xlim(-1.3, 2.3)
        a.set_ylim(-0.9, 1.4)
    ax = axes[2]
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color=GREY, lw=0.8, ls="--")
    ax.axhline(0, color=GREY, lw=0.6)
    ax.axvline(0, color=GREY, lw=0.6)
    ax.scatter(Y[:, 0], Y[:, 1], s=60, color=GREY, zorder=3)
    scatter_axes(ax, 1.2)
    ax.set_title("3  New coordinates Y")
    ax = axes[3]
    for c in range(k):
        ax.scatter(P[lab == c, 0], P[lab == c, 1], s=5, color=CLUSTER_COLORS[c])
    scatter_axes(ax)
    ax.set_xlim(-1.3, 2.3)
    ax.set_ylim(-0.9, 1.4)
    ax.set_title("4  K-means on Y")
    fig.tight_layout(w_pad=3.5)
    return _save(fig, "moons_overview")


def paper_pairs_png() -> dict:
    """Our panels for the slides that sit next to the paper's Figure 1 crops."""
    rows = {(r["dataset"], int(r["k"]), r["method"]): r for r in _read_csv("paper_like_results.csv")}
    out = {}
    for name, k, method, fname in [("twocircles", 2, "njw", "ours_tc_njw"), ("twocircles", 2, "kmeans", "ours_tc_km"),
                                   ("threecircles_joined", 3, "njw", "ours_joined_njw")]:
        P, y, _, _ = PAPER_LIKE_BUILDERS[name](0)
        if method == "njw":
            lab = njw_spectral_clustering(P, k, float(rows[(name, k, "njw")]["sigma"])).labels
        else:
            lab = KMeansConfig().build(k).fit_predict(P)
        core = y >= 0
        a = adjusted_rand_score(y[core], lab[core])
        lab = match_labels(y, lab)
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        for c in np.unique(lab):
            m = (lab == c) & core
            ax.scatter(P[m, 0], P[m, 1], s=5, color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)])
        if not core.all():
            ax.scatter(P[~core, 0], P[~core, 1], s=10, marker="x", color=NAVY, lw=0.8)
        scatter_axes(ax)
        fig.tight_layout(pad=0.2)
        out[fname] = (_save(fig, fname), a)
    return out


def moons_sweep(ratios=(1.0, 2.8, 8.0)) -> dict:
    P, y, k = make_dataset("moons", 0)
    s0 = sigma_knn_median(P)
    frames = []
    for r in ratios:
        res = njw_spectral_clustering(P, k, s0 * r)
        frames.append({"r": r, "ari": adjusted_rand_score(y, res.labels), "labels": match_labels(y, res.labels)})
    keep = np.sort(np.random.default_rng(0).choice(len(y), 220, replace=False))
    curve = {}
    for row in _read_csv("sigma_sensitivity.csv"):
        if row["status"] == "ok":
            curve.setdefault(row["dataset"], []).append((float(row["ratio_r"]), float(row["ARI"])))
    return {"P": P, "y": y, "frames": frames, "keep": keep, "curve": curve, "sigma0": s0}


def build_v3() -> dict:
    D = {"six": six_points()}
    D["img_moons_motivation"], D["moons_motivation"] = compare_png("moons", "truth_km", "moons_truth_km")
    D["img_moons_final"], D["moons_final"] = compare_png("moons", "km_njw", "moons_km_njw")
    D["img_moons_overview"] = moons_overview_png()
    D["pairs"] = paper_pairs_png()
    D["moons_sweep"] = moons_sweep()
    D["img_bridge"], D["bridge"] = bridge_png()
    D["img_wrong_k"], D["wrong_k"] = wrong_k_png()
    D["img_paper_like"], D["paper_like"] = paper_like_png()
    D["img_stage1"], D["stage1"] = stage1_grid_png()
    D["img_sigma_curves"] = sigma_curves_png()
    D["assumptions"] = {r["dataset"]: r for r in _read_csv("stage1_paper_assumptions.csv")}
    D["noise"] = _read_csv("noise_bridge_and_density.csv")
    D["paper_like_rows"] = _read_csv("paper_like_results.csv")
    return D


def build_all() -> dict:
    toy = toy_data()
    D = {"toy": toy}
    D["img_ideal"] = ideal_case_png(toy)
    D["img_perturb"], D["perturb_sigma"] = perturbation_png(toy)
    D["img_motivation"], D["motivation"] = kmeans_motivation_png()
    D["img_stage1"], D["stage1"] = stage1_grid_png()
    D["img_paper_like"], D["paper_like"] = paper_like_png()
    D["sweep"] = sigma_sweep()
    D["img_wrong_k"], D["wrong_k"] = wrong_k_png()
    D["img_bridge"], D["bridge"] = bridge_png()
    D["img_sigma_curves"] = sigma_curves_png()
    D["assumptions"] = {r["dataset"]: r for r in _read_csv("stage1_paper_assumptions.csv")}
    D["noise"] = _read_csv("noise_bridge_and_density.csv")
    D["paper_like_rows"] = _read_csv("paper_like_results.csv")
    return D


if __name__ == "__main__":
    D = build_all()
    t = D["toy"]
    print("toy n", t["n"], "ARI", t["ari"], "evals", np.round(t["evals"], 4))
    print("motivation", D["motivation"])
    print("stage1", D["stage1"])
    print("paper_like", D["paper_like"])
    print("sweep", [(f["r"], round(f["ari"], 3)) for f in D["sweep"]["frames"]])
    print("wrong_k", D["wrong_k"])
    print("bridge", D["bridge"])
