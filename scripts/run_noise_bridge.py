"""Noise-bridge and density-imbalance critical experiments.

A) Moons: clean / thin bridge / strong bridge
B) Blobs: clean vs dense corridor bridge (clearer connectivity pollution)
C) Unbalanced blobs: row_normalize on vs off

ARI on bridged data ignores bridge points (y = -1). Stage-1 defaults unchanged.

Usage (from the project root):
    python scripts/run_noise_bridge.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.spatial.distance import pdist, squareform  # noqa: E402
from sklearn.datasets import make_blobs, make_moons  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from njw_spectral import KMeansConfig, njw_spectral_clustering, sigma_knn_median  # noqa: E402
from njw_spectral.datasets import (  # noqa: E402
    BRIDGE_MOONS_SPEC,
    UNBALANCED_BLOBS_SPEC,
    make_blobs_with_bridge,
    make_moons_with_bridge,
    make_moons_with_gap_noise,
    make_unbalanced_blobs,
)
from njw_spectral.plotting import (  # noqa: E402
    align_labels_for_display,
    save_figure,
    _data_limits,
    _scatter,
)

DATA_SEED = 0
KMEANS_SEED = 0
SIGMA_NEIGHBORS = 7

FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"


def ari_on_labeled(y_eval: np.ndarray, pred: np.ndarray) -> float:
    m = y_eval >= 0
    return float(adjusted_rand_score(y_eval[m], pred[m]))


def connected_components_for_k(points: np.ndarray, k: int) -> tuple[int, np.ndarray, float]:
    """Paper-style CC baseline: search distance threshold T to obtain k components.

    Among thresholds that yield exactly k components, pick the largest T
    (stronger connectivity within components). If none, pick T whose component
    count is closest to k (ties -> larger T). Label-free; uses distances only.
    """
    dist = squareform(pdist(points))
    # Candidate thresholds: unique pairwise distances (capped for speed).
    vals = np.sort(dist[np.triu_indices_from(dist, k=1)])
    # Skip near-duplicates; keep a manageable grid.
    candidates = np.unique(np.round(vals, decimals=6))
    if candidates.size > 400:
        idx = np.linspace(0, candidates.size - 1, 400).astype(int)
        candidates = candidates[idx]

    best = None  # (score, T, n_comp, labels) with score = (-|n-k|, T)
    for T in candidates:
        if T <= 0:
            continue
        adj = (dist > 0) & (dist <= T)
        n_comp, labels = connected_components(adj.astype(bool), directed=False)
        score = (-abs(n_comp - k), float(T))
        if best is None or score > best[0]:
            best = (score, float(T), int(n_comp), labels)
            if n_comp == k:
                # Keep searching for a larger T that still gives k (more robust).
                pass
    assert best is not None
    _, T, n_comp, labels = best
    # Re-select: among all with n_comp==k, largest T; else closest already in best.
    # Second pass for largest T with exact k.
    exact = []
    for T in candidates:
        if T <= 0:
            continue
        adj = (dist > 0) & (dist <= T)
        n_comp, labels = connected_components(adj.astype(bool), directed=False)
        if n_comp == k:
            exact.append((float(T), labels))
    if exact:
        T, labels = max(exact, key=lambda z: z[0])
        return k, labels, T
    return best[2], best[3], best[1]


def run_one(tag: str, experiment: str, pts, y_eval, k, km_cfg, spec: dict) -> tuple[dict, dict]:
    sigma = sigma_knn_median(pts, n_neighbors=SIGMA_NEIGHBORS)
    res = njw_spectral_clustering(pts, k, sigma, kmeans_config=km_cfg)
    km = km_cfg.build(k).fit_predict(pts)
    n_comp, cc_labels, cc_T = connected_components_for_k(pts, k)
    has_bridge = np.any(y_eval < 0)
    score = ari_on_labeled if has_bridge else adjusted_rand_score
    row = {
        "experiment": experiment,
        "setting": tag,
        "n": int(pts.shape[0]),
        "n_bridge": int(np.sum(y_eval < 0)),
        "sigma": sigma,
        "k": k,
        "ARI_njw": score(y_eval, res.labels),
        "ARI_kmeans": score(y_eval, km),
        "ARI_connected_components": score(y_eval, cc_labels),
        "cc_n_components": n_comp,
        "cc_threshold": cc_T,
        "eigengap": res.eigengap if res.eigengap is not None else "",
        "data_seed": DATA_SEED,
        "kmeans_seed": KMEANS_SEED,
        "spec": json.dumps(spec),
    }
    payload = {
        "points": pts,
        "y_eval": y_eval,
        "res": res,
        "sigma": sigma,
        "is_bridge": y_eval < 0,
        "cc_labels": cc_labels,
        "cc_n_components": n_comp,
        "cc_threshold": cc_T,
    }
    return row, payload


def run_moons_bridge(km_cfg) -> tuple[list[dict], dict]:
    pts_c, y_c = make_moons(
        n_samples=BRIDGE_MOONS_SPEC["n_samples"],
        noise=BRIDGE_MOONS_SPEC["noise"],
        random_state=DATA_SEED,
    )
    specs = [("clean", pts_c, y_c, 2, BRIDGE_MOONS_SPEC)]
    for name, n_b in [("thin_bridge", BRIDGE_MOONS_SPEC["n_bridge_thin"]),
                      ("strong_bridge", BRIDGE_MOONS_SPEC["n_bridge_strong"])]:
        pts, y_eval, _, k = make_moons_with_bridge(DATA_SEED, n_bridge=n_b)
        specs.append((name, pts, y_eval, k, {**BRIDGE_MOONS_SPEC, "n_bridge": n_b}))
    pts_g, y_g, _, k_g = make_moons_with_gap_noise(DATA_SEED, n_noise=120)
    specs.append(("gap_noise_cloud", pts_g, y_g, k_g, {**BRIDGE_MOONS_SPEC, "n_noise": 120}))

    rows, payloads = [], {}
    for tag, pts, y_eval, k, spec in specs:
        row, payload = run_one(tag, "moons_bridge", pts, y_eval, k, km_cfg, spec)
        rows.append(row)
        payloads[tag] = payload
    return rows, payloads


def run_blobs_bridge(km_cfg) -> tuple[list[dict], dict]:
    pts_c, y_c = make_blobs(
        n_samples=[200, 200], centers=np.array([[-2.5, 0.0], [2.5, 0.0]]),
        cluster_std=0.35, random_state=DATA_SEED,
    )
    pts_b, y_b, _, k = make_blobs_with_bridge(DATA_SEED, n_bridge=150, bridge_noise=0.25)
    rows, payloads = [], {}
    for tag, pts, y_eval, spec in [
        ("clean", pts_c, y_c, {"n_per_blob": 200, "bridge": False}),
        ("dense_corridor", pts_b, y_b, {"n_per_blob": 200, "n_bridge": 150, "bridge_noise": 0.25}),
    ]:
        row, payload = run_one(tag, "blobs_bridge", pts, y_eval, 2, km_cfg, spec)
        rows.append(row)
        payloads[tag] = payload
    return rows, payloads


def run_unbalanced(km_cfg) -> tuple[list[dict], dict]:
    pts, y, k = make_unbalanced_blobs(DATA_SEED)
    sigma = sigma_knn_median(pts, n_neighbors=SIGMA_NEIGHBORS)
    rows, payloads = [], {"points": pts, "y": y, "sigma": sigma}
    for row_norm in (True, False):
        res = njw_spectral_clustering(pts, k, sigma, row_normalize=row_norm, kmeans_config=km_cfg)
        rows.append({
            "experiment": "density_imbalance",
            "setting": f"row_normalize={row_norm}",
            "n": int(pts.shape[0]),
            "n_bridge": 0,
            "sigma": sigma,
            "k": k,
            "ARI_njw": adjusted_rand_score(y, res.labels),
            "ARI_kmeans": adjusted_rand_score(y, km_cfg.build(k).fit_predict(pts)),
            "ARI_connected_components": "",
            "cc_n_components": "",
            "cc_threshold": "",
            "eigengap": res.eigengap if res.eigengap is not None else "",
            "data_seed": DATA_SEED,
            "kmeans_seed": KMEANS_SEED,
            "spec": json.dumps(UNBALANCED_BLOBS_SPEC),
        })
        payloads[f"row_normalize={row_norm}"] = res
    return rows, payloads


def _panel_raw(ax, payload, title_prefix: str):
    pts, y_eval, res, is_bridge = payload["points"], payload["y_eval"], payload["res"], payload["is_bridge"]
    sigma = payload["sigma"]
    ari = ari_on_labeled(y_eval, res.labels) if np.any(is_bridge) else adjusted_rand_score(y_eval, res.labels)
    pred = res.labels.copy()
    if np.any(~is_bridge):
        pred[~is_bridge] = align_labels_for_display(y_eval[~is_bridge], pred[~is_bridge])
    xlim, ylim = _data_limits(pts)
    core = ~is_bridge
    _scatter(ax, pts[core], pred[core], f"{title_prefix}\nNJW ARI(core)={ari:.3f}, sigma={sigma:.3g}")
    if np.any(is_bridge):
        ax.scatter(pts[is_bridge, 0], pts[is_bridge, 1], s=18, c="black", marker="x", label="bridge", zorder=5)
        ax.legend(fontsize=7, loc="best")
    ax.set(xlim=xlim, ylim=ylim, aspect="equal", xlabel="x1", ylabel="x2")


def plot_moons(payloads: dict, stem: Path) -> list[Path]:
    tags = ["clean", "thin_bridge", "strong_bridge", "gap_noise_cloud"]
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 3.9))
    titles = ["Clean", "Thin bridge (25)", "Strong bridge (80)", "Gap noise cloud (120)"]
    for ax, tag, title in zip(axes, tags, titles):
        _panel_raw(ax, payloads[tag], title)
    fig.suptitle(
        "Moons + noise (our data). ARI uses original moon points only; black x = noise.",
        fontsize=11,
    )
    fig.tight_layout()
    return save_figure(fig, stem)


def plot_blobs(payloads: dict, stem: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for ax, tag, title in zip(axes, ["clean", "dense_corridor"],
                              ["Clean blobs", "Blobs + dense corridor bridge"]):
        _panel_raw(ax, payloads[tag], title)
    fig.suptitle(
        "Blobs + dense corridor: connectivity pollution (ARI on blob points only).",
        fontsize=11,
    )
    fig.tight_layout()
    return save_figure(fig, stem)


def plot_moons_embedding(payloads: dict, stem: Path) -> list[Path]:
    tags = ["clean", "thin_bridge", "strong_bridge", "gap_noise_cloud"]
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.0))
    theta = np.linspace(0, 2 * np.pi, 400)
    for ax, tag in zip(axes, tags):
        p = payloads[tag]
        res, y_eval, is_bridge = p["res"], p["y_eval"], p["is_bridge"]
        ax.plot(np.cos(theta), np.sin(theta), color="0.75", lw=0.8, ls="--")
        for lab, name in [(0, "moon 0"), (1, "moon 1")]:
            m = y_eval == lab
            ax.scatter(res.Y[m, 0], res.Y[m, 1], s=9, label=name)
        if np.any(is_bridge):
            ax.scatter(res.Y[is_bridge, 0], res.Y[is_bridge, 1], s=18, c="black",
                       marker="x", label="noise", zorder=5)
        ari = ari_on_labeled(y_eval, res.labels) if np.any(is_bridge) else adjusted_rand_score(y_eval, res.labels)
        ax.set_title(f"{tag}\nARI(core)={ari:.3f}")
        ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), aspect="equal",
               xlabel="Y[:, 0]", ylabel="Y[:, 1]")
        ax.legend(fontsize=6, loc="best")
    fig.suptitle("Y embeddings for moons noise settings", fontsize=11)
    fig.tight_layout()
    return save_figure(fig, stem)


def plot_unbalanced(payloads: dict, stem: Path) -> list[Path]:
    pts, y, sigma = payloads["points"], payloads["y"], payloads["sigma"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    xlim, ylim = _data_limits(pts)
    _scatter(axes[0], pts, y, "Unbalanced blobs\n(350 vs 50 points)")
    axes[0].set(xlim=xlim, ylim=ylim, aspect="equal", xlabel="x1", ylabel="x2")
    for ax, flag in zip(axes[1:], [True, False]):
        res = payloads[f"row_normalize={flag}"]
        lab = align_labels_for_display(y, res.labels)
        ari = adjusted_rand_score(y, lab)
        _scatter(ax, pts, lab, f"NJW, row_normalize={flag}\nsigma={sigma:.3g}, ARI={ari:.3f}")
        ax.set(xlim=xlim, ylim=ylim, aspect="equal", xlabel="x1", ylabel="x2")
    axes[0].legend(fontsize=7, loc="best", markerscale=1.5)
    fig.suptitle(
        "Mass imbalance: row normalization on vs off (same sigma, k).",
        fontsize=11,
    )
    fig.tight_layout()
    return save_figure(fig, stem)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    km_cfg = KMeansConfig(random_state=KMEANS_SEED)

    moon_rows, moon_payloads = run_moons_bridge(km_cfg)
    blob_rows, blob_payloads = run_blobs_bridge(km_cfg)
    unbal_rows, unbal_payloads = run_unbalanced(km_cfg)
    rows = moon_rows + blob_rows + unbal_rows

    csv_path = TAB_DIR / "noise_bridge_and_density.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    outputs = []
    outputs += plot_moons(moon_payloads, FIG_DIR / "noise_bridge_moons")
    outputs += plot_moons_embedding(moon_payloads, FIG_DIR / "noise_bridge_embedding")
    outputs += plot_blobs(blob_payloads, FIG_DIR / "noise_bridge_blobs")
    outputs += plot_unbalanced(unbal_payloads, FIG_DIR / "density_imbalance_row_norm")

    print(f"{'experiment':<14} {'setting':<18} {'n_br':>5} {'ARI_njw':>8} {'ARI_cc':>8} {'cc_n':>5}")
    for r in rows:
        ari_cc = f"{r['ARI_connected_components']:.3f}" if r["ARI_connected_components"] != "" else "-"
        cc_n = str(r["cc_n_components"]) if r["cc_n_components"] != "" else "-"
        print(f"{r['experiment']:<14} {r['setting']:<18} {r['n_bridge']:>5} {r['ARI_njw']:>8.3f} {ari_cc:>8} {cc_n:>5}")
    print(f"\nWrote {csv_path.relative_to(ROOT)}")
    for p in outputs:
        print(f"Wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
