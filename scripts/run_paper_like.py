"""Attempted Figure-1-style experiments on OUR reconstructed datasets.

Sigma for NJW is chosen by minimum K-means distortion on Y (the paper's rule),
on a grid we declare here. The paper did not publish its grid or its points.
Ground-truth labels are used only for ARI and plot colours.

Usage (from the project root):
    python scripts/run_paper_like.py
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
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from njw_spectral import KMeansConfig, njw_spectral_clustering, sigma_knn_median  # noqa: E402
from njw_spectral.baselines import (  # noqa: E402
    connected_components_for_k,
    kannan_style_max_abs,
    meila_shi_clustering,
)
from njw_spectral.paper_like_data import (  # noqa: E402
    make_fourclouds,
    make_lineandballs,
    make_nips,
    make_squiggles,
    make_threecircles_joined,
    make_twocircles,
)
from njw_spectral.plotting import align_labels_for_display, save_figure, _data_limits, _scatter  # noqa: E402

DATA_SEED = 0
KMEANS_SEED = 0
# Declared by us. The paper searches sigma^2 but does not give the grid.
RATIOS = list(np.geomspace(0.35, 6.0, 12))

# (key, k, builder). threecircles is run at both k=3 and k=2, matching Fig. 1f/1g.
TASKS = [
    ("twocircles", 2, make_twocircles),
    ("threecircles_joined", 3, make_threecircles_joined),
    ("threecircles_joined", 2, make_threecircles_joined),
    ("lineandballs", 3, make_lineandballs),
    ("fourclouds", 2, make_fourclouds),
    ("squiggles", 4, make_squiggles),
    ("nips", 8, make_nips),
]

FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"


def ari_score(y_true: np.ndarray, pred: np.ndarray) -> float:
    mask = y_true >= 0
    return float(adjusted_rand_score(y_true[mask], pred[mask]))


def choose_sigma(points, k, sigma0, km_cfg):
    """Paper's rule: smallest K-means distortion of Y. Labels are not used."""
    records = []
    best = None
    for ratio in RATIOS:
        sigma = float(ratio * sigma0)
        rec = {"ratio": ratio, "sigma": sigma, "status": "ok", "inertia": "", "error": ""}
        try:
            res = njw_spectral_clustering(points, k, sigma, kmeans_config=km_cfg)
            rec["inertia"] = res.kmeans_inertia
            if best is None or res.kmeans_inertia < best[0]:
                best = (res.kmeans_inertia, sigma, res)
        except ValueError as exc:
            rec["status"] = "failed"
            rec["error"] = str(exc).split("\n")[0][:180]
        records.append(rec)
    return best, records


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    km_cfg = KMeansConfig(random_state=KMEANS_SEED)

    built = {}
    rows = []
    search_rows = []
    gallery = []

    for name, k, builder in TASKS:
        if name not in built:
            pts, y, _k_true, spec = builder(DATA_SEED)
            built[name] = (pts, y, spec)
        pts, y, spec = built[name]
        sigma0 = float(sigma_knn_median(pts, n_neighbors=7))
        best, records = choose_sigma(pts, k, sigma0, km_cfg)
        for rec in records:
            search_rows.append({"dataset": name, "k": k, "sigma0": sigma0, **rec})
        if best is None:
            rows.append({
                "dataset": name, "k": k, "method": "njw", "status": "all_sigma_failed",
                "sigma": "", "ARI": "", "detail": "", "sigma0": sigma0,
                "spec": json.dumps(spec),
            })
            continue
        inertia, sigma, njw = best
        methods = {
            "njw": njw.labels,
            "kmeans": km_cfg.build(k).fit_predict(pts),
        }
        try:
            methods["meila_shi"], _ = meila_shi_clustering(pts, k, sigma, km_cfg)
        except ValueError as exc:
            methods["meila_shi"] = exc
        try:
            methods["kannan_style"], _ = kannan_style_max_abs(pts, k, sigma)
        except ValueError as exc:
            methods["kannan_style"] = exc
        cc_labels, n_comp, threshold, exact = connected_components_for_k(pts, k)
        methods["connected_components"] = cc_labels

        for method, labels in methods.items():
            if isinstance(labels, Exception):
                rows.append({
                    "dataset": name, "k": k, "method": method, "status": "failed",
                    "sigma": sigma, "ARI": "", "detail": str(labels).split("\n")[0][:180],
                    "sigma0": sigma0, "njw_inertia": inertia,
                    "spec": json.dumps(spec), "data_seed": DATA_SEED, "kmeans_seed": KMEANS_SEED,
                })
                continue
            detail = ""
            if method == "connected_components":
                detail = f"n_components={n_comp}; T={threshold:.4f}; exact_k={exact}"
            if method == "njw":
                detail = f"chosen_by=min_Y_distortion; inertia={inertia:.6g}; eigengap={njw.eigengap}"
            rows.append({
                "dataset": name, "k": k, "method": method, "status": "ok",
                "sigma": sigma if method != "kmeans" else "",
                "ARI": ari_score(y, labels),
                "detail": detail,
                "sigma0": sigma0,
                "njw_inertia": inertia,
                "n": int(pts.shape[0]),
                "spec": json.dumps(spec),
                "data_seed": DATA_SEED,
                "kmeans_seed": KMEANS_SEED,
                "sigma_rule": "min NJW Y-distortion on declared ratio grid; same sigma for kernel methods",
            })
        gallery.append({
            "name": name, "k": k, "points": pts, "y": y, "labels": njw.labels,
            "sigma": sigma, "ari": ari_score(y, njw.labels),
        })

    csv_path = TAB_DIR / "paper_like_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    search_path = TAB_DIR / "paper_like_sigma_search.csv"
    with search_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(search_rows[0].keys()))
        writer.writeheader()
        writer.writerows(search_rows)

    # Gallery: NJW only, one panel per task.
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, item in zip(axes.ravel(), gallery):
        y, pred = item["y"], item["labels"]
        mask = y >= 0
        shown = np.full(len(y), -1)
        shown[mask] = align_labels_for_display(y[mask], pred[mask])
        # Bridge points (y<0) keep a distinct display id if present.
        if np.any(~mask):
            shown[~mask] = int(shown[mask].max()) + 1 if mask.any() else 0
        _scatter(ax, item["points"], shown,
                 f"{item['name']}, k={item['k']}\nNJW ARI={item['ari']:.3f}, sigma={item['sigma']:.3g}")
        xlim, ylim = _data_limits(item["points"])
        ax.set(xlim=xlim, ylim=ylim, aspect="equal", xlabel="x1", ylabel="x2")
    for ax in axes.ravel()[len(gallery):]:
        ax.axis("off")
    fig.suptitle(
        "NJW on our Figure-1-style datasets (not the paper's points).\n"
        "Sigma picked by minimum Y distortion. Colours re-mapped to our ground truth for display.",
        fontsize=11,
    )
    fig.tight_layout()
    outputs = save_figure(fig, FIG_DIR / "paper_like_njw")

    # Baseline comparison on three settings the paper contrasts.
    compare = [
        ("twocircles", 2),
        ("lineandballs", 3),
        ("threecircles_joined", 3),
    ]
    method_order = ["njw", "kmeans", "meila_shi", "connected_components", "kannan_style"]
    by_key = {(r["dataset"], r["k"], r["method"]): r for r in rows}
    fig, axes = plt.subplots(len(compare), len(method_order), figsize=(16.5, 10.5))
    cache_labels = {}
    for row_i, (name, k) in enumerate(compare):
        pts, y, _spec = built[name]
        sigma0 = float(sigma_knn_median(pts, n_neighbors=7))
        best, _ = choose_sigma(pts, k, sigma0, km_cfg)
        sigma = best[1]
        njw = best[2]
        cache_labels[("njw", name, k)] = njw.labels
        cache_labels[("kmeans", name, k)] = km_cfg.build(k).fit_predict(pts)
        cache_labels[("meila_shi", name, k)] = meila_shi_clustering(pts, k, sigma, km_cfg)[0]
        cache_labels[("kannan_style", name, k)] = kannan_style_max_abs(pts, k, sigma)[0]
        cache_labels[("connected_components", name, k)] = connected_components_for_k(pts, k)[0]
        xlim, ylim = _data_limits(pts)
        for col, method in enumerate(method_order):
            pred = cache_labels[(method, name, k)]
            mask = y >= 0
            shown = np.full(len(y), -1)
            shown[mask] = align_labels_for_display(y[mask], pred[mask])
            ari = by_key[(name, k, method)]["ARI"]
            _scatter(axes[row_i, col], pts, shown, f"{name} k={k}\n{method}\nARI={float(ari):.3f}")
            axes[row_i, col].set(xlim=xlim, ylim=ylim, aspect="equal")
            if row_i == len(compare) - 1:
                axes[row_i, col].set_xlabel("x1")
            if col == 0:
                axes[row_i, col].set_ylabel("x2")
    fig.suptitle(
        "Baselines on three reconstructed settings. Kernel methods share the NJW distortion-selected sigma.\n"
        "kannan_style assigns each point to the P-eigenvector with largest |coordinate|; not Kannan's original code.",
        fontsize=10,
    )
    fig.tight_layout()
    outputs += save_figure(fig, FIG_DIR / "paper_like_baselines")

    print(f"{'dataset':<22} {'k':>2} {'method':<22} {'ARI':>7} {'sigma':>8}")
    for r in rows:
        ari = f"{float(r['ARI']):.3f}" if r["ARI"] != "" else "-"
        sig = f"{float(r['sigma']):.4f}" if r["sigma"] != "" else "-"
        print(f"{r['dataset']:<22} {r['k']:>2} {r['method']:<22} {ari:>7} {sig:>8}")
    print(f"\nWrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {search_path.relative_to(ROOT)}")
    for p in outputs:
        print(f"Wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
