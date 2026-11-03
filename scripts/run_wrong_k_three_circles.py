"""Wrong-k experiment: three concentric rings, NJW asked for k=2 vs k=3.

Our own synthetic data (not the paper's threecircles-joined). Stage-1 defaults
are unchanged. Ground-truth 3-way labels are used only for scoring and plots.

Usage (from the project root):
    python scripts/run_wrong_k_three_circles.py
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # noqa: E402

from njw_spectral import KMeansConfig, njw_spectral_clustering, sigma_knn_median  # noqa: E402
from njw_spectral.datasets import THREE_CIRCLES_SPEC, make_three_circles  # noqa: E402
from njw_spectral.plotting import (  # noqa: E402
    align_labels_for_display,
    save_figure,
    _data_limits,
    _scatter,
)

DATA_SEED = 0
KMEANS_SEED = 0
SIGMA_NEIGHBORS = 7
RING_NAMES = ["inner", "middle", "outer"]

FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"


def best_merge_of_truth(y_true: np.ndarray) -> list[tuple[np.ndarray, str]]:
    """All ways to merge 3 true rings into 2 labels (for interpreting k=2)."""
    merges = []
    pairs = [(0, 1, 2, "merge inner+middle vs outer"),
             (0, 2, 1, "merge inner+outer vs middle"),
             (1, 2, 0, "merge middle+outer vs inner")]
    for a, b, alone, name in pairs:
        y = np.where((y_true == a) | (y_true == b), 0, 1)
        # put the singleton as label 1 for readability; ARI ignores ids
        merges.append((y, name))
    return merges


def ring_composition(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """For each predicted cluster, count points from each true ring."""
    out = {}
    for pred in np.unique(y_pred):
        mask = y_pred == pred
        counts = {RING_NAMES[t]: int(np.sum(mask & (y_true == t))) for t in np.unique(y_true)}
        out[f"pred_{int(pred)}"] = counts
    return out


def dominant_merge_description(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Describe which true rings were grouped together under k=2."""
    # For each true ring, majority predicted label.
    maj = {}
    for t in np.unique(y_true):
        preds, counts = np.unique(y_pred[y_true == t], return_counts=True)
        maj[int(t)] = int(preds[np.argmax(counts)])
    groups: dict[int, list[str]] = {}
    for t, p in maj.items():
        groups.setdefault(p, []).append(RING_NAMES[t])
    parts = ["+".join(names) for names in groups.values()]
    return " vs ".join(parts)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    pts, y3, true_k = make_three_circles(DATA_SEED)
    assert true_k == 3
    sigma = sigma_knn_median(pts, n_neighbors=SIGMA_NEIGHBORS)
    km_cfg = KMeansConfig(random_state=KMEANS_SEED)

    results = {}
    for method, k in [("kmeans", 2), ("kmeans", 3), ("njw", 2), ("njw", 3)]:
        if method == "kmeans":
            labels = km_cfg.build(k).fit_predict(pts)
            extras = {"eigengap": "", "top_eigenvalues": "", "inertia": ""}
        else:
            res = njw_spectral_clustering(pts, k, sigma, kmeans_config=km_cfg)
            labels = res.labels
            extras = {
                "eigengap": res.eigengap if res.eigengap is not None else "",
                "top_eigenvalues": " ".join(f"{v:.6f}" for v in res.eigenvalues),
                "inertia": res.kmeans_inertia,
            }
        key = f"{method}_k{k}"
        results[key] = {
            "labels": labels,
            "ARI_vs_3ring_truth": adjusted_rand_score(y3, labels),
            "NMI_vs_3ring_truth": normalized_mutual_info_score(y3, labels),
            "composition": ring_composition(y3, labels),
            "merge_description": dominant_merge_description(y3, labels) if k == 2 else "k=3 (no merge)",
            **extras,
        }

    # How well does NJW k=2 match each natural 2-way merge of the truth?
    merge_scores = []
    for y_merge, name in best_merge_of_truth(y3):
        merge_scores.append({
            "merge": name,
            "ARI_njw_k2": adjusted_rand_score(y_merge, results["njw_k2"]["labels"]),
            "ARI_kmeans_k2": adjusted_rand_score(y_merge, results["kmeans_k2"]["labels"]),
        })
    best_njw_merge = max(merge_scores, key=lambda d: d["ARI_njw_k2"])

    # ---- CSV summary
    table_rows = []
    for key, info in results.items():
        method, k_str = key.split("_k")
        table_rows.append({
            "method": method,
            "k": int(k_str),
            "sigma": sigma if method == "njw" else "",
            "sigma_rule": f"median {SIGMA_NEIGHBORS}-NN" if method == "njw" else "",
            "ARI_vs_3ring_truth": info["ARI_vs_3ring_truth"],
            "NMI_vs_3ring_truth": info["NMI_vs_3ring_truth"],
            "merge_description": info["merge_description"],
            "composition_json": json.dumps(info["composition"]),
            "eigengap": info["eigengap"],
            "top_eigenvalues": info["top_eigenvalues"],
            "data_seed": DATA_SEED,
            "kmeans_seed": KMEANS_SEED,
            "dataset_spec": json.dumps(THREE_CIRCLES_SPEC),
        })
    csv_path = TAB_DIR / "wrong_k_three_circles.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)

    merge_path = TAB_DIR / "wrong_k_three_circles_merges.csv"
    with merge_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(merge_scores[0].keys()))
        writer.writeheader()
        writer.writerows(merge_scores)

    # ---- Figure: GT | KMeans k=2 | NJW k=2 | NJW k=3
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0))
    xlim, ylim = _data_limits(pts)
    panels = [
        (y3, "Ground truth (3 rings)"),
        (align_labels_for_display(y3, results["kmeans_k2"]["labels"]),
         f"K-means k=2\nARI vs 3-ring truth = {results['kmeans_k2']['ARI_vs_3ring_truth']:.3f}"),
        (align_labels_for_display(y3, results["njw_k2"]["labels"]),
         f"NJW k=2, sigma = {sigma:.3g}\n{results['njw_k2']['merge_description']}\n"
         f"ARI vs 3-ring truth = {results['njw_k2']['ARI_vs_3ring_truth']:.3f}"),
        (align_labels_for_display(y3, results["njw_k3"]["labels"]),
         f"NJW k=3, sigma = {sigma:.3g}\nARI vs 3-ring truth = {results['njw_k3']['ARI_vs_3ring_truth']:.3f}"),
    ]
    for ax, (labels, title) in zip(axes, panels):
        _scatter(ax, pts, labels, title)
        ax.set(xlim=xlim, ylim=ylim, aspect="equal", xlabel="x1", ylabel="x2")
    axes[0].legend(fontsize=7, loc="best", markerscale=1.5)
    fig.suptitle(
        "Three concentric rings (our data): effect of asking for k=2 vs k=3\n"
        "(predicted colours re-mapped to ground-truth palette for display only; "
        "not the paper's threecircles-joined dataset)",
        fontsize=10,
    )
    fig.tight_layout()
    fig_paths = save_figure(fig, FIG_DIR / "wrong_k_three_circles")

    # ---- Optional small Y panel for NJW k=2 (only 2-D embedding)
    res2 = njw_spectral_clustering(pts, 2, sigma, kmeans_config=km_cfg)
    fig2, axes2 = plt.subplots(1, 2, figsize=(9.5, 4.2))
    _scatter(axes2[0], pts, align_labels_for_display(y3, res2.labels),
             f"NJW k=2 on raw coords\n{dominant_merge_description(y3, res2.labels)}")
    axes2[0].set(xlim=xlim, ylim=ylim, aspect="equal", xlabel="x1", ylabel="x2")
    theta = np.linspace(0, 2 * np.pi, 400)
    axes2[1].plot(np.cos(theta), np.sin(theta), color="0.75", lw=0.8, ls="--")
    _scatter(axes2[1], res2.Y, y3, "Rows of Y for k=2\n(colour = 3-ring ground truth)")
    axes2[1].set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), aspect="equal",
                 xlabel="Y[:, 0]", ylabel="Y[:, 1]")
    for ax in axes2:
        ax.legend(fontsize=7, loc="best", markerscale=1.5)
    fig2.suptitle(
        f"Why k=2 merges rings: Y embedding (sigma = {sigma:.3g}). "
        f"Best match among natural merges: {best_njw_merge['merge']} "
        f"(ARI = {best_njw_merge['ARI_njw_k2']:.3f})",
        fontsize=10,
    )
    fig2.tight_layout()
    fig_paths += save_figure(fig2, FIG_DIR / "wrong_k_three_circles_embedding")

    # ---- Small probe: can k=3 recover the three rings for some nearby sigma?
    # Not a full sensitivity study; only to avoid over-claiming that k=3 "cannot work".
    probe_ratios = [0.7, 1.0, 1.4, 2.0, 2.8, 4.0]
    probe_rows = []
    for r in probe_ratios:
        sig = r * sigma
        try:
            res = njw_spectral_clustering(pts, 3, sig, kmeans_config=km_cfg)
            probe_rows.append({
                "ratio_r": r,
                "sigma": sig,
                "status": "ok",
                "ARI_vs_3ring_truth": adjusted_rand_score(y3, res.labels),
                "merge_or_composition": json.dumps(ring_composition(y3, res.labels)),
            })
        except ValueError as exc:
            probe_rows.append({
                "ratio_r": r,
                "sigma": sig,
                "status": str(exc).split("\n")[0][:120],
                "ARI_vs_3ring_truth": "",
                "merge_or_composition": "",
            })
    probe_path = TAB_DIR / "wrong_k_three_circles_k3_sigma_probe.csv"
    with probe_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(probe_rows[0].keys()))
        writer.writeheader()
        writer.writerows(probe_rows)

    # ---- print summary
    print(f"sigma_0 (7-NN median) = {sigma:.6f}")
    print(f"{'method':<10} {'k':>2} {'ARI_vs_3':>8} {'merge / note'}")
    for row in table_rows:
        print(f"{row['method']:<10} {row['k']:>2} {row['ARI_vs_3ring_truth']:>8.3f}  {row['merge_description']}")
    print("\nARI of k=2 predictions vs each natural 2-way merge of truth:")
    for m in merge_scores:
        print(f"  {m['merge']:<32} NJW={m['ARI_njw_k2']:.3f}  K-means={m['ARI_kmeans_k2']:.3f}")
    print("\nNJW k=3 sigma probe (same data):")
    for p in probe_rows:
        ari = f"{p['ARI_vs_3ring_truth']:.3f}" if p["status"] == "ok" else "-"
        print(f"  r={p['ratio_r']:<4} sigma={p['sigma']:.4f}  status={p['status']:<12} ARI={ari}")
    print(f"\nWrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {merge_path.relative_to(ROOT)}")
    print(f"Wrote {probe_path.relative_to(ROOT)}")
    for p in fig_paths:
        print(f"Wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
