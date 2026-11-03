"""Stage 1: K-means vs. our NJW implementation on three synthetic datasets.

Usage (from the project root):
    python scripts/run_stage1.py
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # noqa: E402

from njw_spectral import KMeansConfig, njw_spectral_clustering, sigma_knn_median  # noqa: E402
from njw_spectral.datasets import DATASET_SPECS, DATASET_TITLES, make_dataset  # noqa: E402
from njw_spectral.plotting import plot_circles_pipeline, plot_method_comparison  # noqa: E402

DATA_SEED = 0
KMEANS_SEED = 0
SIGMA_NEIGHBORS = 7
SCALING = "none (raw generator coordinates for both methods)"
DATASETS = ["circles", "moons", "blobs"]

FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    km_cfg = KMeansConfig(random_state=KMEANS_SEED)

    table, fig_rows, circles_run = [], [], None
    for name in DATASETS:
        pts, y, k = make_dataset(name, seed=DATA_SEED)
        # Ground-truth y is used only for k (given to both methods) and for scoring.
        sigma = sigma_knn_median(pts, n_neighbors=SIGMA_NEIGHBORS)

        km_labels = km_cfg.build(k).fit_predict(pts)
        njw = njw_spectral_clustering(pts, k, sigma, kmeans_config=km_cfg)

        common = {
            "dataset": name,
            "n": pts.shape[0],
            "d": pts.shape[1],
            "k": k,
            "data_seed": DATA_SEED,
            "dataset_params": json.dumps({kk: v for kk, v in DATASET_SPECS[name].items()}),
            "scaling": SCALING,
            "kmeans_init": km_cfg.init,
            "kmeans_n_init": km_cfg.n_init,
            "kmeans_seed": km_cfg.random_state,
        }
        table.append({
            **common,
            "method": "kmeans_raw",
            "sigma": "",
            "sigma_rule": "",
            "ARI": adjusted_rand_score(y, km_labels),
            "NMI": normalized_mutual_info_score(y, km_labels),
            "top_k_eigenvalues": "",
            "lambda_k_plus_1": "",
            "eigengap": "",
            "min_row_norm_X": "",
        })
        table.append({
            **common,
            "method": "njw",
            "sigma": sigma,
            "sigma_rule": f"median distance to {SIGMA_NEIGHBORS}-th nearest neighbour (label-free)",
            "ARI": adjusted_rand_score(y, njw.labels),
            "NMI": normalized_mutual_info_score(y, njw.labels),
            "top_k_eigenvalues": " ".join(f"{v:.6f}" for v in njw.eigenvalues),
            "lambda_k_plus_1": njw.next_eigenvalue,
            "eigengap": njw.eigengap,
            "min_row_norm_X": float(njw.row_norms.min()),
        })
        fig_rows.append({
            "title": DATASET_TITLES[name],
            "points": pts,
            "y_true": y,
            "kmeans_labels": km_labels,
            "njw_labels": njw.labels,
            "sigma": sigma,
        })
        if name == "circles":
            circles_run = (pts, y, njw)

    csv_path = TAB_DIR / "stage1_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)

    config = {
        "datasets": {n: DATASET_SPECS[n] for n in DATASETS},
        "data_seed": DATA_SEED,
        "kmeans": {"init": km_cfg.init, "n_init": km_cfg.n_init, "max_iter": km_cfg.max_iter,
                   "random_state": km_cfg.random_state},
        "sigma_rule": f"median distance to {SIGMA_NEIGHBORS}-th nearest neighbour",
        "scaling": SCALING,
        "row_normalize": True,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__},
    }
    (TAB_DIR / "stage1_config.json").write_text(json.dumps(config, indent=2))

    outputs = plot_method_comparison(fig_rows, FIG_DIR / "stage1_comparison")
    outputs += plot_circles_pipeline(*circles_run, FIG_DIR / "stage1_circles_pipeline")

    print(f"{'dataset':<8} {'method':<11} {'sigma':>8} {'ARI':>7} {'NMI':>7} {'eigengap':>9}")
    for r in table:
        sigma = f"{r['sigma']:.4f}" if r["sigma"] != "" else "-"
        gap = f"{r['eigengap']:.4f}" if r["eigengap"] != "" else "-"
        print(f"{r['dataset']:<8} {r['method']:<11} {sigma:>8} {r['ARI']:>7.3f} {r['NMI']:>7.3f} {gap:>9}")
    print(f"\nWrote {csv_path.relative_to(ROOT)}, {(TAB_DIR / 'stage1_config.json').relative_to(ROOT)}")
    for p in outputs:
        print(f"Wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
