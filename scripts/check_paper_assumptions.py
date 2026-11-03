"""Evaluate the paper's Assumptions A1-A4 and Theorem 2 on the stage-1 runs.

Ground-truth labels are used here only to define the "true" clusters S_i that
the theory refers to; they never enter clustering or sigma selection.

Usage (from the project root):
    python scripts/check_paper_assumptions.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from njw_spectral import KMeansConfig, njw_spectral_clustering, sigma_knn_median  # noqa: E402
from njw_spectral.datasets import make_dataset  # noqa: E402
from njw_spectral.diagnostics import (  # noqa: E402
    mean_sq_distance_to_orthogonal_centers,
    paper_assumption_diagnostics,
)

DATA_SEED = 0
SIGMA_NEIGHBORS = 7
DATASETS = ["circles", "moons", "blobs"]


def main() -> None:
    rows = []
    for name in DATASETS:
        pts, y, k = make_dataset(name, seed=DATA_SEED)
        sigma = sigma_knn_median(pts, n_neighbors=SIGMA_NEIGHBORS)
        res = njw_spectral_clustering(pts, k, sigma, kmeans_config=KMeansConfig(random_state=0))
        diag = paper_assumption_diagnostics(res.affinity, y)
        rows.append({
            "dataset": name,
            "k": k,
            "sigma": sigma,
            "lambda_k_plus_1_of_L": res.next_eigenvalue,
            "lambda2_per_true_cluster": " ".join(f"{v:.6f}" for v in diag["lambda2_per_cluster"]),
            "delta": diag["delta"],
            "eps1": diag["eps1"],
            "eps2": diag["eps2"],
            "max_cross_ratio": diag["max_cross_ratio"],
            "C": diag["C"],
            "eps": diag["eps"],
            "required_delta_(2+sqrt2)eps": (2 + 2**0.5) * diag["eps"],
            "theorem2_condition_holds": diag["theorem2_condition_holds"],
            "theorem2_bound": diag["theorem2_bound"],
            "observed_mean_sq_dist_to_orthogonal_centers": mean_sq_distance_to_orthogonal_centers(res.Y, y),
        })

    out = ROOT / "results" / "tables" / "stage1_paper_assumptions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for r in rows:
        print(f"\n[{r['dataset']}] sigma = {r['sigma']:.4f}")
        for key, v in r.items():
            if key not in ("dataset", "sigma"):
                print(f"  {key:<45} {v:.6g}" if isinstance(v, float) else f"  {key:<45} {v}")
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
