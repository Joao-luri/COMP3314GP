"""Sigma sensitivity (stage 2a): fixed data, pre-set relative sigma grid.

Does not change stage-1 defaults. ARI is never used to choose a default sigma.

Usage (from the project root):
    python scripts/run_sigma_sensitivity.py
"""

from __future__ import annotations

import csv
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
from njw_spectral.datasets import DATASET_TITLES, make_dataset  # noqa: E402
from njw_spectral.diagnostics import paper_assumption_diagnostics  # noqa: E402
from njw_spectral.plotting import (  # noqa: E402
    align_labels_for_display,
    save_figure,
    _data_limits,
    _scatter,
)

DATA_SEED = 0
KMEANS_SEED = 0
SIGMA_NEIGHBORS = 7
DATASETS = ["circles", "moons", "blobs"]
# Pre-set relative grid: sigma = r * sigma_0. r=1 is the stage-1 heuristic.
RATIOS = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0]
# Circles case panels: one too-small-ish, stage-1 default, one too-large.
CASE_RATIOS = [0.5, 1.0, 4.0]

FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"


def _classify_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "zero degree" in msg:
        return "zero_degree"
    if "norm" in msg and "row" in msg:
        return "zero_row_norm"
    if "non-finite" in msg or "nan" in msg:
        return "non_finite"
    return "error"


def run_grid() -> list[dict]:
    km_cfg = KMeansConfig(random_state=KMEANS_SEED)
    rows: list[dict] = []
    for name in DATASETS:
        pts, y, k = make_dataset(name, seed=DATA_SEED)
        sigma0 = sigma_knn_median(pts, n_neighbors=SIGMA_NEIGHBORS)
        for r in RATIOS:
            sigma = r * sigma0
            row = {
                "dataset": name,
                "ratio_r": r,
                "sigma0": sigma0,
                "sigma": sigma,
                "k": k,
                "n": pts.shape[0],
                "data_seed": DATA_SEED,
                "kmeans_seed": KMEANS_SEED,
                "status": "ok",
                "ARI": np.nan,
                "NMI": np.nan,
                "kmeans_inertia": np.nan,
                "eigengap": np.nan,
                "lambda_k": np.nan,
                "lambda_k_plus_1": np.nan,
                "delta": np.nan,
                "eps": np.nan,
                "theorem2_holds": "",
                "error_message": "",
            }
            try:
                res = njw_spectral_clustering(pts, k, sigma, kmeans_config=km_cfg)
                row.update({
                    "ARI": adjusted_rand_score(y, res.labels),
                    "NMI": normalized_mutual_info_score(y, res.labels),
                    "kmeans_inertia": res.kmeans_inertia,
                    "eigengap": res.eigengap if res.eigengap is not None else np.nan,
                    "lambda_k": float(res.eigenvalues[-1]),
                    "lambda_k_plus_1": res.next_eigenvalue if res.next_eigenvalue is not None else np.nan,
                })
                diag = paper_assumption_diagnostics(res.affinity, y)
                row.update({
                    "delta": diag["delta"],
                    "eps": diag["eps"],
                    "theorem2_holds": diag["theorem2_condition_holds"],
                })
            except ValueError as exc:
                row["status"] = _classify_error(exc)
                row["error_message"] = str(exc).split("\n")[0][:200]
            rows.append(row)
    return rows


def plot_ari(rows: list[dict], stem: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
    for ax, name in zip(axes, DATASETS):
        sub = [r for r in rows if r["dataset"] == name]
        ok = [r for r in sub if r["status"] == "ok"]
        bad = [r for r in sub if r["status"] != "ok"]
        if ok:
            ax.plot([r["ratio_r"] for r in ok], [r["ARI"] for r in ok], "o-", color="C0", label="NJW ARI (ok)")
        if bad:
            ax.scatter(
                [r["ratio_r"] for r in bad],
                [-0.08] * len(bad),
                marker="x",
                color="C3",
                s=50,
                label="numerical failure",
                zorder=5,
            )
        ax.axvline(1.0, color="0.4", ls="--", lw=1, label="stage-1 sigma (r=1)")
        ax.set_xscale("log")
        ax.set_xlabel(r"r = $\sigma$ / $\sigma_0$")
        ax.set_title(DATASET_TITLES[name])
        ax.set_ylim(-0.15, 1.05)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("ARI")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(
        "Sigma sensitivity: ARI vs relative bandwidth\n"
        r"($\sigma_0$ = median distance to 7th NN; not chosen by ARI)",
        fontsize=11,
    )
    fig.tight_layout()
    return save_figure(fig, stem)


def plot_aux(rows: list[dict], stem: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.2), sharex="col")
    for col, name in enumerate(DATASETS):
        sub = [r for r in rows if r["dataset"] == name and r["status"] == "ok"]
        rs = [r["ratio_r"] for r in sub]
        axes[0, col].plot(rs, [r["eigengap"] for r in sub], "o-", color="C1")
        axes[0, col].axvline(1.0, color="0.4", ls="--", lw=1)
        axes[0, col].set_xscale("log")
        axes[0, col].set_title(DATASET_TITLES[name])
        axes[0, col].set_ylabel(r"eigengap $\lambda_k-\lambda_{k+1}$" if col == 0 else "")
        axes[0, col].grid(True, alpha=0.3)

        axes[1, col].plot(rs, [r["kmeans_inertia"] for r in sub], "o-", color="C2")
        axes[1, col].axvline(1.0, color="0.4", ls="--", lw=1)
        axes[1, col].set_xscale("log")
        axes[1, col].set_xlabel(r"r = $\sigma$ / $\sigma_0$")
        axes[1, col].set_ylabel("Y K-means distortion" if col == 0 else "")
        axes[1, col].grid(True, alpha=0.3)
    fig.suptitle(
        "Auxiliary curves on the same sigma grid\n"
        "(paper picks sigma by small distortion; compare with ARI figure)",
        fontsize=11,
    )
    fig.tight_layout()
    return save_figure(fig, stem)


def plot_cases(stem: Path) -> list[Path]:
    pts, y, k = make_dataset("circles", seed=DATA_SEED)
    sigma0 = sigma_knn_median(pts, n_neighbors=SIGMA_NEIGHBORS)
    km_cfg = KMeansConfig(random_state=KMEANS_SEED)
    fig, axes = plt.subplots(2, len(CASE_RATIOS), figsize=(4.2 * len(CASE_RATIOS), 7.2))
    xlim, ylim = _data_limits(pts)

    for col, r in enumerate(CASE_RATIOS):
        sigma = r * sigma0
        ax0, ax1 = axes[0, col], axes[1, col]
        try:
            res = njw_spectral_clustering(pts, k, sigma, kmeans_config=km_cfg)
            labels = align_labels_for_display(y, res.labels)
            ari = adjusted_rand_score(y, labels)
            _scatter(ax0, pts, labels, f"r = {r:g}, sigma = {sigma:.3g}\nNJW on raw coords (ARI = {ari:.3f})")
            ax0.set(xlim=xlim, ylim=ylim, aspect="equal", xlabel="x1", ylabel="x2")
            theta = np.linspace(0, 2 * np.pi, 400)
            ax1.plot(np.cos(theta), np.sin(theta), color="0.75", lw=0.8, ls="--")
            _scatter(ax1, res.Y, y, "Rows of Y (colour = ground truth)")
            ax1.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), aspect="equal", xlabel="Y[:, 0]", ylabel="Y[:, 1]")
        except ValueError as exc:
            ax0.set_xlim(xlim)
            ax0.set_ylim(ylim)
            ax0.set_aspect("equal")
            ax0.set_title(f"r = {r:g}, sigma = {sigma:.3g}\nFAILED: {_classify_error(exc)}")
            ax0.text(0.5, 0.5, str(exc).split("\n")[0][:80], transform=ax0.transAxes, ha="center", va="center", fontsize=8)
            ax1.set_title("no Y embedding")
            ax1.axis("off")
        if col == 0:
            ax0.legend(fontsize=7, loc="best", markerscale=1.5)
            ax1.legend(fontsize=7, loc="best", markerscale=1.5)

    fig.suptitle(
        "Circles case study: small / stage-1 / large sigma\n"
        "(top: clustering on raw coords; bottom: Y embedding; colours re-mapped for display only)",
        fontsize=11,
    )
    fig.tight_layout()
    return save_figure(fig, stem)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    rows = run_grid()
    csv_path = TAB_DIR / "sigma_sensitivity.csv"
    write_csv(rows, csv_path)

    outputs = []
    outputs += plot_ari(rows, FIG_DIR / "sigma_sensitivity_ari")
    outputs += plot_aux(rows, FIG_DIR / "sigma_sensitivity_aux")
    outputs += plot_cases(FIG_DIR / "sigma_sensitivity_cases")

    print(f"{'dataset':<8} {'r':>5} {'sigma':>8} {'status':<12} {'ARI':>7} {'eigengap':>9} {'inertia':>10}")
    for r in rows:
        ari = f"{r['ARI']:.3f}" if r["status"] == "ok" else "-"
        gap = f"{r['eigengap']:.4f}" if r["status"] == "ok" and np.isfinite(r["eigengap"]) else "-"
        inn = f"{r['kmeans_inertia']:.4f}" if r["status"] == "ok" else "-"
        print(f"{r['dataset']:<8} {r['ratio_r']:>5.2f} {r['sigma']:>8.4f} {r['status']:<12} {ari:>7} {gap:>9} {inn:>10}")
    print(f"\nWrote {csv_path.relative_to(ROOT)}")
    for p in outputs:
        print(f"Wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
