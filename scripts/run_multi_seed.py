"""Repeat the headline experiments over seeds, one source of randomness at a time.

source = "data":   data_seed in SEEDS, kmeans_seed = 0  (new point sets)
source = "kmeans": data_seed = 0, kmeans_seed in SEEDS  (same points, new K-means init)

Covers: stage-1 K-means vs NJW, the sigma curves for moons and circles, the
80-point moons bridge (NJW vs connected components) and three rings with k=3
at sigma0 and 0.7*sigma0. sigma0 is recomputed from each point set with the
stage-1 rule; ARI is never used to choose anything. Seed 0 of either source
reproduces the single-seed numbers used so far.

Usage (from the project root):
    python scripts/run_multi_seed.py
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
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from njw_spectral import KMeansConfig, njw_spectral_clustering, sigma_knn_median  # noqa: E402
from njw_spectral.baselines import connected_components_for_k  # noqa: E402
from njw_spectral.datasets import (  # noqa: E402
    BRIDGE_MOONS_SPEC,
    make_dataset,
    make_moons_with_bridge,
    make_three_circles,
)
from njw_spectral.plotting import save_figure  # noqa: E402

SEEDS = list(range(10))
SOURCES = ["data", "kmeans"]
SIGMA_NEIGHBORS = 7
RATIOS = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0]  # same grid as run_sigma_sensitivity
SIGMA_DATASETS = ["moons", "circles"]
# Values quoted on the critical-analysis slide (data seed 0, K-means seed 0).
SLIDE_POINTS = {"moons": [1.0, 2.8, 8.0], "circles": [1.0, 2.0]}

FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"


def ari(y, pred) -> float:
    m = y >= 0
    return float(adjusted_rand_score(y[m], pred[m]))


def njw_ari(pts, y, k, sigma, km_cfg) -> float:
    try:
        return ari(y, njw_spectral_clustering(pts, k, sigma, kmeans_config=km_cfg).labels)
    except ValueError:
        return float("nan")


def run_one(source: str, seed: int) -> list[dict]:
    data_seed = seed if source == "data" else 0
    km_seed = seed if source == "kmeans" else 0
    km_cfg = KMeansConfig(random_state=km_seed)
    base = {"source": source, "seed": seed, "data_seed": data_seed, "kmeans_seed": km_seed}
    rows = []

    def rec(experiment, setting, method, value, sigma=""):
        rows.append({**base, "experiment": experiment, "setting": setting, "method": method,
                     "sigma": sigma, "ARI": value})

    for name in ["circles", "moons", "blobs"]:
        pts, y, k = make_dataset(name, seed=data_seed)
        s0 = sigma_knn_median(pts, SIGMA_NEIGHBORS)
        rec("stage1", name, "kmeans", ari(y, km_cfg.build(k).fit_predict(pts)))
        rec("stage1", name, "njw", njw_ari(pts, y, k, s0, km_cfg), s0)
        if name in SIGMA_DATASETS:
            for r in RATIOS:
                rec("sigma_curve", f"{name}@{r:g}", "njw", njw_ari(pts, y, k, r * s0, km_cfg), r * s0)

    pts, y_eval, _, k = make_moons_with_bridge(data_seed, n_bridge=BRIDGE_MOONS_SPEC["n_bridge_strong"])
    s0 = sigma_knn_median(pts, SIGMA_NEIGHBORS)
    rec("bridge", "moons_strong_bridge", "njw", njw_ari(pts, y_eval, k, s0, km_cfg), s0)
    cc_labels, _, _, _ = connected_components_for_k(pts, k)
    rec("bridge", "moons_strong_bridge", "connected_components", ari(y_eval, cc_labels))

    pts, y, k = make_three_circles(data_seed)
    s0 = sigma_knn_median(pts, SIGMA_NEIGHBORS)
    for r in (1.0, 0.7):
        rec("three_rings_k3", f"{r:g}*sigma0", "njw", njw_ari(pts, y, k, r * s0, km_cfg), r * s0)
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    keys = list(dict.fromkeys((r["source"], r["experiment"], r["setting"], r["method"]) for r in rows))
    out = []
    for key in keys:
        sub = [r for r in rows if (r["source"], r["experiment"], r["setting"], r["method"]) == key]
        v = np.array([r["ARI"] for r in sub], float)
        ok = v[np.isfinite(v)]
        seed0 = next(r["ARI"] for r in sub if r["seed"] == 0)
        out.append({
            "source": key[0], "experiment": key[1], "setting": key[2], "method": key[3],
            "runs": len(sub), "numerical_failures": int(np.sum(~np.isfinite(v))),
            "ARI_seed0": seed0,
            "ARI_mean": float(ok.mean()) if ok.size else np.nan,
            "ARI_std": float(ok.std(ddof=1)) if ok.size > 1 else np.nan,
            "ARI_min": float(ok.min()) if ok.size else np.nan,
            "ARI_max": float(ok.max()) if ok.size else np.nan,
            "n_ARI_ge_0.99": int(np.sum(ok >= 0.99)),
        })
    return out


def _values(rows, source, experiment, setting, method):
    sub = sorted((r for r in rows if (r["source"], r["experiment"], r["setting"], r["method"])
                  == (source, experiment, setting, method)), key=lambda r: r["seed"])
    return np.array([r["ARI"] for r in sub], float)


def plot_sigma_curves(rows: list[dict], stem: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for i, source in enumerate(SOURCES):
        for j, name in enumerate(SIGMA_DATASETS):
            ax = axes[i, j]
            M = np.array([_values(rows, source, "sigma_curve", f"{name}@{r:g}", "njw") for r in RATIOS]).T
            for s, curve in enumerate(M):
                ax.plot(RATIOS, curve, "-", color="C0" if s else "C3", lw=1.8 if s == 0 else 0.7,
                        alpha=1.0 if s == 0 else 0.45, zorder=3 if s == 0 else 2,
                        label="seed 0 (slide)" if s == 0 else ("other seeds" if s == 1 else None))
            mean, std = np.nanmean(M, axis=0), np.nanstd(M, axis=0, ddof=1)
            ax.fill_between(RATIOS, mean - std, mean + std, color="k", alpha=0.12, lw=0, label="mean ± 1 std")
            ax.plot(RATIOS, mean, "k--", lw=1.2, label="mean")
            for r in SLIDE_POINTS[name]:
                idx = RATIOS.index(r)
                ax.annotate(f"{mean[idx]:.2f}±{std[idx]:.2f}", (r, mean[idx]), textcoords="offset points",
                            xytext=(4, 6), fontsize=7.5)
            ax.axvline(1.0, color="0.5", ls=":", lw=1)
            ax.set_xscale("log")
            ax.set_xticks([0.25, 0.5, 1, 2, 4, 8])
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_ylim(-0.1, 1.08)
            ax.grid(True, alpha=0.3)
            what = "data seeds 0-9, K-means seed 0" if source == "data" else "data seed 0, K-means seeds 0-9"
            ax.set_title(f"{name}: {what}", fontsize=10)
            if i == 1:
                ax.set_xlabel(r"r = $\sigma/\sigma_0$ ($\sigma_0$ recomputed per point set)")
            if j == 0:
                ax.set_ylabel("NJW ARI")
    axes[0, 0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Sigma curves over seeds (same relative grid as the single-seed sigma experiment)", fontsize=11)
    fig.tight_layout()
    return save_figure(fig, stem)


HEADLINES = [
    ("stage1", "circles", "kmeans", "circles\nK-means"),
    ("stage1", "circles", "njw", "circles\nNJW"),
    ("stage1", "moons", "kmeans", "moons\nK-means"),
    ("stage1", "moons", "njw", "moons\nNJW"),
    ("stage1", "blobs", "kmeans", "blobs\nK-means"),
    ("stage1", "blobs", "njw", "blobs\nNJW"),
    ("bridge", "moons_strong_bridge", "njw", "bridge 80\nNJW"),
    ("bridge", "moons_strong_bridge", "connected_components", "bridge 80\nconn. comp."),
    ("three_rings_k3", "1*sigma0", "njw", "3 rings k=3\nNJW σ0"),
    ("three_rings_k3", "0.7*sigma0", "njw", "3 rings k=3\nNJW 0.7σ0"),
    ("sigma_curve", "moons@2.8", "njw", "moons\nNJW 2.8σ0"),
    ("sigma_curve", "moons@8", "njw", "moons\nNJW 8σ0"),
]


def plot_headlines(rows: list[dict], stem: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(14, 4.6))
    rng = np.random.default_rng(0)
    offsets = {"data": -0.17, "kmeans": 0.17}
    names = {"data": "10 data seeds (K-means seed 0)", "kmeans": "10 K-means seeds (data seed 0)"}
    for x, (exp, setting, method, _) in enumerate(HEADLINES):
        for c, source in enumerate(SOURCES):
            v = _values(rows, source, exp, setting, method)
            xs = x + offsets[source] + rng.uniform(-0.07, 0.07, size=v.size)
            ax.scatter(xs[1:], v[1:], s=14, color=f"C{c}", alpha=0.7, lw=0,
                       label=names[source] if x == 0 else None)
            ax.plot([x + offsets[source] - 0.12, x + offsets[source] + 0.12], [np.nanmean(v)] * 2,
                    color=f"C{c}", lw=2)
        v0 = _values(rows, "data", exp, setting, method)[0]
        ax.scatter([x], [v0], marker="*", s=110, color="k", zorder=5,
                   label="seed 0 = value on slides" if x == 0 else None)
        d = _values(rows, "data", exp, setting, method)
        ax.text(x, -0.2, f"{np.nanmean(d):.2f}±{np.nanstd(d, ddof=1):.2f}", ha="center", fontsize=7.5, color="C0")
    for xb in (5.5, 7.5, 9.5):
        ax.axvline(xb, color="0.7", lw=0.8)
    ax.set_xticks(range(len(HEADLINES)))
    ax.set_xticklabels([h[3] for h in HEADLINES], fontsize=8)
    ax.set_ylim(-0.27, 1.08)
    ax.set_ylabel("ARI")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title("Headline numbers over seeds (bar = mean; blue text = mean ± std over data seeds)", fontsize=11)
    fig.tight_layout()
    return save_figure(fig, stem)


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in SOURCES:
        for seed in SEEDS:
            rows += run_one(source, seed)
            print(f"done source={source} seed={seed}", flush=True)
    summary = summarise(rows)
    write_csv(rows, TAB_DIR / "multi_seed_runs.csv")
    write_csv(summary, TAB_DIR / "multi_seed_summary.csv")
    outputs = plot_sigma_curves(rows, FIG_DIR / "multi_seed_sigma_curves")
    outputs += plot_headlines(rows, FIG_DIR / "multi_seed_headlines")

    print(f"\n{'source':<7} {'experiment':<15} {'setting':<20} {'method':<21} {'seed0':>6} {'mean':>6} "
          f"{'std':>6} {'min':>6} {'max':>6} {'>=.99':>5}")
    for s in summary:
        if s["experiment"] == "sigma_curve" and float(s["setting"].split("@")[1]) not in (1.0, 2.0, 2.8, 8.0):
            continue
        print(f"{s['source']:<7} {s['experiment']:<15} {s['setting']:<20} {s['method']:<21} "
              f"{s['ARI_seed0']:>6.3f} {s['ARI_mean']:>6.3f} {s['ARI_std']:>6.3f} {s['ARI_min']:>6.3f} "
              f"{s['ARI_max']:>6.3f} {s['n_ARI_ge_0.99']:>5}")
    for name in ["multi_seed_runs.csv", "multi_seed_summary.csv"]:
        print(f"Wrote results/tables/{name}")
    for p in outputs:
        print(f"Wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
