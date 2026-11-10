"""Final K-means step: paper's 90-degree initialization vs our k-means++ default.

For each task the embedding Y is computed once (it does not depend on K-means),
then only the K-means call on the rows of Y is repeated over INIT_SEEDS with
four schemes. Data, sigma and k are fixed per task. Defaults are unchanged.

Usage (from the project root):
    python scripts/run_kmeans_init.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from njw_spectral import KMeansConfig, njw_spectral_clustering, sigma_knn_median  # noqa: E402
from njw_spectral.datasets import make_dataset, make_three_circles  # noqa: E402
from njw_spectral.paper_like_data import make_nips, make_squiggles, make_threecircles_joined  # noqa: E402
from njw_spectral.plotting import save_figure  # noqa: E402
from run_paper_like import choose_sigma  # noqa: E402

DATA_SEED = 0
SIGMA_NEIGHBORS = 7
INIT_SEEDS = list(range(30))
# A run "reaches the best optimum" if its distortion is within this relative
# tolerance of the smallest distortion seen for that task over all runs.
BEST_RTOL = 1e-6

SCHEMES = [
    ("k-means++ x10\n(our default)", dict(init="k-means++", n_init=10)),
    ("k-means++ x1", dict(init="k-means++", n_init=1)),
    ("random x1", dict(init="random", n_init=1)),
    ("90° x1\n(paper fn. 3)", dict(init="orthogonal", n_init=1)),
]

FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"


def build_tasks() -> list[dict]:
    """(title, Y, y_true, k, sigma rule). Y uses the project-default K-means only to finish NJW."""
    tasks = []

    def add(key, title, pts, y, k, sigma, rule):
        res = njw_spectral_clustering(pts, k, sigma)
        mask = y >= 0
        tasks.append({"key": key, "title": title, "Y": res.Y, "y": y, "mask": mask, "k": k,
                      "sigma": sigma, "sigma_rule": rule, "n": pts.shape[0]})

    for name in ["circles", "moons", "blobs"]:
        pts, y, k = make_dataset(name, seed=DATA_SEED)
        add(name, f"{name} (k={k})", pts, y, k, sigma_knn_median(pts, SIGMA_NEIGHBORS), "sigma0")

    pts, y, k = make_three_circles(DATA_SEED)
    s0 = sigma_knn_median(pts, SIGMA_NEIGHBORS)
    add("three_rings_s0", "three rings (k=3), σ0", pts, y, k, s0, "sigma0")
    add("three_rings_0.7s0", "three rings (k=3), 0.7σ0", pts, y, k, 0.7 * s0, "0.7*sigma0")

    for key, k, builder in [("threecircles_joined", 3, make_threecircles_joined),
                            ("squiggles", 4, make_squiggles), ("nips", 8, make_nips)]:
        pts, y, _, _ = builder(DATA_SEED)
        s0 = float(sigma_knn_median(pts, n_neighbors=SIGMA_NEIGHBORS))
        best, _ = choose_sigma(pts, k, s0, KMeansConfig())
        add(key, f"{key} (k={k})", pts, y, k, best[1], "min Y distortion")
    return tasks


def run(tasks: list[dict]) -> list[dict]:
    rows = []
    for t in tasks:
        task_rows = []
        for label, kw in SCHEMES:
            for seed in INIT_SEEDS:
                km = KMeansConfig(random_state=seed, **kw).build(t["k"], t["Y"]).fit(t["Y"])
                task_rows.append({
                    "task": t["key"],
                    "n": t["n"],
                    "k": t["k"],
                    "sigma": t["sigma"],
                    "sigma_rule": t["sigma_rule"],
                    "scheme": label.replace("\n", " "),
                    "init_seed": seed,
                    "ARI": adjusted_rand_score(t["y"][t["mask"]], km.labels_[t["mask"]]),
                    "Y_distortion": float(km.inertia_),
                    "n_iter": int(km.n_iter_),
                    "data_seed": DATA_SEED,
                })
        best = min(r["Y_distortion"] for r in task_rows)
        for r in task_rows:
            r["excess_distortion"] = r["Y_distortion"] / best - 1.0 if best > 0 else 0.0
            r["reached_best"] = bool(r["Y_distortion"] <= best * (1 + BEST_RTOL) + 1e-12)
        rows += task_rows
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    for task in tasks:
        for label, _ in SCHEMES:
            sch = label.replace("\n", " ")
            sub = [r for r in rows if r["task"] == task and r["scheme"] == sch]
            ari = np.array([r["ARI"] for r in sub])
            out.append({
                "task": task,
                "scheme": sch,
                "runs": len(sub),
                "ARI_mean": float(ari.mean()),
                "ARI_std": float(ari.std(ddof=1)),
                "ARI_min": float(ari.min()),
                "ARI_max": float(ari.max()),
                "frac_reached_best_distortion": float(np.mean([r["reached_best"] for r in sub])),
                "median_n_iter": float(np.median([r["n_iter"] for r in sub])),
            })
    return out


def plot(tasks: list[dict], rows: list[dict], stem: Path) -> list[Path]:
    ncol = 4
    nrow = int(np.ceil(len(tasks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.5 * nrow), sharey=True)
    rng = np.random.default_rng(0)
    for ax, t in zip(axes.ravel(), tasks):
        for i, (label, _) in enumerate(SCHEMES):
            sch = label.replace("\n", " ")
            sub = [r for r in rows if r["task"] == t["key"] and r["scheme"] == sch]
            ari = np.array([r["ARI"] for r in sub])
            hit = np.array([r["reached_best"] for r in sub])
            x = i + rng.uniform(-0.18, 0.18, size=ari.size)
            ax.scatter(x[hit], ari[hit], s=12, color=f"C{i}", alpha=0.75, lw=0)
            ax.scatter(x[~hit], ari[~hit], s=16, facecolors="none", edgecolors=f"C{i}", lw=0.8)
            ax.plot([i - 0.3, i + 0.3], [ari.mean()] * 2, color="k", lw=1.5)
            ax.text(i, 0.33, f"best {hit.sum()}/{hit.size}", ha="center", fontsize=7, color="0.3")
        ax.set_xticks(range(len(SCHEMES)))
        ax.set_xticklabels([s[0] for s in SCHEMES], fontsize=6.5)
        ax.set_ylim(0.28, 1.05)
        ax.set_title(f"{t['title']}\nσ = {t['sigma']:.3g} ({t['sigma_rule']})", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    for ax in axes.ravel()[len(tasks):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("ARI")
    fig.suptitle(
        f"K-means step on the same Y, {len(INIT_SEEDS)} init seeds per scheme (black bar = mean). "
        "Filled = reached the lowest Y distortion seen for that task; hollow = worse local optimum.",
        fontsize=10,
    )
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
    tasks = build_tasks()
    rows = run(tasks)
    summary = summarise(rows)
    write_csv(rows, TAB_DIR / "kmeans_init_runs.csv")
    write_csv(summary, TAB_DIR / "kmeans_init_summary.csv")
    outputs = plot(tasks, rows, FIG_DIR / "kmeans_init_comparison")

    print(f"{'task':<20} {'scheme':<28} {'ARI mean':>8} {'std':>6} {'min':>6} {'best%':>6}")
    for s in summary:
        print(f"{s['task']:<20} {s['scheme']:<28} {s['ARI_mean']:>8.3f} {s['ARI_std']:>6.3f} "
              f"{s['ARI_min']:>6.3f} {100 * s['frac_reached_best_distortion']:>5.0f}%")
    for name in ["kmeans_init_runs.csv", "kmeans_init_summary.csv"]:
        print(f"Wrote results/tables/{name}")
    for p in outputs:
        print(f"Wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
