"""Scaling of our dense NJW implementation: wall time per step and peak memory vs n.

Concentric circles (stage-1 generator, same noise/factor) at growing n. Each n
runs in a fresh child process so its peak RSS is not polluted by earlier sizes.
Times are the minimum over repeats. Stage-1 defaults and results are unchanged.

Usage (from the project root):
    python scripts/run_scaling.py            # n up to 6000
    python scripts/run_scaling.py --max-n 8000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

N_GRID = [250, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000]
FULL_EIGH_MAX_N = 4000  # full spectrum only as a reference point; it is not used by NJW
DATA_SEED = 0
KMEANS_SEED = 0
SIGMA_NEIGHBORS = 7
FIT_MIN_N = 1000  # log-log slopes are fitted on n >= this (small n is overhead-dominated)
STEPS = ["sigma", "affinity", "normalize", "eigh_topk", "kmeans"]
STEP_LABELS = {
    "sigma": "sigma heuristic (7-NN)",
    "affinity": "step 1: A (pairwise dists + exp)",
    "normalize": "step 2: L = D^-1/2 A D^-1/2",
    "eigh_topk": "step 3: top-(k+1) eigh of L",
    "kmeans": "steps 4-5: row norm + K-means",
    "eigh_full": "reference: full eigh of L",
}

FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def child(n: int, repeats: int) -> dict:
    from scipy.linalg import eigh
    from sklearn.datasets import make_circles
    from sklearn.metrics import adjusted_rand_score

    from njw_spectral import KMeansConfig, gaussian_affinity, normalize_rows, normalized_affinity
    from njw_spectral import sigma_knn_median, top_eigenvectors

    pts, y = make_circles(n_samples=n, noise=0.05, factor=0.5, random_state=DATA_SEED)
    k = 2
    km_cfg = KMeansConfig(random_state=KMEANS_SEED)
    rss_before = _peak_rss_bytes()

    best = {s: np.inf for s in STEPS}
    for rep in range(repeats):
        t = {}
        t0 = time.perf_counter()
        sigma = sigma_knn_median(pts, n_neighbors=SIGMA_NEIGHBORS)
        t["sigma"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        A = gaussian_affinity(pts, sigma)
        t["affinity"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        _, L = normalized_affinity(A)
        t["normalize"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        vals, X, next_val = top_eigenvectors(L, k)
        t["eigh_topk"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        Y, _ = normalize_rows(X)
        labels = km_cfg.build(k, Y).fit_predict(Y)
        t["kmeans"] = time.perf_counter() - t0
        best = {s: min(best[s], t[s]) for s in STEPS}
        if rep < repeats - 1:
            del A, L, X, Y
    rss_pipeline = _peak_rss_bytes()

    out = {
        "n": n,
        "k": k,
        "repeats": repeats,
        "sigma": float(sigma),
        "ARI": float(adjusted_rand_score(y, labels)),
        "eigengap": float(vals[-1] - next_val),
        **{f"t_{s}": float(best[s]) for s in STEPS},
        "t_total": float(sum(best.values())),
        "peak_rss_before_MB": rss_before / 2**20,
        "peak_rss_pipeline_MB": rss_pipeline / 2**20,
        "pipeline_extra_MB": (rss_pipeline - rss_before) / 2**20,
        "dense_matrix_MB": n * n * 8 / 2**20,
        "t_eigh_full": "",
    }
    if n <= FULL_EIGH_MAX_N:
        t0 = time.perf_counter()
        eigh(L)
        out["t_eigh_full"] = float(time.perf_counter() - t0)
    return out


def _environment() -> dict:
    env = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import scipy
        import sklearn
        from threadpoolctl import threadpool_info

        env.update(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)
        env["blas"] = [
            {key: info.get(key) for key in ("internal_api", "num_threads", "version")}
            for info in threadpool_info() if info.get("user_api") == "blas"
        ]
    except ImportError:
        pass
    return env


def fit_slope(ns, ts) -> float:
    ns, ts = np.asarray(ns, float), np.asarray(ts, float)
    m = ns >= FIT_MIN_N
    return float(np.polyfit(np.log(ns[m]), np.log(ts[m]), 1)[0])


def plot(rows: list[dict], slopes: dict, stem: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from njw_spectral.plotting import save_figure

    ns = np.array([r["n"] for r in rows], float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    for i, s in enumerate(STEPS):
        ax.loglog(ns, [r[f"t_{s}"] for r in rows], "o-", color=f"C{i}", ms=4,
                  label=f"{STEP_LABELS[s]}  (slope {slopes[s]:.2f})")
    full = [(r["n"], r["t_eigh_full"]) for r in rows if r["t_eigh_full"] != ""]
    if full:
        fx, fy = zip(*full)
        ax.loglog(fx, fy, "s--", color="0.45", ms=4,
                  label=f"{STEP_LABELS['eigh_full']}  (slope {slopes['eigh_full']:.2f})")
    anchor_n, anchor_t = ns[-1], rows[-1]["t_eigh_topk"]
    ref = np.array([ns[0], ns[-1]])
    for p, ls in [(2, ":"), (3, "-.")]:
        ax.loglog(ref, anchor_t * (ref / anchor_n) ** p, ls, color="k", lw=0.9, alpha=0.6,
                  label=f"reference $\\propto n^{p}$")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set(xlabel="n (points)", ylabel="wall time [s] (min over repeats)",
           title="Time per NJW step (dense implementation)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    ax.loglog(ns, [r["pipeline_extra_MB"] for r in rows], "o-", color="C3",
              label="measured: peak RSS increase during one NJW run")
    n_ext = np.geomspace(ns[0], 1e5, 50)
    one = n_ext**2 * 8 / 2**20
    c = np.median([r["pipeline_extra_MB"] / r["dense_matrix_MB"] for r in rows if r["n"] >= FIT_MIN_N])
    ax.loglog(n_ext, one, "--", color="0.4", label=r"one dense $n\times n$ float64 matrix ($8n^2$ bytes)")
    ax.loglog(n_ext, c * one, ":", color="C3", label=f"extrapolated: {c:.1f} x one matrix")
    for gb in (8, 16):
        ax.axhline(gb * 1024, color="0.6", lw=0.8)
        ax.text(n_ext[0] * 1.1, gb * 1024 * 1.12, f"{gb} GB laptop RAM", fontsize=7, color="0.35")
    for n_cross in (8, 16):
        n_star = np.sqrt(n_cross * 2**30 / (8 * c))
        ax.plot([n_star], [n_cross * 1024], "x", color="C3")
        ax.annotate(f"n ≈ {n_star:,.0f}", (n_star, n_cross * 1024), textcoords="offset points",
                    xytext=(-58, 4), fontsize=7, color="C3")
    ax.set(xlabel="n (points)", ylabel="memory [MB]",
           title=r"Memory: dense $A$ and $L$ grow as $n^2$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Concentric circles, k = 2, stage-1 sigma rule; each n in a fresh process "
                 f"(slopes fitted for n >= {FIT_MIN_N})", fontsize=10.5)
    fig.tight_layout()
    return save_figure(fig, stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-n", type=int, default=6000)
    parser.add_argument("--replot", action="store_true", help="redraw the figure from the saved CSV/JSON")
    parser.add_argument("--child", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--repeats", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child is not None:
        print(json.dumps(child(args.child, args.repeats)))
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    if args.replot:
        with (TAB_DIR / "scaling.csv").open() as f:
            rows = [{k: (float(v) if v != "" else "") for k, v in r.items()} for r in csv.DictReader(f)]
        slopes = json.loads((TAB_DIR / "scaling_meta.json").read_text())["loglog_slopes"]
        for p in plot(rows, slopes, FIG_DIR / "scaling_time_memory"):
            print(f"Wrote {p.relative_to(ROOT)}")
        return
    rows = []
    for n in [n for n in N_GRID if n <= args.max_n]:
        repeats = 3 if n <= 2000 else 1
        proc = subprocess.run([sys.executable, __file__, "--child", str(n), "--repeats", str(repeats)],
                              check=True, capture_output=True, text=True)
        row = json.loads(proc.stdout.strip().splitlines()[-1])
        rows.append(row)
        print(f"n={n:>5}  total={row['t_total']:7.2f}s  eigh_topk={row['t_eigh_topk']:7.2f}s  "
              f"extra RSS={row['pipeline_extra_MB']:7.0f} MB  ({row['pipeline_extra_MB'] / row['dense_matrix_MB']:.1f}"
              f" x n^2*8)  ARI={row['ARI']:.3f}", flush=True)

    slopes = {s: fit_slope([r["n"] for r in rows], [r[f"t_{s}"] for r in rows]) for s in STEPS}
    full = [(r["n"], r["t_eigh_full"]) for r in rows if r["t_eigh_full"] != ""]
    slopes["eigh_full"] = fit_slope(*zip(*full)) if full else float("nan")
    slopes["memory"] = fit_slope([r["n"] for r in rows], [r["pipeline_extra_MB"] for r in rows])

    csv_path = TAB_DIR / "scaling.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    meta_path = TAB_DIR / "scaling_meta.json"
    meta_path.write_text(json.dumps({
        "dataset": {"generator": "make_circles", "noise": 0.05, "factor": 0.5, "data_seed": DATA_SEED},
        "k": 2,
        "sigma_rule": f"median distance to {SIGMA_NEIGHBORS}-th nearest neighbour",
        "kmeans": "k-means++, n_init=10, random_state=0",
        "eigensolver": "scipy.linalg.eigh(subset_by_index=top k+1)",
        "fit_min_n": FIT_MIN_N,
        "loglog_slopes": slopes,
        "environment": _environment(),
    }, indent=2))
    outputs = plot(rows, slopes, FIG_DIR / "scaling_time_memory")

    print("\nlog-log slopes (n >= %d): " % FIT_MIN_N + ", ".join(f"{s}={v:.2f}" for s, v in slopes.items()))
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {meta_path.relative_to(ROOT)}")
    for p in outputs:
        print(f"Wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
