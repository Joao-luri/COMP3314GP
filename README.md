# NJW Spectral Clustering — COMP3314 Group Project

**中文详细说明（目录、每个文件做什么）：[README_zh.md](README_zh.md)**

A small, from-scratch implementation and experimental study of

> Andrew Y. Ng, Michael I. Jordan, Yair Weiss.
> **On Spectral Clustering: Analysis and an Algorithm.** NIPS 2001.
> [Official PDF](https://proceedings.neurips.cc/paper_files/paper/2001/file/801272ee79cfde7fa5960571fee36b9b-Paper.pdf)

## Scope (stage 1)

- Our own implementation of the NJW algorithm (Section 2 of the paper).
- K-means on raw coordinates vs. NJW on three freshly generated 2-D datasets
  (concentric circles, two moons, isotropic Gaussian blobs).
- A step-by-step figure of the pipeline on concentric circles
  (raw points, rows of X, rows of Y, final clustering).
- Unit tests for the matrix properties and the paper's "ideal case" (Proposition 1).

The datasets are our own experiments in a *similar setting* to the paper's
Figure 1. They are **not** the paper's data, and this is **not** an exact
reproduction of Figure 1. The paper reports no ARI; all ARI/NMI values here
are our own measurements.

Planned later stages (sigma sensitivity, row-normalization ablation, repeated
seeds, report/slides) are described in [docs/next_stages_zh.md](docs/next_stages_zh.md).

## What we implemented vs. what we call from libraries

| Implemented by us (`src/njw_spectral/algorithm.py`) | Library calls |
|---|---|
| Gaussian affinity `A` with zero diagonal | `scipy.spatial.distance.pdist` (squared distances) |
| Degrees `d`, `L = D^{-1/2} A D^{-1/2}` | `scipy.linalg.eigh` (symmetric eigensolver) |
| Selecting the k algebraically largest eigenvectors | `sklearn.cluster.KMeans` (step 5) |
| Row normalization `X -> Y` (switchable) | `sklearn.metrics` (ARI, NMI) |
| Input validation (sigma, k, zero degree, zero rows, NaN/inf) | `sklearn.neighbors.NearestNeighbors` (sigma heuristic) |
| | `sklearn.datasets` (data generation), Matplotlib |

`sklearn.cluster.SpectralClustering` is **not** used anywhere.

## Installation

Python 3.10+ (tested with Python 3.14.3, NumPy 2.5.4, SciPy 1.18.1,
scikit-learn 1.9.1, Matplotlib 3.11.2 on macOS, CPU only).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python -m pytest tests -q                    # 25 sanity checks
python scripts/run_stage1.py                 # experiment + figures + tables (~20 s on a laptop CPU)
python scripts/check_paper_assumptions.py    # Assumptions A1-A4 / Theorem 2 on the same runs
python scripts/run_sigma_sensitivity.py      # relative sigma grid; does not change stage-1 defaults
python scripts/run_wrong_k_three_circles.py  # three rings with k=2 vs k=3 (our data)
python scripts/run_noise_bridge.py           # noise bridge vs NJW/CC; row-norm ablation
python scripts/run_paper_like.py             # Figure-1-style data (ours) + baselines
```

If joblib warns that it cannot find the number of physical cores, it is
harmless; set `LOKY_MAX_CPU_COUNT=4` to silence it.

## Outputs

| File | Content |
|---|---|
| `results/tables/stage1_results.csv` | ARI, NMI, sigma, eigenvalues, eigengap, seeds, all parameters |
| `results/tables/stage1_config.json` | Full configuration and library versions |
| `results/figures/stage1_comparison.{png,pdf}` | Ground truth / K-means / NJW for the three datasets |
| `results/figures/stage1_circles_pipeline.{png,pdf}` | Raw points, rows of X, rows of Y, final clustering (circles) |
| `results/tables/stage1_paper_assumptions.csv` | delta, eps1, eps2, C, Theorem 2 condition and bound vs. observed tightness |
| `results/tables/sigma_sensitivity.csv` | ARI / eigengap / Y distortion / failures on relative sigma grid |
| `results/figures/sigma_sensitivity_*.{png,pdf}` | ARI curves, aux curves, circles case study |
| `results/tables/wrong_k_three_circles*.csv` | three rings, k=2 vs k=3, merge scores, k=3 sigma probe |
| `results/figures/wrong_k_three_circles*.{png,pdf}` | wrong-k comparison and Y embedding |
| `results/tables/noise_bridge_and_density.csv` | bridge / gap-noise / row-norm results |
| `results/figures/noise_bridge_*.{png,pdf}` | moons/blobs noise-bridge figures |
| `results/figures/density_imbalance_row_norm.{png,pdf}` | row-normalization ablation |
| `results/tables/paper_like_results.csv` | NJW and baselines on reconstructed Figure-1 settings |
| `results/tables/paper_like_sigma_search.csv` | distortion search that chose sigma |
| `results/figures/paper_like_njw.{png,pdf}` | NJW gallery on those settings |
| `results/figures/paper_like_baselines.{png,pdf}` | K-means, Meila-Shi, CC, Kannan-style |

Documentation (Chinese): `docs/learning_notes_zh.md`, `docs/paper_notes_zh.md`,
`docs/sigma_sensitivity_zh.md`, `docs/wrong_k_three_circles_zh.md`,
`docs/noise_bridge_zh.md`, `docs/paper_like_experiments_zh.md`,
`docs/next_stages_zh.md`.

## Experimental setup (stage 1)

- Data: `make_circles(n=400, noise=0.05, factor=0.5)`, `make_moons(n=400, noise=0.05)`,
  `make_blobs(n=450, centers=(-5,0),(0,5),(5,0), cluster_std=1.0)`, all with data seed 0.
- No feature scaling for either method (both see identical raw coordinates and the same k).
- K-means (baseline and NJW step 5): `init="k-means++"`, `n_init=10`, `max_iter=300`, `random_state=0`.
  The paper instead used a single run initialized with centroids ~90 degrees apart (footnote 3).
- sigma: median distance to the 7th nearest neighbour, computed from the points only.
  This is a declared heuristic chosen before running, not the paper's method
  (the paper searches sigma^2 for the smallest K-means distortion of Y), and
  not tuned on ground-truth labels.
- Ground-truth labels are used only to set k and for evaluation/plot colours.

## Stage-1 results (our measurements, single seed)

| Dataset | K-means ARI | NJW ARI | sigma | lambda_k - lambda_{k+1} |
|---|---|---|---|---|
| Concentric circles | -0.002 | 1.000 | 0.105 | 0.0051 |
| Two moons | 0.274 | 1.000 | 0.084 | 0.0036 |
| Gaussian blobs | 1.000 | 1.000 | 0.440 | 0.0879 |

ARI (Adjusted Rand Index) compares two partitions and is invariant to
permuting cluster ids; 1 means identical partitions, about 0 means chance-level agreement.
In the figures, predicted cluster ids are re-mapped to ground-truth colours by
Hungarian matching *for display only*; ARI is unaffected.

### Checking the paper's theory on these runs

Using ground-truth clusters as the S_i in Section 3.2 (analysis only), the
Theorem 2 condition delta > (2 + sqrt 2) eps **fails** on circles and moons,
although NJW clusters both perfectly. It holds on blobs, where the bound
(1.5e-4) exceeds the observed value (3.6e-14) by about ten orders of magnitude.
In all three cases lambda_{k+1}(L) almost equals the largest within-cluster
lambda_2, as the paper's ideal-case analysis predicts.

## Current limitations

- One data seed and one K-means seed; no variance estimates yet.
- sigma comes from one heuristic; sensitivity to sigma is not studied yet.
- Easy, low-noise 2-D toy data only; the small eigengaps on circles/moons suggest
  results may be fragile, which we have not tested yet.
- The paper's sigma search and 90-degree K-means initialization are not implemented.
- Dense O(n^2) affinity and full eigensolver: fine for n of a few thousand, not beyond.
- The paper text was read from the official PDF (OCR-quality text layer;
  equations (4)-(8) and Figure 1 panel titles were checked on rendered page
  images). The paper gives no data, n, sigma values or quantitative metrics for
  Figure 1, so an exact reproduction is not possible from the paper alone.
