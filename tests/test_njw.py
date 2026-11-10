"""Sanity checks for the NJW implementation.

Eigenvectors are never compared column by column: their signs, and the basis
inside a repeated eigenvalue, are arbitrary. We compare clusterings, distances
between rows of Y, and projectors onto the eigen-subspace instead.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from njw_spectral import (  # noqa: E402
    KMeansConfig,
    gaussian_affinity,
    njw_from_affinity,
    njw_spectral_clustering,
    normalize_rows,
    normalized_affinity,
    orthogonal_init,
    sigma_knn_median,
    top_eigenvectors,
)
from njw_spectral.datasets import make_dataset  # noqa: E402
from njw_spectral.diagnostics import (  # noqa: E402
    mean_sq_distance_to_orthogonal_centers,
    paper_assumption_diagnostics,
)


@pytest.fixture(scope="module")
def circles_run():
    pts, y, k = make_dataset("circles", seed=0)
    sigma = sigma_knn_median(pts)
    return pts, y, k, njw_spectral_clustering(pts, k, sigma)


def test_affinity_symmetric_zero_diagonal(circles_run):
    A = circles_run[3].affinity
    assert np.array_equal(A, A.T)
    assert np.all(np.diag(A) == 0)
    assert np.all((A >= 0) & (A <= 1))


def test_L_symmetric_and_matches_definition(circles_run):
    res = circles_run[3]
    assert np.allclose(res.L, res.L.T, atol=1e-14)
    D_inv_sqrt = np.diag(1 / np.sqrt(res.affinity.sum(axis=1)))
    assert np.allclose(res.L, D_inv_sqrt @ res.affinity @ D_inv_sqrt, atol=1e-13)


def test_eigenvectors_orthonormal_small_residual_and_largest(circles_run):
    res = circles_run[3]
    X, lam = res.X, res.eigenvalues
    assert np.allclose(X.T @ X, np.eye(res.k), atol=1e-10)
    residual = np.linalg.norm(res.L @ X - X * lam, axis=0)
    assert residual.max() < 1e-10
    assert np.all(np.diff(lam) <= 0)  # descending
    all_vals = np.linalg.eigvalsh(res.L)
    assert np.allclose(lam, np.sort(all_vals)[::-1][: res.k], atol=1e-10)
    # Top eigenvalue of D^{-1/2} A D^{-1/2} is exactly 1 for a positive A.
    assert abs(lam[0] - 1) < 1e-10


def test_subspace_matches_full_eigensolver(circles_run):
    res = circles_run[3]
    vals, vecs = np.linalg.eigh(res.L)
    X_full = vecs[:, np.argsort(vals)[::-1][: res.k]]
    assert np.allclose(res.X @ res.X.T, X_full @ X_full.T, atol=1e-8)


def test_rows_unit_norm_after_normalization(circles_run):
    Y = circles_run[3].Y
    assert np.allclose(np.linalg.norm(Y, axis=1), 1.0, atol=1e-12)


def test_row_normalization_can_be_disabled(circles_run):
    pts, _, k, _ = circles_run
    res = njw_spectral_clustering(pts, k, sigma_knn_median(pts), row_normalize=False)
    assert np.array_equal(res.Y, res.X)


def _block_affinity(sizes, rng):
    n = sum(sizes)
    A = np.zeros((n, n))
    start = 0
    for s in sizes:
        B = rng.uniform(0.2, 1.0, size=(s, s))
        A[start:start + s, start:start + s] = (B + B.T) / 2
        start += s
    np.fill_diagonal(A, 0)
    y = np.repeat(np.arange(len(sizes)), sizes)
    perm = rng.permutation(n)
    return A[np.ix_(perm, perm)], y[perm]


def test_ideal_block_diagonal_recovers_groups():
    rng = np.random.default_rng(1)
    A, y = _block_affinity([30, 50, 40], rng)
    res = njw_from_affinity(A, k=3)
    assert adjusted_rand_score(y, res.labels) == 1.0
    assert np.allclose(res.eigenvalues, 1.0, atol=1e-10)
    assert res.eigengap > 0.1
    # Proposition 1: rows of Y coincide within a block, are orthogonal across.
    G = res.Y @ res.Y.T
    same = y[:, None] == y[None, :]
    assert np.allclose(G[same], 1.0, atol=1e-8)
    assert np.allclose(G[~same], 0.0, atol=1e-8)


def test_orthogonal_init_picks_one_row_per_block_in_ideal_case():
    rng = np.random.default_rng(1)
    A, y = _block_affinity([30, 50, 40], rng)
    Y = njw_from_affinity(A, k=3).Y
    for seed in range(5):
        centres = orthogonal_init(Y, 3, random_state=seed)
        rows = [int(np.flatnonzero(np.all(np.isclose(Y, c), axis=1))[0]) for c in centres]
        assert len(set(y[rows])) == 3
        assert np.allclose(centres @ centres.T, np.eye(3), atol=1e-8)
    res = njw_from_affinity(A, k=3, kmeans_config=KMeansConfig(init="orthogonal"))
    assert adjusted_rand_score(y, res.labels) == 1.0


def test_orthogonal_init_requires_data():
    with pytest.raises(ValueError):
        KMeansConfig(init="orthogonal").build(2)


def test_diagnostics_ideal_case():
    rng = np.random.default_rng(2)
    A, y = _block_affinity([30, 50, 40], rng)
    diag = paper_assumption_diagnostics(A, y)
    res = njw_from_affinity(A, k=3)
    assert diag["eps1"] == 0 and diag["eps2"] == 0
    assert diag["theorem2_condition_holds"]
    # Ideal case: lambda_{k+1}(L) = max_i lambda_2^(i) (Section 3.2).
    assert np.isclose(res.next_eigenvalue, max(diag["lambda2_per_cluster"]), atol=1e-10)
    assert mean_sq_distance_to_orthogonal_centers(res.Y, y) < 1e-12


def test_ari_invariant_to_label_permutation():
    y = np.array([0, 0, 1, 1, 2, 2])
    assert adjusted_rand_score(y, np.array([2, 2, 0, 0, 1, 1])) == 1.0


@pytest.mark.parametrize("sigma", [0, -1.0, np.nan, np.inf, "abc", None])
def test_invalid_sigma_rejected(sigma):
    with pytest.raises(ValueError, match="sigma"):
        gaussian_affinity(np.random.default_rng(0).normal(size=(10, 2)), sigma)


@pytest.mark.parametrize("k", [0, -1, 11, 2.0, True])
def test_invalid_k_rejected(k):
    with pytest.raises(ValueError, match="k must"):
        njw_spectral_clustering(np.random.default_rng(0).normal(size=(10, 2)), k, 1.0)


def test_non_finite_points_rejected():
    pts = np.random.default_rng(0).normal(size=(10, 2))
    pts[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        njw_spectral_clustering(pts, 2, 1.0)


def test_zero_degree_detected_for_tiny_sigma():
    pts = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 0.0], [10.0, 1.0]])
    with pytest.raises(ValueError, match="zero degree"):
        njw_spectral_clustering(pts, 2, sigma=0.01)


def test_invalid_affinity_rejected():
    A = np.array([[0.0, 1.0], [0.5, 0.0]])
    with pytest.raises(ValueError, match="symmetric"):
        normalized_affinity(A)
    with pytest.raises(ValueError, match="diagonal"):
        normalized_affinity(np.ones((3, 3)))


def test_zero_row_norm_detected():
    X = np.array([[1.0, 0.0], [0.0, 0.0], [0.6, 0.8]])
    with pytest.raises(ValueError, match="norm"):
        normalize_rows(X)


def test_top_eigenvectors_algebraic_not_absolute_order():
    # Eigenvalues 1, 0.5, -0.9: algebraic top-2 is {1, 0.5}, not {1, -0.9}.
    Q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))
    L = Q @ np.diag([1.0, 0.5, -0.9]) @ Q.T
    vals, _, nxt = top_eigenvectors((L + L.T) / 2, 2)
    assert np.allclose(vals, [1.0, 0.5])
    assert np.isclose(nxt, -0.9)
