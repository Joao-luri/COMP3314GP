"""NJW spectral clustering (Ng, Jordan & Weiss, NIPS 2001), implemented from scratch.

Only the eigensolver (scipy.linalg.eigh) and the final K-means step
(sklearn.cluster.KMeans) are delegated to libraries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# A row of X with norm below this is treated as numerically zero: it has no
# direction on the unit sphere, so row normalization is undefined for it.
# This threshold is only used to detect that failure, never to regularize.
DEFAULT_MIN_ROW_NORM = 1e-10

# Tolerance for accepting a user-supplied affinity matrix as symmetric.
SYMMETRY_ATOL = 1e-12


ORTHOGONAL_INIT = "orthogonal"


def orthogonal_init(Y: np.ndarray, k: int, random_state: int = 0) -> np.ndarray:
    """Paper footnote 3: k initial centres taken from the rows of Y.

    The first centre is a uniformly random row. Each further centre is the row
    whose largest cosine to the centres chosen so far is smallest, i.e. the row
    closest to 90 degrees from all of them in the worst case.
    """
    Y = np.asarray(Y, dtype=float)
    n = Y.shape[0]
    k = _validate_k(k, n)
    U = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    rng = np.random.default_rng(random_state)
    chosen = [int(rng.integers(n))]
    max_cos = U @ U[chosen[0]]
    for _ in range(1, k):
        score = max_cos.copy()
        score[chosen] = np.inf
        nxt = int(np.argmin(score))
        chosen.append(nxt)
        max_cos = np.maximum(max_cos, U @ U[nxt])
    return Y[chosen].copy()


@dataclass(frozen=True)
class KMeansConfig:
    """Settings for every K-means call (baseline and NJW step 5).

    init may be any sklearn init string, or "orthogonal" for the paper's
    footnote-3 initialization (needs the data in build; always one run).
    """

    init: str = "k-means++"
    n_init: int = 10
    max_iter: int = 300
    random_state: int = 0

    def build(self, k: int, data: np.ndarray | None = None) -> KMeans:
        if self.init == ORTHOGONAL_INIT:
            if data is None:
                raise ValueError("orthogonal init needs the data to pick rows from")
            return KMeans(
                n_clusters=k,
                init=orthogonal_init(data, k, self.random_state),
                n_init=1,
                max_iter=self.max_iter,
            )
        return KMeans(
            n_clusters=k,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )


@dataclass
class NJWResult:
    """All intermediate quantities of one NJW run (n points, k clusters)."""

    labels: np.ndarray  # (n,)
    affinity: np.ndarray  # A, (n, n)
    degrees: np.ndarray  # d, (n,)
    L: np.ndarray  # D^{-1/2} A D^{-1/2}, (n, n)
    eigenvalues: np.ndarray  # top-k eigenvalues of L, descending, (k,)
    X: np.ndarray  # top-k eigenvectors as columns, (n, k)
    Y: np.ndarray  # row-normalized X (or X itself if disabled), (n, k)
    row_norms: np.ndarray  # ||X_i||, (n,)
    next_eigenvalue: float | None  # lambda_{k+1}, None if k == n
    kmeans_inertia: float  # K-means distortion on the rows of Y
    k: int
    sigma: float | None  # None if a precomputed affinity was given
    row_normalize: bool
    kmeans_config: KMeansConfig = field(default_factory=KMeansConfig)

    @property
    def eigengap(self) -> float | None:
        """lambda_k - lambda_{k+1}; the quantity Theorem 2 needs to be large."""
        if self.next_eigenvalue is None:
            return None
        return float(self.eigenvalues[-1] - self.next_eigenvalue)


# ---------------------------------------------------------------- validation


def _validate_points(points) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError(f"points must be a 2-D array of shape (n, d), got shape {points.shape}")
    if points.shape[0] < 2:
        raise ValueError("need at least 2 points")
    if not np.all(np.isfinite(points)):
        raise ValueError("points contain NaN or infinite values")
    return points


def _validate_sigma(sigma) -> float:
    try:
        sigma = float(sigma)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sigma must be a positive finite number, got {sigma!r}") from exc
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma must be a positive finite number, got {sigma!r}")
    return sigma


def _validate_k(k, n: int) -> int:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError(f"k must be an integer, got {k!r}")
    k = int(k)
    if not 1 <= k <= n:
        raise ValueError(f"k must satisfy 1 <= k <= n = {n}, got k = {k}")
    return k


def _validate_affinity(A) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"affinity must be a square matrix, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("affinity contains NaN or infinite values")
    if np.any(A < 0):
        raise ValueError("affinity must be non-negative")
    if np.any(np.diag(A) != 0):
        raise ValueError("NJW requires A_ii = 0 (zero diagonal)")
    if not np.allclose(A, A.T, rtol=0, atol=SYMMETRY_ATOL):
        raise ValueError("affinity must be symmetric")
    return A


# --------------------------------------------------------------- algorithm steps


def gaussian_affinity(points, sigma: float) -> np.ndarray:
    """Step 1: A_ij = exp(-||x_i - x_j||^2 / (2 sigma^2)) for i != j, A_ii = 0."""
    points = _validate_points(points)
    sigma = _validate_sigma(sigma)
    sq_dists = squareform(pdist(points, metric="sqeuclidean"))
    A = np.exp(-sq_dists / (2.0 * sigma**2))
    np.fill_diagonal(A, 0.0)
    return A


def normalized_affinity(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Step 2: degrees d_i = sum_j A_ij and L = D^{-1/2} A D^{-1/2}."""
    A = _validate_affinity(A)
    degrees = A.sum(axis=1)
    n_isolated = int(np.sum(degrees <= 0))
    if n_isolated:
        raise ValueError(
            f"{n_isolated} point(s) have zero degree, i.e. they are isolated in the "
            "affinity graph and D^{-1/2} is undefined. With a Gaussian affinity this "
            "means sigma is too small relative to the nearest-neighbour distances "
            "(exp underflows to 0); choose a larger sigma."
        )
    inv_sqrt_d = 1.0 / np.sqrt(degrees)
    L = inv_sqrt_d[:, None] * A * inv_sqrt_d[None, :]
    # Exact symmetry in exact arithmetic; symmetrize to remove round-off only.
    L = 0.5 * (L + L.T)
    if not np.all(np.isfinite(L)):
        raise ValueError("L contains non-finite values (degrees too close to zero)")
    return degrees, L


def top_eigenvectors(L: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Step 3: k algebraically largest eigenpairs of symmetric L, descending.

    Also returns lambda_{k+1} (or None if k == n) for eigengap diagnostics.
    eigh returns orthonormal eigenvectors, also within repeated eigenvalues.
    """
    n = L.shape[0]
    k = _validate_k(k, n)
    m = min(k + 1, n)
    vals, vecs = eigh(L, subset_by_index=[n - m, n - 1])
    order = np.argsort(vals)[::-1]  # algebraic order, not absolute value
    vals, vecs = vals[order], vecs[:, order]
    if not (np.all(np.isfinite(vals)) and np.all(np.isfinite(vecs))):
        raise ValueError("eigensolver returned non-finite values")
    next_val = float(vals[k]) if m > k else None
    return vals[:k], vecs[:, :k], next_val


def normalize_rows(X: np.ndarray, min_row_norm: float = DEFAULT_MIN_ROW_NORM):
    """Step 4: Y_ij = X_ij / ||X_i||. Raises if any row is numerically zero."""
    norms = np.linalg.norm(X, axis=1)
    bad = np.flatnonzero(norms < min_row_norm)
    if bad.size:
        raise ValueError(
            f"{bad.size} row(s) of X have norm < {min_row_norm:g} (e.g. rows "
            f"{bad[:5].tolist()}); these points have (numerically) no component in "
            "the top-k eigenspace, so they cannot be projected onto the unit sphere."
        )
    return X / norms[:, None], norms


def njw_from_affinity(
    A: np.ndarray,
    k: int,
    *,
    row_normalize: bool = True,
    kmeans_config: KMeansConfig | None = None,
    min_row_norm: float = DEFAULT_MIN_ROW_NORM,
    sigma: float | None = None,
) -> NJWResult:
    """Steps 2-6 of NJW on a given affinity matrix (useful for ideal-case tests)."""
    kmeans_config = kmeans_config or KMeansConfig()
    degrees, L = normalized_affinity(A)
    k = _validate_k(k, L.shape[0])
    eigenvalues, X, next_val = top_eigenvectors(L, k)

    if row_normalize:
        Y, row_norms = normalize_rows(X, min_row_norm)
    else:
        Y, row_norms = X.copy(), np.linalg.norm(X, axis=1)

    # Steps 5-6: K-means on the rows of Y; row i's label is point i's label.
    km = kmeans_config.build(k, Y).fit(Y)
    return NJWResult(
        labels=km.labels_,
        affinity=np.asarray(A, dtype=float),
        degrees=degrees,
        L=L,
        eigenvalues=eigenvalues,
        X=X,
        Y=Y,
        row_norms=row_norms,
        next_eigenvalue=next_val,
        kmeans_inertia=float(km.inertia_),
        k=k,
        sigma=sigma,
        row_normalize=row_normalize,
        kmeans_config=kmeans_config,
    )


def njw_spectral_clustering(
    points,
    k: int,
    sigma: float,
    *,
    row_normalize: bool = True,
    kmeans_config: KMeansConfig | None = None,
    min_row_norm: float = DEFAULT_MIN_ROW_NORM,
) -> NJWResult:
    """Full NJW algorithm on raw points of shape (n, d)."""
    points = _validate_points(points)
    _validate_k(k, points.shape[0])
    A = gaussian_affinity(points, sigma)
    return njw_from_affinity(
        A,
        k,
        row_normalize=row_normalize,
        kmeans_config=kmeans_config,
        min_row_norm=min_row_norm,
        sigma=float(sigma),
    )


# ------------------------------------------------------------ sigma heuristic


def sigma_knn_median(points, n_neighbors: int = 7) -> float:
    """Label-free heuristic: median distance from each point to its n-th neighbour.

    This is NOT the paper's sigma selection (the paper searches sigma^2 for the
    smallest K-means distortion of Y); it is a simple, declared default.
    """
    points = _validate_points(points)
    if not 1 <= n_neighbors < points.shape[0]:
        raise ValueError("n_neighbors must be in [1, n-1]")
    # +1 because each point is its own nearest neighbour at distance 0.
    dist, _ = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(points).kneighbors(points)
    return _validate_sigma(np.median(dist[:, n_neighbors]))
