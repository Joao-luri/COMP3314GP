"""Baselines discussed in Ng, Jordan & Weiss (NIPS 2001), Section 4.

NJW itself stays in algorithm.py. None of these call sklearn SpectralClustering.

Meila & Shi is specified in the NJW paper: row-normalize A to sum to 1, use those
eigenvectors, and do not row-normalize them before K-means.

Kannan et al. is NOT fully specified in NJW. kannan_style_max_abs is only our
reading of the sentence "identifies clusters with individual singular vectors"
on a row-stochastic affinity. Do not describe it as their original code.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform

from .algorithm import (
    KMeansConfig,
    gaussian_affinity,
    normalized_affinity,
    top_eigenvectors,
)


def meila_shi_clustering(points, k: int, sigma: float, kmeans_config: KMeansConfig | None = None):
    """Meila & Shi as described by NJW: eigenvectors of P = D^{-1} A, no row-norm.

    Eigenvectors of P are D^{-1/2} times eigenvectors of L = D^{-1/2} A D^{-1/2},
    with the same eigenvalues. We compute the symmetric problem, then rescale.
    """
    kmeans_config = kmeans_config or KMeansConfig()
    A = gaussian_affinity(points, sigma)
    degrees, L = normalized_affinity(A)
    eigenvalues, X_L, _ = top_eigenvectors(L, k)
    X = X_L / np.sqrt(degrees)[:, None]
    labels = kmeans_config.build(k).fit_predict(X)
    return labels, {"affinity": A, "degrees": degrees, "X": X, "eigenvalues": eigenvalues}


def kannan_style_max_abs(points, k: int, sigma: float):
    """Assign each point to the eigenvector of P with largest absolute coordinate.

    This matches NJW's high-level contrast with Kannan et al. (clusters tied to
    individual vectors) but is not a port of Kannan, Vempala & Vetta (2000).
    """
    A = gaussian_affinity(points, sigma)
    degrees, L = normalized_affinity(A)
    eigenvalues, X_L, _ = top_eigenvectors(L, k)
    X = X_L / np.sqrt(degrees)[:, None]
    labels = np.argmax(np.abs(X), axis=1)
    return labels, {"affinity": A, "X": X, "eigenvalues": eigenvalues}


def connected_components_for_k(points, k: int) -> tuple[np.ndarray, int, float, bool]:
    """Threshold graph: edge if distance <= T. Search T so there are k components.

    Larger T means more edges. The number of components is non-increasing in T.
    We take the largest T that yields exactly k components. If no such T exists,
    we return the T whose component count is closest to k (ties: larger T).

    Returns labels, n_components, threshold, exact_k.
    """
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    dist = squareform(pdist(points))
    vals = np.unique(np.round(dist[np.triu_indices(n, k=1)], decimals=6))
    vals = vals[vals > 0]
    if vals.size > 500:
        vals = vals[np.linspace(0, vals.size - 1, 500).astype(int)]

    def components(threshold: float):
        adj = (dist > 0) & (dist <= threshold)
        return connected_components(adj, directed=False)

    # Largest T with n_components == k, via scan from large T downward is fine
    # at this grid size; binary search would miss plateaus less clearly.
    exact: tuple[float, np.ndarray] | None = None
    closest: tuple[tuple, float, int, np.ndarray] | None = None
    for threshold in vals:
        n_comp, labels = components(float(threshold))
        if n_comp == k:
            exact = (float(threshold), labels)  # later (larger) T overwrites
        rank = (abs(int(n_comp) - k), -float(threshold))
        if closest is None or rank < closest[0]:
            closest = (rank, float(threshold), int(n_comp), labels)
    if exact is not None:
        return exact[1], k, exact[0], True
    assert closest is not None
    return closest[3], closest[2], closest[1], False
