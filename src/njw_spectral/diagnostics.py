"""Quantities from Section 3 of the paper (Assumptions A1-A4, Theorem 2).

These need a reference partition, so they use ground-truth labels. They are
analysis tools only and are never used for clustering or choosing sigma.
"""

from __future__ import annotations

import numpy as np

from .algorithm import normalized_affinity


def paper_assumption_diagnostics(A: np.ndarray, labels: np.ndarray) -> dict:
    """Evaluate delta, eps1, eps2, C, eps and the Theorem 2 condition.

    The "ideal" A_hat keeps only intra-cluster entries of A; d_hat are its degrees.
    """
    A = np.asarray(A, dtype=float)
    labels = np.asarray(labels)
    clusters = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    k = len(clusters)

    same = labels[:, None] == labels[None, :]
    A_hat = np.where(same, A, 0.0)
    d_hat = A_hat.sum(axis=1)

    # A1: lambda_2 of each block L_hat^(ii); delta = 1 - max_i lambda_2^(i).
    lambda2 = []
    for idx in clusters:
        _, L_ii = normalized_affinity(A_hat[np.ix_(idx, idx)])
        lambda2.append(float(np.sort(np.linalg.eigvalsh(L_ii))[-2]) if idx.size > 1 else np.nan)
    delta = 1.0 - max(lambda2)

    # A2: eps1 = max_{i1 != i2} sum_{j in S_i1, k in S_i2} A_jk^2 / (d_hat_j d_hat_k).
    W = A**2 / np.outer(d_hat, d_hat)
    eps1 = max(
        (W[np.ix_(a, b)].sum() for ia, a in enumerate(clusters) for ib, b in enumerate(clusters) if ia != ib),
        default=0.0,
    )

    # A3: (sum_{k not in S_i} A_jk / d_hat_j) <= eps2 * (sum_{k,l in S_i} A_kl^2/(d_hat_k d_hat_l))^{-1/2}.
    cross_ratio = (A * ~same).sum(axis=1) / d_hat
    eps2 = 0.0
    for idx in clusters:
        block_term = W[np.ix_(idx, idx)].sum() ** -0.5
        eps2 = max(eps2, float(cross_ratio[idx].max() / block_term))

    # A4: d_hat_j >= (sum_{k in S_i} d_hat_k) / (C n_i)  =>  C = max mean(d_hat) / d_hat_j.
    C = max(float(d_hat[idx].mean() / d_hat[idx].min()) for idx in clusters)

    eps = float(np.sqrt(k * (k - 1) * eps1 + k * eps2**2))
    condition = delta > (2 + np.sqrt(2)) * eps
    bound = 4 * C * (4 + 2 * np.sqrt(k)) ** 2 * eps**2 / (delta - np.sqrt(2) * eps) ** 2 if condition else np.nan

    return {
        "lambda2_per_cluster": lambda2,
        "delta": delta,
        "eps1": float(eps1),
        "eps2": eps2,
        "max_cross_ratio": float(cross_ratio.max()),
        "C": C,
        "eps": eps,
        "theorem2_condition_holds": bool(condition),
        "theorem2_bound": float(bound),
    }


def mean_sq_distance_to_orthogonal_centers(Y: np.ndarray, labels: np.ndarray) -> float:
    """Left side of Eq. (8), using the best orthonormal r_i for the given labels.

    The optimal orthonormal centres are the polar factor of the per-cluster
    sums of Y's rows (orthogonal Procrustes).
    """
    ids = np.unique(labels)
    S = np.stack([Y[labels == c].sum(axis=0) for c in ids])  # (k, k)
    U, _, Vt = np.linalg.svd(S)
    R = U @ Vt  # rows are orthonormal r_i
    diffs = Y - R[np.searchsorted(ids, labels)]
    return float(np.mean(np.sum(diffs**2, axis=1)))
