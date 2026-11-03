"""Synthetic 2-D datasets used in the stage-1 experiments.

These are freshly generated with sklearn.datasets; they resemble the kinds of
problems in the paper's Figure 1 but are NOT the paper's data.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons

DATASET_SPECS: dict[str, dict] = {
    "circles": {"generator": "make_circles", "n_samples": 400, "noise": 0.05, "factor": 0.5},
    "moons": {"generator": "make_moons", "n_samples": 400, "noise": 0.05},
    "blobs": {
        "generator": "make_blobs",
        "n_samples": 450,
        "centers": [[-5.0, 0.0], [0.0, 5.0], [5.0, 0.0]],
        "cluster_std": 1.0,
    },
}

DATASET_TITLES = {
    "circles": "Concentric circles",
    "moons": "Two moons",
    "blobs": "Isotropic Gaussian blobs",
}


def make_dataset(name: str, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (points of shape (n, 2), ground-truth labels of shape (n,), k)."""
    spec = DATASET_SPECS[name]
    if name == "circles":
        pts, y = make_circles(
            n_samples=spec["n_samples"], noise=spec["noise"], factor=spec["factor"], random_state=seed
        )
    elif name == "moons":
        pts, y = make_moons(n_samples=spec["n_samples"], noise=spec["noise"], random_state=seed)
    elif name == "blobs":
        pts, y = make_blobs(
            n_samples=spec["n_samples"],
            centers=np.array(spec["centers"]),
            cluster_std=spec["cluster_std"],
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown dataset {name!r}")
    k = int(np.unique(y).size)
    return pts, y, k


# Extra generators used by later critical experiments (not part of stage 1).
THREE_CIRCLES_SPEC = {
    "radii": (0.3, 0.65, 1.0),
    "n_per_ring": 150,
    "noise": 0.03,
}


def make_three_circles(
    seed: int,
    *,
    radii: tuple[float, ...] = THREE_CIRCLES_SPEC["radii"],
    n_per_ring: int = THREE_CIRCLES_SPEC["n_per_ring"],
    noise: float = THREE_CIRCLES_SPEC["noise"],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Three concentric rings. Our data; not the paper's threecircles-joined set.

    Ground-truth labels are 0/1/2 from innermost to outermost. True k = 3.
    """
    rng = np.random.default_rng(seed)
    points, labels = [], []
    for lab, radius in enumerate(radii):
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_per_ring)
        pts = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
        pts += rng.normal(0.0, noise, size=pts.shape)
        points.append(pts)
        labels.append(np.full(n_per_ring, lab, dtype=int))
    return np.vstack(points), np.concatenate(labels), len(radii)


BRIDGE_MOONS_SPEC = {
    "n_samples": 400,
    "noise": 0.05,
    "n_bridge_thin": 25,
    "n_bridge_strong": 80,
    "bridge_noise": 0.02,
}

UNBALANCED_BLOBS_SPEC = {
    "centers": [[-3.0, 0.0], [3.0, 0.0]],
    "n_samples": (350, 50),
    "cluster_std": (0.45, 0.45),
}


def make_moons_with_bridge(
    seed: int,
    *,
    n_samples: int = BRIDGE_MOONS_SPEC["n_samples"],
    noise: float = BRIDGE_MOONS_SPEC["noise"],
    n_bridge: int = BRIDGE_MOONS_SPEC["n_bridge_thin"],
    bridge_noise: float = BRIDGE_MOONS_SPEC["bridge_noise"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Two moons plus a point bridge linking the facing tips.

    Returns
    -------
    points : (n_moons + n_bridge, 2)
    y_eval : labels for scoring; bridge points are -1 (ignored in ARI)
    is_bridge : boolean mask, True on bridge points
    k : 2
    """
    rng = np.random.default_rng(seed)
    pts, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    # Facing tips of the two moons in the standard make_moons layout.
    tip0 = pts[y == 0][np.argmax(pts[y == 0, 0])]
    tip1 = pts[y == 1][np.argmin(pts[y == 1, 0])]
    t = np.linspace(0.02, 0.98, n_bridge)
    bridge = (1 - t)[:, None] * tip0 + t[:, None] * tip1
    bridge += rng.normal(0.0, bridge_noise, size=bridge.shape)
    points = np.vstack([pts, bridge])
    y_eval = np.concatenate([y, np.full(n_bridge, -1, dtype=int)])
    is_bridge = np.concatenate([np.zeros(len(y), dtype=bool), np.ones(n_bridge, dtype=bool)])
    return points, y_eval, is_bridge, 2


def make_blobs_with_bridge(
    seed: int,
    *,
    n_per_blob: int = 200,
    centers: tuple = ((-2.5, 0.0), (2.5, 0.0)),
    cluster_std: float = 0.35,
    n_bridge: int = 60,
    bridge_noise: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Two separated blobs joined by a dense corridor of noise points."""
    rng = np.random.default_rng(seed)
    pts, y = make_blobs(
        n_samples=[n_per_blob, n_per_blob],
        centers=np.array(centers),
        cluster_std=cluster_std,
        random_state=seed,
    )
    c0, c1 = np.array(centers[0]), np.array(centers[1])
    t = np.linspace(0.08, 0.92, n_bridge)
    bridge = (1 - t)[:, None] * c0 + t[:, None] * c1
    bridge += rng.normal(0.0, bridge_noise, size=bridge.shape)
    points = np.vstack([pts, bridge])
    y_eval = np.concatenate([y, np.full(n_bridge, -1, dtype=int)])
    is_bridge = np.concatenate([np.zeros(len(y), dtype=bool), np.ones(n_bridge, dtype=bool)])
    return points, y_eval, is_bridge, 2


def make_moons_with_gap_noise(
    seed: int,
    *,
    n_samples: int = BRIDGE_MOONS_SPEC["n_samples"],
    noise: float = BRIDGE_MOONS_SPEC["noise"],
    n_noise: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Two moons with a diffuse cloud of noise filling the gap between them.

    Unlike a thin 1-D bridge (which a normalized cut can still sever), this
    fills the region between the facing tips with many points, raising
    inter-cluster connectivity more aggressively.
    """
    rng = np.random.default_rng(seed)
    pts, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    tip0 = pts[y == 0][np.argmax(pts[y == 0, 0])]
    tip1 = pts[y == 1][np.argmin(pts[y == 1, 0])]
    mid = 0.5 * (tip0 + tip1)
    # Axis-aligned ellipse covering the gap between the facing tips.
    scale = np.array([0.55 * np.linalg.norm(tip0 - tip1), 0.35])
    cloud = mid + rng.normal(0.0, 1.0, size=(n_noise, 2)) * scale
    points = np.vstack([pts, cloud])
    y_eval = np.concatenate([y, np.full(n_noise, -1, dtype=int)])
    is_bridge = np.concatenate([np.zeros(len(y), dtype=bool), np.ones(n_noise, dtype=bool)])
    return points, y_eval, is_bridge, 2


def make_unbalanced_blobs(
    seed: int,
    *,
    centers=UNBALANCED_BLOBS_SPEC["centers"],
    n_samples=UNBALANCED_BLOBS_SPEC["n_samples"],
    cluster_std=UNBALANCED_BLOBS_SPEC["cluster_std"],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Two Gaussian blobs with very different sizes (density / mass imbalance)."""
    pts, y = make_blobs(
        n_samples=list(n_samples),
        centers=np.array(centers),
        cluster_std=list(cluster_std),
        random_state=seed,
    )
    return pts, y, 2
