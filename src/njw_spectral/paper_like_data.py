"""Synthetic stand-ins for the settings drawn in NJW Figure 1.

The paper does not release coordinates, n, noise, or sigma. These generators
are our reconstructions from the figure titles and the qualitative layout.
They are NOT the original datasets, and a good ARI here is not a reproduction
of Figure 1.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_circles

PAPER_LIKE_NOTE = (
    "Our reconstruction of a Figure 1 setting. Not the paper's point set."
)


def _sample_polyline(poly, n: int, rng: np.random.Generator, jitter: float) -> np.ndarray:
    poly = np.asarray(poly, dtype=float)
    seg = np.diff(poly, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = cum[-1]
    if total <= 0:
        raise ValueError("polyline has zero length")
    u = rng.uniform(0.0, total, size=n)
    idx = np.clip(np.searchsorted(cum, u, side="right") - 1, 0, len(seglen) - 1)
    alpha = (u - cum[idx]) / np.maximum(seglen[idx], 1e-12)
    pts = poly[idx] + alpha[:, None] * seg[idx]
    pts = pts + rng.normal(0.0, jitter, size=pts.shape)
    return pts


def make_twocircles(seed: int = 0) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """Concentric rings, the setting of Figure 1e / 1i. n=300, noise=0.05."""
    pts, y = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=seed)
    spec = {"name": "twocircles", "n_samples": 300, "noise": 0.05, "factor": 0.5, "seed": seed,
            "note": PAPER_LIKE_NOTE}
    return pts, y, 2, spec


def make_threecircles_joined(seed: int = 0) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """Three rings plus short radial bridges (Figure 1f/1g/1j setting, our data).

    Bridge points are labeled -1 and should be ignored when scoring ARI.
    True ring count is 3; the paper also shows a k=2 run on this kind of data.
    """
    rng = np.random.default_rng(seed)
    radii = (0.45, 0.95, 1.45)
    n_per = 120
    points, labels = [], []
    for lab, radius in enumerate(radii):
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_per)
        pts = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
        pts += rng.normal(0.0, 0.02, size=pts.shape)
        points.append(pts)
        labels.append(np.full(n_per, lab, dtype=int))
    # Two short bridges so neighbouring rings are joined, as in the figure name.
    for r0, r1 in ((radii[0], radii[1]), (radii[1], radii[2])):
        t = np.linspace(0.15, 0.85, 10)
        bridge = np.column_stack([((1 - t) * r0 + t * r1), np.zeros_like(t)])
        bridge += rng.normal(0.0, 0.01, size=bridge.shape)
        points.append(bridge)
        labels.append(np.full(len(t), -1, dtype=int))
    spec = {"name": "threecircles_joined", "radii": radii, "n_per_ring": n_per,
            "n_bridge_per_gap": 10, "seed": seed, "note": PAPER_LIKE_NOTE}
    return np.vstack(points), np.concatenate(labels), 3, spec


def make_lineandballs(seed: int = 0) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """Sparse upper arc plus two dense blobs (Figure 1b setting, our data)."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.15 * np.pi, 0.85 * np.pi, 90)
    arc = np.column_stack([2.5 + 2.3 * np.cos(theta), 0.3 + 2.1 * np.sin(theta)])
    arc = arc + rng.normal(0.0, 0.03, size=arc.shape)
    b1 = rng.normal(loc=(1.6, 1.15), scale=0.16, size=(140, 2))
    b2 = rng.normal(loc=(3.4, 1.15), scale=0.16, size=(140, 2))
    pts = np.vstack([arc, b1, b2])
    y = np.concatenate([
        np.zeros(len(arc), dtype=int),
        np.ones(len(b1), dtype=int),
        np.full(len(b2), 2, dtype=int),
    ])
    spec = {"name": "lineandballs", "n_arc": 90, "n_per_ball": 140, "seed": seed,
            "note": PAPER_LIKE_NOTE}
    return pts, y, 3, spec


def make_fourclouds(seed: int = 0) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """Four blobs; ground truth is the 2-cluster left/right grouping of Figure 1c.

    Sparse vertical bridges connect the two left blobs and the two right blobs.
    The gap between left and right is larger.
    """
    rng = np.random.default_rng(seed)
    centers = [(-1.2, -1.0), (-1.2, 1.0), (1.6, -1.0), (1.6, 1.0)]
    clouds = [rng.normal(loc=c, scale=0.18, size=(80, 2)) for c in centers]
    bridges = []
    for x in (-1.2, 1.6):
        t = np.linspace(-0.55, 0.55, 18)
        bridges.append(np.column_stack([np.full_like(t, x), t]) + rng.normal(0, 0.02, size=(len(t), 2)))
    pts = np.vstack(clouds + bridges)
    y = np.concatenate([
        np.zeros(160, dtype=int),  # two left clouds
        np.ones(160, dtype=int),   # two right clouds
        np.zeros(18, dtype=int),
        np.ones(18, dtype=int),
    ])
    spec = {"name": "fourclouds", "k_used_by_paper": 2, "n_per_cloud": 80,
            "n_bridge": 18, "seed": seed, "note": PAPER_LIKE_NOTE + " GT is left vs right, k=2."}
    return pts, y, 2, spec


def make_squiggles(seed: int = 0) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """Four separate wavy strokes (Figure 1d setting, our data; not intertwined)."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 5.0, 100)
    parts, labels = [], []
    for i in range(4):
        y = 1.15 * i + 0.28 * np.sin(1.7 * xs + 0.6 * i)
        pts = np.column_stack([xs, y]) + rng.normal(0.0, 0.025, size=(len(xs), 2))
        parts.append(pts)
        labels.append(np.full(len(xs), i, dtype=int))
    spec = {"name": "squiggles", "n_per_curve": 100, "n_curves": 4, "seed": seed,
            "note": PAPER_LIKE_NOTE + " Strokes do not cross."}
    return np.vstack(parts), np.concatenate(labels), 4, spec


def _letters() -> list[tuple[str, list]]:
    """Polylines for a compact 'NIPS 2001' (Figure 1a), in a roughly [0, 10] x [0, 2] box."""
    return [
        ("N", [[0.0, 0.0], [0.0, 1.4], [0.7, 0.0], [0.7, 1.4]]),
        ("I", [[1.15, 0.0], [1.15, 1.4]]),
        ("P", [[1.6, 0.0], [1.6, 1.4], [2.25, 1.4], [2.25, 0.7], [1.6, 0.7]]),
        ("S", [[3.3, 1.3], [2.6, 1.3], [2.6, 0.75], [3.3, 0.75], [3.3, 0.1], [2.6, 0.1]]),
        ("2", [[4.0, 1.15], [4.7, 1.15], [4.7, 0.7], [4.0, 0.15], [4.7, 0.15]]),
        ("0", [[5.1, 0.15], [5.1, 1.25], [5.8, 1.25], [5.8, 0.15], [5.1, 0.15]]),
        ("0b", [[6.2, 0.15], [6.2, 1.25], [6.9, 1.25], [6.9, 0.15], [6.2, 0.15]]),
        ("1", [[7.5, 0.15], [7.5, 1.3]]),
    ]


def make_nips(seed: int = 0) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """Points sampled along strokes of N, I, P, S, 2, 0, 0, 1. Eight clusters."""
    rng = np.random.default_rng(seed)
    parts, labels = [], []
    for lab, (_name, poly) in enumerate(_letters()):
        parts.append(_sample_polyline(poly, n=70, rng=rng, jitter=0.02))
        labels.append(np.full(70, lab, dtype=int))
    spec = {"name": "nips", "chars": [n for n, _ in _letters()], "n_per_char": 70,
            "seed": seed, "note": PAPER_LIKE_NOTE}
    return np.vstack(parts), np.concatenate(labels), 8, spec


PAPER_LIKE_BUILDERS = {
    "twocircles": make_twocircles,
    "threecircles_joined": make_threecircles_joined,
    "lineandballs": make_lineandballs,
    "fourclouds": make_fourclouds,
    "squiggles": make_squiggles,
    "nips": make_nips,
}
