"""Checks for the Section-4 baselines. They must not call SpectralClustering."""

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from njw_spectral import KMeansConfig  # noqa: E402
from njw_spectral.baselines import (  # noqa: E402
    connected_components_for_k,
    kannan_style_max_abs,
    meila_shi_clustering,
)


def test_meila_shi_separates_two_blobs():
    pts, y = make_blobs(n_samples=80, centers=[[0, 0], [8, 0]], cluster_std=0.3, random_state=0)
    labels, info = meila_shi_clustering(pts, 2, sigma=0.8, kmeans_config=KMeansConfig(random_state=0))
    assert labels.shape == (len(pts),)
    assert info["X"].shape == (len(pts), 2)
    assert adjusted_rand_score(y, labels) == 1.0


def test_connected_components_finds_two_blobs():
    pts, y = make_blobs(n_samples=60, centers=[[0, 0], [10, 0]], cluster_std=0.2, random_state=0)
    labels, n_comp, _threshold, exact = connected_components_for_k(pts, k=2)
    assert exact and n_comp == 2
    assert adjusted_rand_score(y, labels) == 1.0


def test_kannan_style_returns_k_range():
    pts, _y = make_blobs(n_samples=40, centers=3, random_state=0)
    labels, info = kannan_style_max_abs(pts, k=3, sigma=1.0)
    assert labels.shape == (len(pts),)
    assert set(np.unique(labels)).issubset({0, 1, 2})
    assert info["X"].shape[1] == 3
