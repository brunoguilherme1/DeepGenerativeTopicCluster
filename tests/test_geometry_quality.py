"""Unit tests for metrics/geometry_quality.py - written but NOT executed
in this session, per this task's own instructions (part of the
llm_cluster_refinement feature). Tiny synthetic data, no model/LLM."""

import numpy as np

from vaebm_benchmark.metrics.geometry_quality import (
    calinski_harabasz,
    compute_geometry_metrics,
    davies_bouldin,
    silhouette,
)


def _well_separated_data():
    return np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]]), [0, 0, 1, 1]


def test_silhouette_is_high_for_well_separated_clusters():
    representation, labels = _well_separated_data()
    value = silhouette(representation, labels)
    assert value > 0.9


def test_silhouette_returns_none_for_single_cluster():
    representation = np.array([[0.0], [1.0], [2.0]])
    assert silhouette(representation, [0, 0, 0]) is None


def test_silhouette_returns_none_when_every_point_is_its_own_cluster():
    representation = np.array([[0.0], [1.0], [2.0]])
    assert silhouette(representation, [0, 1, 2]) is None


def test_davies_bouldin_is_low_for_well_separated_clusters():
    representation, labels = _well_separated_data()
    value = davies_bouldin(representation, labels)
    assert value < 0.2  # lower is better - see module docstring


def test_calinski_harabasz_is_high_for_well_separated_clusters():
    representation, labels = _well_separated_data()
    value = calinski_harabasz(representation, labels)
    assert value > 100


def test_compute_geometry_metrics_returns_requested_subset():
    representation, labels = _well_separated_data()
    result = compute_geometry_metrics(representation, labels, ["silhouette", "davies_bouldin"])
    assert set(result.keys()) == {"silhouette", "davies_bouldin"}
    assert result["silhouette"] > 0.9
