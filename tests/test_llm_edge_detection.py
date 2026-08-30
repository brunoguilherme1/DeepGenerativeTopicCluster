"""Unit tests for llm/edge_detection.py - written but NOT executed in
this session, per this task's own instructions. Pure numpy logic; no
model/LLM/dataset involved."""

import numpy as np

from vaebm_benchmark.llm.edge_detection import (
    compute_centroids,
    distances_to_own_centroid,
    select_edge_points,
)


def test_compute_centroids_is_the_mean_of_cluster_members():
    representation = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0], [12.0, 10.0]])
    labels = [0, 0, 1, 1]
    centroids = compute_centroids(representation, labels)
    np.testing.assert_allclose(centroids[0], [1.0, 0.0])
    np.testing.assert_allclose(centroids[1], [11.0, 10.0])


def test_distances_to_own_centroid_are_nonnegative_and_zero_at_centroid():
    representation = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])  # all identical -> centroid == every point
    labels = [0, 0, 0]
    distances = distances_to_own_centroid(representation, labels)
    np.testing.assert_allclose(distances, [0.0, 0.0, 0.0])


def test_select_edge_points_picks_the_farthest_within_each_cluster():
    # Cluster 0: points at distance 0, 1, 5 from its centroid-ish region.
    representation = np.array([
        [0.0], [1.0], [5.0],   # cluster 0
        [100.0], [101.0], [105.0],  # cluster 1
    ])
    labels = [0, 0, 0, 1, 1, 1]
    # edge_fraction chosen so exactly 1 point per cluster is selected (round(3*0.34)=1).
    edge_indices = select_edge_points(representation, labels, edge_fraction=0.34)
    assert 2 in edge_indices  # farthest in cluster 0 (value 5.0)
    assert 5 in edge_indices  # farthest in cluster 1 (value 105.0)
    assert len(edge_indices) == 2


def test_select_edge_points_is_per_cluster_not_global():
    """A tiny, tight cluster (1) must still contribute its own edge
    point even though every one of its members is far closer to ITS
    centroid than cluster 0's edge points are to cluster 0's centroid -
    a global top-fraction selection would let cluster 0 dominate and
    starve cluster 1 entirely."""
    representation = np.array([
        [0.0], [50.0], [100.0],  # cluster 0: spread out, large distances to its own centroid
        [200.0], [200.1],        # cluster 1: very tight, tiny distances to its own centroid
    ])
    labels = [0, 0, 0, 1, 1]
    edge_indices = select_edge_points(representation, labels, edge_fraction=0.5)
    cluster_1_indices = {3, 4}
    assert cluster_1_indices & set(edge_indices), "cluster 1 must contribute at least one edge point"


def test_select_edge_points_never_selects_more_than_available_in_small_cluster():
    representation = np.array([[0.0], [1.0]])
    labels = [0, 1]  # two singleton clusters
    edge_indices = select_edge_points(representation, labels, edge_fraction=0.10)
    # At least 1 per non-empty cluster (the function's own documented floor).
    assert set(edge_indices) == {0, 1}


def test_edge_fraction_out_of_range_raises():
    representation = np.array([[0.0], [1.0]])
    labels = [0, 1]
    try:
        select_edge_points(representation, labels, edge_fraction=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        select_edge_points(representation, labels, edge_fraction=1.5)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_select_edge_points_never_uses_ground_truth_labels():
    """Structural check: the function signature has no parameter through
    which ground-truth labels could even be passed - only the model's
    OWN cluster assignment (`labels` here means cluster ids, per the
    module's own docstring, not ground truth)."""
    import inspect

    signature = inspect.signature(select_edge_points)
    assert "ground_truth" not in signature.parameters
    assert "true_labels" not in signature.parameters
