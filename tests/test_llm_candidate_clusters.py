"""Unit tests for llm/candidate_clusters.py - written but NOT executed in
this session, per this task's own instructions."""

import numpy as np

from vaebm_benchmark.llm.candidate_clusters import select_candidate_clusters


def test_current_cluster_is_always_first_candidate():
    centroids = {0: np.array([0.0]), 1: np.array([1.0]), 2: np.array([10.0])}
    doc = np.array([0.1])
    candidates = select_candidate_clusters(doc, current_cluster=0, centroids=centroids, num_candidates=3)
    assert candidates[0] == 0


def test_selects_nearest_competing_centroids_not_farthest():
    centroids = {0: np.array([0.0]), 1: np.array([1.0]), 2: np.array([10.0]), 3: np.array([20.0])}
    doc = np.array([0.0])  # exactly at cluster 0's centroid
    candidates = select_candidate_clusters(doc, current_cluster=0, centroids=centroids, num_candidates=3)
    assert candidates == [0, 1, 2]  # nearest two OTHER clusters, not cluster 3 (farthest)


def test_num_candidates_bounds_total_including_current():
    centroids = {i: np.array([float(i)]) for i in range(10)}
    doc = np.array([5.0])
    candidates = select_candidate_clusters(doc, current_cluster=5, centroids=centroids, num_candidates=3)
    assert len(candidates) == 3
    assert candidates[0] == 5


def test_num_candidates_larger_than_available_clusters_returns_all():
    centroids = {0: np.array([0.0]), 1: np.array([1.0])}
    doc = np.array([0.0])
    candidates = select_candidate_clusters(doc, current_cluster=0, centroids=centroids, num_candidates=5)
    assert set(candidates) == {0, 1}


def test_never_uses_ground_truth_labels():
    import inspect

    signature = inspect.signature(select_candidate_clusters)
    assert "ground_truth" not in signature.parameters
    assert "labels" not in signature.parameters
