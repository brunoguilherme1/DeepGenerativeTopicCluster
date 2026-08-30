"""Candidate cluster selection for the LLM prompt: an edge document's
CURRENT cluster plus its `num_candidates - 1` nearest COMPETING
centroids (by representation-space distance) - never all K clusters
(impractical and unnecessary at K=50/100), and never chosen using
ground-truth labels."""

from __future__ import annotations

import numpy as np


def select_candidate_clusters(
    doc_representation: np.ndarray,
    current_cluster: int,
    centroids: dict[int, np.ndarray],
    num_candidates: int = 3,
) -> list[int]:
    other_clusters = [c for c in centroids if c != current_cluster]
    ranked_others = sorted(other_clusters, key=lambda c: float(np.linalg.norm(doc_representation - centroids[c])))
    nearest_others = ranked_others[: max(0, num_candidates - 1)]
    return [current_cluster] + nearest_others
