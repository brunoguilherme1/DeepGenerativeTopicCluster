"""Edge-point / uncertainty detection (LLMEdgeRefine-style, Feng et al.,
EMNLP 2024): documents farthest from their OWN cluster's centroid, in a
chosen representation, are the least confidently assigned and are the
only ones sent to the LLM - most documents never reach it at all.

Ground-truth labels are NEVER used anywhere in this module - only the
model's own hard cluster assignments and a chosen representation
(native or shared, see experiment/llm_refinement_runner.py).

Selection is PER-CLUSTER (top `edge_fraction` of EACH cluster's own
members, ranked by that cluster's own centroid distance) - not a single
global top-`edge_fraction` across the whole corpus. This is a deliberate
choice, not an oversight: a global ranking could let one large or
diffuse cluster dominate every edge slot while a tighter cluster
contributes none, which would silently exempt some clusters from
refinement entirely.
"""

from __future__ import annotations

import numpy as np


def compute_centroids(representation: np.ndarray, labels: list[int]) -> dict[int, np.ndarray]:
    labels_arr = np.asarray(labels)
    return {
        cluster_id: representation[labels_arr == cluster_id].mean(axis=0)
        for cluster_id in sorted(set(labels))
    }


def distances_to_own_centroid(
    representation: np.ndarray,
    labels: list[int],
    centroids: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    if centroids is None:
        centroids = compute_centroids(representation, labels)
    labels_arr = np.asarray(labels)
    distances = np.empty(len(labels), dtype=float)
    for i, label in enumerate(labels_arr):
        distances[i] = np.linalg.norm(representation[i] - centroids[label])
    return distances


def select_edge_points(
    representation: np.ndarray,
    labels: list[int],
    edge_fraction: float = 0.10,
    centroids: dict[int, np.ndarray] | None = None,
) -> list[int]:
    """Returns document INDICES (into `representation`/`labels`), the
    top `edge_fraction` of EACH cluster's own members ranked by distance
    to that cluster's own centroid (farthest first). At least 1 edge
    point is selected per non-empty cluster, so every cluster gets at
    least some LLM attention even at a very small `edge_fraction`."""
    if not (0.0 < edge_fraction <= 1.0):
        raise ValueError(f"edge_fraction must be in (0, 1], got {edge_fraction}")
    labels_arr = np.asarray(labels)
    if centroids is None:
        centroids = compute_centroids(representation, labels)

    edge_indices: list[int] = []
    for cluster_id in sorted(set(labels)):
        member_indices = np.where(labels_arr == cluster_id)[0]
        if member_indices.size == 0:
            continue
        member_distances = np.linalg.norm(representation[member_indices] - centroids[cluster_id], axis=1)
        n_edge = max(1, int(round(member_indices.size * edge_fraction)))
        ranked = member_indices[np.argsort(member_distances)[::-1][:n_edge]]
        edge_indices.extend(int(i) for i in ranked)
    return sorted(edge_indices)
