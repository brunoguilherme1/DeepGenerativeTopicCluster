"""Internal clustering-quality metrics: describe how well-separated hard
clusters are in a given REPRESENTATION SPACE, without any reference to
ground-truth labels - unlike ACC/NMI/ARI/AMI/Purity
(metrics/clustering_quality.py, external metrics). Used by the
llm_cluster_refinement experiment to check whether LLM-based refinement
improves or degrades geometric cluster separation, not just label
agreement.

IMPORTANT - direction: Silhouette and Calinski-Harabasz are HIGHER-is-
better; Davies-Bouldin is LOWER-is-better (the one inverted-direction
metric here) - a positive "delta_davies_bouldin" (after - before) means
WORSE separation, not better, unlike every other delta this project
computes. Documented explicitly wherever deltas are computed
(experiment/llm_refinement_runner.py) so this is never misread as an
improvement.

IMPORTANT - representation consistency: these metrics are only
comparable BEFORE vs. AFTER refinement when computed against the SAME
representation both times (see experiment/llm_refinement_runner.py's
`--edge-representation` handling) - comparing Silhouette computed in
VAE-BM's own latent mu space against Silhouette computed in a shared
SentenceTransformer space would not be a like-for-like comparison.
"""

from __future__ import annotations

from typing import Optional


def silhouette(representation, labels) -> Optional[float]:
    from sklearn.metrics import silhouette_score

    if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
        return None
    return float(silhouette_score(representation, labels))


def davies_bouldin(representation, labels) -> Optional[float]:
    from sklearn.metrics import davies_bouldin_score

    if len(set(labels)) < 2:
        return None
    return float(davies_bouldin_score(representation, labels))


def calinski_harabasz(representation, labels) -> Optional[float]:
    from sklearn.metrics import calinski_harabasz_score

    if len(set(labels)) < 2:
        return None
    return float(calinski_harabasz_score(representation, labels))


# Metrics where a LOWER value is better - the inverted-direction case
# every delta computation must check explicitly (see module docstring).
LOWER_IS_BETTER = {"davies_bouldin"}

METRIC_FUNCTIONS = {
    "silhouette": silhouette,
    "davies_bouldin": davies_bouldin,
    "calinski_harabasz": calinski_harabasz,
}


def compute_geometry_metrics(representation, labels, metric_names: list[str]) -> dict[str, Optional[float]]:
    return {name: METRIC_FUNCTIONS[name](representation, labels) for name in metric_names}
