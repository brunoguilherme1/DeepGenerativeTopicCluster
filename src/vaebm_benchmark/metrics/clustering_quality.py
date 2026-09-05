"""Document-clustering-quality metrics: NMI/ARI/AMI/Homogeneity/
Completeness/V-measure (scikit-learn's authoritative implementations)
and Purity/ACC (no scikit-learn equivalent, implemented directly from
their standard definitions). Ground-truth labels are used ONLY here, at
evaluation time - never during fit().

Also: Silhouette/Davies-Bouldin/Calinski-Harabasz (geometry.py-style
metrics below) - these take NO ground truth at all; they answer whether
a model's own learned representation contains compact, well-separated
groups, a different question from NMI/ARI/AMI/.../Purity/ACC above
(which ask whether that grouping matches human-labeled classes). Which
representation is "the" feature space for these three is a per-model
choice made by the caller (see experiment/cluster_runner.py's
`representation_source`), not decided here - each function just consumes
whatever `embeddings` array it's given."""

from __future__ import annotations

from collections import Counter


def nmi(predicted_labels, true_labels) -> float:
    from sklearn.metrics import normalized_mutual_info_score

    return float(normalized_mutual_info_score(true_labels, predicted_labels))


def ari(predicted_labels, true_labels) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(true_labels, predicted_labels))


def ami(predicted_labels, true_labels) -> float:
    from sklearn.metrics import adjusted_mutual_info_score

    return float(adjusted_mutual_info_score(true_labels, predicted_labels))


def purity(predicted_labels, true_labels) -> float:
    n = len(true_labels)
    if n == 0:
        raise ValueError("Purity requires at least one document")
    clusters: dict[int, Counter] = {}
    for pred, true in zip(predicted_labels, true_labels):
        clusters.setdefault(pred, Counter())[true] += 1
    correct = sum(counter.most_common(1)[0][1] for counter in clusters.values())
    return correct / n


def accuracy_hungarian(predicted_labels, true_labels) -> float:
    """Clustering accuracy (ACC) via the Hungarian algorithm's optimal
    predicted-cluster -> true-label assignment - the metric short-text
    topic-model papers (e.g. GloCOM) commonly report alongside/instead of
    NMI/Purity for clustering quality."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    y_pred = np.asarray(predicted_labels)
    y_true = np.asarray(true_labels)
    n = y_pred.size
    d = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((d, d), dtype=np.int64)
    for i in range(n):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-w)
    return sum(w[i, j] for i, j in zip(row_ind, col_ind)) / n


def homogeneity(predicted_labels, true_labels) -> float:
    """Rosenberg, A., & Hirschberg, J. (2007). "V-Measure: A Conditional
    Entropy-Based External Cluster Evaluation Measure." EMNLP-CoNLL. 1.0
    iff every predicted cluster contains only members of a single true
    class (says nothing about whether each true class stayed in one
    cluster - see completeness())."""
    from sklearn.metrics import homogeneity_score

    return float(homogeneity_score(true_labels, predicted_labels))


def completeness(predicted_labels, true_labels) -> float:
    """Rosenberg & Hirschberg (2007), same reference as homogeneity()
    above. 1.0 iff every member of a given true class is assigned to the
    same predicted cluster (the "dual" of homogeneity - says nothing
    about whether that cluster also contains other classes)."""
    from sklearn.metrics import completeness_score

    return float(completeness_score(true_labels, predicted_labels))


def v_measure(predicted_labels, true_labels) -> float:
    """Rosenberg & Hirschberg (2007), same reference as homogeneity()/
    completeness() above - their harmonic mean (beta=1.0, scikit-learn's
    own default weighting)."""
    from sklearn.metrics import v_measure_score

    return float(v_measure_score(true_labels, predicted_labels))


def silhouette(embeddings, predicted_labels) -> float:
    """Rousseeuw, P. J. (1987). "Silhouettes: A Graphical Aid to the
    Interpretation and Validation of Cluster Analysis." Journal of
    Computational and Applied Mathematics. Range [-1, 1], higher is
    better-separated. Label-free (no ground truth) - `embeddings` is
    whatever feature space the caller declares as `representation_source`
    (see experiment/cluster_runner.py), not decided here."""
    from sklearn.metrics import silhouette_score

    return float(silhouette_score(embeddings, predicted_labels))


def davies_bouldin(embeddings, predicted_labels) -> float:
    """Davies, D. L., & Bouldin, D. W. (1979). "A Cluster Separation
    Measure." IEEE Transactions on Pattern Analysis and Machine
    Intelligence. Lower is better (0 is the best possible score) - the
    only metric in this module where direction is "minimize". Label-free,
    same `embeddings` convention as silhouette() above."""
    from sklearn.metrics import davies_bouldin_score

    return float(davies_bouldin_score(embeddings, predicted_labels))


def calinski_harabasz(embeddings, predicted_labels) -> float:
    """Caliński, T., & Harabasz, J. (1974). "A Dendrite Method for
    Cluster Analysis." Communications in Statistics. Ratio of
    between-cluster to within-cluster dispersion; unbounded above,
    higher is better. Label-free, same `embeddings` convention as
    silhouette()/davies_bouldin() above."""
    from sklearn.metrics import calinski_harabasz_score

    return float(calinski_harabasz_score(embeddings, predicted_labels))


# Label-based metrics: (predicted_labels, true_labels) - ground truth used
# ONLY here, at evaluation time.
METRIC_FUNCTIONS = {
    "nmi": nmi,
    "ari": ari,
    "ami": ami,
    "purity": purity,
    "acc": accuracy_hungarian,
    "homogeneity": homogeneity,
    "completeness": completeness,
    "v_measure": v_measure,
}

# Geometry (label-free, internal-validity) metrics: (embeddings,
# predicted_labels) - never take true_labels, see each function's own
# docstring.
GEOMETRY_METRIC_FUNCTIONS = {
    "silhouette": silhouette,
    "davies_bouldin": davies_bouldin,
    "calinski_harabasz": calinski_harabasz,
}


def compute_clustering_metrics(
    predicted_labels, true_labels, metric_names: list[str]
) -> dict[str, float]:
    return {name: METRIC_FUNCTIONS[name](predicted_labels, true_labels) for name in metric_names}


def compute_geometry_metrics(
    embeddings, predicted_labels, metric_names: list[str]
) -> dict[str, float]:
    return {name: GEOMETRY_METRIC_FUNCTIONS[name](embeddings, predicted_labels) for name in metric_names}
