"""Document-clustering-quality metrics: NMI/ARI/AMI (scikit-learn's
authoritative implementations) and Purity (no scikit-learn equivalent,
implemented directly from its standard definition). Ground-truth labels are
used ONLY here, at evaluation time - never during fit()."""

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


METRIC_FUNCTIONS = {
    "nmi": nmi,
    "ari": ari,
    "ami": ami,
    "purity": purity,
    "acc": accuracy_hungarian,
}


def compute_clustering_metrics(
    predicted_labels, true_labels, metric_names: list[str]
) -> dict[str, float]:
    return {name: METRIC_FUNCTIONS[name](predicted_labels, true_labels) for name in metric_names}
