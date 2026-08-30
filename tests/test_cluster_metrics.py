"""ACC (unsupervised clustering accuracy, Hungarian-matched) must be
invariant to any permutation of predicted cluster IDs - unsupervised
cluster ids are arbitrary labels, not aligned to ground-truth class ids
by construction."""

import itertools

from vaebm_benchmark.metrics.clustering_quality import accuracy_hungarian, nmi


def test_swapped_cluster_ids_still_score_perfect_accuracy():
    """Exact example from the task spec: predictions use the OPPOSITE
    cluster id convention from the true labels, but every point is still
    correctly grouped - ACC must be 1.0, not 0.0."""
    labels = [0, 0, 1, 1]
    predictions = [1, 1, 0, 0]
    assert accuracy_hungarian(predictions, labels) == 1.0


def test_identical_ids_also_score_perfect_accuracy():
    labels = [0, 0, 1, 1]
    predictions = [0, 0, 1, 1]
    assert accuracy_hungarian(predictions, labels) == 1.0


def test_acc_is_invariant_to_every_permutation_of_predicted_ids():
    labels = [0, 0, 0, 1, 1, 2, 2, 2, 2]
    base_predictions = [2, 2, 2, 0, 0, 1, 1, 1, 1]  # some arbitrary but "correct" grouping
    base_acc = accuracy_hungarian(base_predictions, labels)
    assert base_acc == 1.0

    for perm in itertools.permutations([0, 1, 2]):
        remap = dict(enumerate(perm))
        permuted = [remap[p] for p in base_predictions]
        assert accuracy_hungarian(permuted, labels) == base_acc


def test_acc_with_genuinely_imperfect_clustering():
    # 3 of 4 points correctly grouped under the best permutation.
    labels = [0, 0, 1, 1]
    predictions = [0, 1, 1, 1]
    assert accuracy_hungarian(predictions, labels) == 0.75


def test_nmi_still_symmetric_for_sanity():
    labels = [0, 0, 1, 1]
    predictions = [1, 1, 0, 0]
    assert nmi(predictions, labels) == 1.0
