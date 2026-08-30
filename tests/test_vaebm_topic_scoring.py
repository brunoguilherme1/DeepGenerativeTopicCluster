import numpy as np

from vaebm_benchmark.models._vaebm_topic_scoring import energy_scores


def test_never_observed_word_cannot_outrank_a_negative_but_observed_word():
    """Regression test for the masking bug: word 1 is observed (in one
    doc, with a negative logit); word 0 is never observed in the cluster
    at all. Naive 0/1 masking would give word 0 a score of exactly 0,
    which beats word 1's negative score - wrongly ranking an absent word
    above an observed one. The fix must force word 0 to -inf."""
    logits_k = np.array([[5.0, -3.0]])  # one document, two vocab words
    mask_k = np.array([[0.0, 1.0]])  # word 0 absent from this doc, word 1 present
    counts_k = np.array([0, 1])  # word 0 never observed anywhere in the cluster

    scores = energy_scores(logits_k, mask_k, counts_k)

    assert scores[0] == -np.inf
    assert scores[1] == -3.0
    # argsort-descending must rank the observed (even if negative) word first
    ranked = np.argsort(scores)[::-1]
    assert ranked[0] == 1


def test_observed_words_rank_by_summed_masked_logit():
    logits_k = np.array([[1.0, 2.0, 3.0], [4.0, 0.5, -1.0]])
    mask_k = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    counts_k = np.array([2, 2, 2])

    scores = energy_scores(logits_k, mask_k, counts_k)

    np.testing.assert_allclose(scores, [5.0, 2.5, 2.0])


def test_all_never_observed_gives_all_negative_infinity():
    logits_k = np.array([[1.0, 2.0]])
    mask_k = np.array([[0.0, 0.0]])
    counts_k = np.array([0, 0])

    scores = energy_scores(logits_k, mask_k, counts_k)

    assert np.all(scores == -np.inf)
    assert not np.any(np.isfinite(scores))
