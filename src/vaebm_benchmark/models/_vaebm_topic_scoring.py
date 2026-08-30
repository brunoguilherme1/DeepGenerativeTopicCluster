"""Pure-numpy energy-topic scoring, extracted out of vaebm.py's
`top_words_by_freq_exact()` so it is independently unit-testable without
needing TensorFlow or a fitted model (see tests/test_vaebm_topic_scoring.py).

The bug this guards against: multiplying decoder logits by a 0/1 presence
mask makes a word that is NEVER observed in a cluster score EXACTLY 0 -
indistinguishable from, and able to silently outrank, a genuinely
observed word whose summed logit happens to be negative. This is a
topic-word DISPLAY/ranking fix only; it changes nothing about the trained
model's parameters, ELBO, or KMeans clusters.
"""

from __future__ import annotations

import numpy as np


def energy_scores(logits_k: np.ndarray, mask_k: np.ndarray, counts_k: np.ndarray) -> np.ndarray:
    """logits_k: [n_docs_in_cluster, vocab] raw decoder logits (h_k @ R.T + b).
    mask_k: same shape, 1.0 where that word is present in that document.
    counts_k: [vocab] total count of each word across the cluster's docs
    (used only to detect "never observed anywhere in this cluster").

    Returns a [vocab] score array where never-observed words are -inf, so
    `np.argsort(scores)[::-1]` can never rank an absent word above an
    observed one, regardless of the sign of the observed words' summed
    logits.
    """
    masked = logits_k * mask_k
    scores = np.asarray(masked.sum(axis=0)).ravel()
    never_observed = np.asarray(counts_k).ravel() == 0
    return np.where(never_observed, -np.inf, scores)
