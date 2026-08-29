"""Topic-quality metrics: coherence (C_V, NPMI) via gensim's CoherenceModel
(the reference implementation almost every topic-modeling paper actually
uses), plus Topic Diversity and IRBO.

Each baseline protocol declares which of these its own paper actually
reports (see protocols/fastopic_protocol.py, protocols/glocom_protocol.py) -
this module does not decide that; it only computes whatever is requested,
identically for the baseline and for VAE-BM, so the comparison is apples to
apples.

`reference_corpus` MUST be the same tokenized corpus (same tokenizer, same
document set) for every model being compared under one protocol - coherence
is only meaningful relative to a shared reference, never as an absolute
number across different reference corpora.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional


def coherence(
    topics: list[list[str]],
    reference_corpus: list[list[str]],
    top_n: int = 10,
    measure: str = "c_npmi",
) -> tuple[float, list[float]]:
    """measure: any gensim CoherenceModel `coherence` value, e.g.
    'c_npmi' (NPMI) or 'c_v' (C_V)."""
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel

    truncated = [words[:top_n] for words in topics]
    dictionary = Dictionary(reference_corpus)
    cm = CoherenceModel(
        topics=truncated,
        texts=reference_corpus,
        dictionary=dictionary,
        coherence=measure,
        topn=top_n,
    )
    return float(cm.get_coherence()), [float(x) for x in cm.get_coherence_per_topic()]


def topic_diversity(topics: list[list[str]], top_n: int = 10) -> float:
    """Dieng, Ruiz & Blei (2020), "Topic Modeling in Embedding Spaces."
    TACL. Proportion of unique words across all topics' top-n words."""
    truncated = [words[:top_n] for words in topics]
    unique_words = {w for words in truncated for w in words}
    total_slots = sum(len(words) for words in truncated) or 1
    return len(unique_words) / total_slots


def topic_diversity_glocom(topics: list[list[str]], top_n: int = 10) -> float:
    """GloCOM's own TD implementation (`evaluations/topic_diversity.py::
    compute_TD` in the official repo, github.com/qducnguyen/GloCOM) -
    NOT the same formula as `topic_diversity()` above. It counts only
    words that appear in EXACTLY ONE topic's top-n list (a word appearing
    in 2+ topics contributes 0, not 1, to the numerator), divided by
    K*top_n:  TD = |{words with cross-topic frequency == 1}| / (K * top_n)
    Standard Topic Diversity (Dieng et al. 2020, used above) instead
    counts every unique word once regardless of how many topics repeat it
    - the two metrics diverge whenever any word repeats across topics, so
    do not compare a `topic_diversity()` number against a paper reporting
    this GloCOM-style TD."""
    from collections import Counter

    truncated = [words[:top_n] for words in topics]
    counts = Counter(w for words in truncated for w in words)
    singly_occurring = sum(1 for w, c in counts.items() if c == 1)
    total_slots = sum(len(words) for words in truncated) or 1
    return singly_occurring / total_slots


def _agreement(list1: list[str], list2: list[str], depth: int) -> float:
    set1, set2 = set(list1[:depth]), set(list2[:depth])
    return len(set1 & set2) / depth


def _rbo(list1: list[str], list2: list[str], p: float) -> float:
    """Extrapolated RBO point estimate (Webber, Moffat & Zobel 2010, Eq.
    32), specialized for the equal-length, no-ties case that always holds
    here (every topic's top-n word list has exactly top_n distinct words)."""
    k = len(list1)
    term1 = (1 - p) / p * sum((p**d) * _agreement(list1, list2, d) for d in range(1, k + 1))
    term2 = (p**k) * _agreement(list1, list2, k) if k > 0 else 0.0
    return term1 + term2


def irbo(topics: list[list[str]], top_n: int = 10, p: float = 0.9) -> float:
    """Inverted Rank-Biased Overlap (Bianchi, Terragni & Hovy, ACL 2021).
    1 - average pairwise RBO between every pair of topics' top-n word
    rankings. Default p=0.9 matches the reference implementation."""
    truncated = [words[:top_n] for words in topics]
    if len(truncated) < 2:
        raise ValueError("IRBO requires at least 2 topics to compare")
    scores = [_rbo(t1, t2, p) for t1, t2 in combinations(truncated, 2)]
    return 1.0 - (sum(scores) / len(scores))


METRIC_FUNCTIONS = {
    "npmi": lambda topics, ref, top_n=10: coherence(topics, ref, top_n, "c_npmi")[0],
    "cv": lambda topics, ref, top_n=10: coherence(topics, ref, top_n, "c_v")[0],
    "topic_diversity": lambda topics, ref, top_n=10: topic_diversity(topics, top_n),
    "glocom_td": lambda topics, ref, top_n=10: topic_diversity_glocom(topics, top_n),
    "irbo": lambda topics, ref, top_n=10: irbo(topics, top_n),
}


def compute_topic_metrics(
    topics: list[list[str]],
    reference_corpus: list[list[str]],
    metric_names: list[str],
    top_n: int = 10,
) -> dict[str, float]:
    return {name: METRIC_FUNCTIONS[name](topics, reference_corpus, top_n) for name in metric_names}
