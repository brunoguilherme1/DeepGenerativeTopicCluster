"""Interpretable cluster descriptions (top words + representative
documents) for the LLM prompt, with explicit provenance:
`native_topic_words` when the baseline model exposes its own topic words
(model.get_topics()), `ctfidf_adapter` when derived directly from
assigned documents via class-based TF-IDF (the same representation
mechanism BERTopic's own topic representation uses - Grootendorst, M.
(2022). "BERTopic: Neural topic modeling with a class-based TF-IDF
procedure." arXiv:2203.05794) when no native topic words are available
(or when describing REFINED clusters, whose composition the original
model's own native topics never described - see
experiment/llm_refinement_runner.py's `cluster_derived_topic_metrics`).

Representative documents are chosen as those closest to each cluster's
own centroid, in the SAME representation used for edge-point/candidate-
cluster selection (never a different, unrelated geometry).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np


def class_based_tfidf_topics(documents: list[str], cluster_labels: list[int], top_n: int = 15) -> dict[int, list[str]]:
    """Class-based TF-IDF: each cluster's documents are concatenated into
    one "class document"; term frequency is computed within each class
    document, and IDF compares a term's frequency in one class against
    its frequency across all classes (Grootendorst 2022, Eq. 4-5) -
    NOT the usual per-document TF-IDF. A word specific to a small number
    of clusters scores highly even if it is not rare in the corpus
    overall."""
    from sklearn.feature_extraction.text import CountVectorizer

    grouped: dict[int, list[str]] = defaultdict(list)
    for doc, label in zip(documents, cluster_labels):
        grouped[label].append(doc)
    clusters = sorted(grouped)
    class_documents = [" ".join(grouped[c]) for c in clusters]

    vectorizer = CountVectorizer(max_features=20_000)
    counts = vectorizer.fit_transform(class_documents).toarray().astype(float)  # [n_clusters, vocab]
    vocab = vectorizer.get_feature_names_out()

    words_per_class = counts.sum(axis=1, keepdims=True)
    words_per_class[words_per_class == 0] = 1.0
    term_frequency = counts / words_per_class

    total_term_frequency = counts.sum(axis=0)
    average_words_per_class = float(words_per_class.mean())
    inverse_class_frequency = np.log(1.0 + average_words_per_class / np.maximum(total_term_frequency, 1e-9))

    scores = term_frequency * inverse_class_frequency  # [n_clusters, vocab]
    topics: dict[int, list[str]] = {}
    for i, cluster_id in enumerate(clusters):
        top_idx = np.argsort(scores[i])[::-1][:top_n]
        topics[cluster_id] = vocab[top_idx].tolist()
    return topics


def representative_documents(
    documents: list[str],
    representation: np.ndarray,
    labels: list[int],
    cluster_id: int,
    centroid: np.ndarray,
    top_n: int = 3,
) -> list[str]:
    labels_arr = np.asarray(labels)
    member_indices = np.where(labels_arr == cluster_id)[0]
    if member_indices.size == 0:
        return []
    member_distances = np.linalg.norm(representation[member_indices] - centroid, axis=1)
    closest = member_indices[np.argsort(member_distances)[:top_n]]
    return [documents[i] for i in closest]


def build_cluster_contexts(
    documents: list[str],
    representation: np.ndarray,
    labels: list[int],
    centroids: dict[int, np.ndarray],
    native_topics: dict[int, list[str]] | None = None,
    top_words: int = 15,
    top_docs: int = 3,
) -> tuple[dict[int, dict], str]:
    """Returns ({cluster_id: {"words": [...], "examples": [...]}},
    provenance) - provenance is `"native_topic_words"` when
    `native_topics` is given, else `"ctfidf_adapter"`."""
    if native_topics is not None:
        words_by_cluster = {cid: words[:top_words] for cid, words in native_topics.items()}
        provenance = "native_topic_words"
    else:
        words_by_cluster = class_based_tfidf_topics(documents, labels, top_n=top_words)
        provenance = "ctfidf_adapter"

    contexts = {}
    for cluster_id, centroid in centroids.items():
        contexts[cluster_id] = {
            "words": words_by_cluster.get(cluster_id, []),
            "examples": representative_documents(documents, representation, labels, cluster_id, centroid, top_n=top_docs),
        }
    return contexts, provenance


def hash_cluster_context(context_subset: dict) -> str:
    """Stable hash of the SUBSET of cluster contexts actually offered as
    candidates for one document - part of the LLM decision cache key
    (llm/cache.py), so a decision is only reused when the exact candidate
    context content (not just the candidate cluster ids) is unchanged."""
    encoded = json.dumps(context_subset, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
