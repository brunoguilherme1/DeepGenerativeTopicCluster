"""Unit tests for llm/cluster_context.py - written but NOT executed in
this session, per this task's own instructions. Uses only sklearn's
CountVectorizer (already a project dependency) on tiny synthetic text -
no model, no LLM, no dataset download."""

import numpy as np

from vaebm_benchmark.llm.cluster_context import (
    build_cluster_contexts,
    class_based_tfidf_topics,
    hash_cluster_context,
    representative_documents,
)


def test_class_based_tfidf_finds_cluster_specific_words():
    documents = [
        "cats and dogs are pets", "dogs bark loudly", "cats meow softly",
        "stocks and bonds are investments", "the stock market fell today",
    ]
    labels = [0, 0, 0, 1, 1]
    topics = class_based_tfidf_topics(documents, labels, top_n=5)
    assert set(topics.keys()) == {0, 1}
    # cluster 0 should surface pet-related words, not finance ones
    assert any(word in topics[0] for word in ("dogs", "cats"))
    assert "stock" not in topics[0] and "stocks" not in topics[0]


def test_representative_documents_are_closest_to_centroid():
    documents = ["far document", "near document", "closest document"]
    representation = np.array([[10.0], [1.0], [0.0]])
    centroid = np.array([0.0])
    reps = representative_documents(documents, representation, labels=[0, 0, 0], cluster_id=0, centroid=centroid, top_n=2)
    assert reps == ["closest document", "near document"]  # ranked closest-first


def test_representative_documents_empty_for_missing_cluster():
    documents = ["a", "b"]
    representation = np.array([[0.0], [1.0]])
    reps = representative_documents(documents, representation, labels=[0, 0], cluster_id=99, centroid=np.array([0.0]))
    assert reps == []


def test_build_cluster_contexts_uses_native_topics_when_given():
    documents = ["doc a", "doc b"]
    representation = np.array([[0.0], [1.0]])
    labels = [0, 1]
    centroids = {0: np.array([0.0]), 1: np.array([1.0])}
    native_topics = {0: ["alpha", "beta"], 1: ["gamma", "delta"]}
    contexts, provenance = build_cluster_contexts(documents, representation, labels, centroids, native_topics=native_topics)
    assert provenance == "native_topic_words"
    assert contexts[0]["words"] == ["alpha", "beta"]


def test_build_cluster_contexts_falls_back_to_ctfidf_when_no_native_topics():
    documents = ["cats and dogs", "dogs bark", "stocks rise", "bonds fall"]
    representation = np.array([[0.0], [0.1], [10.0], [10.1]])
    labels = [0, 0, 1, 1]
    centroids = {0: np.array([0.05]), 1: np.array([10.05])}
    contexts, provenance = build_cluster_contexts(documents, representation, labels, centroids, native_topics=None)
    assert provenance == "ctfidf_adapter"
    assert len(contexts[0]["words"]) > 0


def test_cluster_context_hash_is_deterministic_and_order_invariant():
    context_a = {1: {"words": ["x", "y"], "examples": ["doc"]}, 3: {"words": ["z"], "examples": []}}
    context_b = {3: {"words": ["z"], "examples": []}, 1: {"words": ["x", "y"], "examples": ["doc"]}}
    assert hash_cluster_context(context_a) == hash_cluster_context(context_b)


def test_cluster_context_hash_changes_when_content_changes():
    context_v1 = {1: {"words": ["x", "y"], "examples": ["doc"]}}
    context_v2 = {1: {"words": ["x", "z"], "examples": ["doc"]}}  # one word different
    assert hash_cluster_context(context_v1) != hash_cluster_context(context_v2)
