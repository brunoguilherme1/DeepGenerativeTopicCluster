"""SBERT + KMeans adapter for the simplified VAE-BM vs. BERTopic experiment
runner (scripts/run_experiment.py). `embedder` accepts ANY SentenceTransformer/
HuggingFace model name - not one fixed checkpoint - so e.g. all-MiniLM-L6-v2,
all-mpnet-base-v2, or any other encoder can be swapped in from the CLI (see
--sbert-embedder). A lightweight embed-then-cluster baseline with no trained
model of its own, just an off-the-shelf sentence encoder plus KMeans.

Has no native topic-word or soft doc-topic output at all - a genuine
capability gap of a plain hard-clustering pipeline, not an oversight (same
gap BERTopicAdapter's own KMeans-backed configuration has for doc-topics, see
its get_document_topics()). Topic words are instead derived via class-based
TF-IDF (c-TF-IDF, _class_based_tfidf below) - the same formula BERTopic
itself uses to turn cluster membership into topic words: Grootendorst, M.
(2022). "BERTopic: Neural topic modeling with a class-based TF-IDF
procedure." arXiv:2203.05794. Reimplemented here (not imported) since this
project and DTEA (document-topic-evaluatio-arena, which has its own
equivalent sbert_kmeans/mpnet_kmeans + c-TF-IDF adapter) are deliberately
independent - see models/base.py.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Optional

import numpy as np

from vaebm_benchmark.models.base import ProtocolModelAdapter


def _class_based_tfidf(documents: list[str], cluster_labels: list[int], top_n: int = 10) -> dict[int, list[str]]:
    clusters = sorted(set(cluster_labels))
    cluster_word_counts: dict[int, Counter] = {c: Counter() for c in clusters}
    for doc, cluster in zip(documents, cluster_labels):
        cluster_word_counts[cluster].update(doc.split())

    word_cluster_df: Counter = Counter()
    for counts in cluster_word_counts.values():
        word_cluster_df.update(counts.keys())

    num_clusters = len(clusters)
    topics: dict[int, list[str]] = {}
    for cluster in clusters:
        counts = cluster_word_counts[cluster]
        total_words_in_cluster = sum(counts.values()) or 1
        scored = []
        for word, count in counts.items():
            tf = count / total_words_in_cluster
            idf = math.log(1 + num_clusters / word_cluster_df[word])
            scored.append((tf * idf, word))
        scored.sort(reverse=True)
        topics[cluster] = [word for _, word in scored[:top_n]]
    return topics


class SBERTKMeansAdapter(ProtocolModelAdapter):
    def __init__(
        self,
        n_clusters: int,
        embedder: str = "all-MiniLM-L6-v2",  # any SentenceTransformer/HuggingFace model name
        random_state: int = 42,
        n_init: int = 10,
    ) -> None:
        self.n_clusters = n_clusters
        self.embedder_name = embedder
        self.random_state = random_state
        self.n_init = n_init
        self._encoder = None
        self._kmeans = None
        self._train_documents: Optional[list[str]] = None
        self._train_clusters: Optional[list[int]] = None

    def fit(self, documents: list[str]) -> "SBERTKMeansAdapter":
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans

        self._train_documents = list(documents)
        self._encoder = SentenceTransformer(self.embedder_name)
        embeddings = self._encoder.encode(self._train_documents, show_progress_bar=False)
        self._kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=self.n_init)
        self._train_clusters = [int(c) for c in self._kmeans.fit_predict(embeddings)]
        return self

    def get_topics(self, top_n: int = 10) -> list[list[str]]:
        if self._train_documents is None:
            raise RuntimeError("Call fit() before get_topics().")
        topics = _class_based_tfidf(self._train_documents, self._train_clusters, top_n=top_n)
        return [topics.get(k, []) for k in range(self.n_clusters)]

    def get_document_topics(self, documents: list[str]) -> Optional[np.ndarray]:
        return None  # hard clustering only - no soft doc-topic distribution, see module docstring

    def get_document_clusters(self, documents: list[str]) -> list[int]:
        documents = list(documents)
        if documents == self._train_documents:
            return list(self._train_clusters)
        embeddings = self._encoder.encode(documents, show_progress_bar=False)
        return [int(c) for c in self._kmeans.predict(embeddings)]

    def get_document_embeddings(self, documents: list[str]) -> Optional[np.ndarray]:
        if self._encoder is None:
            raise RuntimeError("Call fit() before get_document_embeddings().")
        return np.asarray(self._encoder.encode(list(documents), show_progress_bar=False))
