"""BERTopic adapter for the simplified VAE-BM vs. BERTopic experiment
runner (scripts/run_experiment.py). Wraps the official `bertopic`
package directly - no reimplementation of its c-TF-IDF topic
representation.

FIXED-K ENFORCEMENT (documented per the task's own requirement to state
exactly how K is enforced, and to prefer an officially-supported
mechanism over post-hoc topic merging):

BERTopic's default pipeline is
    SentenceTransformer embed -> UMAP reduce -> HDBSCAN cluster -> c-TF-IDF represent
HDBSCAN chooses its own number of clusters (plus a noise/outlier
"topic -1"), which cannot be compared apples-to-apples against a model
run at a REQUESTED K. BERTopic's own constructor accepts a
`hdbscan_model=` argument that can be ANY object implementing
`.fit(X)` / `.labels_`, NOT only actual HDBSCAN - this is officially
documented BERTopic usage for exactly this situation (see BERTopic's own
"Other clustering models" documentation, e.g. swapping in
`sklearn.cluster.KMeans`). This adapter passes
`KMeans(n_clusters=K, random_state=seed)` as `hdbscan_model`, keeping
UMAP dimensionality reduction as BERTopic's own default (only the
clustering step is swapped, nothing else) - this is a supported
configuration switch, NOT "faking K" via BERTopic's separate `nr_topics`
post-hoc topic-merging parameter (which this adapter does NOT use).

Because KMeans has no notion of an outlier/noise cluster, every document
receives one of the K cluster assignments - there is no "-1" topic to
discard, so `get_document_clusters()` always returns a value in
`[0, K)` for every document, matching VAE-BM's own KMeans-over-mu
behavior for a controlled comparison.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from vaebm_benchmark.models.base import ProtocolModelAdapter


class BERTopicAdapter(ProtocolModelAdapter):
    def __init__(
        self,
        n_clusters: int,
        embedding_model: str = "all-MiniLM-L6-v2",
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_clusters = n_clusters
        self.embedding_model_name = embedding_model
        self.random_state = random_state
        self.verbose = verbose
        self._model = None
        self._train_documents: Optional[list[str]] = None
        self._embedder = None  # lazily constructed only if get_document_embeddings() is actually called

    def fit(self, documents: list[str]) -> "BERTopicAdapter":
        from bertopic import BERTopic
        from sklearn.cluster import KMeans
        from umap import UMAP

        cluster_model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        # UMAP's own stochastic optimization is NOT seeded by KMeans's
        # random_state - without this, re-running with the "same" seed
        # still produces different embeddings/topics (observed directly:
        # two runs at seed=42 gave CV=0.455 vs. 0.495). BERTopic's default
        # UMAP is left as-is otherwise (n_neighbors/n_components/metric
        # defaults unchanged) - only determinism is added.
        reducer_model = UMAP(random_state=self.random_state)
        self._model = BERTopic(
            embedding_model=self.embedding_model_name,
            umap_model=reducer_model,
            hdbscan_model=cluster_model,  # the officially-supported "swap the clustering backend" mechanism - see module docstring
            calculate_probabilities=False,
            verbose=self.verbose,
        )
        self._train_documents = list(documents)
        self._model.fit_transform(self._train_documents)
        return self

    def get_topics(self, top_n: int = 10) -> list[list[str]]:
        topics = []
        for topic_id in range(self.n_clusters):
            words_scores = self._model.get_topic(topic_id)
            if not words_scores:
                # A cluster BERTopic's own c-TF-IDF step found no
                # distinguishing terms for (rare with KMeans, since every
                # cluster is non-empty by construction, but c-TF-IDF can
                # still return an empty list for a degenerate cluster) -
                # keep the topic slot rather than silently reindexing,
                # so topic ids stay aligned with cluster ids.
                topics.append([])
                continue
            topics.append([word for word, _score in words_scores[:top_n]])
        return topics

    def get_document_topics(self, documents: list[str]) -> Optional[np.ndarray]:
        return None  # KMeans-backed BERTopic has no soft doc-topic distribution, same capability gap as sbert_kmeans

    def get_document_clusters(self, documents: list[str]) -> list[int]:
        documents = list(documents)
        if documents == self._train_documents:
            # Already computed during fit_transform() - no need to re-embed/re-predict.
            return [int(t) for t in self._model.topics_]
        topics, _probabilities = self._model.transform(documents)
        return [int(t) for t in topics]

    def get_document_embeddings(self, documents: list[str]) -> Optional[np.ndarray]:
        """BERTopic's own SBERT document embeddings (pre-UMAP-reduction)
        - re-encoded directly via the same embedding model BERTopic
        itself was constructed with, rather than reaching into BERTopic's
        internal `umap_model.embedding_` (which is only populated for the
        exact training documents, not arbitrary held-out ones - this
        method must work for both)."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embedding_model_name)
        return np.asarray(self._embedder.encode(list(documents), show_progress_bar=False))
