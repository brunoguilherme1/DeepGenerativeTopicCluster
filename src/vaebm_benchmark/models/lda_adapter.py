"""LDA adapter: scikit-learn's own `LatentDirichletAllocation` (Hoffman,
Bach & Blei's online variational Bayes, `learning_method="online"` -
scikit-learn's default and the only method that scales past a handful of
documents) - a standard baseline every one of ECRTM/HiCOT's own
comparison tables includes, given here as-is, no reimplementation.

Exposes a GENUINE document-topic distribution: `transform()` returns
`p(topic|document)`, a real probability simplex (rows sum to 1) - unlike
VAE-BM's `mu` (see docs/methodological_notes.md #1), this is exactly the
kind of `theta` experiment/classification_runner.py and
experiment/cluster_runner.py's `assignment_source="argmax_theta"`/
`representation_source="theta"` paths are for.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from vaebm_benchmark.models.base import ProtocolModelAdapter


class LDAAdapter(ProtocolModelAdapter):
    def __init__(
        self,
        n_clusters: int,
        voc_size: int = 5000,
        max_iter: int = 10,  # scikit-learn's own LatentDirichletAllocation default
        learning_method: str = "online",
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.voc_size = voc_size
        self.max_iter = max_iter
        self.learning_method = learning_method
        self.random_state = random_state
        self._vectorizer = None
        self._model = None

    def fit(self, documents: list[str]) -> "LDAAdapter":
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer

        self._vectorizer = CountVectorizer(max_features=self.voc_size, stop_words="english")
        X = self._vectorizer.fit_transform(documents)
        self._model = LatentDirichletAllocation(
            n_components=self.n_clusters,
            max_iter=self.max_iter,
            learning_method=self.learning_method,
            random_state=self.random_state,
        )
        self._model.fit(X)
        return self

    def get_topics(self, top_n: int = 10) -> list[list[str]]:
        vocab = self._vectorizer.get_feature_names_out()
        topics = []
        for component in self._model.components_:
            top_idx = np.argsort(component)[::-1][:top_n]
            topics.append(vocab[top_idx].tolist())
        return topics

    def get_document_topics(self, documents: list[str]) -> Optional[np.ndarray]:
        X = self._vectorizer.transform(documents)
        return self._model.transform(X)

    def get_document_clusters(self, documents: list[str]) -> list[int]:
        theta = self.get_document_topics(documents)
        return [int(i) for i in np.argmax(theta, axis=1)]

    def get_document_embeddings(self, documents: list[str]) -> Optional[np.ndarray]:
        """LDA has no separate embedding space - its own doc-topic
        distribution IS its native document representation, same value
        get_document_topics() returns."""
        return self.get_document_topics(documents)
