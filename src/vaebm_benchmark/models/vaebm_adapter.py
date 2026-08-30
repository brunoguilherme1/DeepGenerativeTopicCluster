"""Thin ProtocolModelAdapter wrapper around VaeBmKMeansFit (models/vaebm.py)
- the model itself is untouched (see that module's docstring); this file
only adapts it to the common interface protocols/*.py drive, and exposes
BOTH topic-word views (energy-based and frequency-based) the supplied
top_words_by_freq_exact() produces, per the task's instruction to preserve
both rather than picking one.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from vaebm_benchmark.models.base import ProtocolModelAdapter
from vaebm_benchmark.models.vaebm import VaeBmKMeansFit


class VAEBMAdapter(ProtocolModelAdapter):
    def __init__(
        self,
        n_clusters: int,
        voc_size: int = 5000,
        units: int = 50,
        epochs: int = 30,
        batch_size: int = 128,
        lr: float = 1e-2,
        random_state: int = 42,
        vectorizer_type: str = "tfidf",  # BoW branch (x_bow): "tfidf" or anything else -> CountVectorizer/"bag" - see vaebm.py fit_predict
        embedder: str = "all-MiniLM-L6-v2",  # embedding branch (e_txt): any SentenceTransformer/HuggingFace model name - a SEPARATE knob from vectorizer_type, never "tfidf"/"bag" itself
        dim: tuple = (1500, 1000, 500),
        dim_emb: tuple = (368,),
        alpha: float = 0.99,
        top_words_mode: str = "energy",  # "energy" or "freq" - which view get_topics() returns
        vocabulary: Optional[list] = None,  # fixes the exact vocab (protocol fidelity) - see vaebm.py fit_predict
    ) -> None:
        self.n_clusters = n_clusters
        self.vectorizer_type = vectorizer_type
        self.embedder_name = embedder
        self.dim = dim
        self.dim_emb = dim_emb
        self.alpha = alpha
        self.top_words_mode = top_words_mode
        self.vocabulary = vocabulary

        self._pipeline = VaeBmKMeansFit(
            voc_size=voc_size,
            units=units,
            n_clusters=n_clusters,
            random_state=random_state,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )
        self._train_documents: Optional[list[str]] = None
        self._mu_train: Optional[np.ndarray] = None
        self._topics_cache: Optional[dict] = None

    def fit(self, documents: list[str]) -> "VAEBMAdapter":
        self._train_documents = list(documents)
        _, mu = self._pipeline.fit_predict(
            documents,
            vectorizer_type=self.vectorizer_type,
            embedder=self.embedder_name,
            dim=self.dim,
            dim_emb=self.dim_emb,
            alpha=self.alpha,
            vocabulary=self.vocabulary,
        )
        self._mu_train = mu
        return self

    def get_topics(self, top_n: int = 10) -> list[list[str]]:
        if self._topics_cache is None:
            self._topics_cache = self._pipeline.top_words_by_freq_exact(
                self._train_documents, top_m=max(top_n, 20)
            )
        words = self._topics_cache[self.top_words_mode]
        return [w[:top_n] for w in words]

    def get_topics_both_views(self, top_n: int = 10) -> dict[str, list[list[str]]]:
        if self._topics_cache is None:
            self._topics_cache = self._pipeline.top_words_by_freq_exact(
                self._train_documents, top_m=max(top_n, 20)
            )
        return {
            "energy": [w[:top_n] for w in self._topics_cache["energy"]],
            "freq": [w[:top_n] for w in self._topics_cache["freq"]],
        }

    def get_document_topics(self, documents: list[str]) -> Optional[np.ndarray]:
        """Returns mu (the latent Gaussian mean), NOT a normalized topic
        distribution - see docs/methodological_notes.md. Callers that need
        a distribution-like object should treat this as a latent
        embedding, e.g. for downstream clustering distance, not as p(z|d)."""
        if documents is self._train_documents or documents == self._train_documents:
            return self._mu_train
        _, mu = self._pipeline.predict(documents)
        return mu.numpy() if hasattr(mu, "numpy") else np.asarray(mu)

    def get_document_clusters(self, documents: list[str]) -> list[int]:
        if documents is self._train_documents or documents == self._train_documents:
            return self._pipeline.kmeans.predict(self._mu_train).tolist()
        labels, _ = self._pipeline.predict(documents)
        return labels

    def get_mu(self, documents: list[str]) -> np.ndarray:
        return self.get_document_topics(documents)

    def get_document_embeddings(self, documents: list[str]) -> Optional[np.ndarray]:
        """VAE-BM's native representation is its latent mu - the same
        value get_document_topics()/get_mu() return, exposed here too
        under the common cross-model interface (models/base.py) that
        experiment/llm_refinement_runner.py's `--edge-representation
        native` relies on."""
        return self.get_document_topics(documents)
