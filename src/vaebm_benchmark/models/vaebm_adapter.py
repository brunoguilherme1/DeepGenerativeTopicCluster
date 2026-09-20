"""Thin ProtocolModelAdapter wrapper around VaeBmKMeansFit (models/vaebm.py)
- the model itself is untouched (see that module's docstring); this file
only adapts it to the common interface protocols/*.py drive, and exposes
BOTH topic-word views (energy-based and frequency-based) the supplied
top_words_by_freq_exact() produces, per the task's instruction to preserve
both rather than picking one.
"""

from __future__ import annotations

from typing import Optional, Sequence

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
        top_words_mode: str = "energy",  # "energy", "freq", or "static" (requires static_embeddings) - which view get_topics() returns
        vocabulary: Optional[list] = None,  # fixes the exact vocab (protocol fidelity) - see vaebm.py fit_predict
        verbose: int = 1,  # 0: silent, 1: one concise summary line, 2: full Keras per-epoch output - see vaebm.py VaeBmKMeansFit
        kl_weight: float = 1.0,  # see vaebm.py::VAEBM's own comment - "mimic GTE" research knob
        freeze_embedding_branch: bool = False,  # see vaebm.py::VAEBM's own comment
        static_embeddings=None,  # [voc_size, d] array row-aligned to `vocabulary`'s own word order - see vaebm.py::top_words_by_freq_exact's own docstring (2026-09-20 "topic-word generation" research pass)
        static_candidate_pool: Optional[int] = None,  # narrows "static" mode's re-ranking pool to the top-N most frequent present words - see vaebm.py::top_words_by_freq_exact's own docstring
    ) -> None:
        self.n_clusters = n_clusters
        self.vectorizer_type = vectorizer_type
        self.embedder_name = embedder
        self.dim = dim
        self.dim_emb = dim_emb
        self.alpha = alpha
        self.top_words_mode = top_words_mode
        self.vocabulary = vocabulary
        self.static_embeddings = static_embeddings
        self.static_candidate_pool = static_candidate_pool

        self._pipeline = VaeBmKMeansFit(
            voc_size=voc_size,
            units=units,
            n_clusters=n_clusters,
            random_state=random_state,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            verbose=verbose,
            kl_weight=kl_weight,
            freeze_embedding_branch=freeze_embedding_branch,
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
                self._train_documents, top_m=max(top_n, 20), static_embeddings=self.static_embeddings,
                static_candidate_pool=self.static_candidate_pool,
            )
        words = self._topics_cache[self.top_words_mode]
        return [w[:top_n] for w in words]

    def get_topics_both_views(self, top_n: int = 10) -> dict[str, list[list[str]]]:
        if self._topics_cache is None:
            self._topics_cache = self._pipeline.top_words_by_freq_exact(
                self._train_documents, top_m=max(top_n, 20), static_embeddings=self.static_embeddings,
                static_candidate_pool=self.static_candidate_pool,
            )
        views = {
            "energy": [w[:top_n] for w in self._topics_cache["energy"]],
            "freq": [w[:top_n] for w in self._topics_cache["freq"]],
        }
        if "static" in self._topics_cache:
            views["static"] = [w[:top_n] for w in self._topics_cache["static"]]
        return views

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


class VAEBMPoEAdapter(ProtocolModelAdapter):
    """Adapter for models/vaebm_poe.py::VaeBmPoEFit - Product-of-Experts
    fusion instead of vaebm.py's fixed alpha. See that module's docstring
    for the full PoE derivation and the labels/oracle-checkpointing
    contract (models/base.py's "fit() never sees labels" rule is
    deliberately, narrowly relaxed here - fit()'s `labels` param, when
    given, selects the best-by-accuracy epoch's checkpoint under a
    wall-clock training budget; it is NEVER used in the loss/gradient).
    Every result this produces should be tagged
    checkpoint_selection="oracle_label_informed" by the caller."""

    def __init__(
        self,
        n_clusters: int,
        voc_size: int = 5000,
        units: int = 50,
        epochs: int = 50,
        batch_size: int = 128,
        lr: float = 1e-3,
        random_state: int = 42,
        vectorizer_type: str = "tfidf",
        embedder: str = "all-MiniLM-L6-v2",
        dim: tuple = (1500, 1000, 500),
        dim_emb: tuple = (368,),
        max_fit_seconds: Optional[float] = None,
        top_words_mode: str = "energy",
        vocabulary: Optional[list] = None,
        verbose: int = 1,
        kl_weight: float = 1.0,
        freeze_embedding_branch: bool = False,
    ) -> None:
        from vaebm_benchmark.models.vaebm_poe import VaeBmPoEFit

        self.n_clusters = n_clusters
        self.vectorizer_type = vectorizer_type
        self.embedder_name = embedder
        self.dim = dim
        self.dim_emb = dim_emb
        self.top_words_mode = top_words_mode
        self.vocabulary = vocabulary

        self._pipeline = VaeBmPoEFit(
            voc_size=voc_size, units=units, n_clusters=n_clusters, random_state=random_state,
            epochs=epochs, batch_size=batch_size, lr=lr, max_fit_seconds=max_fit_seconds, verbose=verbose,
            kl_weight=kl_weight, freeze_embedding_branch=freeze_embedding_branch,
        )
        self._train_documents: Optional[list[str]] = None
        self._mu_train: Optional[np.ndarray] = None
        self._topics_cache: Optional[dict] = None

    # Mirrors HiCOTAdapter's own top-level epochs_completed/epochs
    # attributes (models/hicot_adapter.py) so cluster_runner.py's own
    # generic `getattr(model, "epochs_completed", None)` reporting works
    # here too, plus the oracle-checkpointing epoch/accuracy actually
    # picked (see this class's own docstring).
    @property
    def epochs_completed(self) -> int:
        return self._pipeline.epochs_completed

    @property
    def epochs(self) -> int:
        return self._pipeline.epochs

    @property
    def best_epoch(self) -> int:
        return self._pipeline.best_epoch

    @property
    def best_acc(self) -> float:
        return self._pipeline.best_acc

    def fit(self, documents: list[str], labels: Optional[Sequence[int]] = None) -> "VAEBMPoEAdapter":
        self._train_documents = list(documents)
        _, mu = self._pipeline.fit_predict(
            documents, vectorizer_type=self.vectorizer_type, embedder=self.embedder_name,
            dim=self.dim, dim_emb=self.dim_emb, vocabulary=self.vocabulary, labels=labels,
        )
        self._mu_train = mu
        return self

    def get_topics(self, top_n: int = 10) -> list[list[str]]:
        if self._topics_cache is None:
            self._topics_cache = self._pipeline.top_words_by_freq_exact(self._train_documents, top_m=max(top_n, 20))
        return [w[:top_n] for w in self._topics_cache[self.top_words_mode]]

    def get_topics_both_views(self, top_n: int = 10) -> dict[str, list[list[str]]]:
        if self._topics_cache is None:
            self._topics_cache = self._pipeline.top_words_by_freq_exact(self._train_documents, top_m=max(top_n, 20))
        return {"energy": [w[:top_n] for w in self._topics_cache["energy"]],
                "freq": [w[:top_n] for w in self._topics_cache["freq"]]}

    def get_document_topics(self, documents: list[str]) -> Optional[np.ndarray]:
        if documents is self._train_documents or documents == self._train_documents:
            return self._mu_train
        _, mu = self._pipeline.predict(documents)
        return np.asarray(mu)

    def get_document_clusters(self, documents: list[str]) -> list[int]:
        if documents is self._train_documents or documents == self._train_documents:
            return self._pipeline.kmeans.predict(self._mu_train).tolist()
        labels, _ = self._pipeline.predict(documents)
        return labels

    def get_document_embeddings(self, documents: list[str]) -> Optional[np.ndarray]:
        return self.get_document_topics(documents)


class VAEBMCkptAdapter(ProtocolModelAdapter):
    """Adapter for models/vaebm_ckpt.py::VaeBmCkptFit - the SAME
    fixed-alpha fusion encoder as VAEBMAdapter (alpha is a real
    constructor argument here too, not hardcoded), but trained with
    VAEBMPoEAdapter's own per-epoch oracle-checkpoint-selection loop
    instead of a fixed epoch count. See that module's own docstring for
    why (testing where a fixed-alpha VAE-BM's best epoch lands relative
    to a pure-embedding+KMeans baseline) and the same labels/oracle-
    checkpointing contract VAEBMPoEAdapter documents above."""

    def __init__(
        self,
        n_clusters: int,
        voc_size: int = 5000,
        units: int = 50,
        epochs: int = 50,
        batch_size: int = 128,
        lr: float = 1e-3,
        alpha: float = 0.99,
        random_state: int = 42,
        vectorizer_type: str = "tfidf",
        embedder: str = "all-MiniLM-L6-v2",
        dim: tuple = (1500, 1000, 500),
        dim_emb: tuple = (368,),
        max_fit_seconds: Optional[float] = None,
        top_words_mode: str = "energy",
        vocabulary: Optional[list] = None,
        verbose: int = 1,
        kl_weight: float = 1.0,
        freeze_embedding_branch: bool = False,
        unfreeze_after_epoch: Optional[int] = None,
        post_unfreeze_lr: Optional[float] = None,
        oracle_metric: str = "acc",
    ) -> None:
        from vaebm_benchmark.models.vaebm_ckpt import VaeBmCkptFit

        self.n_clusters = n_clusters
        self.vectorizer_type = vectorizer_type
        self.embedder_name = embedder
        self.dim = dim
        self.dim_emb = dim_emb
        self.top_words_mode = top_words_mode
        self.vocabulary = vocabulary

        self._pipeline = VaeBmCkptFit(
            voc_size=voc_size, units=units, n_clusters=n_clusters, random_state=random_state,
            epochs=epochs, batch_size=batch_size, lr=lr, alpha=alpha,
            max_fit_seconds=max_fit_seconds, verbose=verbose,
            kl_weight=kl_weight, freeze_embedding_branch=freeze_embedding_branch,
            unfreeze_after_epoch=unfreeze_after_epoch, post_unfreeze_lr=post_unfreeze_lr,
            oracle_metric=oracle_metric,
        )
        self._train_documents: Optional[list[str]] = None
        self._mu_train: Optional[np.ndarray] = None
        self._topics_cache: Optional[dict] = None

    @property
    def epochs_completed(self) -> int:
        return self._pipeline.epochs_completed

    @property
    def epochs(self) -> int:
        return self._pipeline.epochs

    @property
    def best_epoch(self) -> int:
        return self._pipeline.best_epoch

    @property
    def best_acc(self) -> float:
        return self._pipeline.best_acc

    def fit(self, documents: list[str], labels: Optional[Sequence[int]] = None) -> "VAEBMCkptAdapter":
        self._train_documents = list(documents)
        _, mu = self._pipeline.fit_predict(
            documents, vectorizer_type=self.vectorizer_type, embedder=self.embedder_name,
            dim=self.dim, dim_emb=self.dim_emb, vocabulary=self.vocabulary, labels=labels,
        )
        self._mu_train = mu
        return self

    def get_topics(self, top_n: int = 10) -> list[list[str]]:
        if self._topics_cache is None:
            self._topics_cache = self._pipeline.top_words_by_freq_exact(self._train_documents, top_m=max(top_n, 20))
        return [w[:top_n] for w in self._topics_cache[self.top_words_mode]]

    def get_topics_both_views(self, top_n: int = 10) -> dict[str, list[list[str]]]:
        if self._topics_cache is None:
            self._topics_cache = self._pipeline.top_words_by_freq_exact(self._train_documents, top_m=max(top_n, 20))
        return {"energy": [w[:top_n] for w in self._topics_cache["energy"]],
                "freq": [w[:top_n] for w in self._topics_cache["freq"]]}

    def get_document_topics(self, documents: list[str]) -> Optional[np.ndarray]:
        if documents is self._train_documents or documents == self._train_documents:
            return self._mu_train
        _, mu = self._pipeline.predict(documents)
        return np.asarray(mu)

    def get_document_clusters(self, documents: list[str]) -> list[int]:
        if documents is self._train_documents or documents == self._train_documents:
            return self._pipeline.kmeans.predict(self._mu_train).tolist()
        labels, _ = self._pipeline.predict(documents)
        return labels

    def get_document_embeddings(self, documents: list[str]) -> Optional[np.ndarray]:
        return self.get_document_topics(documents)


class VAEBMDECAdapter(ProtocolModelAdapter):
    """Adapter for models/vaebm_dec.py::VaeBmDECFit - the SAME alpha-fusion
    encoder as VAEBMAdapter, but Deep Embedded Clustering trained jointly
    with the VAE instead of a post-hoc KMeans call. See that module's
    docstring for the DEC math and the same labels/oracle-checkpointing
    contract VAEBMPoEAdapter documents above."""

    def __init__(
        self,
        n_clusters: int,
        voc_size: int = 5000,
        units: int = 50,
        epochs: int = 50,
        batch_size: int = 128,
        lr: float = 1e-3,
        alpha: float = 0.99,
        lambda_c: float = 0.1,
        random_state: int = 42,
        vectorizer_type: str = "tfidf",
        embedder: str = "all-MiniLM-L6-v2",
        dim: tuple = (1500, 1000, 500),
        dim_emb: tuple = (368,),
        max_fit_seconds: Optional[float] = None,
        top_words_mode: str = "energy",
        vocabulary: Optional[list] = None,
        verbose: int = 1,
        kl_weight: float = 1.0,
        freeze_embedding_branch: bool = False,
    ) -> None:
        from vaebm_benchmark.models.vaebm_dec import VaeBmDECFit

        self.n_clusters = n_clusters
        self.vectorizer_type = vectorizer_type
        self.embedder_name = embedder
        self.dim = dim
        self.dim_emb = dim_emb
        self.top_words_mode = top_words_mode
        self.vocabulary = vocabulary

        self._pipeline = VaeBmDECFit(
            voc_size=voc_size, units=units, n_clusters=n_clusters, random_state=random_state,
            epochs=epochs, batch_size=batch_size, lr=lr, alpha=alpha, lambda_c=lambda_c,
            max_fit_seconds=max_fit_seconds, verbose=verbose,
            kl_weight=kl_weight, freeze_embedding_branch=freeze_embedding_branch,
        )
        self._train_documents: Optional[list[str]] = None
        self._mu_train: Optional[np.ndarray] = None
        self._topics_cache: Optional[dict] = None

    @property
    def epochs_completed(self) -> int:
        return self._pipeline.epochs_completed

    @property
    def epochs(self) -> int:
        return self._pipeline.epochs

    @property
    def best_epoch(self) -> int:
        return self._pipeline.best_epoch

    @property
    def best_acc(self) -> float:
        return self._pipeline.best_acc

    def fit(self, documents: list[str], labels: Optional[Sequence[int]] = None) -> "VAEBMDECAdapter":
        self._train_documents = list(documents)
        _, mu = self._pipeline.fit_predict(
            documents, vectorizer_type=self.vectorizer_type, embedder=self.embedder_name,
            dim=self.dim, dim_emb=self.dim_emb, vocabulary=self.vocabulary, labels=labels,
        )
        self._mu_train = mu
        return self

    def get_topics(self, top_n: int = 10) -> list[list[str]]:
        if self._topics_cache is None:
            self._topics_cache = self._pipeline.top_words_by_freq_exact(self._train_documents, top_m=max(top_n, 20))
        return [w[:top_n] for w in self._topics_cache[self.top_words_mode]]

    def get_topics_both_views(self, top_n: int = 10) -> dict[str, list[list[str]]]:
        if self._topics_cache is None:
            self._topics_cache = self._pipeline.top_words_by_freq_exact(self._train_documents, top_m=max(top_n, 20))
        return {"energy": [w[:top_n] for w in self._topics_cache["energy"]],
                "freq": [w[:top_n] for w in self._topics_cache["freq"]]}

    def get_document_topics(self, documents: list[str]) -> Optional[np.ndarray]:
        if documents is self._train_documents or documents == self._train_documents:
            return self._mu_train
        _, mu = self._pipeline.predict(documents)
        return np.asarray(mu)

    def get_document_clusters(self, documents: list[str]) -> list[int]:
        if documents is self._train_documents or documents == self._train_documents:
            return self._pipeline._predict_labels(self._mu_train).tolist()
        labels, _ = self._pipeline.predict(documents)
        return labels

    def get_document_embeddings(self, documents: list[str]) -> Optional[np.ndarray]:
        return self.get_document_topics(documents)
