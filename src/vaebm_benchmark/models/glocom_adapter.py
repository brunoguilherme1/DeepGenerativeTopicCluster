"""GloCOM adapter: reproduces the official repo's own evaluation protocol -
full-corpus, transductive (no held-out test split; see _glocom_source.py's
module docstring for why - the official repo's own dataloader.py reloads
the training bow.npz as "test" too).

Two ways to fit, in order of faithfulness:

  1. `fit_precomputed(bundle)` - trains on the EXACT precomputed bow.npz /
     global_bow.npz / global_maps.txt / vocab.txt the official repo ships
     for SearchSnippets (see datasets/definitions/glocom_official.py).
     Nothing about the "global clustering context" is re-derived here;
     this is the official artifact, verbatim. Use this whenever the
     official repo ships precomputed arrays for the target dataset.

  2. `fit(documents)` - a from-raw-text fallback for datasets the official
     repo does NOT ship precomputed arrays for (e.g. StackOverflow,
     Biomedical - see configs/datasets/*.yaml). Builds the "global
     clustering context" following the paper's own description (embed
     with a pretrained LM, K-Means-cluster, pool cluster members' BOW
     vectors) - not a guess, but also not verified byte-for-byte against
     an official reference the way path (1) is, since no such reference
     exists for these datasets. See docs/methodological_notes.md.

Default hyperparameters below match `run.py`'s own CLI defaults (verified
by cloning the official repo and reading `run.py::parse_args` directly),
NOT the `GloCOM.__init__` class defaults, which differ for two arguments:
`prior_var` (CLI default 0.1, class default 0.01) and `weight_loss_ECR`
(CLI default 60.0, class default 30.0) - `run.py` always passes its own
CLI value explicitly, so 0.1/60.0 are what the paper's actual reported
numbers were produced with.

See _glocom_source.py and models/base.py for what is/isn't vendored vs.
this repo's own code.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from vaebm_benchmark.models.base import ProtocolModelAdapter
from vaebm_benchmark.models._glocom_source import GloCOM
from vaebm_benchmark.models._topmost_bases import (
    DEFAULT_VOCAB_SIZE_CAP,
    BowDataset,
    resolve_device,
    run_preprocess,
    vectorize_against_vocab,
)


class GloCOMAdapter(ProtocolModelAdapter):
    def __init__(
        self,
        num_topics: int,
        num_global_clusters: int,
        vocab_size_cap: int = DEFAULT_VOCAB_SIZE_CAP,
        embedding_model: str = "all-MiniLM-L6-v2",
        en_units: int = 200,
        dropout: float = 0.0,
        embed_size: int = 200,
        sinkhorn_alpha: float = 20.0,
        sinkhorn_max_iter: int = 100,
        beta_temp: float = 0.2,
        aug_coef: float = 0.5,
        prior_var: float = 0.1,
        weight_loss_ECR: float = 60.0,
        epochs: int = 200,
        learning_rate: float = 0.002,
        batch_size: int = 200,
        num_top_words: int = 15,
        seed: int = 42,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.num_topics = num_topics
        self.num_global_clusters = num_global_clusters
        self.vocab_size_cap = vocab_size_cap
        self.embedding_model_name = embedding_model
        self.en_units = en_units
        self.dropout = dropout
        self.embed_size = embed_size
        self.sinkhorn_alpha = sinkhorn_alpha
        self.sinkhorn_max_iter = sinkhorn_max_iter
        self.beta_temp = beta_temp
        self.aug_coef = aug_coef
        self.prior_var = prior_var
        self.weight_loss_ECR = weight_loss_ECR
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_top_words = num_top_words
        self.seed = seed
        self.device_name = device
        self.verbose = verbose

        self._trainer = None
        self._preprocess = None
        self._vocab: Optional[list[str]] = None
        self._device = None
        self._encoder = None
        self._kmeans = None
        self._cluster_bow: Optional[np.ndarray] = None
        self._precomputed_train_data: Optional[np.ndarray] = None  # set only by fit_precomputed()

    def fit(self, documents: list[str]) -> "GloCOMAdapter":
        """Fallback path for datasets the official repo does NOT ship
        precomputed arrays for - see module docstring. Builds the global
        clustering context from raw text following the paper's own
        description."""
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans

        self._preprocess, self._vocab, local_bow, _train_texts = run_preprocess(
            documents, vocab_size_cap=self.vocab_size_cap, seed=self.seed, verbose=self.verbose
        )

        self._encoder = SentenceTransformer(self.embedding_model_name)
        embeddings = self._encoder.encode(documents, show_progress_bar=False)

        num_global_clusters = min(self.num_global_clusters, len(documents))
        self._kmeans = KMeans(n_clusters=num_global_clusters, random_state=self.seed, n_init=10)
        cluster_ids = self._kmeans.fit_predict(embeddings)

        self._cluster_bow = np.zeros((num_global_clusters, local_bow.shape[1]), dtype="float32")
        for cluster_id in range(num_global_clusters):
            members = cluster_ids == cluster_id
            if members.any():
                self._cluster_bow[cluster_id] = local_bow[members].sum(axis=0)
        global_bow_per_doc = self._cluster_bow[cluster_ids]

        train_data = np.concatenate([local_bow, global_bow_per_doc], axis=1)
        self._train_and_wrap(train_data, self._vocab, pretrained_WE=None)
        return self

    def fit_precomputed(self, bundle) -> "GloCOMAdapter":
        """Trains on the OFFICIAL precomputed artifact bundle (bow.npz /
        global_bow.npz / global_maps.txt / vocab.txt / word_embeddings.npz)
        - see datasets/definitions/glocom_official.py. This is the most
        faithful path: `train_data` below is built EXACTLY the way the
        official `dataloader.py` builds it
        (`np.concatenate((train_bow, global_bow[global_maps]), axis=1)`),
        with no re-derivation of the global clustering context."""
        self._vocab = bundle.vocab
        self._preprocess = None  # unused on this path; get_document_topics() below handles it
        self._precomputed_train_data = bundle  # kept so get_document_topics can look up the same rows

        global_bow_per_doc = bundle.global_bow[bundle.global_maps]
        train_data = np.concatenate([bundle.bow, global_bow_per_doc], axis=1)
        self._train_and_wrap(train_data, bundle.vocab, pretrained_WE=bundle.word_embeddings)
        return self

    def _train_and_wrap(self, train_data: np.ndarray, vocab: list[str], pretrained_WE: Optional[np.ndarray]) -> None:
        from topmost import BasicTrainer

        self._device = resolve_device(self.device_name)
        dataset = BowDataset(train_data, vocab, self._device, batch_size=self.batch_size)

        model = GloCOM(
            vocab_size=len(vocab),
            num_topics=self.num_topics,
            en_units=self.en_units,
            dropout=self.dropout,
            pretrained_WE=pretrained_WE,
            embed_size=self.embed_size,
            sinkhorn_alpha=self.sinkhorn_alpha,
            sinkhorn_max_iter=self.sinkhorn_max_iter,
            beta_temp=self.beta_temp,
            aug_coef=self.aug_coef,
            prior_var=self.prior_var,
            weight_loss_ECR=self.weight_loss_ECR,
        ).to(self._device)

        self._trainer = BasicTrainer(
            model,
            dataset,
            num_top_words=self.num_top_words,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        self._trainer.train()

    def _global_bow_for(self, documents: list[str]) -> np.ndarray:
        embeddings = self._encoder.encode(documents, show_progress_bar=False)
        cluster_ids = self._kmeans.predict(embeddings)
        return self._cluster_bow[cluster_ids]

    def get_topics(self, top_n: int = 10) -> list[list[str]]:
        return [words.split()[:top_n] for words in self._trainer.get_top_words(top_n)]

    def get_document_topics(self, documents: list[str]) -> Optional[np.ndarray]:
        """Under this repo's `full_corpus` split strategy (see
        protocols/glocom_protocol.py), `documents` is always exactly the
        same corpus fit() was called on - train_theta == test_theta, per
        the official repo's own explicit comment
        ("train_theta == test_theta for the short text problem")."""
        import torch

        if self._precomputed_train_data is not None:
            bundle = self._precomputed_train_data
            if documents != bundle.documents:
                raise ValueError(
                    "GloCOMAdapter.fit_precomputed() only supports querying the exact "
                    "same documents it was fit on (this protocol is transductive, per "
                    "the official repo - see docs/methodological_notes.md)."
                )
            global_bow_per_doc = bundle.global_bow[bundle.global_maps]
            combined = np.concatenate([bundle.bow, global_bow_per_doc], axis=1)
        else:
            local_bow = vectorize_against_vocab(self._preprocess, documents, self._vocab)
            global_bow = self._global_bow_for(documents)
            combined = np.concatenate([local_bow, global_bow], axis=1)

        combined_tensor = torch.from_numpy(combined.astype("float32")).to(self._device)
        return self._trainer.test(combined_tensor)

    def get_document_clusters(self, documents: list[str]) -> list[int]:
        doc_topics = self.get_document_topics(documents)
        return [int(i) for i in np.argmax(doc_topics, axis=1)]

    def get_document_embeddings(self, documents: list[str]) -> Optional[np.ndarray]:
        """The SBERT encoder used to build the "global clustering
        context" - a preprocessing utility of GloCOM's own pipeline, not
        a representation specific to its topic-discovery mechanism, but
        a direct native output (same caveat as fastopic_adapter.py's).
        `fit_precomputed()` (the official-artifact GloCOM protocol path)
        never sets `self._encoder`, since it needs no SBERT step at all
        (the global context is already precomputed) - constructed lazily
        here so this method still works in that case."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.embedding_model_name)
        return self._encoder.encode(list(documents), show_progress_bar=False)
