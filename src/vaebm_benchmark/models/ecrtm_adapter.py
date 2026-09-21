"""ECRTM adapter (Wu, Dong, Nguyen & Luu, "Effective Neural Topic
Modeling with Embedding Clustering Regularization," ICML 2023). Ported
2026-09-21 from document-topic-evaluatio-arena's own
`src/dtea/models/definitions/ecrtm_model.py` (the two repos are
intentionally independent, see models/base.py's own docstring) - same
underlying `topmost.ECRTM` + `topmost.BasicTrainer` (the official
TopMost toolkit, from the same authors as the ECRTM paper), reusing
this repo's own pre-existing `_topmost_bases.py` helpers (already
shared with GloCOM/FASTopic here) rather than duplicating them.

Unlike topmost.ETM, topmost.ECRTM's word_embeddings are always a
trainable nn.Parameter regardless of whether pretrained_WE is supplied
(no requires_grad=False branch) - so this adapter does NOT need to
pretrain word2vec embeddings first; random init plus end-to-end
training is exactly what the official model does by default.

Stopword removal is unconditional here, same as GloCOM/FASTopic:
_topmost_bases.py::run_preprocess -> topmost.Preprocess's own default
stopwords="English", never overridden (see models/hicot_adapter.py's
own comment on why this must be unconditional for a baseline).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from vaebm_benchmark.models.base import ProtocolModelAdapter
from vaebm_benchmark.models._topmost_bases import (
    DEFAULT_VOCAB_SIZE_CAP,
    BowDataset,
    resolve_device,
    run_preprocess,
    vectorize_against_vocab,
)


class ECRTMAdapter(ProtocolModelAdapter):
    def __init__(
        self,
        num_topics: int,
        vocab_size_cap: int = DEFAULT_VOCAB_SIZE_CAP,
        en_units: int = 200,
        dropout: float = 0.0,
        embed_size: int = 200,
        beta_temp: float = 0.2,
        weight_loss_ECR: float = 100.0,
        sinkhorn_alpha: float = 20.0,
        sinkhorn_max_iter: int = 1000,
        epochs: int = 200,
        learning_rate: float = 0.002,
        batch_size: int = 200,
        num_top_words: int = 15,
        seed: int = 42,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.num_topics = num_topics
        self.vocab_size_cap = vocab_size_cap
        self.en_units = en_units
        self.dropout = dropout
        self.embed_size = embed_size
        self.beta_temp = beta_temp
        self.weight_loss_ECR = weight_loss_ECR
        self.sinkhorn_alpha = sinkhorn_alpha
        self.sinkhorn_max_iter = sinkhorn_max_iter
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

    def fit(self, documents: list[str]) -> "ECRTMAdapter":
        from topmost import ECRTM, BasicTrainer

        self._preprocess, self._vocab, train_bow, _train_texts = run_preprocess(
            documents, vocab_size_cap=self.vocab_size_cap, seed=self.seed, verbose=self.verbose
        )
        self._device = resolve_device(self.device_name)
        dataset = BowDataset(train_bow, self._vocab, self._device, batch_size=self.batch_size)

        model = ECRTM(
            vocab_size=dataset.vocab_size,
            num_topics=self.num_topics,
            en_units=self.en_units,
            dropout=self.dropout,
            embed_size=self.embed_size,
            beta_temp=self.beta_temp,
            weight_loss_ECR=self.weight_loss_ECR,
            sinkhorn_alpha=self.sinkhorn_alpha,
            sinkhorn_max_iter=self.sinkhorn_max_iter,
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
        return self

    def get_topics(self, top_n: int = 10) -> list[list[str]]:
        return [words.split()[:top_n] for words in self._trainer.get_top_words(top_n)]

    def get_document_topics(self, documents: list[str]) -> Optional[np.ndarray]:
        import torch

        bow = vectorize_against_vocab(self._preprocess, documents, self._vocab)
        bow_tensor = torch.from_numpy(bow.astype("float32")).to(self._device)
        return self._trainer.test(bow_tensor)

    def get_document_clusters(self, documents: list[str]) -> list[int]:
        doc_topics = self.get_document_topics(documents)
        return [int(i) for i in np.argmax(doc_topics, axis=1)]

    def get_document_embeddings(self, documents: list[str]) -> Optional[np.ndarray]:
        return None  # no native embedding space, same capability gap as fastopic/lda/glocom/hicot
