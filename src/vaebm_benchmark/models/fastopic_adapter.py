"""FASTopic adapter. Wraps the official `fastopic` PyPI package (the
authors' own implementation) directly - no reimplementation. Official
repo: https://github.com/bobxwu/FASTopic. Wu, X., Nguyen, T., Zhang, D. C.,
Wang, W. Y., & Luu, A. T. (2024). "FASTopic: Pretrained Transformer is a
Fast, Adaptive, Stable, and Transferable Topic Model." NeurIPS 2024.

Default hyperparameters below (DT_alpha=3.0, TW_alpha=2.0, theta_temp=1.0,
epochs=200, learning_rate=0.002) are the `fastopic` package's OWN
constructor/fit_transform defaults, verified against the paper's Appendix
D (verbatim: "we set epsilon_1 as 1/3 and epsilon_2 as 1/2... tau as 1.0
in Eq. (9)... Adam with 200 epochs and learning rate as 0.002... the same
hyperparameters for all reported experiments") - i.e. calling
`FASTopic(num_topics=K)` with no further overrides already reproduces the
paper's exact configuration for every table; nothing here retunes them.

`low_memory` stays False (the package's own default) unless explicitly
overridden - the paper's own training is full-batch (`fit_transform` sets
`batch_size = len(docs)` whenever `low_memory=False`), so silently
enabling low-memory mini-batching above some doc-count threshold (as a
convenience wrapper might) would itself be a protocol deviation from what
produced the paper's published numbers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from vaebm_benchmark.models.base import ProtocolModelAdapter


class FASTopicAdapter(ProtocolModelAdapter):
    def __init__(
        self,
        num_topics: int,
        num_top_words: int = 15,
        doc_embed_model: str = "all-MiniLM-L6-v2",
        epochs: int = 200,
        learning_rate: float = 0.002,
        vocab_size_cap: Optional[int] = None,
        stopwords: str = "English",
        DT_alpha: float = 3.0,
        TW_alpha: float = 2.0,
        theta_temp: float = 1.0,
        normalize_embeddings: bool = False,
        low_memory: bool = False,
        low_memory_batch_size: Optional[int] = None,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.num_topics = num_topics
        self.num_top_words = num_top_words
        self.doc_embed_model = doc_embed_model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.vocab_size_cap = vocab_size_cap
        self.stopwords = stopwords
        self.DT_alpha = DT_alpha
        self.TW_alpha = TW_alpha
        self.theta_temp = theta_temp
        self.normalize_embeddings = normalize_embeddings
        self.low_memory = low_memory
        self.low_memory_batch_size = low_memory_batch_size
        self.device = device
        self.verbose = verbose
        self._model = None

    def fit(self, documents: list[str]) -> "FASTopicAdapter":
        from fastopic import FASTopic
        from topmost import Preprocess

        preprocess = Preprocess(vocab_size=self.vocab_size_cap, stopwords=self.stopwords, verbose=self.verbose)

        self._model = FASTopic(
            num_topics=self.num_topics,
            preprocess=preprocess,
            num_top_words=self.num_top_words,
            device=self.device,
            normalize_embeddings=self.normalize_embeddings,
            doc_embed_model=self.doc_embed_model,
            DT_alpha=self.DT_alpha,
            TW_alpha=self.TW_alpha,
            theta_temp=self.theta_temp,
            low_memory=self.low_memory,
            low_memory_batch_size=self.low_memory_batch_size,
            verbose=self.verbose,
        )
        self._model.fit_transform(
            list(documents),
            epochs=self.epochs,
            learning_rate=self.learning_rate,
        )
        return self

    def get_topics(self, top_n: int = 10) -> list[list[str]]:
        return [words.split()[:top_n] for words in self._model.get_top_words(top_n)]

    def get_document_topics(self, documents: list[str]) -> Optional[np.ndarray]:
        return self._model.transform(list(documents))

    def get_document_clusters(self, documents: list[str]) -> list[int]:
        doc_topics = self.get_document_topics(documents)
        return [int(i) for i in np.argmax(doc_topics, axis=1)]
