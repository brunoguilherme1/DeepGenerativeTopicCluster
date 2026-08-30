"""Common adapter interface every model in this repo implements (the
baselines FASTopic/GloCOM, and VAE-BM). Deliberately small - this project
only ever needs to run 3 models under 2 protocols, not a general registry
across a large model zoo (see DTEA for that pattern, at
document-topic-evaluatio-arena/src/dtea/models/base.py, which this mirrors
in spirit but not by import - the two repos are intentionally independent).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class ProtocolModelAdapter:
    """fit() never sees labels, on any dataset, under any protocol - every
    model here is unsupervised. Every getter takes `documents` explicitly
    (not just "whatever was last fit") so the same fitted model can be
    queried on train and on held-out documents where a protocol defines a
    split.
    """

    def fit(self, documents: list[str]) -> "ProtocolModelAdapter":
        raise NotImplementedError

    def get_topics(self, top_n: int = 10) -> list[list[str]]:
        """Top-n words per topic/cluster, learned during fit()."""
        raise NotImplementedError

    def get_document_topics(self, documents: list[str]) -> Optional[np.ndarray]:
        """Soft doc-topic distribution [len(documents), K], if the model
        natively produces one. None if it does not (e.g. VAE-BM's mu is a
        latent Gaussian mean, not a topic distribution - see
        docs/methodological_notes.md)."""
        return None

    def get_document_clusters(self, documents: list[str]) -> list[int]:
        """Hard cluster/topic assignment per document."""
        raise NotImplementedError

    def get_document_embeddings(self, documents: list[str]) -> Optional[np.ndarray]:
        """This model's own native geometric representation, if it has
        one meaningful for downstream geometry-based post-processing
        (e.g. experiment/llm_refinement_runner.py's edge-point detection,
        `--edge-representation native`) - NOT necessarily a fixed-size
        "embedding" in the deep-learning sense for every model (VAE-BM's
        native representation is its latent mu, not a separate embedding
        model's output). Returns None if this adapter has none; callers
        must handle that explicitly (e.g. by falling back to
        `--edge-representation shared`) rather than assuming every model
        has a usable native representation."""
        return None
