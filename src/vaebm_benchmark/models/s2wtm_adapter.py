"""S2WTM adapter (Adhya & Sanyal, "S2WTM: Spherical Sliced-Wasserstein
Autoencoder for Topic Modeling," Findings of ACL 2025). Ported
2026-09-21 from document-topic-evaluatio-arena's own
`src/dtea/models/definitions/s2wtm_model.py` (the two repos are
intentionally independent, see models/base.py's own docstring). Wraps
the vendored WAE model (models/_s2wtm_source.py, copied verbatim from
the Arena's own vendoring of the official repo -
https://github.com/AdhyaSuman/S2WTM, a plain OCTIS-based research repo,
not pip-installable - see that file's own docstring for exactly what
was trimmed and why) with this repo's own preprocessing/training loop.

Preprocessing reproduces the official `octis/models/S2WTM.py`'s own
`preprocess()`: a gensim `Dictionary` over the corpus, TF-IDF-weighted
(via `gensim.models.TfidfModel`) bag-of-words vectors - NOT raw counts.

STOPWORD FIX (not present in the Arena's own port): the official
preprocessing's tokenizer is a bare `doc.split()` with no stopword
filtering at all - unlike every TopMost-based adapter here (FASTopic/
GloCOM/ECRTM), which default to `stopwords='English'` unconditionally.
Per this project's own mandatory-stopword-removal-for-baselines policy
(models/hicot_adapter.py's own comment), this adapter tokenizes via
utils/text_preprocessing.py::remove_english_stopwords() BEFORE handing
text to gensim's Dictionary, rather than reproducing the official
preprocessing's own gap.

Training loop reproduces the official `wae_sp.py`'s exact loss
computation (reconstruction + sliced-Wasserstein-on-sphere OT loss,
scaled by the same `lamb` factor) - the loop control flow itself
(checkpointing, verbose logging) is not reproduced since it isn't part
of what makes the result correct.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from vaebm_benchmark.models.base import ProtocolModelAdapter
from vaebm_benchmark.models._s2wtm_source import WAE

DEFAULT_VOCAB_SIZE_CAP = 10_000


def _tokenize(documents: list[str]) -> list[list[str]]:
    from vaebm_benchmark.utils.text_preprocessing import remove_english_stopwords

    return [doc.split() for doc in remove_english_stopwords(documents)]


def _bow_to_dense(bow: list[tuple[int, float]], vocab_size: int) -> np.ndarray:
    vec = np.zeros(vocab_size, dtype="float32")
    for token_id, value in bow:
        vec[token_id] = value
    return vec


class S2WTMAdapter(ProtocolModelAdapter):
    def __init__(
        self,
        num_topics: int,
        vocab_size_cap: int = DEFAULT_VOCAB_SIZE_CAP,
        dropout: float = 0.0,
        dist: str = "unif_sphere",
        temperature: float = 0.7,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        num_projections: int = 500,
        beta: float = 1.0,
        p: int = 2,
        seed: int = 42,
        device: Optional[str] = None,
    ) -> None:
        self.num_topics = num_topics
        self.vocab_size_cap = vocab_size_cap
        self.dropout = dropout
        self.dist = dist
        self.temperature = temperature
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_projections = num_projections
        self.beta = beta
        self.p = p
        self.seed = seed
        self.device_name = device

        self._wae = None
        self._dictionary = None
        self._tfidf_model = None
        self._vocab: Optional[list[str]] = None
        self._device = None

    def fit(self, documents: list[str]) -> "S2WTMAdapter":
        import torch
        from gensim.corpora import Dictionary
        from gensim.models import TfidfModel

        torch.manual_seed(self.seed)

        tokenized = _tokenize(documents)
        dictionary = Dictionary(tokenized)
        dictionary.filter_extremes(no_below=1, no_above=1.0, keep_n=self.vocab_size_cap)
        vocab_size = len(dictionary)

        corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]
        tfidf_model = TfidfModel(corpus)
        weighted = [tfidf_model[bow] for bow in corpus]
        train_matrix = np.stack([_bow_to_dense(bow, vocab_size) for bow in weighted])

        self._dictionary = dictionary
        self._tfidf_model = tfidf_model
        self._vocab = [dictionary[i] for i in range(vocab_size)]

        self._device = torch.device(self.device_name) if self.device_name else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        wae = WAE(
            encode_dims=[vocab_size, 1024, 512, self.num_topics],
            decode_dims=[self.num_topics, 512, vocab_size],
            dropout=self.dropout,
            nonlin="relu",
            dist=self.dist,
            batch_size=self.batch_size,
            temperature=self.temperature,
        ).to(self._device)

        train_tensor = torch.from_numpy(train_matrix).to(self._device)
        optimizer = torch.optim.Adam(wae.parameters(), lr=self.learning_rate)
        n_docs = train_tensor.shape[0]
        wae.train()
        for _epoch in range(self.epochs):
            perm = torch.randperm(n_docs, device=self._device)
            for start in range(0, n_docs, self.batch_size):
                idx = perm[start : start + self.batch_size]
                bows = train_tensor[idx]
                if bows.shape[0] < 2:
                    continue

                bows_recon, theta_q = wae(bows)
                theta_prior = wae.sample(ori_data=bows).to(self._device)

                logsoftmax = torch.log_softmax(bows_recon, dim=1)
                rec_loss = -1.0 * torch.sum(bows * logsoftmax)

                ot_loss = wae.sp_swd_loss(theta_q, theta_prior, num_projections=self.num_projections, device=self._device, p=self.p)

                s = torch.sum(bows) / len(bows)
                lamb = 5.0 * s * torch.log(torch.tensor(1.0 * bows.shape[-1])) / torch.log(torch.tensor(2.0))
                ot_loss = ot_loss * lamb

                loss = rec_loss + ot_loss * self.beta

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self._wae = wae
        return self

    def _encode(self, documents: list[str]) -> np.ndarray:
        import torch

        tokenized = _tokenize(documents)
        vocab_size = len(self._vocab)
        corpus = [self._dictionary.doc2bow(tokens) for tokens in tokenized]
        weighted = [self._tfidf_model[bow] for bow in corpus]
        matrix = np.stack([_bow_to_dense(bow, vocab_size) for bow in weighted])
        bows = torch.from_numpy(matrix).to(self._device)

        self._wae.eval()
        with torch.no_grad():
            theta = torch.softmax(self._wae.encode(bows), dim=1)
        return theta.cpu().numpy()

    def get_topics(self, top_n: int = 10) -> list[list[str]]:
        import torch

        self._wae.eval()
        with torch.no_grad():
            num_topics = self._wae.z_dim
            eye = torch.eye(num_topics).to(self._device)
            word_dist = torch.softmax(self._wae.decode(eye), dim=1)
            _top_values, top_indices = torch.topk(word_dist, top_n, dim=1)
        indices = top_indices.cpu().tolist()
        return [[self._vocab[idx] for idx in row] for row in indices]

    def get_document_topics(self, documents: list[str]) -> Optional[np.ndarray]:
        return self._encode(documents)

    def get_document_clusters(self, documents: list[str]) -> list[int]:
        doc_topics = self.get_document_topics(documents)
        return [int(i) for i in np.argmax(doc_topics, axis=1)]

    def get_document_embeddings(self, documents: list[str]) -> Optional[np.ndarray]:
        return None  # no native embedding space, same capability gap as fastopic/lda/glocom/hicot/ecrtm
