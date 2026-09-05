"""HiCOT adapter, wrapping the vendored model (models/_hicot_source.py -
see its own module docstring for exactly what was vendored verbatim vs.
dropped-as-unused). The training loop/batch construction/hyperparameter
defaults below are THIS adapter's own reimplementation of HiCOT's
upstream `basic_trainer.py`/`datasethandler/basic_dataset_handler.py`
(not vendored - see rationale below), matching its actual runtime
behavior, not just its class-default arguments:

- **Hyperparameter defaults** are taken from the upstream repo's own
  `utils/config.py` argparse defaults (what `main.py` actually runs with)
  where they differ from `HiCOT`'s own class-signature defaults - e.g.
  `weight_loss_ECR=40.0` (argparse) vs. `250.0` (class signature),
  `max_clusters=9` (argparse) vs. `50` (class signature),
  `threshold_cluster=10` (argparse) vs. `30` (class signature). Verified
  by reading `main.py`'s own `HiCOT(...)` construction call, not assumed
  from the class signature alone.
- **`use_pretrainWE` defaults to `False`**, matching upstream's own
  `--use_pretrainWE` argparse default (`action="store_true",
  default=False)` - i.e. even when a real GloVe `word_embeddings.npz` is
  available (see `datasets/definitions/hicot_datasets.py`'s
  `load_hicot_word_embeddings()`), upstream's own default run does NOT
  use it unless explicitly requested. This adapter mirrors that default
  rather than silently opting every run into pretrained embeddings.
- **Document embeddings are computed live** via `sentence_transformers`
  at `fit()` time (default `all-MiniLM-L6-v2`, 384-dim, matching
  `HiCOT`'s own `doc2vec_size=384` default), NOT downloaded from
  upstream's separate `doc2vec/doc_embeddings_384_.npz` artifact (a file
  this project's own `hicot_datasets.py` does not fetch) - a documented
  simplification: mathematically the same encoding process, just not
  reusing the cached file, so this adapter can fit on ANY document list
  (matching every other adapter's generic `fit(documents)` contract),
  not only the pinned `hicot_*` datasets.
- **BoW/vocabulary**: self-fit via `CountVectorizer(max_features=
  voc_size)` by default (like every other adapter here), OR a fixed
  external `vocabulary=` (e.g. `load_hicot_vocab(dataset_id)`) for exact
  vocab-fidelity - the same `vocabulary=` convention `models/vaebm.py`'s
  own `fit_predict()` already uses.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from vaebm_benchmark.models.base import ProtocolModelAdapter


class HiCOTAdapter(ProtocolModelAdapter):
    def __init__(
        self,
        n_clusters: int,
        voc_size: int = 5000,
        doc_embed_model: str = "all-MiniLM-L6-v2",
        epochs: int = 500,  # upstream utils/config.py argparse default
        batch_size: int = 200,  # upstream default
        lr: float = 0.002,  # upstream default
        dropout: float = 0.2,  # upstream argparse default
        en_units: int = 200,  # HiCOT class's own default (not overridden by argparse)
        embed_size: int = 200,  # HiCOT class's own default - matches the official 200-dim GloVe word_embeddings.npz
        beta_temp: float = 0.2,
        weight_loss_ECR: float = 40.0,  # upstream argparse default (--weight_ECR), NOT the class signature's 250.0
        alpha_ECR: float = 20.0,
        alpha_DT: float = 3.0,
        alpha_TP: float = 20.0,
        weight_loss_TP: float = 250.0,
        weight_loss_DT: float = 250.0,  # upstream argparse default (--weight_loss_DT), NOT the class signature's 10.0
        weight_loss_CLC: float = 1.0,
        weight_loss_CLT: float = 1.0,
        max_clusters: int = 9,  # upstream argparse default, NOT the class signature's 50
        threshold_epoch: int = 10,
        threshold_cluster: int = 10,  # upstream argparse default, NOT the class signature's 30
        method_CL: str = "HAC",
        metric_CL: str = "euclidean",
        sinkhorn_max_iter: int = 5000,
        use_pretrained_word_embeddings: bool = False,  # matches upstream --use_pretrainWE's own default
        vocabulary: Optional[Sequence[str]] = None,
        pretrained_word_embeddings: Optional[np.ndarray] = None,  # [voc_size, embed_size], e.g. load_hicot_word_embeddings()
        random_state: int = 42,
        device: Optional[str] = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.voc_size = voc_size
        self.doc_embed_model = doc_embed_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.en_units = en_units
        self.embed_size = embed_size
        self.beta_temp = beta_temp
        self.weight_loss_ECR = weight_loss_ECR
        self.alpha_ECR = alpha_ECR
        self.alpha_DT = alpha_DT
        self.alpha_TP = alpha_TP
        self.weight_loss_TP = weight_loss_TP
        self.weight_loss_DT = weight_loss_DT
        self.weight_loss_CLC = weight_loss_CLC
        self.weight_loss_CLT = weight_loss_CLT
        self.max_clusters = max_clusters
        self.threshold_epoch = threshold_epoch
        self.threshold_cluster = threshold_cluster
        self.method_CL = method_CL
        self.metric_CL = metric_CL
        self.sinkhorn_max_iter = sinkhorn_max_iter
        self.use_pretrained_word_embeddings = use_pretrained_word_embeddings
        self.vocabulary = list(vocabulary) if vocabulary is not None else None
        self.pretrained_word_embeddings = pretrained_word_embeddings
        self.random_state = random_state
        self.device = device

        self._vectorizer = None
        self._encoder = None
        self._model = None
        self._vocab: list[str] = []

    def fit(self, documents: list[str]) -> "HiCOTAdapter":
        import torch
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        from torch.utils.data import DataLoader, TensorDataset

        from vaebm_benchmark.models._hicot_source import HiCOT

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.random_state)

        documents = list(documents)
        vectorizer_kwargs = {"vocabulary": self.vocabulary} if self.vocabulary is not None else {"max_features": self.voc_size}
        self._vectorizer = CountVectorizer(**vectorizer_kwargs)
        X_bow = self._vectorizer.fit_transform(documents).toarray().astype("float32")
        self._vocab = list(self._vectorizer.get_feature_names_out())

        self._encoder = SentenceTransformer(self.doc_embed_model, device=device)
        doc_embeddings_np = self._encoder.encode(documents, show_progress_bar=False)

        pretrained_WE = None
        if self.use_pretrained_word_embeddings and self.pretrained_word_embeddings is not None:
            pretrained_WE = np.asarray(self.pretrained_word_embeddings, dtype="float32")
            if pretrained_WE.shape[0] != X_bow.shape[1]:
                raise ValueError(
                    f"pretrained_word_embeddings has {pretrained_WE.shape[0]} rows but the fitted "
                    f"vocabulary has {X_bow.shape[1]} words - they must be aligned (pass `vocabulary=` "
                    "matching the same word order the embeddings were built from)."
                )

        bow_tensor = torch.from_numpy(X_bow).to(device)
        doc_embeddings_tensor = torch.tensor(doc_embeddings_np, dtype=torch.float).to(device)
        indices_tensor = torch.arange(len(documents)).to(device)

        self._model = HiCOT(
            vocab_size=X_bow.shape[1],
            num_topics=self.n_clusters,
            en_units=self.en_units,
            dropout=self.dropout,
            threshold_epoch=self.threshold_epoch,
            doc2vec_size=doc_embeddings_np.shape[1],
            pretrained_WE=pretrained_WE,
            embed_size=self.embed_size,
            beta_temp=self.beta_temp,
            weight_loss_CLT=self.weight_loss_CLT,
            threshold_cluster=self.threshold_cluster,
            weight_loss_ECR=self.weight_loss_ECR,
            alpha_ECR=self.alpha_ECR,
            weight_loss_TP=self.weight_loss_TP,
            alpha_TP=self.alpha_TP,
            alpha_DT=self.alpha_DT,
            weight_loss_DT=self.weight_loss_DT,
            vocab=self._vocab,
            doc_embeddings=doc_embeddings_tensor,
            weight_loss_CLC=self.weight_loss_CLC,
            max_clusters=self.max_clusters,
            method_CL=self.method_CL,
            metric_CL=self.metric_CL,
            sinkhorn_max_iter=self.sinkhorn_max_iter,
        ).to(device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        train_dataset = TensorDataset(bow_tensor, indices_tensor, doc_embeddings_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self._model.train()
        for epoch in range(1, self.epochs + 1):  # 1-based, matching upstream basic_trainer.py::train()
            for batch_bow, batch_indices, batch_doc_embeddings in train_dataloader:
                rst_dict = self._model(batch_indices, [batch_bow], epoch_id=epoch, doc_embeddings=batch_doc_embeddings)
                loss = rst_dict["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        self._model.eval()
        return self

    def _theta(self, documents: list[str]) -> np.ndarray:
        import torch

        device = next(self._model.parameters()).device
        X = self._vectorizer.transform(list(documents)).toarray().astype("float32")
        bow_tensor = torch.from_numpy(X).to(device)
        theta_batches = []
        with torch.no_grad():
            for start in range(0, bow_tensor.shape[0], self.batch_size):
                batch = bow_tensor[start : start + self.batch_size]
                theta_batches.append(self._model.get_theta(batch).cpu().numpy())
        return np.concatenate(theta_batches, axis=0)

    def get_topics(self, top_n: int = 10) -> list[list[str]]:
        beta = self._model.get_beta().detach().cpu().numpy()  # [num_topics, vocab_size]
        vocab = np.asarray(self._vocab)
        topics = []
        for row in beta:
            top_idx = np.argsort(row)[::-1][:top_n]
            topics.append(vocab[top_idx].tolist())
        return topics

    def get_document_topics(self, documents: list[str]) -> Optional[np.ndarray]:
        return self._theta(documents)

    def get_document_clusters(self, documents: list[str]) -> list[int]:
        theta = self._theta(documents)
        return [int(i) for i in np.argmax(theta, axis=1)]

    def get_document_embeddings(self, documents: list[str]) -> Optional[np.ndarray]:
        """The same SentenceTransformer encoding fit() used to build
        `doc_embeddings` - HiCOT's own document-side input, not a
        representation specific to its topic-discovery mechanism, but a
        direct, always-available native output (mirrors FASTopicAdapter's
        own get_document_embeddings() docstring on this same point)."""
        return self._encoder.encode(list(documents), show_progress_bar=False)
