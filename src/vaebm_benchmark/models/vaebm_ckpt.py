"""VAE-BM with oracle-checkpoint-selected epoch: the SAME fixed-alpha
fusion architecture as models/vaebm.py (Encoder/VAEBM, unchanged - alpha
is already a first-class constructor argument there, just always called
with the notebook's own default of 0.99 elsewhere in this codebase), but
trained with the custom per-epoch-best-checkpoint-tracking loop
models/vaebm_poe.py::VaeBmPoEFit introduced (recomputing clustering
accuracy after every epoch and keeping the best-so-far weights instead
of just the final epoch's).

Purpose (2026-09-18, user-authorized): test whether a fixed-alpha VAE-BM
run at alpha=0 (=> mu_f = mu_emb, i.e. the lexical/BoW branch contributes
NOTHING and the fused representation is driven entirely by the
embedding branch - see the mu/log_sigma blend in vaebm.py::Encoder.call)
with the GTE-large embedder and a very low learning rate can approximate
the SBERT-GTE+KMeans baseline's own behavior (cluster directly on a
pretrained embedding, no lexical signal, no VAE training pulling the
representation toward a reconstruction objective). alpha/lr are both
left as regular constructor arguments here (not hardcoded to 0/1e-5) -
this file implements "any fixed alpha, with oracle checkpoint
selection", not solely the alpha=0 replication experiment; the specific
alpha/lr/embedder values for that experiment are set by the sweep
script/env vars that build this model, not by this file.

Decoder/ELBO/reparameterization/topic-word scoring are UNCHANGED from
vaebm.py (imported, not reimplemented) - the only thing this file adds
on top of plain vaebm.py is the epoch-level bookkeeping loop, copied
near-verbatim from vaebm_poe.py::VaeBmPoEFit (see that module's own
docstring for the oracle-labels contract this shares unchanged: labels,
when given, are used ONLY to rank already-computed epochs after the
fact, never in the loss/gradient).
"""

from __future__ import annotations

import os

# Must run BEFORE `import tensorflow` below - see vaebm.py's own
# module-level guard/docstring comment for why.
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")

import time
from typing import Optional, Sequence, Union

import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow.keras.optimizers import Adam

from vaebm_benchmark.metrics.clustering_quality import compute_clustering_metrics
from vaebm_benchmark.models._vaebm_shared import build_bow_and_embeddings, transform_bow_and_embeddings
from vaebm_benchmark.models.vaebm import VAEBM


class VaeBmCkptFit:
    """fit_predict()-equivalent pipeline for VAEBM (plain fixed-alpha
    fusion) with the SAME "stop after N wall-clock seconds, keeping the
    best-by-label-accuracy epoch" custom training loop
    vaebm_poe.py::VaeBmPoEFit uses - see this module's own docstring for
    why (testing where a fixed-alpha VAE-BM's OWN best epoch lands
    relative to a pure-embedding+KMeans baseline, not just its final
    epoch after a fixed training budget)."""

    def __init__(self, voc_size=5000, units=50, n_clusters=8, random_state=42,
                 epochs=50, batch_size=128, lr=1e-3, alpha=0.99,
                 max_fit_seconds: Optional[float] = None, verbose=1):
        self.voc_size = voc_size
        self.units = units
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.alpha = alpha
        self.max_fit_seconds = max_fit_seconds
        self.verbose = verbose

        self.vectorizer = None
        self.embedder = None
        self.model: Optional[VAEBM] = None
        self.kmeans: Optional[KMeans] = None
        self.epochs_completed = 0
        self.best_epoch = 0
        self.best_acc = -1.0

    def fit_predict(self, texts: Sequence[str], vectorizer_type="tfidf",
                     embedder: Union[str, np.ndarray] = "all-MiniLM-L6-v2",
                     dim=(1500, 1000, 500), dim_emb=(368,),
                     vocabulary: Optional[Sequence[str]] = None,
                     labels: Optional[Sequence[int]] = None):
        X_bow, E, self.vectorizer, self.embedder = build_bow_and_embeddings(
            texts, self.voc_size, vectorizer_type, embedder, vocabulary,
        )

        self.model = VAEBM(
            units=self.units, voc=X_bow.shape[1], dim=dim, dim_emb=dim_emb, alpha=self.alpha,
            vectorizer_type=vectorizer_type, embedder=embedder,
        )
        optimizer = Adam(self.lr)
        _ = self.model([X_bow[:1], E[:1]], training=False)

        n = X_bow.shape[0]
        rng = np.random.default_rng(self.random_state)
        best_weights = None
        best_kmeans = None
        best_metric = -1.0  # "acc" if labels given, else -loss (higher is better either way)
        fit_start = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(n)
            epoch_losses = []
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                xb = tf.constant(X_bow[idx])
                eb = tf.constant(E[idx])
                with tf.GradientTape() as tape:
                    elbo = self.model([xb, eb], training=True)
                    loss = -tf.reduce_mean(elbo)
                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                epoch_losses.append(float(loss.numpy()))
            self.epochs_completed = epoch

            mu, _, _ = self.model.encoder([X_bow, E], training=False)
            Z = mu.numpy()
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=22, n_init="auto")
            preds = kmeans.fit_predict(Z)

            mean_loss = float(np.mean(epoch_losses))
            if labels is not None:
                acc = compute_clustering_metrics(preds, labels, ["acc"])["acc"]
                metric = acc
            else:
                acc = None
                metric = -mean_loss

            if self.verbose >= 1:
                acc_str = f" acc={acc:.4f}" if acc is not None else ""
                print(f"[VAE-BM-Ckpt alpha={self.alpha}] epoch {epoch}/{self.epochs} loss={mean_loss:.4f}{acc_str}", flush=True)

            if metric > best_metric:
                best_metric = metric
                best_weights = [w.numpy().copy() for w in self.model.trainable_variables]
                best_kmeans = kmeans
                self.best_epoch = epoch
                self.best_acc = acc if acc is not None else float("nan")

            elapsed = time.perf_counter() - fit_start
            if self.max_fit_seconds is not None and elapsed > self.max_fit_seconds:
                if self.verbose >= 1:
                    print(f"[VAE-BM-Ckpt] max_fit_seconds={self.max_fit_seconds}s reached after epoch "
                          f"{epoch}/{self.epochs} - stopping early, keeping best epoch {self.best_epoch}.", flush=True)
                break

        if best_weights is not None:
            for var, val in zip(self.model.trainable_variables, best_weights):
                var.assign(val)
        self.kmeans = best_kmeans if best_kmeans is not None else kmeans

        mu, _, _ = self.model.encoder([X_bow, E], training=False)
        Z = mu.numpy()
        return self.kmeans.predict(Z).tolist(), Z

    def predict(self, texts: Sequence[str]):
        X_bow, E = transform_bow_and_embeddings(texts, self.vectorizer, self.embedder)
        mu, _, _ = self.model.encoder([X_bow, E], training=False)
        Z = mu.numpy()
        return self.kmeans.predict(Z).tolist(), Z

    def top_words_by_freq_exact(self, texts: Sequence[str], top_m: int = 20):
        from vaebm_benchmark.models._vaebm_topic_scoring import energy_scores

        X = self.vectorizer.transform(texts)
        from sklearn.feature_extraction.text import TfidfVectorizer
        if isinstance(self.vectorizer, TfidfVectorizer):
            X = X.tocoo(copy=False)
            X.data = np.log1p(X.data)
            X = X.tocsr()
        X_bow = X.toarray().astype(np.float32)
        E = self.embedder.encode(texts, batch_size=128, convert_to_numpy=True).astype(np.float32)

        mu, _, _ = self.model.encoder([X_bow, E], training=False)
        Z = mu.numpy()
        preds = self.kmeans.predict(Z)

        vocab = self.vectorizer.get_feature_names_out()
        R = self.model.decoder.R.numpy()
        b = self.model.decoder.b.numpy()

        top_words_energy, top_words_freq = [], []
        for k in range(self.n_clusters):
            idx_k = np.where(preds == k)[0]
            if idx_k.size == 0:
                continue
            X_k = X[idx_k]
            h_k = Z[idx_k]
            counts_k = np.asarray(X_k.sum(axis=0)).ravel()
            if counts_k.sum() == 0:
                continue
            top_idx_freq = np.argsort(counts_k)[::-1][:top_m]
            top_words_freq.append(vocab[top_idx_freq].tolist())

            logits_k = h_k @ R.T + b
            mask_k = (X_k.toarray() > 0).astype(np.float32)
            scores = energy_scores(logits_k, mask_k, counts_k)
            if not np.any(np.isfinite(scores)):
                continue
            top_idx_energy = np.argsort(scores)[::-1][:top_m]
            top_words_energy.append(vocab[top_idx_energy].tolist())

        return {"energy": top_words_energy, "freq": top_words_freq}
