"""VAE-BM-PoE: same two-branch (BoW + sentence-embedding) VAE-BM as
models/vaebm.py, EXCEPT the two branches' Gaussian posteriors are fused
via Product-of-Experts (precision-weighted) instead of a fixed alpha
blend - see docs/methodological_notes.md #15 for the derivation this
implements verbatim:

    tau_b = exp(-2*log_sigma_bow)          # BoW branch precision
    tau_e = exp(-2*log_sigma_emb)          # embedding branch precision
    tau_f = 1 + tau_b + tau_e              # + unit prior precision
    sigma_f^2 = 1 / tau_f
    mu_f = sigma_f^2 * (tau_b*mu_bow + tau_e*mu_emb)   # prior mean = 0

Decoder/ELBO/reparameterization are UNCHANGED from vaebm.py (imported,
not reimplemented) - only the fusion step differs, so any quality delta
between "vaebm" and "vaebm_poe" is attributable to that one change.

User-authorized (2026-09-16) deviation from models/base.py's own
"fit() never sees labels" rule: this model's fit() accepts an OPTIONAL
`labels` argument used ONLY to pick which epoch's checkpoint to keep
(oracle model selection under a fixed wall-clock budget, since this is
the unsupervised `cluster` experiment with no train/val/test split to
select on instead - see cluster_runner.py's own comment at the one
call site that passes labels here). Labels are NEVER used in the loss/
gradient - only to rank already-computed epochs after the fact. Every
persisted result row this produces is tagged
checkpoint_selection="oracle_label_informed" precisely so it is never
mistaken for a fully unsupervised number directly comparable to every
other model's (which pick their sole/final epoch with no such benefit).
"""

from __future__ import annotations

import os

# Must run BEFORE `import tensorflow` below - see vaebm.py's own
# module-level guard/docstring comment for why (auto-JIT needs
# libdevice.10.bc, absent from a pip-only CUDA install). vaebm.py sets
# this too, but only takes effect if it happens to be imported before
# tensorflow itself - not guaranteed here, since this module may be the
# first to trigger `import tensorflow` in a process that never imports
# vaebm.py (e.g. running vaebm_poe alone). Not a change to this model's
# math - a compute-backend setting only.
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")

import time
from typing import Optional, Sequence, Union

import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow.keras import Model, initializers, layers
from tensorflow.keras.optimizers import Adam

from vaebm_benchmark.metrics.clustering_quality import compute_clustering_metrics
from vaebm_benchmark.models._vaebm_shared import build_bow_and_embeddings, transform_bow_and_embeddings
from vaebm_benchmark.models.vaebm import Decoder


class PoEEncoder(Model):
    """Same two-branch MLP structure as vaebm.py::Encoder, fused via
    Product-of-Experts instead of a fixed alpha blend - see this
    module's own docstring for the exact math."""

    def __init__(
        self,
        units: int = 50,
        voc: int = 2000,
        dim: Sequence[int] = (1000, 368),
        dim_emb: Sequence[int] = (1000, 368),
    ):
        super().__init__()
        tf.random.set_seed(1234)
        self.units = units
        self.voc = voc

        # Keras 3's Sequential refuses to build with zero layers - see
        # vaebm.py::Encoder's own comment on this same fix. dim=()/
        # dim_emb=() are kept working as "no hidden layer" by skipping the
        # Sequential entirely - see call() below.
        self.mlp_bow = None
        if len(dim) > 0:
            self.mlp_bow = tf.keras.Sequential(name="MLP_BOW")
            for i, width in enumerate(dim):
                self.mlp_bow.add(layers.Dense(
                    width, activation=tf.nn.tanh,
                    kernel_initializer=initializers.Identity(gain=0.99999) if i == 10 else "glorot_uniform",
                    bias_initializer="zeros", name=f"bow_dense_{i}",
                ))
        self.mu_bow = layers.Dense(units, activation=None, kernel_initializer=initializers.Identity(gain=0.99999),
                                    bias_initializer="zeros", name="mu_bow")
        self.log_sigma_bow = layers.Dense(units, activation=None, kernel_initializer="glorot_uniform",
                                           bias_initializer="zeros", name="log_sigma_bow")

        self.mlp_emb = None
        if len(dim_emb) > 0:
            self.mlp_emb = tf.keras.Sequential(name="MLP_EMB")
            for i, width in enumerate(dim_emb):
                self.mlp_emb.add(layers.Dense(
                    width, activation=tf.nn.tanh,
                    kernel_initializer=initializers.Identity(gain=0.99999) if i == 0 else "glorot_uniform",
                    bias_initializer="zeros", name=f"emb_dense_{i}",
                ))
        self.mu_emb = layers.Dense(units, activation=None, kernel_initializer=initializers.Identity(gain=0.99999),
                                    bias_initializer="zeros", name="mu_emb")
        self.log_sigma_emb = layers.Dense(units, activation=None, kernel_initializer="glorot_uniform",
                                           bias_initializer="zeros", name="log_sigma_emb")

    def call(self, inputs_tuple, training: Optional[bool] = None):
        x_bow, e_txt = inputs_tuple[0], inputs_tuple[1]
        batch_size = tf.shape(x_bow)[0]

        h_bow = self.mlp_bow(x_bow) if self.mlp_bow is not None else x_bow
        mu_bow = self.mu_bow(h_bow)
        log_sigma_bow = self.log_sigma_bow(h_bow)

        h_emb = self.mlp_emb(e_txt) if self.mlp_emb is not None else e_txt
        mu_emb = self.mu_emb(h_emb)
        log_sigma_emb = self.log_sigma_emb(h_emb)

        # Precisions (unit prior precision = 1, prior mean = 0).
        tau_b = tf.exp(-2.0 * log_sigma_bow)
        tau_e = tf.exp(-2.0 * log_sigma_emb)
        tau_f = 1.0 + tau_b + tau_e

        sigma2_f = 1.0 / tau_f
        mu = sigma2_f * (tau_b * mu_bow + tau_e * mu_emb)
        log_sigma = 0.5 * tf.math.log(sigma2_f)

        eps = tf.random.normal((batch_size, self.units), mean=0.0, stddev=1.0)
        z = mu + tf.exp(log_sigma) * eps
        return mu, log_sigma, z


class VAEBMPoE(Model):
    """Same forward pass/ELBO as vaebm.py::VAEBM, using PoEEncoder instead
    of the fixed-alpha Encoder. Decoder is imported unchanged."""

    def __init__(self, units=50, voc=2000, dim=(1500, 1000, 300), dim_emb=(368,),
                 kl_weight=1.0, freeze_embedding_branch=False):
        super().__init__()
        self.kl_weight = kl_weight
        self.encoder = PoEEncoder(units=units, voc=voc, dim=dim, dim_emb=dim_emb)
        self.decoder = Decoder(units=units, voc=voc)
        if freeze_embedding_branch:
            if self.encoder.mlp_emb is not None:
                self.encoder.mlp_emb.trainable = False
            self.encoder.mu_emb.trainable = False
            self.encoder.log_sigma_emb.trainable = False

    def call(self, inputs_tuple, training: Optional[bool] = None):
        x_bow, e_txt = inputs_tuple
        mu, log_sigma, z = self.encoder([x_bow, e_txt])
        recon = self.decoder([z, x_bow])
        kl = 0.5 * tf.reduce_sum(
            tf.square(mu) + tf.exp(2.0 * log_sigma) - 1.0 - 2.0 * log_sigma, axis=1,
        )
        return recon - self.kl_weight * kl  # per-sample ELBO


class VaeBmPoEFit:
    """fit_predict()-equivalent pipeline for VAEBMPoE - a custom training
    loop (not Keras .fit(), which has no hook for "stop after N wall-clock
    seconds, keeping the best-by-label-accuracy epoch") - see module
    docstring for the labels/oracle-checkpointing rationale."""

    def __init__(self, voc_size=5000, units=50, n_clusters=8, random_state=42,
                 epochs=50, batch_size=128, lr=1e-3, max_fit_seconds: Optional[float] = None, verbose=1,
                 kl_weight=1.0, freeze_embedding_branch=False):
        self.voc_size = voc_size
        self.units = units
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_fit_seconds = max_fit_seconds
        self.verbose = verbose
        self.kl_weight = kl_weight
        self.freeze_embedding_branch = freeze_embedding_branch

        self.vectorizer = None
        self.embedder = None
        self.model: Optional[VAEBMPoE] = None
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

        self.model = VAEBMPoE(units=self.units, voc=X_bow.shape[1], dim=dim, dim_emb=dim_emb,
                              kl_weight=self.kl_weight, freeze_embedding_branch=self.freeze_embedding_branch)
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
                print(f"[VAE-BM-PoE] epoch {epoch}/{self.epochs} loss={mean_loss:.4f}{acc_str}", flush=True)

            if metric > best_metric:
                best_metric = metric
                best_weights = [w.numpy().copy() for w in self.model.trainable_variables]
                best_kmeans = kmeans
                self.best_epoch = epoch
                self.best_acc = acc if acc is not None else float("nan")

            elapsed = time.perf_counter() - fit_start
            if self.max_fit_seconds is not None and elapsed > self.max_fit_seconds:
                if self.verbose >= 1:
                    print(f"[VAE-BM-PoE] max_fit_seconds={self.max_fit_seconds}s reached after epoch "
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
