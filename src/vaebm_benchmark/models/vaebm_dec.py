"""VAE-BM-DEC: the SAME two-branch alpha-fusion VAE-BM as models/vaebm.py
(Encoder/Decoder/VAEBM imported unchanged - "normal vaebm", not a new
architecture), except clustering is Deep Embedded Clustering (Xie,
Girshick & Farhadi, 2016) trained JOINTLY with the VAE instead of a
post-hoc KMeans call after fit() finishes.

DEC recap (nu=1, the paper's own default):
    q_ik = (1 + ||z_i - c_k||^2)^-1 / sum_j (1 + ||z_i - c_j||^2)^-1   (soft assignment, Student-t kernel)
    p_ik = (q_ik^2 / f_k) / sum_j (q_ij^2 / f_j),  f_k = sum_i q_ik   (sharpened target distribution)
    L_cluster = KL(P || Q) = sum_i sum_k p_ik * log(p_ik / q_ik)

Total loss per batch: L = L_vae (recon + KL, unchanged ELBO) + lambda_c * L_cluster.
Centroids C are trainable variables, updated by the SAME optimizer step
as the encoder/decoder weights - clustering shapes the latent geometry
during training rather than hoping post-hoc KMeans finds good clusters
in whatever geometry the VAE happened to learn on its own.

Epoch 1 is VAE-only (no clustering loss yet - centroids don't exist
until initialized via KMeans on epoch 1's own mu, the paper's own
pretrain-then-cluster convention, compressed to a single epoch here
given the 30-minute budget). P is recomputed once per epoch (not per
batch - the paper's own "update target distribution every T
iterations" convention, T=1 epoch here) from that epoch's own Q.

Same oracle-checkpointing / labels-only-for-selection contract as
vaebm_poe.py - see that module's docstring for the full rationale
(models/base.py's "fit() never sees labels" rule, user-authorized
2026-09-16 exception for these two new models only, always tagged
checkpoint_selection="oracle_label_informed" in persisted results).
"""

from __future__ import annotations

import time
from typing import Optional, Sequence, Union

import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.optimizers import Adam

from vaebm_benchmark.metrics.clustering_quality import compute_clustering_metrics
from vaebm_benchmark.models._vaebm_shared import build_bow_and_embeddings, transform_bow_and_embeddings
from vaebm_benchmark.models._vaebm_topic_scoring import energy_scores
from vaebm_benchmark.models.vaebm import VAEBM


def _student_t_assignment(z: tf.Tensor, centroids: tf.Tensor) -> tf.Tensor:
    """q_ik, nu=1 (Student-t kernel), as in Xie et al. 2016 Eq. 1."""
    sq_dist = tf.reduce_sum(tf.square(tf.expand_dims(z, 1) - tf.expand_dims(centroids, 0)), axis=2)  # [B, K]
    numer = 1.0 / (1.0 + sq_dist)
    return numer / tf.reduce_sum(numer, axis=1, keepdims=True)


def _target_distribution(q: np.ndarray) -> np.ndarray:
    """p_ik, as in Xie et al. 2016 Eq. 3 - computed with numpy on the full
    (non-batched) Q since it needs the global f_k = sum_i q_ik."""
    weight = (q ** 2) / q.sum(axis=0, keepdims=True)
    return weight / weight.sum(axis=1, keepdims=True)


class VaeBmDECFit:
    def __init__(self, voc_size=5000, units=50, n_clusters=8, random_state=42,
                 epochs=50, batch_size=128, lr=1e-3, alpha=0.99, lambda_c=0.1,
                 max_fit_seconds: Optional[float] = None, verbose=1):
        self.voc_size = voc_size
        self.units = units
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.alpha = alpha
        self.lambda_c = lambda_c
        self.max_fit_seconds = max_fit_seconds
        self.verbose = verbose

        self.vectorizer = None
        self.embedder = None
        self.model: Optional[VAEBM] = None
        self.centroids: Optional[tf.Variable] = None
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

        self.model = VAEBM(units=self.units, voc=X_bow.shape[1], dim=dim, dim_emb=dim_emb, alpha=self.alpha,
                            vectorizer_type=vectorizer_type, embedder=embedder)
        optimizer = Adam(self.lr)
        _ = self.model([X_bow[:1], E[:1]], training=False)

        n = X_bow.shape[0]
        rng = np.random.default_rng(self.random_state)
        best_weights = None
        best_centroids = None
        best_preds = None
        best_metric = -1.0
        fit_start = time.perf_counter()
        p_target = None  # set once centroids exist

        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(n)
            epoch_losses = []
            has_centroids = self.centroids is not None

            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                xb = tf.constant(X_bow[idx])
                eb = tf.constant(E[idx])
                p_batch = tf.constant(p_target[idx]) if has_centroids else None

                with tf.GradientTape() as tape:
                    mu, log_sigma, z = self.model.encoder([xb, eb])
                    recon = self.model.decoder([z, xb])
                    kl = 0.5 * tf.reduce_sum(
                        tf.square(mu) + tf.exp(2.0 * log_sigma) - 1.0 - 2.0 * log_sigma, axis=1,
                    )
                    vae_loss = -tf.reduce_mean(recon - kl)

                    if has_centroids:
                        q_batch = _student_t_assignment(mu, self.centroids)
                        cluster_loss = tf.reduce_mean(
                            tf.reduce_sum(p_batch * tf.math.log(p_batch / (q_batch + 1e-10) + 1e-10), axis=1)
                        )
                        loss = vae_loss + self.lambda_c * cluster_loss
                    else:
                        loss = vae_loss

                trainable = self.model.trainable_variables + ([self.centroids] if has_centroids else [])
                grads = tape.gradient(loss, trainable)
                optimizer.apply_gradients(zip(grads, trainable))
                epoch_losses.append(float(loss.numpy()))
            self.epochs_completed = epoch

            mu_full, _, _ = self.model.encoder([X_bow, E], training=False)
            Z = mu_full.numpy()

            if self.centroids is None:
                # First pass: initialize centroids via KMeans on this
                # epoch's own mu (Xie et al.'s own pretrain-then-cluster
                # init, compressed to epoch 1 here).
                init_kmeans = KMeans(n_clusters=self.n_clusters, random_state=22, n_init="auto")
                preds = init_kmeans.fit_predict(Z)
                self.centroids = tf.Variable(init_kmeans.cluster_centers_.astype(np.float32), trainable=True, name="dec_centroids")
                q_full = _student_t_assignment(tf.constant(Z), self.centroids).numpy()
            else:
                q_full = _student_t_assignment(tf.constant(Z), self.centroids).numpy()
                preds = np.argmax(q_full, axis=1)

            p_target = _target_distribution(q_full)  # for the NEXT epoch's batches

            mean_loss = float(np.mean(epoch_losses))
            if labels is not None:
                acc = compute_clustering_metrics(preds, labels, ["acc"])["acc"]
                metric = acc
            else:
                acc = None
                metric = -mean_loss

            if self.verbose >= 1:
                acc_str = f" acc={acc:.4f}" if acc is not None else ""
                print(f"[VAE-BM-DEC] epoch {epoch}/{self.epochs} loss={mean_loss:.4f}{acc_str}", flush=True)

            if metric > best_metric:
                best_metric = metric
                best_weights = [w.numpy().copy() for w in self.model.trainable_variables]
                best_centroids = self.centroids.numpy().copy()
                best_preds = preds.copy()
                self.best_epoch = epoch
                self.best_acc = acc if acc is not None else float("nan")

            elapsed = time.perf_counter() - fit_start
            if self.max_fit_seconds is not None and elapsed > self.max_fit_seconds:
                if self.verbose >= 1:
                    print(f"[VAE-BM-DEC] max_fit_seconds={self.max_fit_seconds}s reached after epoch "
                          f"{epoch}/{self.epochs} - stopping early, keeping best epoch {self.best_epoch}.", flush=True)
                break

        if best_weights is not None:
            for var, val in zip(self.model.trainable_variables, best_weights):
                var.assign(val)
            self.centroids.assign(best_centroids)

        mu_full, _, _ = self.model.encoder([X_bow, E], training=False)
        Z = mu_full.numpy()
        q_full = _student_t_assignment(tf.constant(Z), self.centroids).numpy()
        final_preds = np.argmax(q_full, axis=1) if best_preds is None else best_preds
        return final_preds.tolist(), Z

    def predict(self, texts: Sequence[str]):
        X_bow, E = transform_bow_and_embeddings(texts, self.vectorizer, self.embedder)
        mu, _, _ = self.model.encoder([X_bow, E], training=False)
        Z = mu.numpy()
        q = _student_t_assignment(tf.constant(Z), self.centroids).numpy()
        return np.argmax(q, axis=1).tolist(), Z

    def _predict_labels(self, Z: np.ndarray) -> np.ndarray:
        q = _student_t_assignment(tf.constant(Z), self.centroids).numpy()
        return np.argmax(q, axis=1)

    def top_words_by_freq_exact(self, texts: Sequence[str], top_m: int = 20):
        X = self.vectorizer.transform(texts)
        if isinstance(self.vectorizer, TfidfVectorizer):
            X = X.tocoo(copy=False)
            X.data = np.log1p(X.data)
            X = X.tocsr()
        X_bow = X.toarray().astype(np.float32)
        E = self.embedder.encode(texts, batch_size=128, convert_to_numpy=True).astype(np.float32)

        mu, _, _ = self.model.encoder([X_bow, E], training=False)
        Z = mu.numpy()
        preds = self._predict_labels(Z)

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
