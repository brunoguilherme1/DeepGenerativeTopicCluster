"""VAE-BM: VAE + Bag-of-Words "energy" decoder + KMeans over the latent
mean mu, with an optional similarity-distillation-style second encoder
branch that blends in pretrained sentence embeddings.

This is the model AS SUPPLIED for this benchmark (from the user's own
1_0_Baselines_&_Results.ipynb, cell "VAE-BM-Distill" / vaebm_fit.py) -
copied here verbatim, unmodified, save for this module docstring and
translating the original inline Portuguese comments to English. Per the
task's own instructions: "Do not silently change the mathematical
formulation of VAE-BM. ... If you find a methodological/model issue,
document it separately rather than silently changing the model." Any such
issue found while integrating this model into baseline protocols is
recorded in docs/methodological_notes.md, NOT fixed here.

Two genuinely separate, independent knobs on `VaeBmKMeansFit.fit_predict()`
- not to be conflated (an earlier revision of this file briefly did):
`vectorizer_type` ("tfidf" or anything else -> CountVectorizer/"bag")
controls the BoW branch (`x_bow`); `embedder` controls the SEPARATE
embedding branch (`e_txt`) and only ever accepts a real embedding - any
SentenceTransformer/HuggingFace model name (a plain string), or a
precomputed `[N, d]` array. `embedder` is never "tfidf"/"bag" - that
choice belongs to `vectorizer_type` alone.

In particular (see docs/methodological_notes.md for the full writeup):
mu (the encoder's latent mean) is a latent Gaussian representation, not a
normalized topic distribution - KMeans over mu produces hard document
clusters, and get_topics_energy()/get_topics_freq() below produce topic
word lists from those clusters, but neither mu itself nor the KMeans
assignment should be read as p(z|d) or a topic-proportion vector the way
FASTopic's transform() or GloCOM's get_theta() are.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, initializers
from tensorflow.keras.optimizers import Adam

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans

from vaebm_benchmark.models._vaebm_topic_scoring import energy_scores

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore


# =============================================================================
#                               ENCODER / DECODER
# =============================================================================


class Encoder(Model):
    """Variational encoder with two branches:
      - x_bow: BoW/TF-IDF input [B, voc]
      - e_txt: sentence embeddings [B, d_emb]

    Produces mu, log_sigma (final, alpha-blended across branches) and a
    reparameterized sample z.
    """

    def __init__(
        self,
        units: int = 50,
        voc: int = 2000,
        dim: Sequence[int] = (1000, 368),
        dim_emb: Sequence[int] = (1000, 368),
        alpha: float = 0.99,  # fixed weight between the two branches
    ):
        super().__init__()
        tf.random.set_seed(1234)
        self.units = units
        self.voc = voc
        self.alpha = alpha

        # BoW branch MLP
        self.mlp_bow = tf.keras.Sequential(name="MLP_BOW")
        for i, width in enumerate(dim):
            self.mlp_bow.add(layers.Dense(
                width,
                activation=tf.nn.tanh,
                kernel_initializer=initializers.Identity(gain=0.99999) if i == 10 else "glorot_uniform",
                bias_initializer="zeros",
                name=f"bow_dense_{i}"
            ))

        self.mu_bow = layers.Dense(
            units, activation=None,
            kernel_initializer=initializers.Identity(gain=0.99999),
            bias_initializer="zeros",
            name="mu_bow"
        )
        self.log_sigma_bow = layers.Dense(
            units, activation=None,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name="log_sigma_bow"
        )

        # Embedding branch MLP
        self.mlp_emb = tf.keras.Sequential(name="MLP_EMB")
        for i, width in enumerate(dim_emb):
            self.mlp_emb.add(layers.Dense(
                width,
                activation=tf.nn.tanh,
                kernel_initializer=initializers.Identity(gain=0.99999) if i == 0 else "glorot_uniform",
                bias_initializer="zeros",
                name=f"emb_dense_{i}"
            ))

        self.mu_emb = layers.Dense(
            units, activation=None,
            kernel_initializer=initializers.Identity(gain=0.99999),
            bias_initializer="zeros",
            name="mu_emb"
        )
        self.log_sigma_emb = layers.Dense(
            units, activation=None,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name="log_sigma_emb"
        )

    def call(self, inputs_tuple, training: Optional[bool] = None):
        x_bow, e_txt = inputs_tuple[0], inputs_tuple[1]
        batch_size = tf.shape(x_bow)[0]

        h_bow = self.mlp_bow(x_bow)
        mu_bow = self.mu_bow(h_bow)
        log_sigma_bow = self.log_sigma_bow(h_bow)

        h_emb = self.mlp_emb(e_txt)
        mu_emb = self.mu_emb(h_emb)
        log_sigma_emb = self.log_sigma_emb(h_emb)

        alpha = self.alpha  # could be made learnable
        mu = alpha * mu_bow + (1.0 - alpha) * mu_emb
        log_sigma = alpha * log_sigma_bow + (1.0 - alpha) * log_sigma_emb

        eps = tf.random.normal((batch_size, self.units), mean=0.0, stddev=1.0)
        z = mu + tf.exp(log_sigma) * eps
        return mu, log_sigma, z


class Decoder(Model):
    """Energy-style BoW decoder: logits = z @ R^T + b, log-softmax over the
    vocabulary, summed only at observed positions (x_bow > 0) - an
    approximate per-sample log-likelihood, not a full generative decoder.
    """

    def __init__(self, units: int = 50, voc: int = 2000):
        super().__init__()
        self.units = units
        self.voc = voc

        lim = 0.5 / max(1, units)
        self.R = self.add_weight(
            name="RWord",
            shape=(self.voc, self.units),
            initializer=tf.keras.initializers.RandomUniform(minval=-lim, maxval=lim),
            trainable=True,
        )
        self.b = self.add_weight(
            name="biasWord",
            shape=(self.voc,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs, training: Optional[bool] = None):
        z, x_bow = inputs  # [B, units], [B, voc]

        logits = tf.matmul(z, tf.transpose(self.R))  # [B, voc]
        logits = tf.nn.bias_add(logits, self.b)

        log_probs = tf.nn.log_softmax(logits, axis=1)  # [B, voc]

        contrib = tf.multiply(log_probs, x_bow)  # [B, voc]
        recon_per_sample = tf.reduce_sum(contrib, axis=1)  # [B]
        return recon_per_sample


class VAEBM(Model):
    """VAE (Encoder + BoW energy Decoder) returning per-sample ELBO:
      ELBO(x) = Recon(x|z) + (-KL[q(z|x)||p(z)])
    """

    def __init__(
        self,
        units: int = 50,
        voc: int = 2000,
        dim: Sequence[int] = (1500, 1000, 300),
        dim_emb: Sequence[int] = (368,),
        alpha=0.99,
        # Two genuinely separate, independent knobs - not to be confused
        # with each other. Neither changes this class's forward pass/ELBO
        # math: by the time VAEBM is constructed, x_bow/e_txt are already
        # fitted/computed arrays (see VaeBmKMeansFit.fit_predict) - these
        # are stored here purely so a VAEBM instance carries a record of
        # what config produced it (introspection/logging), not consumed
        # in call() below.
        vectorizer_type: str = "tfidf",  # BoW branch (x_bow): "tfidf", or anything else -> CountVectorizer/"bag"
        embedder: Union[str, np.ndarray] = "thenlper/gte-small",  # embedding branch (e_txt): any SentenceTransformer/HuggingFace model name, or a precomputed array - never "tfidf"/"bag"
    ):
        super().__init__()
        self.vectorizer_type = vectorizer_type
        self.embedder_name = embedder if isinstance(embedder, str) else None
        self.encoder = Encoder(units=units, voc=voc, dim=dim, dim_emb=dim_emb, alpha=alpha)
        self.decoder = Decoder(units=units, voc=voc)

    def call(self, inputs_tuple, training: Optional[bool] = None):
        x_bow, e_txt = inputs_tuple  # [B, voc], [B, d_emb]

        mu, log_sigma, z = self.encoder([x_bow, e_txt])

        recon = self.decoder([z, x_bow])  # [B]

        # -KL = 0.5 * sum(1 + logvar - mu^2 - exp(logvar)), logvar = 2*log_sigma
        kl = 0.5 * tf.reduce_sum(
            tf.square(mu) + tf.exp(2.0 * log_sigma) - 1.0 - 2.0 * log_sigma,
            axis=1,
        )
        elbo = recon - kl
        return elbo


@tf.function
def lossCluster(_, elbo_per_sample: tf.Tensor) -> tf.Tensor:
    """Keras minimizes; maximizing the ELBO means minimizing its negative
    mean."""
    return -tf.reduce_mean(elbo_per_sample)


# =============================================================================
#                                   PIPELINE
# =============================================================================


class VaeBmKMeansFit:
    """End-to-end pipeline: vectorize -> embed -> fit VAEBM (maximize ELBO)
    -> KMeans over mu -> document clusters. predict() repeats the same
    fitted pipeline on new documents.
    """

    def __init__(
        self,
        voc_size: int = 5000,
        units: int = 50,
        n_clusters: int = 8,
        random_state: int = 42,
        epochs: int = 30,
        batch_size: int = 128,
        lr: float = 1e-2,
        # 0: silent (no training output at all - only the caller's own
        # final result table prints). 1: one concise summary line per
        # fit_predict() call (dataset size, vectorizer/embedder, epochs
        # actually run, final loss). 2: everything Keras itself reports -
        # a per-epoch progress bar plus the EarlyStopping/ModelCheckpoint
        # callbacks' own verbose logging.
        verbose: int = 1,
    ):
        self.voc_size = voc_size
        self.units = units
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose

        self.vectorizer: Optional[Union[TfidfVectorizer, CountVectorizer]] = None
        self.model: Optional[VAEBM] = None
        self.kmeans: Optional[KMeans] = None

        self.embedder: Optional["SentenceTransformer"] = None
        self.teacher_embedder: Optional["SentenceTransformer"] = None

    def fit_predict(
        self,
        texts: Sequence[str],
        vectorizer_type: str = "tfidf",
        embedder: Union[str, np.ndarray] = "thenlper/gte-small",
        dim: Sequence[int] = (1500, 1000, 500),
        dim_emb: Sequence[int] = (368,),
        init_R_from_vocab: bool = False,
        init_R_gain: float = 1.0,
        alpha=0.99,
        checkpoint_dir: Optional[str] = None,
        vocabulary: Optional[Sequence[str]] = None,
    ) -> tuple[List[int], np.ndarray]:
        """`vocabulary`, if given, fixes the exact vocabulary (e.g. a
        baseline protocol's own vocab.txt) instead of letting the
        vectorizer derive one from `texts` capped at `self.voc_size` -
        purely a preprocessing-matching hook for protocol fidelity (see
        protocols/*.py), not a change to the model's math: sklearn's
        vectorizers accept a fixed `vocabulary=` natively, and everything
        downstream (X_bow shape/values) is unaffected either way.

        When `vocabulary` is given, `tokenizer=str.split` and
        `lowercase=False` are ALSO forced (not sklearn's own defaults) -
        the released artifacts this is used against (GloCOM's texts.txt,
        FASTopic/topmost's train_texts.txt) are already tokenized/
        lowercased/filtered exactly once, via simple whitespace-split, to
        build their own official count matrices. sklearn's default
        tokenizer is a DIFFERENT regex (`\\b\\w\\w+\\b` - drops
        single-character tokens, splits on punctuation/underscores
        differently) that could silently undercount some vocabulary
        entries relative to the official artifact even with the same
        `vocabulary=` list - this makes the resulting BoW counts
        PROVABLY identical to the official ones (same tokenization
        rule, same fixed vocabulary), not merely "should be equivalent."
        See protocols/*.py `vocabulary_for()`/checks() for how this is
        surfaced as a MATCH rather than assumed."""
        if vocabulary is not None:
            vectorizer_kwargs = {"vocabulary": list(vocabulary), "tokenizer": str.split, "lowercase": False, "token_pattern": None}
        else:
            vectorizer_kwargs = {"max_features": self.voc_size}
        if vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer(norm=None, **vectorizer_kwargs)
            X = self.vectorizer.fit_transform(texts)
            X = X.tocoo(copy=False); X.data = np.log1p(X.data); X = X.tocsr()
        else:
            if vocabulary is None:
                vectorizer_kwargs["stop_words"] = "english"
            self.vectorizer = CountVectorizer(**vectorizer_kwargs)
            X = self.vectorizer.fit_transform(texts)
        X_bow = X.toarray().astype(np.float32)

        if isinstance(embedder, str):
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers is not available.")
            self.embedder = SentenceTransformer(embedder)
            E = self.embedder.encode(texts, batch_size=32, convert_to_numpy=True).astype(np.float32)
        else:
            E = np.asarray(embedder, dtype=np.float32)
            self.embedder = None

        self.model = VAEBM(
            units=self.units, voc=X_bow.shape[1], dim=dim, dim_emb=dim_emb, alpha=alpha,
            vectorizer_type=vectorizer_type, embedder=embedder,
        )
        self.model.compile(optimizer=Adam(self.lr), loss=lossCluster)

        _ = self.model([X_bow[:1], E[:1]], training=False)

        if init_R_from_vocab:
            if self.embedder is None:
                raise RuntimeError("init_R_from_vocab=True requires a STRING embedder (SentenceTransformer).")
            vocab = self.vectorizer.get_feature_names_out()
            E_vocab = self.embedder.encode(list(vocab), batch_size=256, convert_to_numpy=True).astype(np.float32)
            units = int(self.model.decoder.R.shape[1])
            d = int(E_vocab.shape[1])

            if d != units:
                rng = np.random.default_rng(self.random_state)
                P = rng.standard_normal((d, units)).astype(np.float32) / np.sqrt(max(1, d))
                R_init = E_vocab @ P
            else:
                R_init = E_vocab

            norms = np.linalg.norm(R_init, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            R_init = (R_init / norms) * float(init_R_gain)

            self.model.decoder.R.assign(R_init)

        import shutil
        import tempfile

        # A unique directory per fit_predict() call - NOT a fixed filename
        # in the shared system temp dir (what the model as originally
        # supplied used, and what this wrapper used until this fix). A
        # fixed path collides across runs with different vocab sizes /
        # architectures (e.g. this project's GloCOM protocol, vocab=4618,
        # then FASTopic protocol, vocab=10000, sharing one machine's temp
        # dir): the second run's `load_weights` would try to load the
        # first run's incompatible checkpoint and fail with a shape
        # mismatch. This is a real bug in the checkpoint path, not a
        # change to the model's math - see docs/methodological_notes.md.
        run_ckpt_dir = checkpoint_dir or tempfile.mkdtemp(prefix="vaebm_ckpt_")
        ckpt_path = f"{run_ckpt_dir}/vaebm_best.weights.h5"
        keras_verbose = 1 if self.verbose >= 2 else 0
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="loss", patience=1, restore_best_weights=True, mode="min", verbose=keras_verbose),
            tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, monitor="loss", save_best_only=True,
                                               save_weights_only=True, mode="min", verbose=keras_verbose),
            tf.keras.callbacks.TerminateOnNaN(),
        ]
        history = self.model.fit([X_bow, E], X_bow,
                       epochs=self.epochs, batch_size=self.batch_size,
                       shuffle=True, callbacks=callbacks, verbose=keras_verbose)

        if self.verbose >= 1:
            loss_history = history.history.get("loss", [])
            embedder_label = embedder if isinstance(embedder, str) else "precomputed"
            final_loss = f"{loss_history[-1]:.4f}" if loss_history else "n/a"
            print(f"[VAE-BM] docs={len(texts)} voc={X_bow.shape[1]} vectorizer_type={vectorizer_type} "
                  f"embedder={embedder_label} epochs_run={len(loss_history)}/{self.epochs} final_loss={final_loss}")

        # EarlyStopping(restore_best_weights=True) already restored the
        # in-memory model to its best epoch; reloading from disk here is
        # redundant but harmless (same weights) EXCEPT when the run never
        # improved past epoch 1 (e.g. loss went straight to NaN) and
        # ModelCheckpoint therefore never wrote a file - guard on that
        # rather than assuming ckpt_path always exists.
        if tf.io.gfile.exists(ckpt_path):
            self.model.load_weights(ckpt_path)
        if checkpoint_dir is None:
            shutil.rmtree(run_ckpt_dir, ignore_errors=True)

        mu, _, _ = self.model.encoder([X_bow, E], training=False)
        Z = mu.numpy()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=22, n_init="auto")
        self.kmeans.fit(Z)
        return self.kmeans.predict(Z).tolist(), Z

    def predict(self, texts: Sequence[str]):
        if self.vectorizer is None or self.model is None or self.kmeans is None:
            raise RuntimeError("Call fit_predict() before predict().")

        X = self.vectorizer.transform(texts)
        if isinstance(self.vectorizer, TfidfVectorizer):
            X = X.tocoo(copy=False)
            X.data = np.log1p(X.data)
            X = X.tocsr()
        X_bow = X.toarray().astype(np.float32)

        if self.embedder is None:
            raise RuntimeError("Embedder was not initialized.")
        E = self.embedder.encode(texts, batch_size=128, convert_to_numpy=True).astype(np.float32)

        mu, _, _ = self.model.encoder([X_bow, E], training=False)
        Z = mu.numpy()
        return self.kmeans.predict(Z).tolist(), mu

    def top_words_by_freq_exact(self, texts: Sequence[str], top_m: int = 20):
        """Returns {"energy": [...], "freq": [...]} - per-cluster top words
        by decoder energy (R/b logits, masked to observed terms) and by raw
        frequency, respectively."""
        if self.model is None or self.vectorizer is None or self.kmeans is None:
            raise RuntimeError("model, vectorizer and kmeans must already be fit.")

        X = self.vectorizer.transform(texts)
        if isinstance(self.vectorizer, TfidfVectorizer):
            X = X.tocoo(copy=False)
            X.data = np.log1p(X.data)
            X = X.tocsr()
        X_all = X
        X_bow = X_all.toarray().astype(np.float32)

        E = self.embedder.encode(texts, batch_size=128, convert_to_numpy=True).astype(np.float32)

        mu, _, _ = self.model.encoder([X_bow, E], training=False)
        Z = mu.numpy()
        preds = self.kmeans.predict(Z)

        vocab = self.vectorizer.get_feature_names_out()
        K = self.n_clusters

        top_words_energy = []
        top_words_freq = []
        empty_clusters = 0

        R = self.model.decoder.R.numpy()
        b = self.model.decoder.b.numpy()

        for k in range(K):
            idx_k = np.where(preds == k)[0]
            if idx_k.size == 0:
                empty_clusters += 1
                continue

            X_k = X_all[idx_k]
            h_k = Z[idx_k]

            counts_k = np.asarray(X_k.sum(axis=0)).ravel()
            if counts_k.sum() == 0:
                empty_clusters += 1
                continue
            top_idx_freq = np.argsort(counts_k)[::-1][:top_m]
            top_words_freq.append(vocab[top_idx_freq].tolist())

            logits_k = h_k @ R.T + b
            mask_k = (X_k.toarray() > 0).astype(np.float32)
            # See _vaebm_topic_scoring.energy_scores's docstring for why
            # never-observed words are forced to -inf rather than left at
            # the 0 that plain mask multiplication would give them - a
            # topic-word DISPLAY/ranking fix, not a change to the trained
            # model's parameters, ELBO, or KMeans clusters.
            scores = energy_scores(logits_k, mask_k, counts_k)

            if not np.any(np.isfinite(scores)):
                empty_clusters += 1
                continue

            top_idx_energy = np.argsort(scores)[::-1][:top_m]
            top_words_energy.append(vocab[top_idx_energy].tolist())

        if empty_clusters > 0:
            print(f"   {empty_clusters} empty clusters ignored for TC/TD.")

        return {"energy": top_words_energy, "freq": top_words_freq}
