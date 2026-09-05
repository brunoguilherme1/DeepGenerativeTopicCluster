"""Shared model registry for the `classification` and `cluster`
experiments (scripts/run_experiment.py --experiment classification /
--experiment cluster) - model set {vaebm, fastopic, lda, hicot}, each
exposing a genuine representation (`theta` for fastopic/lda/hicot, `mu`
for VAE-BM - see docs/methodological_notes.md #1 on why `mu` is not
`theta`) that both experiments consume identically, so a model added
here is available to both without duplicating builder logic.

Epoch counts below are REDUCED from each model's own paper defaults for
a generic cross-dataset smoke run - same convention
experiment/cluster_runner.py's own pre-existing `_build_fastopic`/
`_build_glocom` already use ("reduced from the paper's 200, a generic
cross-dataset smoke default, not a paper reproduction"). Not a claim
these defaults reproduce any paper's own published numbers.

`representation_source`/`assignment_source`/`topic_source` are looked up
by model NAME here, never by probing whether `get_document_topics()`
returns non-`None` - VAEBMAdapter's own `get_document_topics()` always
returns `mu` (never `None`; see models/vaebm_adapter.py's own
docstring), so a naive non-`None` check would silently treat `mu` as
`theta`. This mirrors experiment/runner.py's own
`_assignment_source_for_model()` (built for the `ecrtm_hicot` topic
protocol) - same reasoning, extended to this model set.
"""

from __future__ import annotations

MODEL_NAMES = ["vaebm", "fastopic", "lda", "hicot"]


def build_vaebm(k: int, seed: int, voc_size: int, dataset_id: str = None):
    from vaebm_benchmark.models.vaebm_adapter import VAEBMAdapter

    return VAEBMAdapter(
        n_clusters=k,
        voc_size=voc_size,
        units=50,
        epochs=30,
        batch_size=128,
        lr=1e-3,  # see docs/methodological_notes.md #8
        random_state=seed,
        vectorizer_type="tfidf",
        embedder="all-MiniLM-L6-v2",
        dim=(1500, 1000, 500),
        dim_emb=(368,),
        alpha=0.99,
        top_words_mode="energy",
    )


def build_fastopic(k: int, seed: int, voc_size: int, dataset_id: str = None):
    from vaebm_benchmark.models.fastopic_adapter import FASTopicAdapter

    return FASTopicAdapter(
        num_topics=k,
        vocab_size_cap=voc_size,
        epochs=20,  # reduced from the paper's 200, see module docstring
        learning_rate=0.002,
        doc_embed_model="all-MiniLM-L6-v2",
        low_memory=False,
    )


def build_lda(k: int, seed: int, voc_size: int, dataset_id: str = None):
    from vaebm_benchmark.models.lda_adapter import LDAAdapter

    return LDAAdapter(n_clusters=k, voc_size=voc_size, random_state=seed)


def build_hicot(k: int, seed: int, voc_size: int, dataset_id: str = None):
    from vaebm_benchmark.models.hicot_adapter import HiCOTAdapter

    kwargs = dict(
        n_clusters=k,
        voc_size=voc_size,
        random_state=seed,
        epochs=50,  # reduced from upstream's own 500, see module docstring
    )
    if dataset_id is not None and dataset_id.startswith("hicot_"):
        # Use HiCOT's own official vocab + 200-dim GloVe embeddings for
        # its own datasets - the exact artifacts this project already
        # downloads verbatim (datasets/definitions/hicot_datasets.py) -
        # rather than a self-fit vocabulary/random embedding init.
        from vaebm_benchmark.datasets.definitions.hicot_datasets import (
            load_hicot_vocab,
            load_hicot_word_embeddings,
        )

        vocab = load_hicot_vocab(dataset_id)
        embeddings = load_hicot_word_embeddings(dataset_id).toarray()
        kwargs.update(
            vocabulary=vocab,
            voc_size=len(vocab),
            pretrained_word_embeddings=embeddings,
            use_pretrained_word_embeddings=True,
        )
    return HiCOTAdapter(**kwargs)


MODEL_BUILDERS = {
    "vaebm": build_vaebm,
    "fastopic": build_fastopic,
    "lda": build_lda,
    "hicot": build_hicot,
}


def build_model(model_name: str, k: int, seed: int, voc_size: int, dataset_id: str = None):
    if model_name not in MODEL_BUILDERS:
        raise KeyError(f"Unknown model '{model_name}'. Available: {sorted(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[model_name](k, seed, voc_size, dataset_id=dataset_id)


# "theta" (a genuine document-topic probability simplex) vs. "mu"
# (VAE-BM's latent Gaussian mean, NOT a topic distribution - see
# docs/methodological_notes.md #1) vs. "embeddings" (a plain SBERT
# embedding space, for a KMeans-over-embeddings model with neither) - the
# feature space classification/cluster use as each model's document
# representation. Covers "bertopic"/"glocom" too (experiment/
# cluster_runner.py's own pre-existing models, not in MODEL_NAMES above)
# so that module can reuse these same three lookups for every model it
# supports, not just this file's own four.
def representation_source_for_model(model_name: str) -> str:
    if model_name == "vaebm":
        return "mu"
    if model_name in ("fastopic", "lda", "hicot", "glocom"):
        return "theta"
    if model_name == "bertopic":
        return "embeddings"
    return "unknown"


# How hard cluster assignments are derived - "argmax_theta" for a genuine
# theta, "kmeans_on_latent_mu" for VAE-BM (its EXISTING behavior, kept
# as-is - never softmax mu into a fake theta), "kmeans_on_embeddings" for
# a plain embed-then-cluster model with neither.
def assignment_source_for_model(model_name: str) -> str:
    if model_name == "vaebm":
        return "kmeans_on_latent_mu"
    if model_name in ("fastopic", "lda", "hicot", "glocom"):
        return "argmax_theta"
    if model_name == "bertopic":
        return "kmeans_on_embeddings"
    return "unknown"


# Every model covered here produces topic words natively (VAE-BM's own
# decoder energy/freq view, FASTopic's/GloCOM's own extraction, LDA's own
# components_, HiCOT's own beta, BERTopic's own c-TF-IDF) - never
# "cluster-derived" for this set (that only applies to sbert_kmeans
# elsewhere in this repo, not covered by this module).
def topic_source_for_model(model_name: str) -> str:
    return "native"
