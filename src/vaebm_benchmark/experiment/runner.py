"""Simplified, symmetric model-vs-model experiment runner: VAE-BM vs.
BERTopic, across any number of (model, dataset, K) combinations, in the
paper-table style requested (rows=models, columns grouped by dataset,
metrics C_V/Purity/NMI/TD per K block).

Deliberately independent of protocols/*.py (the FASTopic/GloCOM
paper-fidelity track): this runner does not pin a per-paper vocabulary/
checksum/split - it is a direct, controlled comparison of two models on
the SAME shared corpus and the SAME requested K, using identical metric
implementations for both (see metrics/topic_quality.py,
metrics/clustering_quality.py - reused, not reimplemented).

Model names: "bertopic", "sbert_kmeans" (SBERTKMeansAdapter's own supplied
defaults - any SentenceTransformer/HuggingFace model via --sbert-embedder,
see set_sbert_kmeans_defaults), "vaebm" (VAEBMAdapter's own supplied
defaults), and any number of additional named VAE-BM configurations
registered at run time via `register_vaebm_variants()` (see
`--vaebm-configs` in scripts/run_experiment.py) - each an arbitrary set of
VAEBMAdapter constructor overrides (alpha, units, dim, dim_emb, epochs,
batch_size, lr, vectorizer_type, embedder, top_words_mode, verbose -
anything VAEBMAdapter's own `__init__` accepts except
n_clusters/voc_size/random_state, which stay controlled by this runner's
own --k/--voc-size/--seed for every model uniformly). This is what lets a
single run compare 5, 10, or more VAE-BM configurations side by side
without a code change or a new hardcoded branch per variant.

VAE-BM topic words: this runner always uses the ENERGY view
(`top_words_mode="energy"`, VAE-BM's own learned decoder signal) for the
CV/TD columns - chosen as the primary, fixed definition per this
project's own instructions ("choose one primary method and keep it fixed
across all experiments"). The frequency view is also computed and saved
in each result's raw JSON, but never used in the printed table.

`--protocol ecrtm_hicot` (see scripts/run_experiment.py): an OPTIONAL
metric-computation mode aligning topic_evaluation with ECRTM (Wu et al.,
ICML 2023) and HiCOT (2025) as closely as possible WITHOUT touching
datasets or preprocessing (neither paper's exact dataset artifact/
preprocessing pipeline is reproduced here - see
docs/methodological_notes.md #10): top_n=15 (both papers'), TD via the
fixed-K*top_n-denominator Dieng definition
(`topic_diversity_dieng_fixed_k`). The default `protocol="generic"`
reproduces this runner's original behavior byte-for-byte (top_n=10, the
original `topic_diversity()`) - passing `--protocol ecrtm_hicot` never
changes a "generic" run's own numbers.

`cv_method` (`--cv-method {local,palmetto}`, `run_single`/`run_sweep`'s
own parameter) decouples WHICH function computes C_V from `protocol`
(which still controls top_n/TD). `None` (nothing passed - the default,
for EITHER protocol including `ecrtm_hicot`) ALWAYS means local-corpus
gensim C_V: Palmetto is a ~5.1GB one-time download
(`scripts/setup_palmetto.py`) and is NEVER triggered unless the caller
EXPLICITLY passes `cv_method="palmetto"` - `--protocol ecrtm_hicot`
alone does not imply it. Only when `cv_method` is explicitly
`"palmetto"` does `scripts/run_experiment.py` auto-install Palmetto/the
Wikipedia index first (via `scripts/setup_palmetto.py::
ensure_palmetto_ready`) if not already present, rather than silently
recording every `cv` as `None` - see that script's own module docstring.
`palmetto_cv()` itself is still never silently substituted with the
local-corpus number if Palmetto remains unavailable after that (e.g.
the install failed) - `cv` is recorded as `None`/`N/A`.

Every result also now records `assignment_source` (how document clusters
were assigned) and `topic_source` (where get_topics() words came from) -
independent of `protocol`, since these describe what a model actually
does, not a metric-computation choice. `assignment_source` is
"kmeans_on_latent_mu" for VAE-BM and "kmeans_on_embeddings" for
sbert_kmeans/bertopic - NEVER "argmax_theta" for any of these three,
even though VAEBMAdapter's own `get_document_topics()` always returns
something (mu, reused there as a storage slot for other callers) rather
than None: `_assignment_source_for_model()` gates on the MODEL NAME, not
on whether `get_document_topics()` happens to return non-None, so mu is
never mistaken for a genuine theta. "argmax_theta" is reserved for a
model this repo has no explicit knowledge of yet (e.g. a future real
ECRTM baseline) whose `get_document_topics()` returns an actual
probability-simplex distribution.
"""

from __future__ import annotations

import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ExperimentResult:
    model: str
    dataset: str
    k: int
    cv: Optional[float]
    purity: Optional[float]
    nmi: Optional[float]
    td: Optional[float]
    seed: int
    runtime_seconds: float
    status: str  # "ok" | "error"
    error: str = ""
    topics_energy: list = field(default_factory=list)
    topics_freq: list = field(default_factory=list)
    # Paper-alignment metadata - see this module's own docstring and
    # docs/methodological_notes.md #10. `evaluation_protocol`/`top_n`/
    # `cv_source`/`td_definition` describe how CV/TD were computed for
    # THIS result (vary with the `protocol` argument to run_single());
    # `assignment_source`/`topic_source` describe what the model itself
    # does (fixed per model, independent of `protocol`).
    evaluation_protocol: str = "generic"
    top_n: int = 10
    cv_source: str = "cv_local_corpus"
    td_definition: str = "unique_over_returned_slots"
    assignment_source: str = ""
    topic_source: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# VAEBMAdapter's own supplied defaults (what bare "vaebm" uses) - the
# base every named variant's overrides (see register_vaebm_variants) are
# layered on top of. NOT the notebook's own default lr (1e-2) - this
# project already established (docs/methodological_notes.md #8) that
# 1e-2 diverges to inf/NaN at these vocab scales; 1e-3 trains stably,
# and is the base every variant gets unless it overrides `lr` itself.
_VAEBM_DEFAULTS = dict(
    # environment-configurable (VAEBM_UNITS, default "50") - see
    # experiment/cluster_runner.py::_build_vaebm's own comment.
    units=int(os.environ.get("VAEBM_UNITS", "50")),
    # Environment-configurable (VAEBM_EPOCHS, default "50", unchanged prior
    # behavior) - mirrors VAEBM_ALPHA below. Raised from 30 (2026-09-16,
    # user-authorized) - matches experiment/scientific_models.py::build_vaebm's
    # own comment.
    epochs=int(os.environ.get("VAEBM_EPOCHS", "50")),
    # environment-configurable (VAEBM_BATCH_SIZE, default "128" - unchanged
    # prior behavior) - 2026-09-20 parameter audit Tier 1.
    batch_size=int(os.environ.get("VAEBM_BATCH_SIZE", "128")),
    # environment-configurable (VAEBM_LR, default "1e-3") - see
    # experiment/cluster_runner.py::_build_vaebm's own comment.
    lr=float(os.environ.get("VAEBM_LR", "1e-3")),
    vectorizer_type="tfidf",
    # environment-configurable (VAEBM_EMBEDDER, default unchanged) - see
    # experiment/scientific_models.py::_vaebm_embedder's own docstring;
    # --vaebm-embedder still overrides this per-run via set_vaebm_defaults()
    # below, same as before.
    embedder=os.environ.get("VAEBM_EMBEDDER", "all-MiniLM-L6-v2"),
    dim=(1500, 1000, 500),
    # environment-configurable (VAEBM_DIM_EMB, default "368"; "" for no
    # hidden layer at all) - see experiment/cluster_runner.py::_build_vaebm's
    # own comment.
    dim_emb=tuple(int(x) for x in os.environ.get("VAEBM_DIM_EMB", "368").split(",") if x.strip()),
    # environment-configurable (VAEBM_ALPHA, default "0.99" - VAE-BM's own
    # established default, unchanged prior behavior) - see
    # experiment/cluster_runner.py::_build_vaebm's own comment.
    alpha=float(os.environ.get("VAEBM_ALPHA", "0.99")),
    # environment-configurable (VAEBM_KL_WEIGHT, default "1.0" - the
    # ORIGINAL unweighted ELBO, unchanged prior behavior) - beta-VAE-style
    # weight on the KL term, see models/vaebm.py::VAEBM's own comment
    # (2026-09-19 "mimic GTE" research pass).
    kl_weight=float(os.environ.get("VAEBM_KL_WEIGHT", "1.0")),
    # environment-configurable (VAEBM_FREEZE_EMB, default "0"/unset -
    # unchanged prior behavior) - freezes the embedding branch right after
    # construction, see models/vaebm.py::VAEBM's own comment.
    freeze_embedding_branch=os.environ.get("VAEBM_FREEZE_EMB", "0").strip().lower() in ("1", "true", "yes"),
    # environment-configurable (VAEBM_TOP_WORDS_MODE, default "energy" -
    # unchanged prior behavior) - "energy" (decoder R-matrix-driven) or
    # "freq" (per-cluster word-frequency counts, decoupled from decoder
    # training entirely) - see models/vaebm.py's own get_topics_energy()/
    # get_topics_freq() (2026-09-19 "mimic GTE" research pass).
    top_words_mode=os.environ.get("VAEBM_TOP_WORDS_MODE", "energy"),
    verbose=1,
)

# NOT part of _VAEBM_DEFAULTS: vaebm_poe/vaebm_dec/vaebm_ckpt's own
# _build_model branches below spread _VAEBM_DEFAULTS directly into
# VAEBMPoEAdapter/VAEBMDECAdapter/VAEBMCkptAdapter, none of which accept
# these four params - adding them to the shared dict would break all
# three. Applied only to bare "vaebm" (and its registered variants,
# which share VAEBMAdapter). 2026-09-20 parameter audit Tier 1 -
# kmeans_seed/kmeans_n_init override the previously-hardcoded KMeans
# random_state=22 (see models/vaebm.py::VaeBmKMeansFit's own comment);
# normalize_mu/normalize_emb fill the "no cosine/spherical KMeans" gap
# that audit flagged as confirmed absent. All default to unchanged prior
# behavior.
def _parse_df_threshold(raw: str):
    """min_df/max_df accept EITHER an absolute doc count (int) or a
    proportion in (0, 1] (float) per sklearn's own vectorizer contract -
    "5" -> 5 docs, "0.5" -> 50% of docs. Empty string -> None (no
    filtering, unchanged prior behavior)."""
    raw = raw.strip()
    if not raw:
        return None
    value = float(raw)
    return int(value) if value.is_integer() and "." not in raw else value


_VAEBM_ONLY_DEFAULTS = dict(
    kmeans_seed=(int(os.environ["VAEBM_KMEANS_SEED"]) if os.environ.get("VAEBM_KMEANS_SEED", "").strip() else None),
    kmeans_n_init=(
        int(v) if (v := os.environ.get("VAEBM_KMEANS_N_INIT", "auto").strip()).isdigit() else v
    ),
    normalize_mu=os.environ.get("VAEBM_NORMALIZE_MU", "0").strip().lower() in ("1", "true", "yes"),
    normalize_emb=os.environ.get("VAEBM_NORMALIZE_EMB", "0").strip().lower() in ("1", "true", "yes"),
    # environment-configurable (VAEBM_MIN_DF/VAEBM_MAX_DF, default unset ->
    # None -> sklearn's own defaults, unchanged prior behavior) - data-
    # driven vocabulary filtering, see models/vaebm.py::VaeBmKMeansFit's
    # own comment (2026-09-20 idea backlog #2, docs/vaebm_idea_backlog.md).
    min_df=_parse_df_threshold(os.environ.get("VAEBM_MIN_DF", "")),
    max_df=_parse_df_threshold(os.environ.get("VAEBM_MAX_DF", "")),
    # environment-configurable (VAEBM_LAMBDA_RELEVANCE, default unset ->
    # None -> "relevance" mode not computed at all, unchanged prior
    # behavior) - see models/vaebm.py::top_words_by_freq_exact's own
    # docstring (2026-09-20 idea backlog #19). Must be paired with
    # VAEBM_TOP_WORDS_MODE=relevance to actually be SELECTED as the
    # reported topic-word view - setting only one has no visible effect
    # (both env vars gate independent things: whether the mode is
    # computed at all vs. which already-computed mode gets returned).
    lambda_relevance=(float(os.environ["VAEBM_LAMBDA_RELEVANCE"]) if os.environ.get("VAEBM_LAMBDA_RELEVANCE", "").strip() else None),
)

# Experiment-level, never per-variant: every model in a sweep gets the
# SAME requested k/vocabulary-cap/seed (see run_single/run_sweep below),
# so a --vaebm-configs override touching any of these would silently
# fight the sweep's own --k/--voc-size/--seed - rejected explicitly in
# register_vaebm_variants rather than silently overridden or ignored.
_VAEBM_SWEEP_CONTROLLED_PARAMS = {"n_clusters", "voc_size", "random_state"}

# Populated by register_vaebm_variants() - name -> VAEBMAdapter kwarg
# overrides layered on top of _VAEBM_DEFAULTS. Empty until a caller
# (scripts/run_experiment.py's --vaebm-configs) registers something.
_VAEBM_VARIANT_OVERRIDES: dict[str, dict] = {}

# SBERTKMeansAdapter's own supplied defaults (what bare "sbert_kmeans"
# uses) - overridden in place by set_sbert_kmeans_defaults(), e.g. from
# --sbert-embedder, so ANY SentenceTransformer/HuggingFace model can be
# swapped in without a code change.
_SBERT_KMEANS_DEFAULTS = dict(embedder="all-MiniLM-L6-v2")

KNOWN_MODELS = ["vaebm", "vaebm_poe", "vaebm_dec", "vaebm_ckpt", "bertopic", "sbert_kmeans", "ecrtm"]


def _valid_vaebm_params() -> set[str]:
    """VAEBMAdapter's own constructor parameter names (via `inspect`, not
    a hand-maintained duplicate list), minus the sweep-controlled ones -
    the single source of truth both register_vaebm_variants() and
    set_vaebm_defaults() validate against."""
    import inspect

    from vaebm_benchmark.models.vaebm_adapter import VAEBMAdapter

    return set(inspect.signature(VAEBMAdapter.__init__).parameters) - {"self"} - _VAEBM_SWEEP_CONTROLLED_PARAMS


def _check_vaebm_overrides(overrides: dict, valid_params: set[str], label: str) -> None:
    controlled = _VAEBM_SWEEP_CONTROLLED_PARAMS & set(overrides)
    if controlled:
        raise ValueError(
            f"{label} overrides {sorted(controlled)} - these are controlled by this runner's own "
            "--k/--voc-size/--seed for every model uniformly, not settable here."
        )
    unknown = set(overrides) - valid_params
    if unknown:
        raise ValueError(f"Unknown VAEBMAdapter parameter(s) {sorted(unknown)} for {label}. Valid parameters: {sorted(valid_params)}")


def set_vaebm_defaults(**overrides) -> None:
    """Overrides _VAEBM_DEFAULTS in place - e.g. from top-level
    --vaebm-embedder/--vaebm-vectorizer-type - changing what EVERY
    vaebm-family model uses (bare "vaebm" and any registered variant that
    doesn't itself override the same key) without needing a full
    --vaebm-configs entry for a simple global change. Per-variant
    overrides from register_vaebm_variants still win over these, since
    _build_model applies _VAEBM_DEFAULTS first and layers variant
    overrides on top - call this BEFORE register_vaebm_variants if both
    are used together, so that ordering reads naturally (it doesn't
    actually matter: the two update different dicts)."""
    _check_vaebm_overrides(overrides, _valid_vaebm_params(), "set_vaebm_defaults()")
    _VAEBM_DEFAULTS.update(overrides)


# Populated by register_sbert_kmeans_variants() - name -> SBERTKMeansAdapter
# kwarg overrides layered on top of _SBERT_KMEANS_DEFAULTS. Empty until a
# caller (scripts/run_experiment.py's --sbert-configs) registers something.
_SBERT_KMEANS_VARIANT_OVERRIDES: dict[str, dict] = {}


def _valid_sbert_kmeans_params() -> set[str]:
    """SBERTKMeansAdapter's own constructor parameter names (via `inspect`,
    mirroring _valid_vaebm_params above), minus the sweep-controlled ones -
    the single source of truth both register_sbert_kmeans_variants() and
    set_sbert_kmeans_defaults() validate against."""
    import inspect

    from vaebm_benchmark.models.sbert_kmeans_adapter import SBERTKMeansAdapter

    return set(inspect.signature(SBERTKMeansAdapter.__init__).parameters) - {"self", "n_clusters", "random_state"}


def set_sbert_kmeans_defaults(**overrides) -> None:
    """Overrides _SBERT_KMEANS_DEFAULTS in place - e.g. from --sbert-embedder -
    validated against SBERTKMeansAdapter's own constructor signature so a
    typo'd parameter name fails immediately instead of being silently
    ignored."""
    valid_params = _valid_sbert_kmeans_params()
    unknown = set(overrides) - valid_params
    if unknown:
        raise ValueError(f"Unknown SBERTKMeansAdapter parameter(s) {sorted(unknown)}. Valid parameters: {sorted(valid_params)}")
    _SBERT_KMEANS_DEFAULTS.update(overrides)


def register_sbert_kmeans_variants(variants: dict[str, dict]) -> None:
    """Registers additional named sbert_kmeans configurations - e.g. parsed
    from --sbert-configs - each an arbitrary dict of SBERTKMeansAdapter
    constructor overrides layered on top of _SBERT_KMEANS_DEFAULTS (the
    same base "sbert_kmeans" itself uses), mirroring
    register_vaebm_variants above. Lets a single run compare any number of
    embedder choices (e.g. all-mpnet-base-v2 vs. t5-large) side by side
    under distinct model names, entirely from a config file/CLI argument."""
    valid_params = _valid_sbert_kmeans_params()

    for name, overrides in variants.items():
        unknown = set(overrides) - valid_params
        if unknown:
            raise ValueError(f"Unknown SBERTKMeansAdapter parameter(s) {sorted(unknown)} for variant '{name}'. Valid parameters: {sorted(valid_params)}")
        _SBERT_KMEANS_VARIANT_OVERRIDES[name] = overrides
        if name not in KNOWN_MODELS:
            KNOWN_MODELS.append(name)


def register_vaebm_variants(variants: dict[str, dict]) -> None:
    """Registers additional named VAE-BM configurations - e.g. parsed
    from --vaebm-configs - each an arbitrary dict of VAEBMAdapter
    constructor overrides layered on top of _VAEBM_DEFAULTS (the same
    base "vaebm" itself uses). This is what lets a single run compare an
    unbounded number of VAE-BM configurations (5, 10, 50) under distinct
    model names, entirely from a config file/CLI argument - no code
    change or new hardcoded branch per variant.

    Validated against VAEBMAdapter's OWN constructor signature (via
    `inspect`, not a hand-maintained duplicate list) so a typo'd
    parameter name fails immediately with a clear message instead of
    being silently ignored deep inside a training run."""
    valid_params = _valid_vaebm_params()

    for name, overrides in variants.items():
        _check_vaebm_overrides(overrides, valid_params, f"variant '{name}'")
        _VAEBM_VARIANT_OVERRIDES[name] = overrides
        if name not in KNOWN_MODELS:
            KNOWN_MODELS.append(name)


def _compute_hicot_glove_doc_embeddings(dataset_id: str, documents: list[str]):
    """Per-document embedding = the mean of HiCOT's own 200-dim GloVe
    vectors (load_hicot_word_embeddings) over the document's own words
    that appear in HiCOT's own vocab.txt (load_hicot_vocab) - simple
    `str.split()` tokenization, matching this project's own established
    convention for HiCOT-vocab-aligned text (see models/vaebm.py::
    VaeBmKMeansFit.fit_predict's own `vocabulary=` docstring on why
    str.split()/no-lowercasing is used against HiCOT's own artifacts).
    A document with zero matching words gets an all-zero vector (KMeans
    still runs, just with that document as an equidistant outlier - rare
    in practice given HiCOT's own vocab.txt was built FROM these same
    corpora). Used only by VAEBM_DOC_EMBEDDER=hicot_glove_avg above -
    this is a genuine "static embeddings AS the document/clustering
    representation" test, not the existing topic-word-ranking use of
    these same GloVe vectors (VAEBM_USE_HICOT_STATIC_EMB)."""
    import numpy as np

    from vaebm_benchmark.datasets.definitions.hicot_datasets import load_hicot_vocab, load_hicot_word_embeddings

    vocab = load_hicot_vocab(dataset_id)
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    word_embeddings = np.asarray(load_hicot_word_embeddings(dataset_id).todense(), dtype=np.float32)

    doc_embeddings = np.zeros((len(documents), word_embeddings.shape[1]), dtype=np.float32)
    for i, doc in enumerate(documents):
        idxs = [word_to_idx[w] for w in doc.split() if w in word_to_idx]
        if idxs:
            doc_embeddings[i] = word_embeddings[idxs].mean(axis=0)
    return doc_embeddings


def _build_model(model_name: str, k: int, seed: int, voc_size: int):
    if model_name == "vaebm" or model_name in _VAEBM_VARIANT_OVERRIDES:
        from vaebm_benchmark.models.vaebm_adapter import VAEBMAdapter

        params = dict(_VAEBM_DEFAULTS)
        params.update(_VAEBM_ONLY_DEFAULTS)
        params.update(_VAEBM_VARIANT_OVERRIDES.get(model_name, {}))
        return VAEBMAdapter(n_clusters=k, voc_size=voc_size, random_state=seed, **params)
    if model_name == "vaebm_poe":
        # Product-of-Experts fusion variant (models/vaebm_poe.py) - shares
        # _VAEBM_DEFAULTS' own embedder/dim/units/lr (including
        # VAEBM_EMBEDDER env-var overrides), not a separate defaults dict,
        # since every knob it takes besides max_fit_seconds is identical
        # to plain "vaebm"'s own.
        from vaebm_benchmark.models.vaebm_adapter import VAEBMPoEAdapter

        raw = os.environ.get("VAEBM_POE_MAX_FIT_SECONDS", "1200").strip().lower()
        max_fit_seconds = None if raw in ("0", "none", "") else float(raw)
        params = {k2: v for k2, v in _VAEBM_DEFAULTS.items() if k2 != "alpha"}
        return VAEBMPoEAdapter(n_clusters=k, voc_size=voc_size, random_state=seed, max_fit_seconds=max_fit_seconds, **params)
    if model_name == "vaebm_dec":
        # Joint Deep Embedded Clustering variant (models/vaebm_dec.py) -
        # same _VAEBM_DEFAULTS base as "vaebm" (it uses the identical
        # alpha-fusion encoder), plus lambda_c/max_fit_seconds.
        from vaebm_benchmark.models.vaebm_adapter import VAEBMDECAdapter

        raw = os.environ.get("VAEBM_DEC_MAX_FIT_SECONDS", "1200").strip().lower()
        max_fit_seconds = None if raw in ("0", "none", "") else float(raw)
        # environment-configurable (VAEBM_DEC_LAMBDA_C, default "0.1" -
        # unchanged prior behavior) - weight on the DEC clustering loss
        # term, see models/vaebm_dec.py's own docstring. 2026-09-20
        # parameter audit Tier 1.
        lambda_c = float(os.environ.get("VAEBM_DEC_LAMBDA_C", "0.1"))
        params = dict(_VAEBM_DEFAULTS)
        return VAEBMDECAdapter(n_clusters=k, voc_size=voc_size, random_state=seed, lambda_c=lambda_c, max_fit_seconds=max_fit_seconds, **params)
    if model_name == "vaebm_ckpt":
        # Fixed-alpha VAE-BM with oracle-checkpoint selection
        # (models/vaebm_ckpt.py) - shares _VAEBM_DEFAULTS' own
        # embedder/dim/units/lr (including VAEBM_EMBEDDER overrides), like
        # vaebm_poe/vaebm_dec above. alpha/lr are environment-configurable
        # here too (VAEBM_CKPT_ALPHA/VAEBM_CKPT_LR) since this experiment's
        # own run_single() never passes labels to fit() (see below), so
        # this variant always falls back to loss-based epoch selection
        # here regardless of alpha - the oracle-labels path only ever
        # triggers in the cluster experiment (cluster_runner.py explicitly
        # opts in per model name).
        from vaebm_benchmark.models.vaebm_adapter import VAEBMCkptAdapter

        raw = os.environ.get("VAEBM_CKPT_MAX_FIT_SECONDS", "1200").strip().lower()
        max_fit_seconds = None if raw in ("0", "none", "") else float(raw)
        alpha = float(os.environ.get("VAEBM_CKPT_ALPHA", "0.99"))
        lr = float(os.environ.get("VAEBM_CKPT_LR", str(_VAEBM_DEFAULTS["lr"])))
        # 2026-09-19 "let the network learn a bit" research pass - see
        # models/vaebm_ckpt.py::VaeBmCkptFit's own comment. All default to
        # unchanged prior behavior (no unfreezing, "acc" oracle metric).
        raw_unfreeze = os.environ.get("VAEBM_CKPT_UNFREEZE_AFTER_EPOCH", "").strip()
        unfreeze_after_epoch = int(raw_unfreeze) if raw_unfreeze else None
        raw_post_lr = os.environ.get("VAEBM_CKPT_POST_UNFREEZE_LR", "").strip()
        post_unfreeze_lr = float(raw_post_lr) if raw_post_lr else None
        oracle_metric = os.environ.get("VAEBM_CKPT_ORACLE_METRIC", "acc")
        params = {k2: v for k2, v in _VAEBM_DEFAULTS.items() if k2 not in ("alpha", "lr")}
        return VAEBMCkptAdapter(n_clusters=k, voc_size=voc_size, random_state=seed, alpha=alpha, lr=lr,
                                 max_fit_seconds=max_fit_seconds,
                                 unfreeze_after_epoch=unfreeze_after_epoch,
                                 post_unfreeze_lr=post_unfreeze_lr, oracle_metric=oracle_metric, **params)
    if model_name == "bertopic":
        from vaebm_benchmark.models.bertopic_adapter import BERTopicAdapter

        return BERTopicAdapter(n_clusters=k, embedding_model="all-MiniLM-L6-v2", random_state=seed)
    if model_name == "sbert_kmeans" or model_name in _SBERT_KMEANS_VARIANT_OVERRIDES:
        from vaebm_benchmark.models.sbert_kmeans_adapter import SBERTKMeansAdapter

        params = dict(_SBERT_KMEANS_DEFAULTS)
        params.update(_SBERT_KMEANS_VARIANT_OVERRIDES.get(model_name, {}))
        return SBERTKMeansAdapter(n_clusters=k, random_state=seed, **params)
    if model_name == "ecrtm":
        # Self-runs ECRTM (Wu et al., ICML 2023) under this repo's own
        # topic-experiment protocol (2026-09-21, replacing the Topic
        # tables' own ECRTM row, currently numbers copied verbatim from
        # HiCOT's own published table). Deliberately uses its own
        # self-fit vocabulary (voc_size cap, unconditional stopword
        # removal via topmost.Preprocess's own default - see
        # models/ecrtm_adapter.py's own docstring), matching this
        # project's own VAE-BM headline sweep's own vocabulary
        # methodology (self-fit, NOT VAEBM_USE_HICOT_VOCAB's opt-in
        # official-vocab path, off by default for VAE-BM's own headline
        # numbers too) - internally consistent with what we report for
        # ourselves, rather than reproducing HiCOT's own paper's exact
        # (different, unverified from our side) vocab choice.
        from vaebm_benchmark.models.ecrtm_adapter import ECRTMAdapter

        return ECRTMAdapter(num_topics=k, vocab_size_cap=voc_size, seed=seed)
    raise KeyError(f"Unknown model '{model_name}'. Available: {', '.join(KNOWN_MODELS)}")


def _assignment_source_for_model(model_name: str) -> str:
    """The known-model-family gate run_single() uses to decide clustering
    assignment - see its own comment on why this is a name-based lookup,
    not a check on what get_document_topics() returns. "unknown" for any
    model name not one of vaebm/bertopic/sbert_kmeans (or a registered
    variant of one) - the only case run_single() will actually try
    get_document_topics()/argmax for. Keyed by model family, not
    introspectable generically: "the geometric space KMeans clustered in"
    is a fact about each adapter's own fit() (VAE-BM's latent mu vs. a
    plain/UMAP-reduced sentence embedding), not something derivable from
    the adapter interface alone."""
    if model_name in ("vaebm", "vaebm_poe", "vaebm_dec", "vaebm_ckpt") or model_name in _VAEBM_VARIANT_OVERRIDES:
        return "kmeans_on_latent_mu"
    if model_name == "sbert_kmeans" or model_name in _SBERT_KMEANS_VARIANT_OVERRIDES or model_name == "bertopic":
        return "kmeans_on_embeddings"
    return "unknown"


def _topic_source_for_model(model_name: str) -> str:
    """"native" if get_topics() is the model's own learned/native output
    (VAE-BM's decoder energy/freq view, BERTopic's own c-TF-IDF);
    "cluster-derived" for sbert_kmeans, whose words are computed here
    from cluster membership via class-based TF-IDF - see
    models/sbert_kmeans_adapter.py's own module docstring."""
    if model_name == "sbert_kmeans" or model_name in _SBERT_KMEANS_VARIANT_OVERRIDES:
        return "cluster-derived"
    return "native"


def run_single(
    model_name: str, dataset_id: str, k: int, seed: int = 42, voc_size: int = 5000,
    protocol: str = "generic", cv_method: Optional[str] = None,
) -> ExperimentResult:
    from vaebm_benchmark.datasets.simple_registry import load_dataset
    from vaebm_benchmark.metrics.clustering_quality import nmi as compute_nmi
    from vaebm_benchmark.metrics.clustering_quality import purity as compute_purity
    from vaebm_benchmark.metrics.palmetto import PalmettoUnavailable, palmetto_cv
    from vaebm_benchmark.metrics.topic_quality import coherence, topic_diversity, topic_diversity_dieng_fixed_k
    from vaebm_benchmark.utils.seeding import set_all_seeds

    if protocol not in ("generic", "ecrtm_hicot"):
        raise ValueError(f"Unknown protocol '{protocol}'. Available: generic, ecrtm_hicot")
    if cv_method is not None and cv_method not in ("local", "palmetto"):
        raise ValueError(f"Unknown cv_method '{cv_method}'. Available: local, palmetto (or None to follow --protocol's own default)")

    # top_n/td_definition are still governed SOLELY by `protocol`, exactly
    # as before this parameter existed - see this module's own docstring
    # and docs/methodological_notes.md #10.
    if protocol == "ecrtm_hicot":
        top_n = 15
        td_definition = "dieng_unique_words_top15"
    else:
        top_n = 10
        td_definition = "unique_over_returned_slots"

    # `cv_method` decouples WHICH function computes C_V from `protocol`
    # (which still controls top_n/TD). Palmetto is a ~5.1GB one-time
    # download (scripts/setup_palmetto.py) - it is NEVER triggered
    # unless the caller EXPLICITLY passes `cv_method="palmetto"`. `None`
    # (nothing passed - the default, for EITHER protocol, including
    # "ecrtm_hicot") always means local-corpus gensim C_V - no auto-
    # install, no large download, regardless of `--protocol`. Pass
    # `cv_method="local"` explicitly too if you want to be unambiguous
    # about it under `ecrtm_hicot`. See docs/methodological_notes.md #10.
    effective_cv_method = cv_method if cv_method == "palmetto" else "local"
    cv_source = "palmetto_wikipedia" if effective_cv_method == "palmetto" else "cv_local_corpus"

    start = time.perf_counter()
    try:
        set_all_seeds(seed)
        documents, labels, _num_classes = load_dataset(dataset_id)

        model = _build_model(model_name, k, seed, voc_size)

        # Environment-gated, opt-in, OFF by default (unchanged prior
        # behavior): replaces the CLUSTERING embedding itself (not just
        # topic-word ranking, see VAEBM_USE_HICOT_STATIC_EMB below) with a
        # per-document average of HiCOT's own 200-dim GloVe word vectors -
        # a genuine "static embeddings as document embeddings" test, per
        # the user's 2026-09-20 directive's own explicit framing that this
        # should be deliberate/tested, never a silent default. Only
        # meaningful for vaebm-family models (VAEBMAdapter's own
        # embedder_name is passed straight to VaeBmKMeansFit.fit_predict's
        # `embedder` param, which already branches on isinstance(...,str)
        # vs. a precomputed array - models/vaebm.py was never touched for
        # this, the array-acceptance path already existed). Deliberately
        # NOT applied to sbert_kmeans - its own fit() always does
        # SentenceTransformer(self.embedder_name), unconditionally
        # treating it as a model name string; feeding it an array would
        # crash, not silently misbehave, so this is gated to the
        # vaebm-family model names explicitly rather than via hasattr.
        if os.environ.get("VAEBM_DOC_EMBEDDER", "").strip().lower() == "hicot_glove_avg":
            if model_name not in ("vaebm", "vaebm_poe", "vaebm_dec", "vaebm_ckpt") and model_name not in _VAEBM_VARIANT_OVERRIDES:
                raise ValueError(
                    f"VAEBM_DOC_EMBEDDER=hicot_glove_avg requires a vaebm-family model, got {model_name!r}"
                )
            if not dataset_id.startswith("hicot_"):
                raise ValueError(
                    f"VAEBM_DOC_EMBEDDER=hicot_glove_avg requires a hicot_* dataset_id, got {dataset_id!r} "
                    "- HiCOT's own vocab.txt/word_embeddings.npz only exist for its own 5 datasets."
                )
            model.embedder_name = _compute_hicot_glove_doc_embeddings(dataset_id, documents)

        # Environment-gated, opt-in, OFF by default (unchanged prior
        # behavior): fixes VAE-BM's own vocabulary to HiCOT's own official
        # vocab.txt (matching ECRTM's own Table 9 vocab-size column) and/or
        # scores topic words by cosine similarity to each cluster's own
        # frequency-weighted centroid in HiCOT's own 200-dim GloVe space
        # (matching ECRTM's Appendix B) instead of fitting a fresh
        # TfidfVectorizer vocabulary from these texts - both artifacts were
        # already downloaded/available (datasets/definitions/
        # hicot_datasets.py::load_hicot_vocab/load_hicot_word_embeddings)
        # but never wired into this runner before this pass (see their own
        # "NOT wired into experiment/runner.py automatically" docstrings).
        # 2026-09-20 "topic-word generation" research pass (see
        # docs/vaebm_mimic_gte_research_log.md). VAEBM_USE_HICOT_STATIC_EMB
        # implies the vocab fix too (the embeddings are row-aligned to it,
        # so using one without the other would silently misalign words).
        use_hicot_vocab = os.environ.get("VAEBM_USE_HICOT_VOCAB", "0").strip().lower() in ("1", "true", "yes")
        use_hicot_static_emb = os.environ.get("VAEBM_USE_HICOT_STATIC_EMB", "0").strip().lower() in ("1", "true", "yes")
        if (use_hicot_vocab or use_hicot_static_emb) and hasattr(model, "vocabulary"):
            from vaebm_benchmark.datasets.definitions.hicot_datasets import (
                load_hicot_vocab, load_hicot_word_embeddings,
            )

            if not dataset_id.startswith("hicot_"):
                raise ValueError(
                    f"VAEBM_USE_HICOT_VOCAB/VAEBM_USE_HICOT_STATIC_EMB require a hicot_* dataset_id, got {dataset_id!r} "
                    "- HiCOT's own vocab.txt/word_embeddings.npz only exist for its own 5 datasets."
                )
            model.vocabulary = load_hicot_vocab(dataset_id)
            if use_hicot_static_emb and hasattr(model, "static_embeddings"):
                model.static_embeddings = load_hicot_word_embeddings(dataset_id)
                # Round 14 finding: re-ranking over EVERY word present in a
                # cluster (often hundreds) diluted relevance and
                # underperformed plain frequency ranking on every dataset -
                # narrow to the top-N most frequent present words first
                # (default 40; VAEBM_STATIC_CANDIDATE_POOL to override).
                if hasattr(model, "static_candidate_pool"):
                    raw_pool = os.environ.get("VAEBM_STATIC_CANDIDATE_POOL", "40").strip()
                    model.static_candidate_pool = int(raw_pool) if raw_pool else None
                # Round 15 finding: the hard pool cutoff helped some
                # datasets (search_snippets, agnews) and hurt others
                # (20ng) non-monotonically - VAEBM_STATIC_HYBRID_WEIGHT
                # (default unset/None - unchanged prior behavior) replaces
                # pure similarity ranking with a soft frequency+similarity
                # blend instead, see top_words_by_freq_exact's own comment.
                if hasattr(model, "static_hybrid_weight"):
                    raw_hybrid = os.environ.get("VAEBM_STATIC_HYBRID_WEIGHT", "").strip()
                    model.static_hybrid_weight = float(raw_hybrid) if raw_hybrid else None

        # Environment-gated, opt-in, OFF by default (unchanged prior
        # behavior for every existing result): excludes standard English
        # stopwords (+ an optional extra custom list) from EVERY topic-word
        # mode. 2026-09-20 diagnostic finding (see
        # docs/vaebm_mimic_gte_research_log.md): plain (non-hicot) datasets'
        # default TfidfVectorizer applies no stopword filter, so their
        # topic-word lists can be dominated by function words ("the", "and",
        # "of") and un-stripped HTML artifacts ("br") that score
        # artificially high on Palmetto C_V (they co-occur with everything)
        # without reflecting genuine topic quality - this is suspected to be
        # why plain `imdb` appeared to beat HiCOT's own Cv target while
        # hicot_imdb (already stopword-free upstream) did not. Applies to
        # ANY dataset (not gated on hicot_*), independent of the
        # VAEBM_USE_HICOT_VOCAB/STATIC_EMB knobs above.
        if os.environ.get("VAEBM_EXCLUDE_STOPWORDS", "0").strip().lower() in ("1", "true", "yes") and hasattr(model, "exclude_words"):
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

            extra_raw = os.environ.get("VAEBM_EXTRA_EXCLUDE_WORDS", "").strip()
            extra = {w.strip().lower() for w in extra_raw.split(",") if w.strip()}
            model.exclude_words = set(ENGLISH_STOP_WORDS) | extra

        # Environment-gated, opt-in, OFF by default (unchanged prior
        # behavior for every existing result): the topic experiment's own
        # "labels never passed to fit()" rule is deliberately, narrowly
        # relaxed ONLY when VAEBM_TOPIC_ORACLE_LABELS=1 AND the model is
        # one of the oracle-checkpoint-selection family (vaebm_poe/
        # vaebm_dec/vaebm_ckpt) - mirrors cluster_runner.py's own,
        # pre-existing opt-in for the cluster experiment. 2026-09-19
        # "let the network learn a bit" research pass (see
        # docs/vaebm_mimic_gte_research_log.md) - labels are NEVER used in
        # the loss/gradient, only to rank already-computed epochs after
        # the fact (see models/vaebm_ckpt.py's own docstring). Any result
        # produced this way is oracle-label-informed, not a blind/
        # unsupervised number, and must not be conflated with this
        # experiment's own normal (label-blind) results.
        use_oracle_labels = (
            model_name in ("vaebm_poe", "vaebm_dec", "vaebm_ckpt")
            and os.environ.get("VAEBM_TOPIC_ORACLE_LABELS", "0").strip().lower() in ("1", "true", "yes")
        )
        if use_oracle_labels:
            model.fit(documents, labels=labels)
        else:
            model.fit(documents)

        topics_energy = model.get_topics(top_n=top_n)
        topics_freq = model.get_topics_both_views(top_n=top_n)["freq"] if hasattr(model, "get_topics_both_views") else []

        # Assignment: argmax(theta) ONLY for a model this repo has no
        # explicit knowledge about (_assignment_source_for_model returns
        # "unknown") AND whose get_document_topics() actually returns
        # something. This is NOT a generic "is get_document_topics() not
        # None" check - VAEBMAdapter.get_document_topics() ALWAYS returns
        # mu (never None; it's reused as the storage slot for
        # get_mu()/get_document_embeddings() elsewhere in this codebase),
        # even though mu is explicitly NOT a topic distribution (see
        # docs/methodological_notes.md #1) - treating that as theta would
        # be exactly the "pretend mu is theta" mistake this must avoid.
        # Gating on the known-model-family check below guarantees VAE-BM/
        # BERTopic/sbert_kmeans (and their registered variants) NEVER take
        # this branch, regardless of what their own get_document_topics()
        # happens to return; only a genuinely new, not-yet-special-cased
        # model (e.g. a future real ECRTM baseline) can.
        known_assignment_source = _assignment_source_for_model(model_name)
        document_topics = model.get_document_topics(documents) if known_assignment_source == "unknown" else None
        if document_topics is not None:
            clusters = [int(c) for c in np.argmax(np.asarray(document_topics), axis=1)]
            assignment_source = "argmax_theta"
        else:
            clusters = model.get_document_clusters(documents)
            assignment_source = known_assignment_source
        topic_source = _topic_source_for_model(model_name)

        non_empty_topics = [t for t in topics_energy if t]

        if effective_cv_method == "palmetto":
            try:
                cv = palmetto_cv(non_empty_topics, top_n=top_n) if non_empty_topics else None
            except PalmettoUnavailable:
                cv = None  # never silently substituted with cv_local_corpus - see module docstring
        else:
            reference_corpus = [doc.split() for doc in documents]
            try:
                cv = coherence(non_empty_topics, reference_corpus, top_n=top_n, measure="c_v")[0] if non_empty_topics else None
            except Exception:
                cv = None

        if protocol == "ecrtm_hicot":
            try:
                td = topic_diversity_dieng_fixed_k(topics_energy, k=k, top_n=top_n) if topics_energy else None
            except Exception:
                td = None
        else:
            try:
                td = topic_diversity(non_empty_topics, top_n=top_n) if non_empty_topics else None
            except Exception:
                td = None

        purity_value = compute_purity(clusters, labels)
        nmi_value = compute_nmi(clusters, labels)

        runtime = time.perf_counter() - start
        return ExperimentResult(
            model=model_name, dataset=dataset_id, k=k, cv=cv, purity=purity_value, nmi=nmi_value, td=td,
            seed=seed, runtime_seconds=runtime, status="ok",
            topics_energy=topics_energy, topics_freq=topics_freq,
            evaluation_protocol=protocol, top_n=top_n, cv_source=cv_source, td_definition=td_definition,
            assignment_source=assignment_source, topic_source=topic_source,
        )
    except Exception as exc:  # noqa: BLE001 - one failed combination must not abort the whole sweep
        runtime = time.perf_counter() - start
        return ExperimentResult(
            model=model_name, dataset=dataset_id, k=k, cv=None, purity=None, nmi=None, td=None,
            seed=seed, runtime_seconds=runtime, status="error",
            error=f"{exc}\n{traceback.format_exc(limit=3)}",
            evaluation_protocol=protocol, top_n=top_n, cv_source=cv_source, td_definition=td_definition,
        )
    finally:
        # Best-effort GPU/accelerator memory release before the NEXT
        # (model, dataset, k) combination tries to fit - a heavy model
        # (e.g. BGE-M3) can otherwise OOM purely because an EARLIER
        # model in the same sweep left memory allocated/fragmented, not
        # because the current combination itself doesn't fit. Runs after
        # BOTH a successful fit and a caught exception (an OOM'd model
        # is exactly the case that most needs this). `model` may be
        # unbound if `_build_model`/`load_dataset` itself raised before
        # ever assigning it.
        from vaebm_benchmark.utils.gpu_memory import release_accelerator_memory

        try:
            del model
        except NameError:
            pass
        release_accelerator_memory()


def run_sweep(
    models: list[str], datasets: list[str], ks: list[int], seed: int = 42,
    protocol: str = "generic", cv_method: Optional[str] = None,
) -> list[ExperimentResult]:
    """Prints a one-line status for each (model, dataset, k) combination
    AS IT FINISHES (not only after the whole sweep completes), so
    progress is visible during a long run and partial results survive
    even if a later combination is interrupted - the final aggregated
    table (scripts/run_experiment.py's own printing) still runs
    afterward on the complete returned list, unchanged."""
    results = []
    total = len(ks) * len(datasets) * len(models)
    count = 0
    for k in ks:
        for dataset_id in datasets:
            for model_name in models:
                count += 1
                result = run_single(model_name, dataset_id, k, seed=seed, protocol=protocol, cv_method=cv_method)
                results.append(result)
                if result.status == "ok":
                    print(f"[{count}/{total}] model={model_name} dataset={dataset_id} k={k}: ok "
                          f"(cv={result.cv} purity={result.purity} nmi={result.nmi} td={result.td})", flush=True)
                else:
                    print(f"[{count}/{total}] model={model_name} dataset={dataset_id} k={k}: ERROR "
                          f"{result.error.splitlines()[0]}", flush=True)
    return results
