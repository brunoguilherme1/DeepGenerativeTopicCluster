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
"""

from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Optional


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

    def as_dict(self) -> dict:
        return asdict(self)


# VAEBMAdapter's own supplied defaults (what bare "vaebm" uses) - the
# base every named variant's overrides (see register_vaebm_variants) are
# layered on top of. NOT the notebook's own default lr (1e-2) - this
# project already established (docs/methodological_notes.md #8) that
# 1e-2 diverges to inf/NaN at these vocab scales; 1e-3 trains stably,
# and is the base every variant gets unless it overrides `lr` itself.
_VAEBM_DEFAULTS = dict(
    units=50,
    epochs=30,
    batch_size=128,
    lr=1e-3,
    vectorizer_type="tfidf",
    embedder="all-MiniLM-L6-v2",
    dim=(1500, 1000, 500),
    dim_emb=(368,),
    alpha=0.99,
    top_words_mode="energy",
    verbose=1,
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

KNOWN_MODELS = ["vaebm", "bertopic", "sbert_kmeans"]


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


def _build_model(model_name: str, k: int, seed: int, voc_size: int):
    if model_name == "vaebm" or model_name in _VAEBM_VARIANT_OVERRIDES:
        from vaebm_benchmark.models.vaebm_adapter import VAEBMAdapter

        params = dict(_VAEBM_DEFAULTS)
        params.update(_VAEBM_VARIANT_OVERRIDES.get(model_name, {}))
        return VAEBMAdapter(n_clusters=k, voc_size=voc_size, random_state=seed, **params)
    if model_name == "bertopic":
        from vaebm_benchmark.models.bertopic_adapter import BERTopicAdapter

        return BERTopicAdapter(n_clusters=k, embedding_model="all-MiniLM-L6-v2", random_state=seed)
    if model_name == "sbert_kmeans" or model_name in _SBERT_KMEANS_VARIANT_OVERRIDES:
        from vaebm_benchmark.models.sbert_kmeans_adapter import SBERTKMeansAdapter

        params = dict(_SBERT_KMEANS_DEFAULTS)
        params.update(_SBERT_KMEANS_VARIANT_OVERRIDES.get(model_name, {}))
        return SBERTKMeansAdapter(n_clusters=k, random_state=seed, **params)
    raise KeyError(f"Unknown model '{model_name}'. Available: {', '.join(KNOWN_MODELS)}")


def run_single(model_name: str, dataset_id: str, k: int, seed: int = 42, voc_size: int = 5000) -> ExperimentResult:
    from vaebm_benchmark.datasets.simple_registry import load_dataset
    from vaebm_benchmark.metrics.clustering_quality import nmi as compute_nmi
    from vaebm_benchmark.metrics.clustering_quality import purity as compute_purity
    from vaebm_benchmark.metrics.topic_quality import coherence, topic_diversity
    from vaebm_benchmark.utils.seeding import set_all_seeds

    start = time.perf_counter()
    try:
        set_all_seeds(seed)
        documents, labels, _num_classes = load_dataset(dataset_id)

        model = _build_model(model_name, k, seed, voc_size)
        model.fit(documents)

        topics_energy = model.get_topics(top_n=10)
        topics_freq = model.get_topics_both_views(top_n=10)["freq"] if hasattr(model, "get_topics_both_views") else []
        clusters = model.get_document_clusters(documents)

        reference_corpus = [doc.split() for doc in documents]
        non_empty_topics = [t for t in topics_energy if t]
        try:
            cv = coherence(non_empty_topics, reference_corpus, top_n=10, measure="c_v")[0] if non_empty_topics else None
        except Exception:
            cv = None
        try:
            td = topic_diversity(non_empty_topics, top_n=10) if non_empty_topics else None
        except Exception:
            td = None

        purity_value = compute_purity(clusters, labels)
        nmi_value = compute_nmi(clusters, labels)

        runtime = time.perf_counter() - start
        return ExperimentResult(
            model=model_name, dataset=dataset_id, k=k, cv=cv, purity=purity_value, nmi=nmi_value, td=td,
            seed=seed, runtime_seconds=runtime, status="ok",
            topics_energy=topics_energy, topics_freq=topics_freq,
        )
    except Exception as exc:  # noqa: BLE001 - one failed combination must not abort the whole sweep
        runtime = time.perf_counter() - start
        return ExperimentResult(
            model=model_name, dataset=dataset_id, k=k, cv=None, purity=None, nmi=None, td=None,
            seed=seed, runtime_seconds=runtime, status="error",
            error=f"{exc}\n{traceback.format_exc(limit=3)}",
        )


def run_sweep(models: list[str], datasets: list[str], ks: list[int], seed: int = 42) -> list[ExperimentResult]:
    results = []
    for k in ks:
        for dataset_id in datasets:
            for model_name in models:
                results.append(run_single(model_name, dataset_id, k, seed=seed))
    return results
