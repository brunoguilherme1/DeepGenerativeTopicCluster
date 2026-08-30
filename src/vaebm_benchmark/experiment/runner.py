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


def _build_model(model_name: str, k: int, seed: int, voc_size: int):
    if model_name == "vaebm":
        from vaebm_benchmark.models.vaebm_adapter import VAEBMAdapter

        return VAEBMAdapter(
            n_clusters=k,
            voc_size=voc_size,
            units=50,
            epochs=30,  # VaeBmKMeansFit's own supplied default
            batch_size=128,
            # NOT the supplied notebook's own default (1e-2) - this
            # project already established (docs/methodological_notes.md
            # #8) that 1e-2 diverges to inf/NaN at these vocab scales.
            # 1e-3 trains stably; documented here, not silent.
            lr=1e-3,
            random_state=seed,
            vectorizer_type="tfidf",
            embedder="all-MiniLM-L6-v2",
            dim=(1500, 1000, 500),
            dim_emb=(368,),
            alpha=0.99,
            top_words_mode="energy",
        )
    if model_name == "bertopic":
        from vaebm_benchmark.models.bertopic_adapter import BERTopicAdapter

        return BERTopicAdapter(n_clusters=k, embedding_model="all-MiniLM-L6-v2", random_state=seed)
    raise KeyError(f"Unknown model '{model_name}'. Available: vaebm, bertopic")


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
