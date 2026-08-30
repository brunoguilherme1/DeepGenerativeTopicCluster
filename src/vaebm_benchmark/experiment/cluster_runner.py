"""Cluster experiment: pure document-clustering-quality comparison
(ACC + NMI only - no C_V/TD/Purity, see experiment/runner.py for the
topic-quality experiment those belong to).

    documents -> model.fit() -> model.get_document_clusters() -> compare
    against ground-truth labels (ACC via Hungarian-matched accuracy, NMI)

Labels are used ONLY after fitting, to (a) determine the benchmark's own
class cardinality K (`requested_k = num_classes`, never a value chosen by
looking at how well a model does) and (b) score ACC/NMI - never during
`model.fit()`.

MODEL REGISTRY, NOT BRANCHING: `CLUSTER_MODEL_BUILDERS` maps a model name
to a small builder function; `run_single()` below never inspects a model
name itself - it only ever calls the two capability methods every
adapter already implements (`fit(documents)`, `get_document_clusters
(documents)`, both from models/base.py::ProtocolModelAdapter). Adding a
future baseline to this experiment is exactly one new entry in that dict
plus (if needed) a small `_build_<name>` function - nothing else in this
module changes.

FASTopic/GloCOM here use their EXISTING adapters' generic, non-protocol-
pinned `fit(documents)` path (no released official BoW/vocab artifact
injected - see models/fastopic_adapter.py / models/glocom_adapter.py) -
appropriate for this experiment, which compares models on a SHARED
generic corpus, not a specific paper's own pinned dataset artifact.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class ClusterResult:
    experiment: str
    model: str
    dataset: str
    seed: int
    requested_k: int
    actual_k: Optional[int]
    num_classes: int
    acc: Optional[float]
    nmi: Optional[float]
    runtime_seconds: float
    status: str  # "ok" | "error"
    error: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _build_vaebm(k: int, seed: int, voc_size: int):
    from vaebm_benchmark.models.vaebm_adapter import VAEBMAdapter

    return VAEBMAdapter(
        n_clusters=k,
        voc_size=voc_size,
        units=50,
        epochs=30,  # VaeBmKMeansFit's own supplied default
        batch_size=128,
        lr=1e-3,  # see docs/methodological_notes.md #8 - 1e-2 diverges at these vocab scales
        random_state=seed,
        vectorizer_type="tfidf",
        embedder="all-MiniLM-L6-v2",
        dim=(1500, 1000, 500),
        dim_emb=(368,),
        alpha=0.99,
        top_words_mode="energy",
    )


def _build_bertopic(k: int, seed: int, voc_size: int):
    from vaebm_benchmark.models.bertopic_adapter import BERTopicAdapter

    return BERTopicAdapter(n_clusters=k, embedding_model="all-MiniLM-L6-v2", random_state=seed)


def _build_fastopic(k: int, seed: int, voc_size: int):
    from vaebm_benchmark.models.fastopic_adapter import FASTopicAdapter

    return FASTopicAdapter(
        num_topics=k,
        vocab_size_cap=voc_size,
        epochs=20,  # reduced from the paper's 200 - a generic cross-dataset smoke default, not a paper reproduction
        learning_rate=0.002,
        doc_embed_model="all-MiniLM-L6-v2",
        low_memory=False,
    )
    # released_train_bow/released_vocab intentionally left unset - this is
    # the generic (not protocol-pinned) path, appropriate for an arbitrary
    # shared corpus in this experiment.


def _build_glocom(k: int, seed: int, voc_size: int):
    from vaebm_benchmark.models.glocom_adapter import GloCOMAdapter

    return GloCOMAdapter(
        num_topics=k,
        num_global_clusters=40,  # the paper's own choice for its short-text datasets (see protocols/glocom_protocol.py)
        vocab_size_cap=voc_size,
        epochs=20,  # reduced from the paper's 200 - a generic cross-dataset smoke default, not a paper reproduction
        learning_rate=0.002,
        batch_size=200,
        seed=seed,
    )
    # Uses GloCOMAdapter.fit(documents) - the from-raw-text reconstruction
    # path, NOT fit_precomputed() (which is pinned to the official
    # SearchSnippets artifact for the GloCOM protocol track).


CLUSTER_MODEL_BUILDERS = {
    "vaebm": _build_vaebm,
    "bertopic": _build_bertopic,
    "fastopic": _build_fastopic,
    "glocom": _build_glocom,
}


def list_cluster_models() -> list[str]:
    return sorted(CLUSTER_MODEL_BUILDERS)


def run_single(model_name: str, dataset_id: str, seed: int = 42, voc_size: int = 5000) -> ClusterResult:
    from vaebm_benchmark.datasets.simple_registry import load_dataset, resolve_dataset_id
    from vaebm_benchmark.metrics.clustering_quality import accuracy_hungarian, nmi as compute_nmi
    from vaebm_benchmark.utils.seeding import set_all_seeds

    resolved_dataset_id = resolve_dataset_id(dataset_id)
    start = time.perf_counter()
    try:
        if model_name not in CLUSTER_MODEL_BUILDERS:
            raise KeyError(f"Unknown model '{model_name}'. Available: {list_cluster_models()}")

        set_all_seeds(seed)
        documents, labels, num_classes = load_dataset(resolved_dataset_id)
        requested_k = num_classes  # labels inspected ONLY to obtain K, never passed to fit()

        model = CLUSTER_MODEL_BUILDERS[model_name](requested_k, seed, voc_size)
        model.fit(documents)  # labels never passed here

        clusters = model.get_document_clusters(documents)
        actual_k = len(set(clusters))

        acc = accuracy_hungarian(clusters, labels)
        nmi_value = compute_nmi(clusters, labels)

        runtime = time.perf_counter() - start
        return ClusterResult(
            experiment="cluster", model=model_name, dataset=dataset_id, seed=seed,
            requested_k=requested_k, actual_k=actual_k, num_classes=num_classes,
            acc=acc, nmi=nmi_value, runtime_seconds=runtime, status="ok",
        )
    except Exception as exc:  # noqa: BLE001 - one failed combination must not abort the whole sweep
        runtime = time.perf_counter() - start
        return ClusterResult(
            experiment="cluster", model=model_name, dataset=dataset_id, seed=seed,
            requested_k=0, actual_k=None, num_classes=0, acc=None, nmi=None,
            runtime_seconds=runtime, status="error", error=f"{exc}\n{traceback.format_exc(limit=3)}",
        )


def run_sweep(
    models: list[str],
    datasets: list[str],
    seed: int = 42,
    voc_size: int = 5000,
    seeds: Optional[list[int]] = None,
) -> list[ClusterResult]:
    """Runs every (model, dataset) combination once per seed in `seeds`
    (default: just `[seed]`, i.e. today's single-run behavior) and
    returns the FLAT, per-seed result list - one ClusterResult per
    (model, dataset, seed). Averaging across seeds (for a reference-style
    "averaged over N random runs" table) is a REPORTING concern, not a
    running one - see experiment/cluster_report.py::aggregate_cluster_results,
    which consumes exactly this flat list. Nothing here is lost or
    pre-averaged; every individual seed's result is still in the
    returned list and gets persisted to cluster_results.csv/json."""
    seed_list = seeds if seeds else [seed]
    results = []
    for dataset_id in datasets:
        for model_name in models:
            for s in seed_list:
                results.append(run_single(model_name, dataset_id, seed=s, voc_size=voc_size))
    return results
