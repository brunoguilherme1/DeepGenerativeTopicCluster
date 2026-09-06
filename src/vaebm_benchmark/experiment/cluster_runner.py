"""Cluster experiment: unsupervised document-clustering-quality
comparison, using the FULL dataset transductively (fit AND evaluate on
the same combined corpus, NOT the HiCOT train/test split - see
experiment/classification_runner.py for the experiment that uses that
split; the same "full corpus" convention experiment/runner.py's topic
experiment and ECRTM's own Table 2/3 already use, see
docs/methodological_notes.md #11/#12).

    documents -> model.fit() -> hard cluster assignment -> compare
    against ground-truth labels (label-based metrics) AND against the
    model's own feature space (label-free geometry metrics)

Labels are used ONLY after fitting: (a) to determine the benchmark's own
class cardinality K (`requested_k = num_classes`, never a value chosen by
looking at how well a model does), and (b) to score the label-based
metrics below - never during `model.fit()`. Silhouette/Davies-Bouldin/
Calinski-Harabasz never see labels at all (see
metrics/clustering_quality.py's own module docstring on this
distinction).

Metrics computed:
  - Label-based (ground truth vs. predicted cluster): ACC (Hungarian),
    NMI, ARI, AMI, Homogeneity, Completeness, V-measure, Purity.
  - Label-free/geometry (predicted cluster vs. the model's OWN feature
    space only): Silhouette, Davies-Bouldin, Calinski-Harabasz.

`assignment_source`/`representation_source` (see experiment/
scientific_models.py) record, per model, HOW its hard cluster assignment
was derived and WHICH feature space the geometry metrics were computed
over - gated on model NAME, never on whether get_document_topics()
returns non-None (VAEBMAdapter's own get_document_topics() always
returns mu, never None - see docs/methodological_notes.md #1/#10):
"argmax_theta" for fastopic/glocom/lda/hicot (a genuine theta),
"kmeans_on_latent_mu" for vaebm (its EXISTING, unchanged KMeans-on-mu
behavior - mu is never softmaxed into a fake theta), "kmeans_on_embeddings"
for bertopic (its own SBERT-embedding KMeans swap).

MODEL REGISTRY, NOT BRANCHING: `CLUSTER_MODEL_BUILDERS` maps a model name
to a small builder function; `run_single()` below dispatches on
`assignment_source` (itself looked up by model name, see above), not on
the model name directly - adding a future baseline is one new entry in
`CLUSTER_MODEL_BUILDERS` plus a `representation_source_for_model()`/
`assignment_source_for_model()` case in scientific_models.py.

FASTopic/GloCOM/LDA/HiCOT here use their EXISTING adapters' generic,
non-protocol-pinned `fit(documents)` path (no released official BoW/
vocab/word-embeddings artifact injected for HiCOT's own datasets, unlike
experiment/classification_runner.py, which does inject them - see
scientific_models.py::build_hicot) - appropriate for this experiment,
which compares models on a SHARED generic corpus, not a specific paper's
own pinned dataset artifact.
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
    # Added alongside ari/ami/.../assignment_source below (originally
    # only acc/nmi existed) - defaulted and placed after `error` so
    # existing construction call sites (tests, this module's own
    # pre-existing callers) that only pass the original fields keep
    # working unchanged; new callers pass these as keyword arguments too.
    ari: Optional[float] = None
    ami: Optional[float] = None
    homogeneity: Optional[float] = None
    completeness: Optional[float] = None
    v_measure: Optional[float] = None
    purity: Optional[float] = None
    silhouette: Optional[float] = None
    davies_bouldin: Optional[float] = None
    calinski_harabasz: Optional[float] = None
    representation_source: str = ""
    assignment_source: str = ""

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


def _build_lda(k: int, seed: int, voc_size: int):
    from vaebm_benchmark.experiment.scientific_models import build_lda

    return build_lda(k, seed, voc_size)


def _build_hicot(k: int, seed: int, voc_size: int):
    """Generic (not protocol-pinned) path - self-fits its own vocabulary/
    word-embedding init, same reasoning as _build_fastopic/_build_glocom
    above. experiment/classification_runner.py uses HiCOT's own official
    vocab/word-embeddings for hicot_* datasets instead - see
    scientific_models.py::build_hicot's `dataset_id` parameter, left
    unset (None) here."""
    from vaebm_benchmark.experiment.scientific_models import build_hicot

    return build_hicot(k, seed, voc_size)


def _build_sbert_kmeans(k: int, seed: int, voc_size: int):
    from vaebm_benchmark.experiment.scientific_models import build_sbert_kmeans

    return build_sbert_kmeans(k, seed, voc_size)


CLUSTER_MODEL_BUILDERS = {
    "vaebm": _build_vaebm,
    "bertopic": _build_bertopic,
    "fastopic": _build_fastopic,
    "glocom": _build_glocom,
    "lda": _build_lda,
    "hicot": _build_hicot,
    "sbert_kmeans": _build_sbert_kmeans,
}


def list_cluster_models() -> list[str]:
    return sorted(CLUSTER_MODEL_BUILDERS)


LABEL_METRIC_IDS = ["acc", "nmi", "ari", "ami", "homogeneity", "completeness", "v_measure", "purity"]
GEOMETRY_METRIC_IDS = ["silhouette", "davies_bouldin", "calinski_harabasz"]


def run_single(model_name: str, dataset_id: str, seed: int = 42, voc_size: int = 5000) -> ClusterResult:
    import numpy as np

    from vaebm_benchmark.datasets.simple_registry import load_dataset, resolve_dataset_id
    from vaebm_benchmark.experiment.scientific_models import assignment_source_for_model, representation_source_for_model
    from vaebm_benchmark.metrics.clustering_quality import compute_clustering_metrics, compute_geometry_metrics
    from vaebm_benchmark.utils.seeding import set_all_seeds

    resolved_dataset_id = resolve_dataset_id(dataset_id)
    empty_metrics = {name: None for name in LABEL_METRIC_IDS + GEOMETRY_METRIC_IDS}
    start = time.perf_counter()
    try:
        if model_name not in CLUSTER_MODEL_BUILDERS:
            raise KeyError(f"Unknown model '{model_name}'. Available: {list_cluster_models()}")

        set_all_seeds(seed)
        documents, labels, num_classes = load_dataset(resolved_dataset_id)
        requested_k = num_classes  # labels inspected ONLY to obtain K, never passed to fit()

        model = CLUSTER_MODEL_BUILDERS[model_name](requested_k, seed, voc_size)
        model.fit(documents)  # labels never passed here

        representation_source = representation_source_for_model(model_name)
        assignment_source = assignment_source_for_model(model_name)

        # Hard cluster assignment - "argmax_theta" for a genuine theta
        # (fastopic/glocom/lda/hicot), the model's OWN existing
        # get_document_clusters() otherwise (vaebm's KMeans-on-mu,
        # bertopic's KMeans-on-embeddings - both UNCHANGED). Same
        # feature_space also feeds the label-free geometry metrics below.
        if assignment_source == "argmax_theta":
            feature_space = model.get_document_topics(documents)
            clusters = [int(i) for i in np.argmax(np.asarray(feature_space), axis=1)]
        else:
            clusters = model.get_document_clusters(documents)
            feature_space = model.get_document_embeddings(documents)  # mu (vaebm) or embeddings (bertopic)
        actual_k = len(set(clusters))

        label_metrics = compute_clustering_metrics(clusters, labels, LABEL_METRIC_IDS)
        try:
            geometry_metrics = compute_geometry_metrics(feature_space, clusters, GEOMETRY_METRIC_IDS)
        except Exception:
            # Silhouette/DB/CH require >=2 non-trivial clusters - a
            # degenerate assignment (e.g. actual_k < 2) fails these
            # specifically without invalidating the label-based metrics
            # above, which don't have this requirement.
            geometry_metrics = {name: None for name in GEOMETRY_METRIC_IDS}

        runtime = time.perf_counter() - start
        return ClusterResult(
            experiment="cluster", model=model_name, dataset=dataset_id, seed=seed,
            requested_k=requested_k, actual_k=actual_k, num_classes=num_classes,
            representation_source=representation_source, assignment_source=assignment_source,
            runtime_seconds=runtime, status="ok",
            **label_metrics, **geometry_metrics,
        )
    except Exception as exc:  # noqa: BLE001 - one failed combination must not abort the whole sweep
        runtime = time.perf_counter() - start
        return ClusterResult(
            experiment="cluster", model=model_name, dataset=dataset_id, seed=seed,
            requested_k=0, actual_k=None, num_classes=0,
            representation_source="", assignment_source="",
            runtime_seconds=runtime, status="error", error=f"{exc}\n{traceback.format_exc(limit=3)}",
            **empty_metrics,
        )
    finally:
        # Best-effort GPU/accelerator memory release before the NEXT
        # (model, dataset, seed) combination tries to fit - see
        # utils/gpu_memory.py's own module docstring for why.
        from vaebm_benchmark.utils.gpu_memory import release_accelerator_memory

        try:
            del model
        except NameError:
            pass
        release_accelerator_memory()


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
    returned list and gets persisted to cluster_results.csv/json.

    Prints a one-line status per (model, dataset, seed) combination AS
    IT FINISHES, so progress is visible during a long run and partial
    results survive even if a later combination is interrupted."""
    seed_list = seeds if seeds else [seed]
    results = []
    total = len(datasets) * len(models) * len(seed_list)
    count = 0
    for dataset_id in datasets:
        for model_name in models:
            for s in seed_list:
                count += 1
                result = run_single(model_name, dataset_id, seed=s, voc_size=voc_size)
                results.append(result)
                if result.status == "ok":
                    print(f"[{count}/{total}] model={model_name} dataset={dataset_id} seed={s}: ok "
                          f"acc={result.acc} nmi={result.nmi}", flush=True)
                else:
                    print(f"[{count}/{total}] model={model_name} dataset={dataset_id} seed={s}: ERROR "
                          f"{result.error.splitlines()[0]}", flush=True)
    return results
