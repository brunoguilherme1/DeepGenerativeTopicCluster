"""llm_cluster_refinement experiment: an LLM-based post-clustering
refinement stage (LLMEdgeRefine-style, Feng et al., EMNLP 2024) applied
on top of the hard clusters any registered baseline produces (reuses
experiment/cluster_runner.py's own `CLUSTER_MODEL_BUILDERS` registry -
model-agnostic by construction, not a new registry to keep in sync).

    documents -> baseline model -> hard clusters (BEFORE)
              -> document representation (native or shared)
              -> detect edge/uncertain points (llm/edge_detection.py)
              -> LLM refinement (llm/{client,prompt,parsing,cache}.py)
              -> refined hard clusters (AFTER)
              -> evaluation (before vs. after, external + internal metrics)

The LLM is a POST-PROCESSING stage only - this module never touches a
baseline model's own training procedure (models/*.py are used exactly as
the `cluster` experiment already uses them, via `model.fit(documents)`
then `model.get_document_clusters(documents)`).

SCOPE NOTE: per this feature's own task instructions, this module is
implemented and NOT executed/tested against a real dataset or a real LLM
in this session - no local run, no smoke test, no downloaded model
weights. It is written to the same standard of correctness as every
other module in this project, verified by direct code reading and (where
noted) lightweight, LLM-free unit tests of its pure-logic pieces
(llm/edge_detection.py, llm/candidate_clusters.py, llm/parsing.py,
llm/cache.py, llm/cluster_context.py) - never executed by this session
either, per the same instruction; see tests/test_llm_*.py.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from vaebm_benchmark.experiment.cluster_runner import CLUSTER_MODEL_BUILDERS


@dataclass
class RefinementConfig:
    llm_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    max_new_tokens: int = 64
    device: str = "auto"
    quantization: str = "4bit"  # "4bit" | "none"

    edge_fraction: float = 0.10
    edge_representation: str = "shared"  # "native" | "shared"
    shared_embedding_model: str = "all-MiniLM-L6-v2"

    candidate_clusters: int = 3
    min_confidence: float = 0.70
    refinement_iterations: int = 1

    top_words: int = 15
    top_docs: int = 3
    voc_size: int = 5000

    compute_topic_metrics: bool = False
    topic_metric_top_n: int = 10

    checkpoint_every: int = 20
    resume: bool = False
    cache_dir: Path = field(default_factory=lambda: Path("results/llm_cache"))


@dataclass
class LLMRefinementResult:
    experiment: str
    model: str
    dataset: str
    seed: int
    requested_k: int
    actual_k_before: Optional[int]
    actual_k_after: Optional[int]

    acc_before: Optional[float] = None
    acc_after: Optional[float] = None
    delta_acc: Optional[float] = None
    nmi_before: Optional[float] = None
    nmi_after: Optional[float] = None
    delta_nmi: Optional[float] = None
    ari_before: Optional[float] = None
    ari_after: Optional[float] = None
    delta_ari: Optional[float] = None
    ami_before: Optional[float] = None
    ami_after: Optional[float] = None
    delta_ami: Optional[float] = None
    purity_before: Optional[float] = None
    purity_after: Optional[float] = None
    delta_purity: Optional[float] = None

    # NOTE direction: davies_bouldin is LOWER-is-better - a positive
    # delta_davies_bouldin (after - before) means WORSE separation, the
    # one inverted-direction delta among these three (see
    # metrics/geometry_quality.py's module docstring).
    silhouette_before: Optional[float] = None
    silhouette_after: Optional[float] = None
    delta_silhouette: Optional[float] = None
    davies_bouldin_before: Optional[float] = None
    davies_bouldin_after: Optional[float] = None
    delta_davies_bouldin: Optional[float] = None
    calinski_harabasz_before: Optional[float] = None
    calinski_harabasz_after: Optional[float] = None
    delta_calinski_harabasz: Optional[float] = None

    # Topic-quality metrics - kept as THREE SEPARATE families, never merged:
    #   native_topic_metrics: the baseline model's OWN topics (unaffected by refinement)
    #   cluster_derived_topic_metrics_before / _after: c-TF-IDF topics derived from
    #     the hard cluster assignment itself, before/after refinement - NEVER
    #     described as "native FASTopic/GloCOM topics".
    native_topic_metrics: Optional[dict] = None
    cluster_derived_topic_metrics_before: Optional[dict] = None
    cluster_derived_topic_metrics_after: Optional[dict] = None

    # LLM cost / reproducibility metadata
    llm_model: str = ""
    quantization: str = ""
    temperature: float = 0.0
    max_new_tokens: int = 0
    edge_representation: str = ""
    candidate_clusters_config: int = 0
    min_confidence: float = 0.0
    refinement_iterations: int = 0
    cluster_context_provenance: str = ""
    number_of_documents: int = 0
    number_of_edge_points: int = 0
    number_of_llm_calls: int = 0
    number_of_cache_hits: int = 0
    number_of_reassignments: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    percentage_documents_sent_to_llm: float = 0.0
    number_of_parse_failures: int = 0

    runtime_seconds: float = 0.0
    status: str = "ok"
    error: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _compute_run_key(model_name: str, dataset_id: str, seed: int, config: RefinementConfig) -> str:
    """A deterministic hash identifying this exact (model, dataset, seed,
    refinement-config) combination - used to namespace `--resume`
    progress files (llm/resume.py) so resuming never mixes progress from
    an unrelated configuration."""
    payload = {
        "model": model_name, "dataset": dataset_id, "seed": seed,
        "llm_model": config.llm_model, "quantization": config.quantization,
        "edge_fraction": config.edge_fraction, "edge_representation": config.edge_representation,
        "candidate_clusters": config.candidate_clusters, "min_confidence": config.min_confidence,
        "refinement_iterations": config.refinement_iterations,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def resolve_representation(model, documents: list[str], config: RefinementConfig, shared_embedder=None) -> np.ndarray:
    """Resolves the geometry used for edge-point detection, candidate-
    cluster selection, representative-document selection, and (by
    default, when `edge_representation == "shared"`) the internal
    Silhouette/Davies-Bouldin/Calinski-Harabasz metrics too - see
    RefinementConfig.compute_topic_metrics's sibling note in
    metrics/geometry_quality.py about representation consistency."""
    if config.edge_representation == "native":
        embeddings = model.get_document_embeddings(documents)
        if embeddings is None:
            raise RuntimeError(
                f"{type(model).__name__} has no native document representation "
                "(get_document_embeddings() returned None) - use --edge-representation shared instead."
            )
        return np.asarray(embeddings)
    if config.edge_representation != "shared":
        raise ValueError(f"edge_representation must be 'native' or 'shared', got {config.edge_representation!r}")
    if shared_embedder is None:
        from sentence_transformers import SentenceTransformer

        shared_embedder = SentenceTransformer(config.shared_embedding_model)
    return np.asarray(shared_embedder.encode(list(documents), show_progress_bar=False))


def _external_and_internal_metrics(clusters: list[int], labels: list[int], representation: np.ndarray) -> dict:
    from vaebm_benchmark.metrics.clustering_quality import accuracy_hungarian, ami, ari, nmi, purity
    from vaebm_benchmark.metrics.geometry_quality import calinski_harabasz, davies_bouldin, silhouette

    return {
        "acc": accuracy_hungarian(clusters, labels),
        "nmi": nmi(clusters, labels),
        "ari": ari(clusters, labels),
        "ami": ami(clusters, labels),
        "purity": purity(clusters, labels),
        "silhouette": silhouette(representation, clusters),
        "davies_bouldin": davies_bouldin(representation, clusters),
        "calinski_harabasz": calinski_harabasz(representation, clusters),
    }


def _topic_metrics_from_words(topics: list[list[str]], reference_corpus: list[list[str]], top_n: int) -> dict:
    from vaebm_benchmark.metrics.topic_quality import coherence, irbo, topic_diversity

    non_empty = [t for t in topics if t]
    if len(non_empty) < 2:
        return {"cv": None, "npmi": None, "td": None, "irbo": None}
    return {
        "cv": coherence(non_empty, reference_corpus, top_n=top_n, measure="c_v")[0],
        "npmi": coherence(non_empty, reference_corpus, top_n=top_n, measure="c_npmi")[0],
        "td": topic_diversity(non_empty, top_n=top_n),
        "irbo": irbo(non_empty, top_n=top_n),
    }


def run_single(
    model_name: str,
    dataset_id: str,
    seed: int = 42,
    config: Optional[RefinementConfig] = None,
    llm_client_box: Optional[list] = None,
) -> LLMRefinementResult:
    """`llm_client_box`, if given, is a mutable one-item list used as a
    shared holder: `llm_client_box[0]` is read as a possibly-already-
    constructed LLMClient to reuse, and (if this call constructs a new
    one) written back so a caller sweeping many (model, dataset)
    combinations - see run_sweep() below - loads a 7B model AT MOST ONCE
    across the whole sweep, not once per combination. Never constructed
    at all unless at least one edge point actually needs it (see
    llm/client.py's own lazy-loading, which applies regardless of how the
    client reached this function)."""
    config = config or RefinementConfig()
    start = time.perf_counter()

    try:
        from vaebm_benchmark.datasets.simple_registry import load_dataset, resolve_dataset_id
        from vaebm_benchmark.llm.cache import LLMDecisionCache, make_cache_key
        from vaebm_benchmark.llm.candidate_clusters import select_candidate_clusters
        from vaebm_benchmark.llm.client import LLMClient
        from vaebm_benchmark.llm.cluster_context import build_cluster_contexts, hash_cluster_context
        from vaebm_benchmark.llm.edge_detection import compute_centroids, select_edge_points
        from vaebm_benchmark.llm.parsing import parse_llm_response
        from vaebm_benchmark.llm.prompt import build_prompt
        from vaebm_benchmark.llm.reassignment import decide_reassignment
        from vaebm_benchmark.llm.resume import clear_progress, load_progress, save_progress
        from vaebm_benchmark.utils.seeding import set_all_seeds

        if model_name not in CLUSTER_MODEL_BUILDERS:
            raise KeyError(f"Unknown model '{model_name}'. Available: {sorted(CLUSTER_MODEL_BUILDERS)}")

        resolved_dataset_id = resolve_dataset_id(dataset_id)
        set_all_seeds(seed)
        documents, labels, num_classes = load_dataset(resolved_dataset_id)
        requested_k = num_classes
        n_docs = len(documents)

        # --- baseline fit (labels never passed) ---
        model = CLUSTER_MODEL_BUILDERS[model_name](requested_k, seed, config.voc_size)
        model.fit(documents)
        clusters_before = list(model.get_document_clusters(documents))
        actual_k_before = len(set(clusters_before))

        representation = resolve_representation(model, documents, config)

        try:
            native_topics_list = model.get_topics(top_n=config.top_words)
            native_topics = {i: words for i, words in enumerate(native_topics_list)} if native_topics_list else None
        except NotImplementedError:
            native_topics = None

        metrics_before = _external_and_internal_metrics(clusters_before, labels, representation)

        # --- refinement loop ---
        run_key = _compute_run_key(model_name, dataset_id, seed, config)
        cache = LLMDecisionCache(config.cache_dir)
        progress = load_progress(config.cache_dir, run_key) if config.resume else {"iteration": 0, "decisions": {}}

        llm = llm_client_box[0] if llm_client_box else None
        number_of_llm_calls = 0
        number_of_cache_hits = 0
        number_of_reassignments = 0
        number_of_parse_failures = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_edge_points = 0
        decisions_made = 0
        decision_log: list[dict] = []

        current_clusters = list(clusters_before)
        start_iteration = progress.get("iteration", 0) if config.resume else 0

        for iteration in range(start_iteration, config.refinement_iterations):
            centroids = compute_centroids(representation, current_clusters)
            edge_indices = select_edge_points(representation, current_clusters, config.edge_fraction, centroids)
            total_edge_points += len(edge_indices)

            contexts, provenance = build_cluster_contexts(
                documents, representation, current_clusters, centroids,
                native_topics=native_topics, top_words=config.top_words, top_docs=config.top_docs,
            )

            resumed_decisions = progress.get("decisions", {}) if (config.resume and iteration == start_iteration) else {}
            new_clusters = list(current_clusters)

            for position, idx in enumerate(edge_indices):
                doc_key = str(idx)
                if doc_key in resumed_decisions:
                    entry = resumed_decisions[doc_key]
                    new_clusters[idx] = entry["final_cluster"]
                    if entry["changed"]:
                        number_of_reassignments += 1
                    decision_log.append(entry)
                    continue

                current_cluster = current_clusters[idx]
                candidates = select_candidate_clusters(representation[idx], current_cluster, centroids, config.candidate_clusters)
                candidate_contexts = {cid: contexts[cid] for cid in candidates}
                context_hash = hash_cluster_context(candidate_contexts)
                cache_key = make_cache_key(config.llm_model, documents[idx], candidates, context_hash)

                cached = cache.get(cache_key)
                if cached is not None:
                    number_of_cache_hits += 1
                    decision = cached
                else:
                    if llm is None:
                        llm = LLMClient(
                            model_name=config.llm_model, device=config.device,
                            quantization=config.quantization, max_new_tokens=config.max_new_tokens,
                        )
                        if llm_client_box is not None:
                            llm_client_box[0] = llm
                    prompt = build_prompt(documents[idx], current_cluster, candidate_contexts)
                    generation = llm.generate(prompt)
                    number_of_llm_calls += 1
                    total_prompt_tokens += generation.prompt_tokens
                    total_completion_tokens += generation.completion_tokens

                    parsed = parse_llm_response(generation.text, candidates)
                    if not parsed.valid:
                        number_of_parse_failures += 1
                    decision = {
                        "cluster_id": parsed.cluster_id, "confidence": parsed.confidence,
                        "valid": parsed.valid, "parse_error": parsed.parse_error,
                    }
                    cache.set(cache_key, decision)

                reassignment = decide_reassignment(
                    valid=decision["valid"], llm_cluster_id=decision["cluster_id"],
                    llm_confidence=decision["confidence"], current_cluster=current_cluster,
                    min_confidence=config.min_confidence,
                )
                if reassignment.changed:
                    number_of_reassignments += 1
                new_clusters[idx] = reassignment.final_cluster

                entry = {
                    "document_index": idx, "original_cluster": reassignment.original_cluster,
                    "suggested_cluster": reassignment.suggested_cluster, "final_cluster": reassignment.final_cluster,
                    "confidence": reassignment.confidence, "changed": reassignment.changed,
                }
                decision_log.append(entry)
                decisions_made += 1

                if config.resume and decisions_made % config.checkpoint_every == 0:
                    cache.flush()
                    progress["iteration"] = iteration
                    progress.setdefault("decisions", {})[doc_key] = entry
                    save_progress(config.cache_dir, run_key, progress)

            current_clusters = new_clusters
            cache.flush()
            if config.resume:
                progress = {"iteration": iteration + 1, "decisions": {}}
                save_progress(config.cache_dir, run_key, progress)

        if config.resume:
            clear_progress(config.cache_dir, run_key)

        clusters_after = current_clusters
        actual_k_after = len(set(clusters_after))
        metrics_after = _external_and_internal_metrics(clusters_after, labels, representation)

        def _delta(name: str) -> Optional[float]:
            before, after = metrics_before[name], metrics_after[name]
            return None if (before is None or after is None) else after - before

        native_topic_metrics = None
        cluster_derived_before = None
        cluster_derived_after = None
        if config.compute_topic_metrics:
            from vaebm_benchmark.llm.cluster_context import class_based_tfidf_topics

            reference_corpus = [doc.split() for doc in documents]
            if native_topics is not None:
                native_topic_metrics = _topic_metrics_from_words(
                    list(native_topics.values()), reference_corpus, config.topic_metric_top_n
                )
            derived_before_words = class_based_tfidf_topics(documents, clusters_before, top_n=config.topic_metric_top_n)
            cluster_derived_before = _topic_metrics_from_words(
                list(derived_before_words.values()), reference_corpus, config.topic_metric_top_n
            )
            derived_after_words = class_based_tfidf_topics(documents, clusters_after, top_n=config.topic_metric_top_n)
            cluster_derived_after = _topic_metrics_from_words(
                list(derived_after_words.values()), reference_corpus, config.topic_metric_top_n
            )

        total_tokens = total_prompt_tokens + total_completion_tokens
        pct_sent = 100.0 * total_edge_points / n_docs if n_docs else 0.0
        runtime = time.perf_counter() - start

        return LLMRefinementResult(
            experiment="llm_cluster_refinement", model=model_name, dataset=dataset_id, seed=seed,
            requested_k=requested_k, actual_k_before=actual_k_before, actual_k_after=actual_k_after,
            acc_before=metrics_before["acc"], acc_after=metrics_after["acc"], delta_acc=_delta("acc"),
            nmi_before=metrics_before["nmi"], nmi_after=metrics_after["nmi"], delta_nmi=_delta("nmi"),
            ari_before=metrics_before["ari"], ari_after=metrics_after["ari"], delta_ari=_delta("ari"),
            ami_before=metrics_before["ami"], ami_after=metrics_after["ami"], delta_ami=_delta("ami"),
            purity_before=metrics_before["purity"], purity_after=metrics_after["purity"], delta_purity=_delta("purity"),
            silhouette_before=metrics_before["silhouette"], silhouette_after=metrics_after["silhouette"],
            delta_silhouette=_delta("silhouette"),
            davies_bouldin_before=metrics_before["davies_bouldin"], davies_bouldin_after=metrics_after["davies_bouldin"],
            delta_davies_bouldin=_delta("davies_bouldin"),
            calinski_harabasz_before=metrics_before["calinski_harabasz"], calinski_harabasz_after=metrics_after["calinski_harabasz"],
            delta_calinski_harabasz=_delta("calinski_harabasz"),
            native_topic_metrics=native_topic_metrics,
            cluster_derived_topic_metrics_before=cluster_derived_before,
            cluster_derived_topic_metrics_after=cluster_derived_after,
            llm_model=config.llm_model, quantization=config.quantization, temperature=0.0,
            max_new_tokens=config.max_new_tokens, edge_representation=config.edge_representation,
            candidate_clusters_config=config.candidate_clusters, min_confidence=config.min_confidence,
            refinement_iterations=config.refinement_iterations,
            cluster_context_provenance="native_topic_words" if native_topics is not None else "ctfidf_adapter",
            number_of_documents=n_docs, number_of_edge_points=total_edge_points,
            number_of_llm_calls=number_of_llm_calls, number_of_cache_hits=number_of_cache_hits,
            number_of_reassignments=number_of_reassignments,
            prompt_tokens=total_prompt_tokens, completion_tokens=total_completion_tokens, total_tokens=total_tokens,
            percentage_documents_sent_to_llm=pct_sent, number_of_parse_failures=number_of_parse_failures,
            runtime_seconds=runtime, status="ok",
        )
    except Exception as exc:  # noqa: BLE001 - one failed combination must not abort a whole sweep
        runtime = time.perf_counter() - start
        return LLMRefinementResult(
            experiment="llm_cluster_refinement", model=model_name, dataset=dataset_id, seed=seed,
            requested_k=0, actual_k_before=None, actual_k_after=None,
            runtime_seconds=runtime, status="error", error=f"{exc}\n{traceback.format_exc(limit=3)}",
        )


def run_sweep(
    models: list[str],
    datasets: list[str],
    seed: int = 42,
    config: Optional[RefinementConfig] = None,
) -> list[LLMRefinementResult]:
    """Shares ONE LLMClient across the whole sweep via a mutable holder
    (see run_single's `llm_client_box` parameter) - constructed lazily on
    first actual use, then reused - so a 7B model is loaded AT MOST ONCE,
    not once per (model, dataset) combination."""
    config = config or RefinementConfig()
    llm_client_box: list = [None]
    results = []
    for dataset_id in datasets:
        for model_name in models:
            results.append(run_single(model_name, dataset_id, seed=seed, config=config, llm_client_box=llm_client_box))
    return results
