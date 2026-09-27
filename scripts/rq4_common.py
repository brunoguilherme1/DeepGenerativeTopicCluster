"""Shared infrastructure for RQ4 (results/rq4_llm/*): logging, LLM response
caching, checkpointing, model/dataset construction, topic-word
normalization, shared document embeddings, and the full metric suite -
used identically by Experiment A (2025 topic refinement) and Experiment B
(2026 cluster refinement) so every row in results/rq4_llm/partial_results.*
carries the same, complete schema regardless of which experiment produced
it.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "rq4_llm"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PARTIAL_JSON = RESULTS_DIR / "partial_results.json"
PARTIAL_CSV = RESULTS_DIR / "partial_results.csv"
CHECKPOINT_JSON = RESULTS_DIR / "checkpoint.json"
LLM_CACHE_JSONL = RESULTS_DIR / "llm_cache.jsonl"
RUN_LOG = RESULTS_DIR / "run.log"

TOPIC_METRIC_KEYS = ["cv", "npmi", "td", "irbo"]
EXTERNAL_METRIC_KEYS = ["acc", "nmi", "ari", "ami", "homogeneity", "completeness", "v_measure", "purity"]
INTERNAL_METRIC_KEYS = ["silhouette", "davies_bouldin", "calinski_harabasz"]
ALL_METRIC_KEYS = TOPIC_METRIC_KEYS + EXTERNAL_METRIC_KEYS + INTERNAL_METRIC_KEYS

IDENTITY_FIELDS = ["method", "backbone", "dataset", "seed", "k", "llm_model"]
_METRIC_COLUMNS = [f"{prefix}_{key}" for key in ALL_METRIC_KEYS for prefix in ("base", "refined", "delta")]
TRAILING_FIELDS = ["llm_calls", "cache_hits", "prompt_tokens", "completion_tokens", "runtime", "status", "error", "metrics_json"]
CSV_FIELDS = IDENTITY_FIELDS + _METRIC_COLUMNS + TRAILING_FIELDS


def log(msg: str) -> None:
    with open(RUN_LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def load_cache() -> dict:
    cache = {}
    if LLM_CACHE_JSONL.exists():
        with open(LLM_CACHE_JSONL) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cache[rec["key"]] = rec["response"]
    return cache


def append_cache(key: str, prompt: str, response: str) -> None:
    with open(LLM_CACHE_JSONL, "a") as f:
        f.write(json.dumps({"key": key, "prompt": prompt, "response": response}) + "\n")


def cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def llm_generate_cached(llm_client, prompt: str, cache: dict, run_stats: dict) -> str:
    """Single point of LLM invocation shared by both experiments: checks the
    persistent cache first, else calls the LLM, appends to cache, and tracks
    call/token counters in run_stats (mutated in place)."""
    key = cache_key(prompt)
    if key in cache:
        run_stats["cache_hits"] += 1
        return cache[key]
    result = llm_client.generate(prompt)
    append_cache(key, prompt, result.text)
    cache[key] = result.text
    run_stats["llm_calls"] += 1
    run_stats["total_prompt_tokens"] += result.prompt_tokens
    run_stats["total_completion_tokens"] += result.completion_tokens
    return result.text


def new_run_stats() -> dict:
    return {"llm_calls": 0, "cache_hits": 0, "total_prompt_tokens": 0, "total_completion_tokens": 0}


def build_combo_key(method: str, model: str, dataset: str, k: int, seed: int, max_docs: int | None = None) -> str:
    """Checkpoint/dedup key. Includes k/seed/scope so a smoke test (--max-docs
    set) and the corresponding full run never collide or falsely SKIP each
    other, unlike the original method:model:dataset-only key."""
    scope = f"smoke{max_docs}" if max_docs is not None else "full"
    return f"{method}:{model}:{dataset}:k{k}:seed{seed}:{scope}"


def load_checkpoint() -> set:
    if CHECKPOINT_JSON.exists():
        return set(json.load(open(CHECKPOINT_JSON)).get("done", []))
    return set()


def save_checkpoint(done: set) -> None:
    CHECKPOINT_JSON.write_text(json.dumps({"done": sorted(done), "ts": time.time()}, indent=2))


def normalize_topic_words(topics: list) -> list[list[str]]:
    """Different backbones' get_topics() return subtly different element
    types for the same conceptual "top words per topic" - FASTopic returns
    plain python str; others (observed with HiCOT) can return (word, score)
    tuples or numpy string types, which gensim's CoherenceModel cannot
    interpret ("unable to interpret topic as either a list of tokens or a
    list of ids"). Normalizing here, once, in the single shared model-
    building path, means every downstream metric function (Experiment A AND
    B) can assume a plain list[list[str]] regardless of backbone."""
    normalized = []
    for topic in topics:
        words = []
        for w in topic:
            if isinstance(w, (tuple, list)):
                w = w[0]
            words.append(str(w))
        normalized.append(words)
    return normalized


def build_model_and_assignment(model_name: str, dataset_id: str, seed: int, k: int,
                                top_n_words: int = 10, max_docs: int | None = None):
    """Fit `model_name` on `dataset_id`, return (documents, labels, clusters,
    topics, feature_space). Shared by both experiments so fastopic/hicot/vaebm
    are built identically (same k, seed, vocab_size) everywhere in RQ4.
    `topics` is always list[list[str]] (see normalize_topic_words).

    max_docs (smoke-test only): deterministically truncate to the first
    max_docs documents before fitting - same truncation for every backbone,
    so a smoke test still compares them fairly, just on a toy subset."""
    from vaebm_benchmark.datasets.simple_registry import load_dataset
    from vaebm_benchmark.experiment.cluster_runner import CLUSTER_MODEL_BUILDERS
    from vaebm_benchmark.experiment.scientific_models import assignment_source_for_model
    from vaebm_benchmark.utils.seeding import set_all_seeds
    import numpy as np

    set_all_seeds(seed)
    documents, labels, num_classes = load_dataset(dataset_id)
    if max_docs is not None:
        documents, labels = documents[:max_docs], labels[:max_docs]
        k = min(k, max(2, len(set(labels))))
    model = CLUSTER_MODEL_BUILDERS[model_name](k, seed, 5000)
    model.fit(documents)

    assignment_source = assignment_source_for_model(model_name)
    feature_space = None
    if assignment_source == "argmax_theta":
        feature_space = np.asarray(model.get_document_topics(documents))
        clusters = [int(i) for i in np.argmax(feature_space, axis=1)]
    else:
        clusters = model.get_document_clusters(documents)

    topics = normalize_topic_words(model.get_topics(top_n=top_n_words))
    return documents, labels, clusters, topics, feature_space


_SBERT_SINGLETON: dict = {}


def get_sbert(model_name: str = "all-MiniLM-L6-v2"):
    if "model" not in _SBERT_SINGLETON:
        from sentence_transformers import SentenceTransformer
        _SBERT_SINGLETON["model"] = SentenceTransformer(model_name)
    return _SBERT_SINGLETON["model"]


_DOC_EMBED_CACHE: dict = {}


def get_doc_embeddings(dataset_id: str, documents: list[str]):
    """Cached per dataset_id: identical documents recur across every backbone
    fit on the same dataset_id, so embeddings are computed once and reused -
    this also guarantees every backbone (and both Experiment A and B) uses
    the exact same document representation for internal clustering metrics,
    per the 'same representation before/after, per backbone' requirement."""
    if dataset_id not in _DOC_EMBED_CACHE:
        sbert = get_sbert()
        _DOC_EMBED_CACHE[dataset_id] = sbert.encode(documents, show_progress_bar=False, normalize_embeddings=True)
    return _DOC_EMBED_CACHE[dataset_id]


def compute_full_metric_suite(topics: list[list[str]] | None, tokenized_corpus: list[list[str]],
                               clusters, labels, doc_emb) -> dict:
    """The one full metric suite both experiments compute for both the base
    and the refined state: topic quality (CV/NPMI/TD/IRBO, skipped/None if
    topics is None or has <2 topics), external clustering (ACC/NMI/ARI/AMI/
    Homogeneity/Completeness/V-measure/Purity), and internal clustering
    (Silhouette/Davies-Bouldin/Calinski-Harabasz, on doc_emb - the SAME
    shared SBERT representation for a given dataset every time this is
    called, so before/after deltas reflect assignment changes only)."""
    from vaebm_benchmark.metrics.topic_quality import coherence, topic_diversity, irbo
    from vaebm_benchmark.metrics.clustering_quality import compute_clustering_metrics, compute_geometry_metrics

    result: dict = {}
    if topics and len(topics) >= 2:
        cv, _ = coherence(topics, tokenized_corpus, top_n=10, measure="c_v")
        npmi, _ = coherence(topics, tokenized_corpus, top_n=10, measure="c_npmi")
        result["cv"] = cv
        result["npmi"] = npmi
        result["td"] = topic_diversity(topics, top_n=10)
        result["irbo"] = irbo(topics, top_n=10)
    else:
        for key in TOPIC_METRIC_KEYS:
            result[key] = None

    result.update(compute_clustering_metrics(clusters, labels, EXTERNAL_METRIC_KEYS))

    try:
        result.update(compute_geometry_metrics(doc_emb, clusters, INTERNAL_METRIC_KEYS))
    except Exception as exc:  # noqa: BLE001
        for key in INTERNAL_METRIC_KEYS:
            result[key] = None
        result["geometry_error"] = f"{exc}"

    return result


def append_result(method: str, backbone: str, dataset: str, k: int, seed: int, llm_model: str,
                   run_stats: dict, runtime_s: float, status: str, error: str | None,
                   base_metrics: dict | None, refined_metrics: dict | None,
                   extra: dict | None = None) -> None:
    import csv

    base_metrics = base_metrics or {}
    refined_metrics = refined_metrics or {}
    row = {
        "method": method, "backbone": backbone, "dataset": dataset, "seed": seed, "k": k, "llm_model": llm_model,
        "llm_calls": run_stats.get("llm_calls"), "cache_hits": run_stats.get("cache_hits"),
        "prompt_tokens": run_stats.get("total_prompt_tokens"), "completion_tokens": run_stats.get("total_completion_tokens"),
        "runtime": round(runtime_s, 1), "status": status, "error": error,
    }
    for key in ALL_METRIC_KEYS:
        b, r = base_metrics.get(key), refined_metrics.get(key)
        row[f"base_{key}"] = b
        row[f"refined_{key}"] = r
        row[f"delta_{key}"] = (r - b) if isinstance(b, (int, float)) and isinstance(r, (int, float)) else None

    metrics_blob = {"base": base_metrics, "refined": refined_metrics}
    if extra:
        metrics_blob.update(extra)
    row["metrics_json"] = json.dumps(metrics_blob, default=str)

    is_new = not PARTIAL_CSV.exists()
    with open(PARTIAL_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow({k_: row.get(k_) for k_ in CSV_FIELDS})

    all_rows = []
    if PARTIAL_JSON.exists():
        all_rows = json.loads(PARTIAL_JSON.read_text())
    row_full = dict(row)
    row_full["metrics"] = metrics_blob
    all_rows.append(row_full)
    PARTIAL_JSON.write_text(json.dumps(all_rows, indent=2, default=str))
