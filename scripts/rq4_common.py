"""Shared infrastructure for RQ4 (results/rq4_llm/*): logging, LLM response
caching, checkpointing, and a single unified result schema so Experiment A
(2025 topic refinement) and Experiment B (2026 cluster refinement) can write
to the SAME results/rq4_llm/{partial_results.json,partial_results.csv,
checkpoint.json,llm_cache.jsonl,run.log} without schema collisions - each
row is tagged with a "method" field ("2025_topic_refinement" or
"2026_cluster_refinement") and carries a JSON-encoded "metrics_json" blob,
since the two experiments track different metric sets.
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

CSV_FIELDS = ["method", "model", "dataset", "k", "seed", "llm_calls", "cache_hits",
              "prompt_tokens", "completion_tokens", "runtime_s", "status", "error", "metrics_json"]


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


def load_checkpoint() -> set:
    if CHECKPOINT_JSON.exists():
        return set(json.load(open(CHECKPOINT_JSON)).get("done", []))
    return set()


def save_checkpoint(done: set) -> None:
    CHECKPOINT_JSON.write_text(json.dumps({"done": sorted(done), "ts": time.time()}, indent=2))


def append_result(method: str, model: str, dataset: str, k: int, seed: int,
                   run_stats: dict, runtime_s: float, status: str, error: str | None,
                   metrics: dict) -> None:
    import csv

    row = {
        "method": method, "model": model, "dataset": dataset, "k": k, "seed": seed,
        "llm_calls": run_stats.get("llm_calls"), "cache_hits": run_stats.get("cache_hits"),
        "prompt_tokens": run_stats.get("total_prompt_tokens"), "completion_tokens": run_stats.get("total_completion_tokens"),
        "runtime_s": round(runtime_s, 1), "status": status, "error": error,
        "metrics_json": json.dumps(metrics, default=str),
    }

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
    row_full["metrics"] = metrics
    all_rows.append(row_full)
    PARTIAL_JSON.write_text(json.dumps(all_rows, indent=2, default=str))


def build_model_and_assignment(model_name: str, dataset_id: str, seed: int, k: int,
                                top_n_words: int = 10, max_docs: int | None = None):
    """Fit `model_name` on `dataset_id`, return (documents, labels, clusters,
    topics, feature_space). Shared by both experiments so fastopic/hicot/vaebm
    are built identically (same k, seed, vocab_size) everywhere in RQ4.

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

    topics = model.get_topics(top_n=top_n_words)
    return documents, labels, clusters, topics, feature_space
