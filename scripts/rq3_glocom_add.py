#!/usr/bin/env python
"""RQ3 Phase 3: add GloCOM as a baseline.

GloCOMAdapter's own class defaults already match the official paper/CLI
defaults exactly for every hyperparameter EXCEPT epochs (see
glocom_adapter.py's own module docstring: prior_var=0.1, weight_loss_ECR=60.0
already ARE the paper's own run.py CLI values, verified by cloning the
official repo). The one real deviation is cluster_runner.py's own
_build_glocom using epochs=20 as "a generic cross-dataset smoke default, not
a paper reproduction" (vs. the paper's own 200). The small, paper-grounded,
unsupervised search here is therefore epochs in {20, 100, 200}, selected by
C_V only (never ACC/NMI/ARI/Purity/labels) - not a broad hyperparameter
search, since every other knob already matches the paper.

Usage:
    python scripts/rq3_glocom_add.py --datasets 20ng bbc_news m10
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import rq4_common as common

RESULTS_DIR = REPO_ROOT / "results" / "rq3_glocom"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = RESULTS_DIR / "rq3_glocom_results.csv"
SEARCH_CSV = RESULTS_DIR / "rq3_glocom_search.csv"
RUN_LOG = RESULTS_DIR / "rq3_glocom_run.log"

SEED = 42
VOC_SIZE = 5000
EPOCHS_GRID = [20, 100, 200]
ALL_DATASETS = ["20ng", "bbc_news", "m10", "stack_overflow", "biomedical", "tweet", "banking77"]
RESULT_FIELDS = (["dataset", "k", "actual_k", "seed", "num_documents", "selected_epochs"]
                  + list(common.ALL_METRIC_KEYS) + ["runtime_s", "status", "error"])
SEARCH_FIELDS = ["dataset", "epochs", "cv", "acc", "nmi", "runtime_s", "status", "error", "selected"]


def log(msg: str) -> None:
    with open(RUN_LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def load_done() -> set:
    if not RESULTS_CSV.exists():
        return set()
    with open(RESULTS_CSV) as f:
        return {row["dataset"] for row in csv.DictReader(f)}


def append_search(row: dict) -> None:
    is_new = not SEARCH_CSV.exists()
    with open(SEARCH_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SEARCH_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in SEARCH_FIELDS})


def append_result(row: dict) -> None:
    is_new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in RESULT_FIELDS})


def build_glocom(k: int, seed: int, epochs: int):
    from vaebm_benchmark.models.glocom_adapter import GloCOMAdapter

    return GloCOMAdapter(num_topics=k, num_global_clusters=40, vocab_size_cap=VOC_SIZE,
                          epochs=epochs, seed=seed)


def fit_and_evaluate(model, documents, labels, dataset_id: str) -> dict:
    model.fit(documents)
    feature_space = np.asarray(model.get_document_topics(documents))
    clusters = [int(i) for i in np.argmax(feature_space, axis=1)]
    topics = common.normalize_topic_words(model.get_topics(top_n=10))
    tokenized_corpus = [d.lower().split() for d in documents]
    doc_emb = common.get_doc_embeddings(dataset_id, documents)
    metrics = common.compute_full_metric_suite(topics, tokenized_corpus, clusters, labels, doc_emb)
    metrics["actual_k"] = len(set(clusters))
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    args = p.parse_args()

    from vaebm_benchmark.datasets.simple_registry import load_dataset

    done = load_done()
    log(f"=== RQ3 GloCOM add start datasets={args.datasets} epochs_grid={EPOCHS_GRID} ===")

    for dataset_id in args.datasets:
        if dataset_id in done:
            log(f"SKIP {dataset_id} (already done)")
            continue
        documents, labels, k = load_dataset(dataset_id)
        log(f"Dataset {dataset_id}: k={k} n_docs={len(documents)}")

        best_epochs, best_metrics, best_cv = None, None, -1e18
        for epochs in EPOCHS_GRID:
            start = time.perf_counter()
            try:
                model = build_glocom(k, SEED, epochs)
                metrics = fit_and_evaluate(model, documents, labels, dataset_id)
                runtime_s = time.perf_counter() - start
                cv = metrics.get("cv")
                cv_for_selection = cv if isinstance(cv, (int, float)) and not np.isnan(cv) else -1e18
                log(f"  SEARCH {dataset_id} epochs={epochs} cv={cv} "
                    f"(acc={metrics.get('acc'):.4f} nmi={metrics.get('nmi'):.4f} - logged only) runtime={runtime_s:.1f}s")
                append_search({"dataset": dataset_id, "epochs": epochs, "cv": cv, "acc": metrics.get("acc"),
                                "nmi": metrics.get("nmi"), "runtime_s": runtime_s, "status": "ok", "selected": False})
                if cv_for_selection > best_cv:
                    best_cv, best_epochs, best_metrics = cv_for_selection, epochs, metrics
            except Exception as exc:  # noqa: BLE001
                log(f"  SEARCH-FAIL {dataset_id} epochs={epochs} error={exc!r}")
                append_search({"dataset": dataset_id, "epochs": epochs, "status": "error", "error": f"{exc}", "selected": False})

        if best_epochs is None:
            log(f"ERROR {dataset_id}: every epochs value failed, skipping")
            continue

        log(f"SELECTED {dataset_id} epochs={best_epochs} cv={best_cv}")
        row = {"dataset": dataset_id, "k": k, "actual_k": best_metrics.pop("actual_k", None), "seed": SEED,
               "num_documents": len(documents), "selected_epochs": best_epochs, "status": "ok", "error": None,
               "runtime_s": None}
        row.update(best_metrics)
        append_result(row)

    log("=== RQ3 GloCOM add done ===")


if __name__ == "__main__":
    main()
