#!/usr/bin/env python
"""RQ2 Phase 3: add GloCOM as a baseline (FASTopic protocol: K=50, 3
datasets, cluster + classification). Same reasoning as
rq3_glocom_add.py's own docstring - GloCOMAdapter's class defaults already
match the paper/official-CLI defaults for everything except epochs
(generic 20 vs paper's 200), so the small unsupervised search here is
epochs in {20, 100, 200}, selected via cluster-task C_V only, then reused
as-is for classification (no separate downstream-label-informed selection).

Usage:
    python scripts/rq2_glocom_add.py --datasets fastopic_20ng
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import rq4_common as common

RESULTS_DIR = REPO_ROOT / "results" / "rq2_glocom"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = RESULTS_DIR / "rq2_glocom_results.csv"
SEARCH_CSV = RESULTS_DIR / "rq2_glocom_search.csv"
RUN_LOG = RESULTS_DIR / "rq2_glocom_run.log"

DATASETS = ["fastopic_20ng", "fastopic_nyt", "fastopic_wos_reconstructed"]
K = 50
SEED = 42
EPOCHS_GRID = [20, 100, 200]

SEARCH_FIELDS = ["dataset", "epochs", "cv", "purity", "nmi", "runtime_s", "status", "error", "selected"]
RESULT_FIELDS = ["dataset", "k", "seed", "selected_epochs", "task", "purity", "nmi", "accuracy", "f1",
                  "cv", "runtime_s", "status", "error"]


def log(msg: str) -> None:
    with open(RUN_LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


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

    return GloCOMAdapter(num_topics=k, num_global_clusters=40, vocab_size_cap=5000, epochs=epochs, seed=seed)


def cluster_metrics_and_cv(protocol, dataset_id: str, epochs: int):
    documents, labels = protocol.prepare_all_documents_and_labels(dataset_id)
    model = build_glocom(K, SEED, epochs)
    start = time.perf_counter()
    model.fit(documents)
    runtime_s = time.perf_counter() - start

    feature_space = np.asarray(model.get_document_topics(documents))
    clusters = [int(i) for i in np.argmax(feature_space, axis=1)]
    topics = common.normalize_topic_words(model.get_topics(top_n=10))
    tokenized_corpus = [d.lower().split() for d in documents]

    from vaebm_benchmark.metrics.clustering_quality import compute_clustering_metrics
    ext = compute_clustering_metrics(clusters, labels, ["purity", "nmi"])

    augmented_corpus = tokenized_corpus + topics
    from vaebm_benchmark.metrics.topic_quality import coherence
    cv, _ = coherence(topics, augmented_corpus, top_n=10, measure="c_v") if len(topics) >= 2 else (None, None)

    return {"purity": ext["purity"], "nmi": ext["nmi"], "cv": cv, "runtime_s": runtime_s}


def classification_metrics(protocol, dataset_id: str, epochs: int):
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score

    train_docs = protocol.prepare_dataset(dataset_id)
    train_labels = protocol._load(dataset_id).train_labels
    test_docs = protocol.prepare_eval_documents(dataset_id)
    test_labels = protocol.prepare_labels(dataset_id)

    model = build_glocom(K, SEED, epochs)
    start = time.perf_counter()
    model.fit(train_docs)
    runtime_s = time.perf_counter() - start

    train_theta = np.asarray(model.get_document_topics(train_docs))
    test_theta = np.asarray(model.get_document_topics(test_docs))

    clf = SVC(kernel="linear", C=1.0)
    clf.fit(train_theta, train_labels)
    preds = clf.predict(test_theta)
    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average="macro")
    return {"accuracy": acc, "f1": f1, "runtime_s": runtime_s}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    args = p.parse_args()

    from vaebm_benchmark.protocols.fastopic_protocol import FASTopicProtocol

    protocol = FASTopicProtocol(smoke_test=False)
    log(f"=== RQ2 GloCOM add start datasets={args.datasets} k={K} seed={SEED} epochs_grid={EPOCHS_GRID} ===")

    for dataset_id in args.datasets:
        log(f"--- searching glocom:{dataset_id} ({len(EPOCHS_GRID)} epochs values, cluster task, C_V-only selection) ---")
        best_epochs, best_metrics, best_cv = None, None, -1e18
        for epochs in EPOCHS_GRID:
            try:
                metrics = cluster_metrics_and_cv(protocol, dataset_id, epochs)
                cv = metrics["cv"]
                cv_for_selection = cv if isinstance(cv, (int, float)) and not np.isnan(cv) else -1e18
                log(f"  SEARCH glocom:{dataset_id} epochs={epochs} cv={cv} "
                    f"(purity={metrics['purity']:.4f} nmi={metrics['nmi']:.4f} - logged only) runtime={metrics['runtime_s']:.1f}s")
                append_search({"dataset": dataset_id, "epochs": epochs, "cv": cv, "purity": metrics["purity"],
                                "nmi": metrics["nmi"], "runtime_s": metrics["runtime_s"], "status": "ok", "selected": False})
                if cv_for_selection > best_cv:
                    best_cv, best_epochs, best_metrics = cv_for_selection, epochs, metrics
            except Exception as exc:  # noqa: BLE001
                log(f"  SEARCH-FAIL glocom:{dataset_id} epochs={epochs} error={exc!r}")
                append_search({"dataset": dataset_id, "epochs": epochs, "status": "error", "error": f"{exc}", "selected": False})

        if best_epochs is None:
            log(f"ERROR glocom:{dataset_id}: every epochs value failed, skipping")
            continue

        log(f"SELECTED glocom:{dataset_id} epochs={best_epochs} cv={best_cv}")
        append_result({"dataset": dataset_id, "k": K, "seed": SEED, "selected_epochs": best_epochs,
                        "task": "cluster", "purity": best_metrics["purity"], "nmi": best_metrics["nmi"],
                        "cv": best_metrics["cv"], "runtime_s": best_metrics["runtime_s"], "status": "ok", "error": None})

        try:
            clf_metrics = classification_metrics(protocol, dataset_id, best_epochs)
            log(f"OK classification glocom:{dataset_id} acc={clf_metrics['accuracy']:.4f} f1={clf_metrics['f1']:.4f}")
            append_result({"dataset": dataset_id, "k": K, "seed": SEED, "selected_epochs": best_epochs,
                            "task": "classification", "accuracy": clf_metrics["accuracy"], "f1": clf_metrics["f1"],
                            "runtime_s": clf_metrics["runtime_s"], "status": "ok", "error": None})
        except Exception as exc:  # noqa: BLE001
            log(f"ERROR classification glocom:{dataset_id} error={exc!r}")
            append_result({"dataset": dataset_id, "k": K, "seed": SEED, "selected_epochs": best_epochs,
                            "task": "classification", "status": "error", "error": f"{exc}"})
            import traceback
            traceback.print_exc()

    log("=== RQ2 GloCOM add done ===")


if __name__ == "__main__":
    main()
