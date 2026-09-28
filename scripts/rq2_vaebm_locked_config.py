#!/usr/bin/env python
"""RQ2 quality fix, part 2: rerun VAE-BM on the 3 FASTopic-protocol datasets
with the TRUE locked configuration main.tex's method section actually
describes (gte-large embedder, frozen embedding branch, alpha=0.0,
units=1024, relevance top-words, lambda_relevance=0.1 - matching
docs/rq1_final_protocol.md), replacing the currently-reported
`hicot_vaebm_fastopic_search/trials.csv` row for VAE-BM, which used
units=50 and (in the "gte_baseline" row actually being reported) an
UNFROZEN embedding branch - a real, previously undisclosed deviation from
the paper's own stated method.

This is NOT a hyperparameter search (the config is already fixed/decided
elsewhere) - one fit per dataset per task, no selection step.

Usage:
    python scripts/rq2_vaebm_locked_config.py --datasets fastopic_20ng fastopic_nyt fastopic_wos_reconstructed
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

RESULTS_DIR = REPO_ROOT / "results" / "rq2_vaebm_locked"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = RESULTS_DIR / "rq2_vaebm_locked_results.csv"
RUN_LOG = RESULTS_DIR / "run.log"

DATASETS = ["fastopic_20ng", "fastopic_nyt", "fastopic_wos_reconstructed"]
K = 50  # FASTopic protocol's own fixed K
SEED = 42
LOCKED_CONFIG = dict(
    embedder="thenlper/gte-large", units=1024, alpha=0.0, freeze_embedding_branch=True,
    top_words_mode="relevance", lambda_relevance=0.1, epochs=1,
)
RESULT_FIELDS = ["dataset", "k", "seed", "task", "purity", "nmi", "accuracy", "f1", "cv", "runtime_s", "status", "error"]


def log(msg: str) -> None:
    with open(RUN_LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def append_result(row: dict) -> None:
    is_new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in RESULT_FIELDS})


def build_vaebm(k: int, seed: int):
    from vaebm_benchmark.models.vaebm_adapter import VAEBMAdapter

    return VAEBMAdapter(n_clusters=k, random_state=seed, **LOCKED_CONFIG)


def cluster_metrics_and_cv(protocol, dataset_id: str):
    documents, labels = protocol.prepare_all_documents_and_labels(dataset_id)
    model = build_vaebm(K, SEED)
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


def classification_metrics(protocol, dataset_id: str):
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score

    train_docs = protocol.prepare_dataset(dataset_id)
    train_labels = protocol._load(dataset_id).train_labels
    test_docs = protocol.prepare_eval_documents(dataset_id)
    test_labels = protocol.prepare_labels(dataset_id)

    model = build_vaebm(K, SEED)
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
    log(f"=== RQ2 VAE-BM locked-config rerun start datasets={args.datasets} k={K} seed={SEED} config={LOCKED_CONFIG} ===")

    for dataset_id in args.datasets:
        try:
            metrics = cluster_metrics_and_cv(protocol, dataset_id)
            log(f"OK cluster vaebm:{dataset_id} purity={metrics['purity']:.4f} nmi={metrics['nmi']:.4f} "
                f"cv={metrics['cv']} runtime={metrics['runtime_s']:.1f}s")
            append_result({"dataset": dataset_id, "k": K, "seed": SEED, "task": "cluster",
                            "purity": metrics["purity"], "nmi": metrics["nmi"], "cv": metrics["cv"],
                            "runtime_s": metrics["runtime_s"], "status": "ok", "error": None})
        except Exception as exc:  # noqa: BLE001
            log(f"ERROR cluster vaebm:{dataset_id} error={exc!r}")
            append_result({"dataset": dataset_id, "k": K, "seed": SEED, "task": "cluster", "status": "error", "error": f"{exc}"})
            import traceback
            traceback.print_exc()
            continue

        try:
            clf_metrics = classification_metrics(protocol, dataset_id)
            log(f"OK classification vaebm:{dataset_id} acc={clf_metrics['accuracy']:.4f} f1={clf_metrics['f1']:.4f}")
            append_result({"dataset": dataset_id, "k": K, "seed": SEED, "task": "classification",
                            "accuracy": clf_metrics["accuracy"], "f1": clf_metrics["f1"],
                            "runtime_s": clf_metrics["runtime_s"], "status": "ok", "error": None})
        except Exception as exc:  # noqa: BLE001
            log(f"ERROR classification vaebm:{dataset_id} error={exc!r}")
            append_result({"dataset": dataset_id, "k": K, "seed": SEED, "task": "classification", "status": "error", "error": f"{exc}"})
            import traceback
            traceback.print_exc()

    log("=== RQ2 VAE-BM locked-config rerun done ===")


if __name__ == "__main__":
    main()
