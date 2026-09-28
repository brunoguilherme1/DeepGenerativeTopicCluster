#!/usr/bin/env python
"""RQ2 quality fix: HiCOT's existing FASTopic-protocol results used
weight_loss_DT=250.0 ("paper_default" in scripts/run_hicot_vaebm_fastopic_search.py),
which is 25-500x outside HiCOT's own paper-documented range ([0.5, 0.7, 1, 2, 5, 10],
Findings ACL 2025 Appendix G.1/G.2) - confirmed via audit to be this repo's own
hardcoded adapter default, not a value re-derived from the paper, and confirmed
elsewhere in this same repo (RQ3's search grid) as the outlier candidate. Also,
every one of those runs' wall-clock runtime clustered right at the 1800s
max_fit_seconds cap, strong circumstantial evidence the target epochs=200 was
never reached (HiCOT's own paper trains for 400 epochs).

This reruns HiCOT (only - FASTopic's RQ2 results are already protocol-valid and
untouched) across all 3 FASTopic-protocol datasets (fastopic_20ng, fastopic_nyt,
fastopic_wos_reconstructed), for BOTH the cluster and classification tasks
(matching what RQ2 requires), with:
  - a small (3-config) grid over weight_loss_DT in {1, 2, 5} (paper's own
    documented range), weight_loss_ECR held at 40.0 (already within the paper's
    range, not flagged as a problem),
  - a longer max_fit_seconds (3600s instead of 1800s) to reduce (not guarantee
    elimination of) the wall-clock-driven undertraining risk,
  - selection via UNSUPERVISED C_V only (gensim, local corpus) - never
    accuracy/F1/purity/NMI/labels, computed on the CLUSTER task's own topics.
    The SAME winning config is then also used for the classification task (no
    separate, classification-specific hyperparameter selection - avoids any
    risk of implicitly optimizing on downstream labels).

Usage:
    python scripts/rq2_hicot_quality_fix.py
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

RESULTS_DIR = REPO_ROOT / "results" / "rq2_hicot_fix"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = RESULTS_DIR / "rq2_hicot_fix_results.csv"
SEARCH_CSV = RESULTS_DIR / "rq2_hicot_fix_search.csv"
RUN_LOG = RESULTS_DIR / "rq2_hicot_fix_run.log"

DATASETS = ["fastopic_20ng", "fastopic_nyt", "fastopic_wos_reconstructed"]
K = 50  # FASTopic protocol's own fixed K
SEED = 42
MAX_FIT_SECONDS = 3600
GRID = [
    {"weight_loss_ECR": 40.0, "weight_loss_DT": 1.0},
    {"weight_loss_ECR": 40.0, "weight_loss_DT": 2.0},
    {"weight_loss_ECR": 40.0, "weight_loss_DT": 5.0},
]

SEARCH_FIELDS = ["dataset", "config_json", "cv", "purity", "nmi", "accuracy", "f1", "runtime_s", "status", "error", "selected"]
RESULT_FIELDS = ["dataset", "k", "seed", "selected_config", "task", "purity", "nmi", "accuracy", "f1",
                  "cv", "runtime_s", "status", "error"]


def log(msg: str) -> None:
    with open(RUN_LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def append_search_row(row: dict) -> None:
    is_new = not SEARCH_CSV.exists()
    with open(SEARCH_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SEARCH_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in SEARCH_FIELDS})


def append_result_row(row: dict) -> None:
    is_new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in RESULT_FIELDS})


def build_hicot(k: int, seed: int, config: dict, voc_size: int = 5000):
    from vaebm_benchmark.models.hicot_adapter import HiCOTAdapter

    return HiCOTAdapter(n_clusters=k, voc_size=voc_size, epochs=200, sinkhorn_max_iter=100,
                         max_fit_seconds=MAX_FIT_SECONDS, random_state=seed, **config)


def cluster_metrics_and_cv(protocol, dataset_id: str, config: dict):
    """Transductive cluster fit (protocol's own convention), full metrics + C_V."""
    documents, labels = protocol.prepare_all_documents_and_labels(dataset_id)
    model = build_hicot(K, SEED, config)
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

    return {"purity": ext["purity"], "nmi": ext["nmi"], "cv": cv, "runtime_s": runtime_s}, model


def classification_metrics(protocol, dataset_id: str, config: dict):
    """Train/test split (protocol's own convention) - reuses the SAME config (already
    selected via cluster-task C_V), no separate hyperparameter selection here."""
    from sklearn.svm import SVC

    train_docs = protocol.prepare_dataset(dataset_id)
    train_labels = protocol._load(dataset_id).train_labels
    test_docs = protocol.prepare_eval_documents(dataset_id)
    test_labels = protocol.prepare_labels(dataset_id)

    model = build_hicot(K, SEED, config)
    start = time.perf_counter()
    model.fit(train_docs)
    runtime_s = time.perf_counter() - start

    train_theta = np.asarray(model.get_document_topics(train_docs))
    test_theta = np.asarray(model.get_document_topics(test_docs))

    clf = SVC(kernel="linear", C=1.0)
    clf.fit(train_theta, train_labels)
    preds = clf.predict(test_theta)

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average="macro")
    return {"accuracy": acc, "f1": f1, "runtime_s": runtime_s}


def main():
    from vaebm_benchmark.protocols.fastopic_protocol import FASTopicProtocol

    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=None, help="Limit to specific dataset(s) (default: all 3).")
    args = p.parse_args()
    datasets = args.datasets if args.datasets else DATASETS

    protocol = FASTopicProtocol(smoke_test=False)
    log(f"=== RQ2 HiCOT quality fix start datasets={datasets} k={K} seed={SEED} grid={GRID} ===")

    for dataset_id in datasets:
        log(f"--- searching hicot:{dataset_id} ({len(GRID)} configs, cluster task, C_V-only selection) ---")
        best_config, best_metrics, best_cv, best_model = None, None, -1e18, None
        for config in GRID:
            try:
                metrics, model = cluster_metrics_and_cv(protocol, dataset_id, config)
                cv = metrics["cv"]
                cv_for_selection = cv if isinstance(cv, (int, float)) and not np.isnan(cv) else -1e18
                log(f"  SEARCH hicot:{dataset_id} config={config} cv={cv} "
                    f"(purity={metrics['purity']:.4f} nmi={metrics['nmi']:.4f} - logged only) runtime={metrics['runtime_s']:.1f}s")
                append_search_row({"dataset": dataset_id, "config_json": json.dumps(config), "cv": cv,
                                    "purity": metrics["purity"], "nmi": metrics["nmi"],
                                    "runtime_s": metrics["runtime_s"], "status": "ok", "selected": False})
                if cv_for_selection > best_cv:
                    best_cv, best_config, best_metrics, best_model = cv_for_selection, config, metrics, model
            except Exception as exc:  # noqa: BLE001
                log(f"  SEARCH-FAIL hicot:{dataset_id} config={config} error={exc!r}")
                append_search_row({"dataset": dataset_id, "config_json": json.dumps(config), "status": "error",
                                    "error": f"{exc}", "selected": False})

        if best_config is None:
            log(f"ERROR hicot:{dataset_id}: every config failed, skipping")
            continue

        log(f"SELECTED hicot:{dataset_id} config={best_config} cv={best_cv}")
        append_result_row({"dataset": dataset_id, "k": K, "seed": SEED, "selected_config": json.dumps(best_config),
                            "task": "cluster", "purity": best_metrics["purity"], "nmi": best_metrics["nmi"],
                            "cv": best_metrics["cv"], "runtime_s": best_metrics["runtime_s"], "status": "ok", "error": None})

        try:
            clf_metrics = classification_metrics(protocol, dataset_id, best_config)
            log(f"OK classification hicot:{dataset_id} acc={clf_metrics['accuracy']:.4f} f1={clf_metrics['f1']:.4f}")
            append_result_row({"dataset": dataset_id, "k": K, "seed": SEED, "selected_config": json.dumps(best_config),
                                "task": "classification", "accuracy": clf_metrics["accuracy"], "f1": clf_metrics["f1"],
                                "runtime_s": clf_metrics["runtime_s"], "status": "ok", "error": None})
        except Exception as exc:  # noqa: BLE001
            log(f"ERROR classification hicot:{dataset_id} error={exc!r}")
            append_result_row({"dataset": dataset_id, "k": K, "seed": SEED, "selected_config": json.dumps(best_config),
                                "task": "classification", "status": "error", "error": f"{exc}"})
            import traceback
            traceback.print_exc()

    log("=== RQ2 HiCOT quality fix done ===")


if __name__ == "__main__":
    main()
