#!/usr/bin/env python
"""One-off: GTE-large + KMeans + SVM classification ablation on HiCOT's
official-split hicot_* datasets (20ng, agnews, imdb - search_snippets/
google_news have no official train/test split). Companion to the
zero-training-cost topic ablation (--experiment topic --models
sbert_kmeans --sbert-embedder thenlper/gte-large), which the CLI allows
directly; this one exists only because run_experiment.py's CLI hard-blocks
--sbert-embedder outside --experiment topic, not because anything about
the underlying pipeline actually requires it.

Usage:
    python scripts/run_gte_kmeans_classif_ablation.py --datasets hicot_20ng hicot_agnews hicot_imdb
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

RESULTS_DIR = REPO_ROOT / "results" / "rq1_gte_kmeans_ablation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = RESULTS_DIR / "classification_results.csv"
FIELDS = ["dataset", "k", "seed", "accuracy", "f1", "runtime_s", "status", "error"]
K = 50
SEED = 42


def append_result(row: dict) -> None:
    is_new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in FIELDS})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True)
    args = p.parse_args()

    from vaebm_benchmark.models.sbert_kmeans_adapter import SBERTKMeansAdapter
    from vaebm_benchmark.datasets.definitions.hicot_datasets import load_hicot_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score

    for dataset_id in args.datasets:
        start = time.perf_counter()
        try:
            train_docs, train_labels, test_docs, test_labels, _num_classes = load_hicot_split(dataset_id)
            model = SBERTKMeansAdapter(n_clusters=K, embedder="thenlper/gte-large", random_state=SEED)
            model.fit(train_docs)
            train_emb = model.get_document_embeddings(train_docs)
            test_emb = model.get_document_embeddings(test_docs)
            clf = SVC(kernel="linear", C=1.0)
            clf.fit(train_emb, train_labels)
            preds = clf.predict(test_emb)
            acc = accuracy_score(test_labels, preds)
            f1 = f1_score(test_labels, preds, average="macro")
            runtime_s = time.perf_counter() - start
            print(f"OK {dataset_id} acc={acc:.4f} f1={f1:.4f} runtime={runtime_s:.1f}s", flush=True)
            append_result({"dataset": dataset_id, "k": K, "seed": SEED, "accuracy": acc, "f1": f1,
                            "runtime_s": runtime_s, "status": "ok", "error": None})
        except Exception as exc:  # noqa: BLE001
            runtime_s = time.perf_counter() - start
            print(f"ERROR {dataset_id} error={exc!r}", flush=True)
            append_result({"dataset": dataset_id, "k": K, "seed": SEED, "runtime_s": runtime_s,
                            "status": "error", "error": f"{exc}"})
            import traceback
            traceback.print_exc()

    print("=== GTE+KMeans classification ablation done ===")


if __name__ == "__main__":
    main()
