#!/usr/bin/env python
"""Smoke test for protocols/fastopic_protocol.py: runs VAE-BM and HiCOT,
smoke_test=True (few epochs), on all 3 datasets (fastopic_20ng,
fastopic_nyt, fastopic_wos_reconstructed), for both tasks:
  - cluster: transductive full-corpus fit, Purity/NMI
    (argmax(theta) for HiCOT, KMeans(mu) for VAE-BM)
  - classification: fit on official train split only, SVC(gamma='scale')
    on theta/mu, Accuracy + macro-F1 on official test split

This is NOT the full hyperparameter search - it's a quick pipeline
validation (does every combination run end-to-end without error, and are
the numbers in a sane range) before committing to a longer run.

Usage:
    python scripts/smoke_test_fastopic_protocol.py
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SEED = 42
DATASETS = ["fastopic_20ng", "fastopic_nyt", "fastopic_wos_reconstructed"]
MODELS = ["vaebm", "hicot"]


def run_cluster(protocol, model_name, dataset_id):
    from sklearn.metrics import normalized_mutual_info_score
    from sklearn.metrics.cluster import contingency_matrix
    import numpy as np

    all_texts, all_labels = protocol.prepare_all_documents_and_labels(dataset_id)
    model = protocol.build_model(model_name, dataset_id, SEED)
    model.fit(all_texts)
    preds = np.asarray(model.get_document_clusters(all_texts))
    labels_arr = np.asarray(all_labels)

    cm = contingency_matrix(labels_arr, preds)
    purity = float(np.sum(np.amax(cm, axis=0)) / np.sum(cm))
    nmi = float(normalized_mutual_info_score(labels_arr, preds))
    return {"purity": purity, "nmi": nmi}


def run_classification(protocol, model_name, dataset_id):
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.svm import SVC

    train_texts = protocol.prepare_dataset(dataset_id)
    test_texts = protocol.prepare_eval_documents(dataset_id)
    bundle = protocol._load(dataset_id)
    train_labels = bundle.train_labels
    test_labels = bundle.test_labels

    model = protocol.build_model(model_name, dataset_id, SEED)
    model.fit(train_texts)
    train_repr = model.get_document_topics(train_texts)
    test_repr = model.get_document_topics(test_texts)

    clf = SVC(gamma="scale")
    clf.fit(train_repr, train_labels)
    preds = clf.predict(test_repr)
    acc = float(accuracy_score(test_labels, preds))
    f1 = float(f1_score(test_labels, preds, average="macro"))
    return {"accuracy": acc, "f1": f1}


def main():
    from vaebm_benchmark.protocols.fastopic_protocol import FASTopicProtocol

    protocol = FASTopicProtocol(smoke_test=True)
    print(f"=== FASTopic protocol smoke test (K={protocol.topic_count}) ===")
    for check in protocol.checks():
        print(f"  [{check.status.value}] {check.field}: {check.note}")
    print()

    results = []
    for dataset_id in DATASETS:
        for model_name in MODELS:
            for task, fn in [("cluster", run_cluster), ("classification", run_classification)]:
                start = time.perf_counter()
                print(f"START task={task} model={model_name} dataset={dataset_id} ...", flush=True)
                try:
                    r = fn(protocol, model_name, dataset_id)
                    runtime = time.perf_counter() - start
                    print(f"OK   task={task} model={model_name} dataset={dataset_id} result={r} runtime={runtime:.1f}s")
                    results.append({"task": task, "model": model_name, "dataset": dataset_id, "status": "ok", "result": r, "runtime_s": runtime})
                except Exception as exc:  # noqa: BLE001
                    runtime = time.perf_counter() - start
                    print(f"ERROR task={task} model={model_name} dataset={dataset_id} error={exc!r}")
                    traceback.print_exc()
                    results.append({"task": task, "model": model_name, "dataset": dataset_id, "status": "error", "error": repr(exc), "runtime_s": runtime})

    print("\n=== SUMMARY ===")
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"{ok}/{len(results)} combinations succeeded")
    for r in results:
        tag = "OK  " if r["status"] == "ok" else "FAIL"
        detail = r.get("result", r.get("error"))
        print(f"  {tag} {r['task']:15s} {r['model']:6s} {r['dataset']:28s} {detail}")

    print("\n=== PUBLISHED REFERENCE (FASTopic paper, for comparison only) ===")
    for pr in protocol.published_results:
        print(f"  {pr.dataset_id:28s} {pr.metric:10s} = {pr.value}  ({pr.source})")


if __name__ == "__main__":
    main()
