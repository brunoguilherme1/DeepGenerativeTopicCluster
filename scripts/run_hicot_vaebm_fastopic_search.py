#!/usr/bin/env python
"""Small hyperparameter search for VAE-BM and HiCOT on the 3 FASTopic-protocol
datasets (fastopic_20ng, fastopic_nyt, fastopic_wos_reconstructed), full
(non-smoke) training, both cluster (Purity/NMI, transductive full-corpus fit)
and classification (Accuracy/macro-F1, official train/test split, SVC) tasks.

VAE-BM grid deliberately varies embedder (all-MiniLM-L6-v2, matching FASTopic's
own doc_embed_model, vs. gte-large, a stronger encoder) and
freeze_embedding_branch, per the project's own "can VAE-BM beat FASTopic's
published numbers" research question. HiCOT grid is a small variation around
its own paper-documented defaults (protocols/fastopic_protocol.py's
build_hicot cfg) - HiCOT is FASTopic's actual VAE-based competitor baseline,
run here (not as a rerun of a FASTopic-paper reference row) to compare
against our own VAE-BM on identical data/protocol.

NOT exhaustive - "a little hyperparameter search" per instruction, not the
full multi-day sweep run_fastopic_protocol_search.py was built for.

Must run on labuai (GPU), never locally - real training, not a smoke test.

Usage:
    python scripts/run_hicot_vaebm_fastopic_search.py
"""
from __future__ import annotations

# MUST import/use torch (via sentence_transformers) BEFORE tensorflow ever
# creates a real GPU device in this process - verified directly on labuai
# (2026-09-22): once TensorFlow has claimed a GPU device (which happens
# lazily, the first time a real Keras model/op runs, not at plain `import
# tensorflow`), importing sentence_transformers afterward segfaults
# (SIGSEGV) at the import itself, before any torch code runs. Reverse order
# (sentence_transformers first, forcing torch's CUDA init, then tensorflow)
# is stable - confirmed with a real Keras .fit() call after. This is the
# OPPOSITE of an earlier (wrong) fix attempt that imported tensorflow first.
from sentence_transformers import SentenceTransformer as _WarmUpSentenceTransformer  # noqa: F401
import torch as _torch

# GPU split, not just import order: TensorFlow defaults to grabbing ~all of
# a GPU's memory on first use and never releasing it for the rest of the
# process (verified 2026-09-23 on labuai - first combo succeeded, every
# later combo needing PyTorch got OutOfMemoryError with <200MiB free on a
# 10.9GiB card TF had already claimed 10.7+GiB of). 4 GPUs are idle here, so
# TF is pinned to GPU 0 (with memory growth, so it doesn't even hog that
# one) and torch/HiCOT/sentence-transformers default to GPU 1 - the two
# frameworks never share a device's memory pool at all.
import tensorflow as tf

_gpus = tf.config.list_physical_devices("GPU")
if _gpus:
    tf.config.set_visible_devices(_gpus[0], "GPU")
    tf.config.experimental.set_memory_growth(_gpus[0], True)
if _torch.cuda.is_available() and _torch.cuda.device_count() > 1:
    _torch.cuda.set_device(1)

import csv
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SEED = 42
DATASETS = ["fastopic_20ng", "fastopic_nyt", "fastopic_wos_reconstructed"]

VAEBM_GRID = [
    {"name": "mini_baseline", "embedder": "all-MiniLM-L6-v2", "alpha": 0.0, "units": 50, "freeze_embedding_branch": False},
    {"name": "gte_baseline", "embedder": "thenlper/gte-large", "alpha": 0.0, "units": 50, "freeze_embedding_branch": False},
    {"name": "gte_frozen", "embedder": "thenlper/gte-large", "alpha": 0.0, "units": 50, "freeze_embedding_branch": True},
]

HICOT_GRID = [
    {"name": "paper_default", "lr": 0.002, "weight_loss_ECR": 40.0, "weight_loss_DT": 250.0, "en_units": 200, "max_fit_seconds": 1800},
    {"name": "wider_units", "lr": 0.002, "weight_loss_ECR": 40.0, "weight_loss_DT": 250.0, "en_units": 300, "max_fit_seconds": 1800},
]

RESULTS_DIR = REPO_ROOT / "results" / "hicot_vaebm_fastopic_search"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = RESULTS_DIR / "trials.csv"
CSV_FIELDS = ["dataset", "model", "config_name", "config", "task", "status",
              "purity", "nmi", "accuracy", "f1", "runtime_s", "error"]


def _append_row(row: dict) -> None:
    is_new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def run_cluster(protocol, model, dataset_id):
    from sklearn.metrics import normalized_mutual_info_score
    from sklearn.metrics.cluster import contingency_matrix
    import numpy as np

    all_texts, all_labels = protocol.prepare_all_documents_and_labels(dataset_id)
    model.fit(all_texts)
    preds = np.asarray(model.get_document_clusters(all_texts))
    labels_arr = np.asarray(all_labels)

    cm = contingency_matrix(labels_arr, preds)
    purity = float(np.sum(np.amax(cm, axis=0)) / np.sum(cm))
    nmi = float(normalized_mutual_info_score(labels_arr, preds))
    return {"purity": purity, "nmi": nmi}


def run_classification(protocol, model, dataset_id):
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.svm import SVC

    train_texts = protocol.prepare_dataset(dataset_id)
    test_texts = protocol.prepare_eval_documents(dataset_id)
    bundle = protocol._load(dataset_id)
    train_labels = bundle.train_labels
    test_labels = bundle.test_labels

    model.fit(train_texts)
    train_repr = model.get_document_topics(train_texts)
    test_repr = model.get_document_topics(test_texts)

    clf = SVC(gamma="scale")
    clf.fit(train_repr, train_labels)
    preds = clf.predict(test_repr)
    acc = float(accuracy_score(test_labels, preds))
    f1 = float(f1_score(test_labels, preds, average="macro"))
    return {"accuracy": acc, "f1": f1}


def run_combo(protocol, model_name, dataset_id, config):
    build_fn = protocol.build_vaebm if model_name == "vaebm" else protocol.build_hicot
    for task, fn in [("cluster", run_cluster), ("classification", run_classification)]:
        start = time.perf_counter()
        row = {"dataset": dataset_id, "model": model_name, "config_name": config["name"],
               "config": {k: v for k, v in config.items() if k != "name"}, "task": task}
        try:
            model = build_fn(dataset_id, SEED, {k: v for k, v in config.items() if k != "name"})
            result = fn(protocol, model, dataset_id)
            row.update(result)
            row["status"] = "ok"
            row["runtime_s"] = round(time.perf_counter() - start, 1)
            print(f"OK   {dataset_id:28s} {model_name:6s} {config['name']:15s} {task:15s} "
                  f"{result} runtime={row['runtime_s']}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = repr(exc)
            row["runtime_s"] = round(time.perf_counter() - start, 1)
            print(f"ERROR {dataset_id:28s} {model_name:6s} {config['name']:15s} {task:15s} error={exc!r}", flush=True)
            traceback.print_exc()
        _append_row(row)


def main():
    from vaebm_benchmark.protocols.fastopic_protocol import FASTopicProtocol

    protocol = FASTopicProtocol(smoke_test=False)  # full epochs: vaebm=50, hicot=200
    print(f"=== HiCOT/VAE-BM hyperparameter search on FASTopic datasets (full training) ===")
    print(f"VAE-BM configs: {[c['name'] for c in VAEBM_GRID]}")
    print(f"HiCOT configs:  {[c['name'] for c in HICOT_GRID]}")
    print(f"Results -> {CSV_PATH}\n", flush=True)

    for dataset_id in DATASETS:
        for config in VAEBM_GRID:
            run_combo(protocol, "vaebm", dataset_id, config)
        for config in HICOT_GRID:
            run_combo(protocol, "hicot", dataset_id, config)

    print("\n=== DONE - see trials.csv for full results ===")


if __name__ == "__main__":
    main()
