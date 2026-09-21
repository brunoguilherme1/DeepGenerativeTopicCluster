#!/usr/bin/env python
"""Classification for the two datasets (20ng, imdb) that are structurally
too slow for run_vaebm_classification_locked_12ds.py's per-seed-refit
approach - see docs/vaebm_leaderboard.md's 2026-09-21 entry: 20ng's own
single VAE-BM fit took 1374s in the cluster experiment (long
newsgroup posts -> expensive gte-large encode), and IMDB's bge-large
encode of 50k documents takes ~50 min - refitting 5x sequentially
(the general run_single_random_split() protocol's own convention,
classification_runner.py's own docstring: "the TOPIC MODEL itself is
refit once per seed") blows any single-combo timeout regardless of
the classifier used (confirmed: both timed out at 3600s even after
the SVC->LinearSVC fix landed).

This script instead fits VAE-BM ONCE per dataset (on the FULL corpus,
not a train-only split) and reuses the resulting `mu` across all 5
seeds - re-splitting (mu, labels) and refitting only the (cheap)
LinearSVC per seed. This is NOT a shortcut that changes the result
under the locked architecture specifically: at alpha=0 with a frozen,
identity-initialized embedding branch, mu = mu_emb = the raw
embedder's own output for that document, independent of the BoW
vectorizer's own fit, of training-epoch count, AND of which other
documents happen to be in the fitting corpus (there is no
document-to-document interaction anywhere in that computation path -
mu is a pure per-document function of the embedder). Fitting on the
full corpus instead of a train-only split therefore cannot leak test
information into mu the way it could for a model whose representation
actually depends on the training set's composition (e.g. anything
with alpha>0, or a real BoW-driven topic distribution) - this
optimization is specific to this exact locked configuration and is
NOT applied to (and would NOT be valid for) any other model in this
codebase's classification protocol.

Usage:
    python scripts/run_vaebm_classification_fast_slow_datasets.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SEEDS = [1, 2, 3, 4, 5]

DATASETS = [
    ("20ng", {"VAEBM_NORMALIZE_MU": "1", "VAEBM_EMBEDDER": "thenlper/gte-large"}),
    ("imdb", {"VAEBM_EMBEDDER": "BAAI/bge-large-en-v1.5"}),
]

BASE_ENV = {
    "VAEBM_UNITS": "1024",
    "VAEBM_DIM_EMB": "",
    "VAEBM_FREEZE_EMB": "1",
    "VAEBM_ALPHA": "0.0",
    "VAEBM_LR": "1e-4",
    "VAEBM_EPOCHS": "1",
    "TF_FORCE_GPU_ALLOW_GROWTH": "true",
}


def main() -> None:
    out_path = REPO_ROOT / "results" / "vaebm_classif_fast_slow_datasets.json"
    rows = []

    for dataset_id, overrides in DATASETS:
        env = dict(BASE_ENV)
        env.update(overrides)
        os.environ.update(env)

        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.svm import LinearSVC

        from vaebm_benchmark.datasets.simple_registry import load_dataset, resolve_dataset_id
        from vaebm_benchmark.experiment.scientific_models import build_model
        from vaebm_benchmark.utils.seeding import set_all_seeds
        from vaebm_benchmark.utils.gpu_memory import release_accelerator_memory

        resolved_id = resolve_dataset_id(dataset_id)
        documents, labels, num_classes = load_dataset(resolved_id)
        print(f"[{dataset_id}] {len(documents)} docs, {num_classes} classes", flush=True)

        set_all_seeds(42)
        t0 = time.perf_counter()
        model = build_model("vaebm", num_classes, 42, 5000, dataset_id=resolved_id)
        model.fit(documents)  # full corpus - see module docstring for why this is valid here
        mu = model.get_document_embeddings(documents)
        fit_runtime = time.perf_counter() - t0
        print(f"[{dataset_id}] fit+embed done in {fit_runtime:.1f}s, mu.shape={getattr(mu, 'shape', None)}", flush=True)
        del model
        release_accelerator_memory()

        for seed in SEEDS:
            t0 = time.perf_counter()
            set_all_seeds(seed)
            stratified = True
            try:
                mu_train, mu_test, train_labels, test_labels = train_test_split(
                    mu, labels, test_size=0.2, random_state=seed, stratify=labels,
                )
            except ValueError as split_exc:
                if "least populated class" not in str(split_exc):
                    raise
                stratified = False
                mu_train, mu_test, train_labels, test_labels = train_test_split(
                    mu, labels, test_size=0.2, random_state=seed,
                )

            clf = LinearSVC(C=1.0, random_state=seed, max_iter=10000, dual="auto")
            clf.fit(mu_train, train_labels)
            preds = clf.predict(mu_test)
            accuracy = float(accuracy_score(test_labels, preds))
            f1 = float(f1_score(test_labels, preds, average="macro"))
            runtime = time.perf_counter() - t0

            row = {
                "experiment": "classification", "model": "vaebm", "dataset": dataset_id,
                "k": num_classes, "seed": seed, "accuracy": accuracy, "f1": f1,
                "representation_source": "mu", "num_train_docs": len(mu_train),
                "num_test_docs": len(mu_test), "runtime_seconds": runtime, "status": "ok",
                "error": "", "split_stratified": stratified,
            }
            rows.append(row)
            print(f"[{dataset_id}] seed={seed}: acc={accuracy:.4f} f1={f1:.4f} ({runtime:.1f}s)", flush=True)

        out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"Wrote {len(rows)} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
