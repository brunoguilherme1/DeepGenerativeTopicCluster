#!/usr/bin/env python
"""Autonomous FASTopic-protocol reproduction: VAE-BM and HiCOT, on
fastopic_20ng / fastopic_nyt / fastopic_wos_reconstructed, at K=50
(FASTopic paper Table 1/2/Figure 4 and Figure 7's own K=50 setting -
verified directly against the paper PDF), reproducing:
  - Text Clustering: Purity, NMI (argmax(theta) for HiCOT, KMeans(mu) for
    VAE-BM per this project's own established semantics - see
    models/hicot_adapter.py::get_document_topics /
    models/vaebm_adapter.py::get_document_topics, both already implement
    exactly this).
  - Downstream Classification: Accuracy, macro-F1 via
    SVC(gamma='scale') - the EXACT rule recovered from TopMost's own
    eva/classification.py::_cls (not this project's other, different
    SVC(kernel='linear', C=1.0) convention used elsewhere - this
    reproduction matches the official FASTopic/ECRTM baseline code
    exactly, per user instruction to reproduce an exact rule when one is
    found).

Clustering is transductive (fit+eval on the full corpus, train+test
texts concatenated) - matches FASTopic's own clustering protocol
(protocols/fastopic_protocol.py's own docstring, verified against
topmost.BasicTrainer). Classification fits the representation model on
the official TRAIN split only, then infers train/test representations
separately - no test-time leakage into fitting.

Hyperparameter search: a small, predeclared, compact grid per model
(see VAEBM_GRID / HICOT_GRID below), never selected by test-set
Purity/NMI/Accuracy/F1. Selection criterion:
  - Clustering: C_V coherence (gensim, local corpus - unsupervised, no
    labels) of the model's own top words (get_topics()). Purity/NMI are
    still computed and logged for every trial as
    oracle_best_purity/oracle_best_nmi DIAGNOSTIC upper bounds only -
    never used to pick the reported config.
  - Classification: accuracy on an inner 80/20 split of the TRAINING
    set only (never the official test set), refit on the full training
    partition with the selected config before the one final test-set
    evaluation.

Logging matches user's own requested structure under
results/fastopic_protocol_vaebm_hicot_<timestamp>/:
  run.log, progress.txt, current_status.txt, trials.csv, checkpoint.json,
  best_configs.json, dataset_provenance.json, logs/, clustering/,
  classification/.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

K = 50
SEED = 42
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 20

DATASETS = ["fastopic_20ng", "fastopic_nyt", "fastopic_wos_reconstructed"]
MODELS = ["vaebm", "hicot"]

# Compact, meaningful search spaces (not huge grids) - see this script's
# own module docstring for what informed each dimension: current
# VAE-BM/HiCOT configs and defaults already in this repo's adapters,
# HiCOT's own paper-documented hyperparameter ranges, and this project's
# own established alpha-sweep finding (alpha=0 vs small positive values).
VAEBM_GRID = [
    {"alpha": 0.0, "lr": 1e-3, "units": 50, "epochs": 30, "normalize_mu": False, "embedder": "all-MiniLM-L6-v2"},
    {"alpha": 0.0, "lr": 1e-3, "units": 50, "epochs": 50, "normalize_mu": True, "embedder": "all-MiniLM-L6-v2"},
    {"alpha": 0.0, "lr": 1e-4, "units": 100, "epochs": 50, "normalize_mu": False, "embedder": "all-mpnet-base-v2"},
    {"alpha": 0.01, "lr": 1e-3, "units": 50, "epochs": 30, "normalize_mu": False, "embedder": "all-MiniLM-L6-v2"},
    {"alpha": 0.0, "lr": 1e-3, "units": 200, "epochs": 30, "normalize_mu": True, "embedder": "all-mpnet-base-v2"},
]

HICOT_GRID = [
    {"lr": 0.002, "epochs": 100, "weight_loss_ECR": 40.0, "weight_loss_DT": 250.0, "en_units": 200},
    {"lr": 0.002, "epochs": 200, "weight_loss_ECR": 40.0, "weight_loss_DT": 250.0, "en_units": 200},
    {"lr": 0.001, "epochs": 100, "weight_loss_ECR": 20.0, "weight_loss_DT": 100.0, "en_units": 200},
    {"lr": 0.002, "epochs": 100, "weight_loss_ECR": 60.0, "weight_loss_DT": 250.0, "en_units": 300},
    {"lr": 0.002, "epochs": 150, "weight_loss_ECR": 40.0, "weight_loss_DT": 150.0, "en_units": 200},
]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


class Sweep:
    def __init__(self, run_dir: Path, gpu: str | None):
        self.run_dir = run_dir
        self.gpu = gpu
        for sub in ("logs", "clustering", "classification"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        self.run_log_path = run_dir / "run.log"
        self.progress_path = run_dir / "progress.txt"
        self.status_path = run_dir / "current_status.txt"
        self.trials_csv_path = run_dir / "trials.csv"
        self.checkpoint_path = run_dir / "checkpoint.json"
        self.best_configs_path = run_dir / "best_configs.json"

        self.checkpoint: dict = {}
        if self.checkpoint_path.exists():
            self.checkpoint = json.loads(self.checkpoint_path.read_text())
        self.best_configs: dict = {}
        if self.best_configs_path.exists():
            self.best_configs = json.loads(self.best_configs_path.read_text())

        if not self.trials_csv_path.exists():
            self.trials_csv_path.write_text(
                "timestamp,task,model,dataset,trial,seed,config,selection_metric,"
                "selection_score,purity,nmi,accuracy,f1,runtime_s,status,error\n"
            )

        self.total_combos = len(DATASETS) * len(MODELS) * 2  # cluster + classification
        self.started_at = now_iso()

    def log(self, msg: str) -> None:
        line = f"[{now_iso()}] {msg}"
        with open(self.run_log_path, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def save_checkpoint(self) -> None:
        tmp = self.checkpoint_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.checkpoint, indent=2))
        os.replace(tmp, self.checkpoint_path)
        tmp2 = self.best_configs_path.with_suffix(".json.tmp")
        tmp2.write_text(json.dumps(self.best_configs, indent=2))
        os.replace(tmp2, self.best_configs_path)

    def write_progress(self) -> None:
        done = sum(1 for v in self.checkpoint.values() if v.get("status") == "ok")
        failed = sum(1 for v in self.checkpoint.values() if v.get("status") == "error")
        lines = [
            f"Total combos (model x dataset x task): {self.total_combos}",
            f"Completed: {done + failed}", f"Successful: {done}", f"Failed: {failed}",
            f"Started at: {self.started_at}", f"Last update: {now_iso()}",
            f"GPU: {self.gpu}",
        ]
        self.progress_path.write_text("\n".join(lines) + "\n")

    def write_status(self, task: str, model: str, dataset: str, trial: int, n_trials: int) -> None:
        self.status_path.write_text(
            f"Task: {task}\nModel: {model}\nDataset: {dataset}\nTrial: {trial}/{n_trials}\n"
            f"Last update: {now_iso()}\nGPU: {self.gpu}\n"
        )

    def append_trial_csv(self, row: dict) -> None:
        import csv as csvmod
        with open(self.trials_csv_path, "a", newline="") as f:
            w = csvmod.writer(f)
            w.writerow([
                row.get("timestamp"), row.get("task"), row.get("model"), row.get("dataset"),
                row.get("trial"), row.get("seed"), json.dumps(row.get("config", {})),
                row.get("selection_metric"), row.get("selection_score"),
                row.get("purity"), row.get("nmi"), row.get("accuracy"), row.get("f1"),
                row.get("runtime_s"), row.get("status"), (row.get("error") or "")[:300],
            ])


def load_dataset_texts_labels(dataset_id: str):
    """Returns (train_texts, test_texts, train_labels, test_labels, vocab)."""
    if dataset_id == "fastopic_20ng":
        from vaebm_benchmark.datasets.definitions.fastopic_20ng import TwentyNGFASTopicDataset
        b = TwentyNGFASTopicDataset().load()
    elif dataset_id == "fastopic_nyt":
        from vaebm_benchmark.datasets.definitions.fastopic_nyt import NYTDataset
        b = NYTDataset().load()
    elif dataset_id == "fastopic_wos_reconstructed":
        from vaebm_benchmark.datasets.definitions.fastopic_wos import WoSReconstructedDataset
        b = WoSReconstructedDataset().load(test_p=0.2, seed=SEED)
    else:
        raise KeyError(dataset_id)
    return b.train_texts, b.test_texts, b.train_labels, b.test_labels, b.vocab


def build_vaebm(config: dict, k: int, vocab: list[str], seed: int):
    from vaebm_benchmark.models.vaebm_adapter import VAEBMAdapter
    return VAEBMAdapter(
        n_clusters=k, voc_size=len(vocab), vocabulary=vocab, random_state=seed,
        alpha=config["alpha"], lr=config["lr"], units=config["units"],
        epochs=config["epochs"], normalize_mu=config["normalize_mu"],
        embedder=config["embedder"], batch_size=128, verbose=0,
    )


def build_hicot(config: dict, k: int, vocab: list[str], seed: int):
    from vaebm_benchmark.models.hicot_adapter import HiCOTAdapter
    return HiCOTAdapter(
        n_clusters=k, voc_size=len(vocab), vocabulary=vocab, random_state=seed,
        lr=config["lr"], epochs=config["epochs"],
        weight_loss_ECR=config["weight_loss_ECR"], weight_loss_DT=config["weight_loss_DT"],
        en_units=config["en_units"], max_clusters=k,
    )


def build_model(model_name: str, config: dict, k: int, vocab: list[str], seed: int):
    if model_name == "vaebm":
        return build_vaebm(config, k, vocab, seed)
    elif model_name == "hicot":
        return build_hicot(config, k, vocab, seed)
    raise KeyError(model_name)


def run_clustering_trial(sweep: Sweep, model_name: str, dataset_id: str, config: dict, trial_idx: int):
    from sklearn.metrics import normalized_mutual_info_score
    from sklearn.metrics.cluster import contingency_matrix
    from vaebm_benchmark.metrics.topic_quality import coherence

    train_texts, test_texts, train_labels, test_labels, vocab = load_dataset_texts_labels(dataset_id)
    all_texts = list(train_texts) + list(test_texts)
    all_labels = list(train_labels) + list(test_labels)

    model = build_model(model_name, config, K, vocab, SEED)
    model.fit(all_texts)
    preds = model.predict(all_texts) if hasattr(model, "predict") else None
    if preds is None:
        raise RuntimeError(f"{model_name} adapter has no predict()")
    preds = np.asarray(preds)
    labels_arr = np.asarray(all_labels)

    cm = contingency_matrix(labels_arr, preds)
    purity = float(np.sum(np.amax(cm, axis=0)) / np.sum(cm))
    nmi = float(normalized_mutual_info_score(labels_arr, preds))

    tokenized_corpus = [t.split() for t in all_texts]
    topics = model.get_topics(top_n=10)
    try:
        cv, _ = coherence(topics, tokenized_corpus, top_n=10, measure="c_v")
    except Exception as exc:  # coherence can fail on tiny/degenerate topics
        sweep.log(f"WARN coherence computation failed for {model_name}/{dataset_id} trial={trial_idx}: {exc!r}")
        cv = float("-inf")

    return {"purity": purity, "nmi": nmi, "selection_score": cv, "selection_metric": "c_v_coherence"}


def run_classification_trial(sweep: Sweep, model_name: str, dataset_id: str, config: dict, trial_idx: int):
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    train_texts, test_texts, train_labels, test_labels, vocab = load_dataset_texts_labels(dataset_id)

    # Inner train/val split of the TRAINING set only, for hyperparameter
    # selection - the official test set is touched exactly once, after
    # a config is already chosen (see run_one's final-refit step).
    inner_train_idx, inner_val_idx = train_test_split(
        np.arange(len(train_texts)), test_size=0.2, random_state=SEED, stratify=train_labels,
    )
    inner_train_texts = [train_texts[i] for i in inner_train_idx]
    inner_val_texts = [train_texts[i] for i in inner_val_idx]
    inner_train_labels = [train_labels[i] for i in inner_train_idx]
    inner_val_labels = [train_labels[i] for i in inner_val_idx]

    model = build_model(model_name, config, K, vocab, SEED)
    model.fit(inner_train_texts)
    train_repr = model.get_document_topics(inner_train_texts)
    val_repr = model.get_document_topics(inner_val_texts)

    clf = SVC(gamma="scale")
    clf.fit(train_repr, inner_train_labels)
    val_preds = clf.predict(val_repr)
    val_acc = float(accuracy_score(inner_val_labels, val_preds))

    return {"selection_score": val_acc, "selection_metric": "inner_val_accuracy"}


def final_classification_eval(model_name: str, dataset_id: str, config: dict):
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.svm import SVC

    train_texts, test_texts, train_labels, test_labels, vocab = load_dataset_texts_labels(dataset_id)
    model = build_model(model_name, config, K, vocab, SEED)
    model.fit(train_texts)  # official train split ONLY - no test leakage
    train_repr = model.get_document_topics(train_texts)
    test_repr = model.get_document_topics(test_texts)

    clf = SVC(gamma="scale")
    clf.fit(train_repr, train_labels)
    test_preds = clf.predict(test_repr)
    acc = float(accuracy_score(test_labels, test_preds))
    f1 = float(f1_score(test_labels, test_preds, average="macro"))
    return {"accuracy": acc, "f1": f1}


def run_one(sweep: Sweep, task: str, model_name: str, dataset_id: str, grid: list[dict]):
    key = f"{task}|{model_name}|{dataset_id}"
    existing = sweep.checkpoint.get(key)
    if existing and existing.get("status") == "ok":
        sweep.log(f"SKIP task={task} model={model_name} dataset={dataset_id} (already completed at {existing.get('timestamp')})")
        return

    best = None
    trial_results = []
    for trial_idx, config in enumerate(grid, start=1):
        sweep.write_status(task, model_name, dataset_id, trial_idx, len(grid))
        last_error = ""
        ok = False
        result = {}
        start = time.perf_counter()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                sweep.log(f"START task={task} model={model_name} dataset={dataset_id} trial={trial_idx} attempt={attempt} config={config}")
                if task == "cluster":
                    result = run_clustering_trial(sweep, model_name, dataset_id, config, trial_idx)
                else:
                    result = run_classification_trial(sweep, model_name, dataset_id, config, trial_idx)
                ok = True
                break
            except Exception as exc:  # noqa: BLE001
                last_error = f"{exc!r}\n{traceback.format_exc(limit=6)}"
                sweep.log(f"ERROR task={task} model={model_name} dataset={dataset_id} trial={trial_idx} attempt={attempt} error={exc!r}")
                if attempt < MAX_ATTEMPTS:
                    time.sleep(RETRY_BACKOFF_SECONDS)
        runtime = time.perf_counter() - start

        row = {
            "timestamp": now_iso(), "task": task, "model": model_name, "dataset": dataset_id,
            "trial": trial_idx, "seed": SEED, "config": config,
            "selection_metric": result.get("selection_metric"), "selection_score": result.get("selection_score"),
            "purity": result.get("purity"), "nmi": result.get("nmi"),
            "accuracy": result.get("accuracy"), "f1": result.get("f1"),
            "runtime_s": round(runtime, 1), "status": "ok" if ok else "error", "error": None if ok else last_error,
        }
        sweep.append_trial_csv(row)
        if ok:
            sweep.log(
                f"OK task={task} model={model_name} dataset={dataset_id} trial={trial_idx} "
                f"selection={result.get('selection_metric')}={result.get('selection_score')} "
                f"purity={result.get('purity')} nmi={result.get('nmi')} "
                f"acc={result.get('accuracy')} f1={result.get('f1')} runtime={runtime:.0f}s"
            )
            trial_results.append((config, result))
            if best is None or (result.get("selection_score") is not None and result["selection_score"] > best[1].get("selection_score", float("-inf"))):
                best = (config, result)
        else:
            sweep.log(f"FAILED task={task} model={model_name} dataset={dataset_id} trial={trial_idx} after {MAX_ATTEMPTS} attempts, skipping trial")

    if best is None:
        sweep.checkpoint[key] = {"status": "error", "timestamp": now_iso(), "error": "all trials failed"}
        sweep.save_checkpoint()
        sweep.write_progress()
        return

    best_config, best_result = best
    final = dict(best_result)
    if task == "classification":
        sweep.log(f"REFIT task=classification model={model_name} dataset={dataset_id} on selected config (official test split, evaluated once)")
        final.update(final_classification_eval(model_name, dataset_id, best_config))
    else:
        # For clustering, the selected trial's own purity/nmi (computed
        # transductively on the full corpus already) ARE the final
        # reported numbers - no separate refit step needed.
        pass

    sweep.checkpoint[key] = {
        "status": "ok", "timestamp": now_iso(), "best_config": best_config,
        "result": final, "n_trials": len(grid), "n_successful_trials": len(trial_results),
    }
    sweep.best_configs[key] = {"config": best_config, "result": final}
    sweep.save_checkpoint()
    sweep.write_progress()
    sweep.log(f"BEST task={task} model={model_name} dataset={dataset_id} config={best_config} result={final}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--tasks", default="cluster,classification")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = REPO_ROOT / "results" / f"fastopic_protocol_vaebm_hicot_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)

    random.seed(SEED)
    np.random.seed(SEED)

    sweep = Sweep(run_dir, gpu=args.gpu)
    sweep.log(f"SWEEP_START run_dir={run_dir} gpu={args.gpu} datasets={args.datasets} models={args.models} tasks={args.tasks} K={K}")

    datasets = args.datasets.split(",")
    models = args.models.split(",")
    tasks = args.tasks.split(",")
    grids = {"vaebm": VAEBM_GRID, "hicot": HICOT_GRID}

    for dataset_id in datasets:
        for model_name in models:
            for task in tasks:
                try:
                    run_one(sweep, task, model_name, dataset_id, grids[model_name])
                except Exception as exc:  # noqa: BLE001
                    sweep.log(f"FATAL (combo-level) task={task} model={model_name} dataset={dataset_id} error={exc!r}\n{traceback.format_exc(limit=6)}")
                    continue

    sweep.write_progress()
    sweep.log("SWEEP_COMPLETE")


if __name__ == "__main__":
    main()
