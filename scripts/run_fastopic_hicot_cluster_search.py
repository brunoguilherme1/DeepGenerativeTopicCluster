#!/usr/bin/env python
"""Autonomous hyperparameter search for the DOCUMENT CLUSTERING task,
FASTopic and HiCOT only, on the project's own established 12-dataset
cluster set (the same list scripts/run_baseline_sweep_12ds.py uses).

Replicates experiment/cluster_runner.py::run_single's own body (dataset
load -> K=num_classes -> fit (labels never passed) -> argmax_theta hard
assignment -> compute_clustering_metrics/compute_geometry_metrics), but
with the model built from a SEARCHED hyperparameter config each trial
instead of cluster_runner's fixed CLUSTER_MODEL_BUILDERS defaults.

Model selection is via internal_rank = mean(rank_silhouette desc,
rank_davies_bouldin asc, rank_calinski_harabasz desc) - labels/ACC/NMI/etc
are computed for every trial but NEVER read until after the winner is
already chosen. Per (model, dataset) pair: 60-minute wall-clock budget,
broad random draws first then single-dimension neighbor refinement
around the current best, skipping already-tried configs. HiCOT gets its
own internal max_fit_seconds cap (cooperative, checked every epoch);
FASTopic has no such hook, so a SIGALRM-based external timeout also
guards every trial for both models (fires between epochs, caught as
an ordinary exception - never corrupts run.log/trials state, matches
this project's normal one-failure-doesn't-abort-the-sweep pattern).

Usage (one process per physical GPU):
    python scripts/run_fastopic_hicot_cluster_search.py --gpu 0 --pairs fastopic:20ng hicot:20ng fastopic:agnews_short --run-dir results/cluster_hparam_fastopic_hicot_20260925 --suffix gpu0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path

_parser_gpu = argparse.ArgumentParser(add_help=False)
_parser_gpu.add_argument("--gpu", type=int, default=None)
_gpu_args, _ = _parser_gpu.parse_known_args()
if _gpu_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_args.gpu)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SEED = 42
VOC_SIZE = 5000
TIME_BUDGET_S = 60 * 60
GRACE_S = 120  # small grace period to let an in-flight trial finish
N_BROAD = 5  # initial random draws before switching to neighbor refinement

ALL_DATASETS = [
    "20ng", "agnews_short", "google_news_t", "imdb", "search_snippets",
    "bbc_news", "tweet", "stack_overflow", "biomedical", "banking77",
    "m10", "pascal_flickr",
]
ALL_MODELS = ["fastopic", "hicot"]

LABEL_METRIC_IDS = ["acc", "nmi", "ari", "ami", "homogeneity", "completeness", "v_measure", "purity"]
GEOMETRY_METRIC_IDS = ["silhouette", "davies_bouldin", "calinski_harabasz"]
ALL_METRIC_IDS = LABEL_METRIC_IDS + GEOMETRY_METRIC_IDS

# Search spaces - built from each adapter's own documented/paper-default
# hyperparameters (see fastopic_adapter.py/hicot_adapter.py docstrings),
# not arbitrary values. Deliberately NOT a full grid (108/729 combos) -
# a bounded time-aware search draws from these lists instead.
FASTOPIC_SPACE = {
    "DT_alpha": [1.0, 3.0, 6.0],  # paper default 3.0
    "TW_alpha": [1.0, 2.0, 4.0],  # paper default 2.0
    "learning_rate": [0.001, 0.002, 0.005],  # paper default 0.002
    "epochs": [100, 200],  # paper uses 200; cluster_runner's own default (20) was a smoke value, not searched
    "normalize_embeddings": [False, True],
}
HICOT_SPACE = {
    "lr": [0.001, 0.002, 0.005],  # adapter default 0.002
    "weight_loss_ECR": [20.0, 40.0, 80.0],  # adapter default 40.0
    "weight_loss_DT": [125.0, 250.0, 500.0],  # adapter default 250.0
    "en_units": [100, 200, 300],  # adapter default 200
    "epochs": [50, 100, 200],  # adapter default 500 (paper); cluster_runner's own default (50) kept as a grid point
    "threshold_epoch": [5, 10, 20],  # adapter default 10 - paper's clustering-activation epoch M
}
SEARCH_SPACES = {"fastopic": FASTOPIC_SPACE, "hicot": HICOT_SPACE}


class TrialTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise TrialTimeout("trial exceeded its per-trial wall-clock guard")


def build_model(model_name: str, k: int, config: dict, per_trial_budget_s: float):
    if model_name == "fastopic":
        from vaebm_benchmark.models.fastopic_adapter import FASTopicAdapter

        return FASTopicAdapter(num_topics=k, vocab_size_cap=VOC_SIZE, device="cuda", **config)
    elif model_name == "hicot":
        from vaebm_benchmark.models.hicot_adapter import HiCOTAdapter

        return HiCOTAdapter(
            n_clusters=k, voc_size=VOC_SIZE, random_state=SEED, device="cuda",
            max_fit_seconds=max(30.0, per_trial_budget_s - 15.0),  # cooperative safety net, belt-and-suspenders with SIGALRM
            **config,
        )
    raise KeyError(model_name)


def run_trial(model_name: str, dataset_id: str, config: dict, per_trial_budget_s: float) -> dict:
    """Mirrors experiment/cluster_runner.py::run_single's own body (same
    imports/functions), with the model built from `config` instead of
    CLUSTER_MODEL_BUILDERS's fixed defaults. Never passes labels to fit()."""
    import numpy as np

    from vaebm_benchmark.datasets.simple_registry import load_dataset, resolve_dataset_id
    from vaebm_benchmark.experiment.scientific_models import assignment_source_for_model, representation_source_for_model
    from vaebm_benchmark.metrics.clustering_quality import compute_clustering_metrics, compute_geometry_metrics
    from vaebm_benchmark.utils.gpu_memory import release_accelerator_memory
    from vaebm_benchmark.utils.seeding import set_all_seeds

    resolved = resolve_dataset_id(dataset_id)
    start = time.perf_counter()
    model = None
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(int(per_trial_budget_s))
    try:
        set_all_seeds(SEED)
        documents, labels, num_classes = load_dataset(resolved)
        requested_k = num_classes  # labels inspected ONLY for K, never passed to fit()

        if model_name == "hicot":
            from vaebm_benchmark.utils.text_preprocessing import remove_english_stopwords

            fit_documents = remove_english_stopwords(documents)
            preprocessing_source = "stopwords_removed_en"
        else:
            fit_documents = documents
            preprocessing_source = "none"

        model = build_model(model_name, requested_k, config, per_trial_budget_s)
        model.fit(fit_documents)  # labels never passed
        training_epochs_completed = getattr(model, "epochs_completed", None)

        representation_source = representation_source_for_model(model_name)
        assignment_source = assignment_source_for_model(model_name)
        assert assignment_source == "argmax_theta", f"unexpected assignment_source for {model_name}"

        feature_space = model.get_document_topics(fit_documents)
        clusters = [int(i) for i in np.argmax(np.asarray(feature_space), axis=1)]
        actual_k = len(set(clusters))

        label_metrics = compute_clustering_metrics(clusters, labels, LABEL_METRIC_IDS)
        try:
            if actual_k < 2:
                raise ValueError("degenerate single-cluster output")
            geometry_metrics = compute_geometry_metrics(feature_space, clusters, GEOMETRY_METRIC_IDS)
        except Exception:
            geometry_metrics = {name: None for name in GEOMETRY_METRIC_IDS}

        runtime = time.perf_counter() - start
        return {
            "model": model_name, "dataset": dataset_id, "k": requested_k, "seed": SEED,
            "config": config, "runtime_seconds": round(runtime, 1),
            "representation_source": representation_source, "assignment_source": assignment_source,
            "preprocessing_source": preprocessing_source,
            "training_epochs_completed": training_epochs_completed,
            "actual_k": actual_k, "num_classes": num_classes,
            "status": "ok", "error": None,
            **label_metrics, **geometry_metrics,
        }
    except Exception as exc:  # noqa: BLE001 - one bad trial must not abort the search
        runtime = time.perf_counter() - start
        return {
            "model": model_name, "dataset": dataset_id, "k": None, "seed": SEED,
            "config": config, "runtime_seconds": round(runtime, 1),
            "representation_source": "", "assignment_source": "", "preprocessing_source": "",
            "training_epochs_completed": None, "actual_k": None, "num_classes": None,
            "status": "error", "error": f"{exc}\n{traceback.format_exc(limit=3)}",
            **{name: None for name in ALL_METRIC_IDS},
        }
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        try:
            del model
        except NameError:
            pass
        release_accelerator_memory()


def compute_internal_ranks(trials: list[dict]) -> None:
    """In-place: sets trial['internal_rank'] for every trial with a
    computable Silhouette (i.e. status=ok and actual_k>=2). Ties broken
    later by Silhouette descending, per spec."""
    valid = [t for t in trials if t["status"] == "ok" and t.get("silhouette") is not None]
    if not valid:
        return
    sil_order = sorted(valid, key=lambda t: -t["silhouette"])
    db_order = sorted(valid, key=lambda t: t["davies_bouldin"])
    ch_order = sorted(valid, key=lambda t: -t["calinski_harabasz"])
    rank_sil = {id(t): i + 1 for i, t in enumerate(sil_order)}
    rank_db = {id(t): i + 1 for i, t in enumerate(db_order)}
    rank_ch = {id(t): i + 1 for i, t in enumerate(ch_order)}
    for t in valid:
        t["internal_rank"] = (rank_sil[id(t)] + rank_db[id(t)] + rank_ch[id(t)]) / 3.0


def pick_best(trials: list[dict]) -> dict | None:
    ranked = [t for t in trials if t.get("internal_rank") is not None]
    if not ranked:
        return None
    ranked.sort(key=lambda t: (t["internal_rank"], -t["silhouette"]))
    return ranked[0]


def sample_config(space: dict, tried: set[tuple], rng: random.Random) -> dict:
    for _ in range(200):
        cfg = {k: rng.choice(v) for k, v in space.items()}
        key = tuple(sorted(cfg.items()))
        if key not in tried:
            return cfg
    return {k: rng.choice(v) for k, v in space.items()}  # give up dedup-ing, space likely exhausted


def sample_neighbor(space: dict, base: dict, tried: set[tuple], rng: random.Random) -> dict:
    dims = list(space.keys())
    rng.shuffle(dims)
    for dim in dims:
        values = space[dim]
        cur_idx = values.index(base[dim]) if base[dim] in values else 0
        candidates = [i for i in (cur_idx - 1, cur_idx + 1) if 0 <= i < len(values)]
        rng.shuffle(candidates)
        for idx in candidates:
            cfg = dict(base)
            cfg[dim] = values[idx]
            key = tuple(sorted(cfg.items()))
            if key not in tried:
                return cfg
    return sample_config(space, tried, rng)  # every neighbor already tried - fall back to a fresh draw


class RunLogger:
    def __init__(self, run_dir: Path, suffix: str):
        self.run_dir = run_dir
        self.suffix = suffix
        self.log_path = run_dir / "run.log"
        self.progress_path = run_dir / f"progress_{suffix}.txt"
        self.status_path = run_dir / f"current_status_{suffix}.txt"
        self.trials_csv = run_dir / f"trials_{suffix}.csv"
        self.trials_json = run_dir / f"trials_{suffix}.json"
        self.best_configs_json = run_dir / f"best_configs_{suffix}.json"
        self.checkpoint_json = run_dir / f"checkpoint_{suffix}.json"
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        self._trials_json_state: list[dict] = []
        self._best_configs_state: dict = {}

    def log(self, line: str) -> None:
        with open(self.log_path, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def status(self, text: str) -> None:
        self.status_path.write_text(text)

    def progress(self, text: str) -> None:
        self.progress_path.write_text(text)

    def append_trial(self, trial: dict, trial_idx: int) -> None:
        fieldnames = [
            "model", "dataset", "k", "seed", "config", "runtime_seconds",
            "acc", "nmi", "ari", "ami", "homogeneity", "completeness", "v_measure", "purity",
            "silhouette", "davies_bouldin", "calinski_harabasz", "internal_rank",
            "representation_source", "assignment_source", "preprocessing_source",
            "training_epochs_completed", "actual_k", "num_classes", "status", "error",
        ]
        is_new = not self.trials_csv.exists()
        row = {k: (json.dumps(v) if k == "config" else v) for k, v in trial.items() if k in fieldnames}
        for k in fieldnames:
            row.setdefault(k, None)
        with open(self.trials_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if is_new:
                w.writeheader()
            w.writerow(row)

        self._trials_json_state.append(trial)
        self.trials_json.write_text(json.dumps(self._trials_json_state, indent=2, default=str))

        with open(self.run_dir / "logs" / f"trial_{self.suffix}_{trial_idx:04d}.log", "w") as f:
            f.write(json.dumps(trial, indent=2, default=str))

    def append_best_config(self, model: str, dataset: str, record: dict) -> None:
        self._best_configs_state[f"{model}:{dataset}"] = record
        self.best_configs_json.write_text(json.dumps(self._best_configs_state, indent=2, default=str))

    def checkpoint(self, done_pairs: list[str]) -> None:
        self.checkpoint_json.write_text(json.dumps({"done_pairs": done_pairs, "ts": time.time()}, indent=2))


def search_one_pair(model_name: str, dataset_id: str, logger: RunLogger, pair_idx: int, total_pairs: int) -> dict:
    space = SEARCH_SPACES[model_name]
    rng = random.Random(hash((model_name, dataset_id)) & 0xFFFFFFFF)
    tried: set[tuple] = set()
    trials: list[dict] = []
    deadline = time.perf_counter() + TIME_BUDGET_S
    trial_idx = 0
    best_so_far = None

    while True:
        now = time.perf_counter()
        remaining = deadline - now
        elapsed_m = (TIME_BUDGET_S - remaining) / 60.0
        if remaining <= 0:
            logger.log(f"TIME_LIMIT model={model_name} dataset={dataset_id} elapsed={elapsed_m:.0f}m")
            break
        # Only start a new trial if there's more than a small grace window left,
        # OR this is the very first trial for this pair (always allowed once).
        if remaining < GRACE_S and trial_idx > 0:
            logger.log(f"TIME_LIMIT model={model_name} dataset={dataset_id} elapsed={elapsed_m:.0f}m")
            break

        if trial_idx < N_BROAD or best_so_far is None:
            config = sample_config(space, tried, rng)
        else:
            config = sample_neighbor(space, best_so_far["config"], tried, rng)
        tried.add(tuple(sorted(config.items())))

        per_trial_budget = max(60.0, min(remaining, TIME_BUDGET_S / max(3, N_BROAD)))
        logger.status(
            f"pair {pair_idx}/{total_pairs} model={model_name} dataset={dataset_id} "
            f"trial={trial_idx} elapsed={elapsed_m:.0f}m/60m config={config}"
        )
        logger.log(f"START model={model_name} dataset={dataset_id} trial={trial_idx} elapsed={elapsed_m:.0f}m/60m")

        trial = run_trial(model_name, dataset_id, config, per_trial_budget)
        # Retry transient failures (OOM/network) up to 2 times; deterministic
        # failures (e.g. bad config -> immediate exception on first attempt
        # with near-zero runtime AND a config-shaped error) are recorded once.
        attempts = 1
        while (
            trial["status"] == "error"
            and attempts < 3
            and any(s in (trial["error"] or "") for s in ("CUDA out of memory", "Connection", "HTTPError", "Timeout"))
        ):
            attempts += 1
            trial = run_trial(model_name, dataset_id, config, per_trial_budget)

        trials.append(trial)
        compute_internal_ranks(trials)
        logger.append_trial(trial, trial_idx)

        if trial["status"] == "ok" and trial.get("silhouette") is not None:
            logger.log(
                f"OK model={model_name} dataset={dataset_id} trial={trial_idx} "
                f"silhouette={trial['silhouette']:.4f} db={trial['davies_bouldin']:.4f} "
                f"ch={trial['calinski_harabasz']:.1f}"
            )
        else:
            logger.log(f"ERROR model={model_name} dataset={dataset_id} trial={trial_idx} error={trial['error']}")

        candidate_best = pick_best(trials)
        if candidate_best is not None and (best_so_far is None or candidate_best is trial):
            if best_so_far is None or candidate_best["internal_rank"] < best_so_far["internal_rank"]:
                best_so_far = candidate_best
                logger.log(
                    f"BEST model={model_name} dataset={dataset_id} trial={trial_idx} "
                    f"internal_rank={best_so_far['internal_rank']:.2f}"
                )
        trial_idx += 1

    if best_so_far is None:
        logger.log(f"FINAL model={model_name} dataset={dataset_id} best_trial=NONE (no valid trial)")
        return {"model": model_name, "dataset": dataset_id, "best_config": None, "trials": trials}

    logger.log(f"FINAL model={model_name} dataset={dataset_id} best_trial={trials.index(best_so_far)}")
    logger.append_best_config(model_name, dataset_id, {
        "config": best_so_far["config"], "trials_completed": len(trials),
        "search_runtime_s": round(sum(t["runtime_seconds"] for t in trials), 1),
        "selection_silhouette": best_so_far["silhouette"], "selection_db": best_so_far["davies_bouldin"],
        "selection_ch": best_so_far["calinski_harabasz"], "internal_rank": best_so_far["internal_rank"],
    })
    return {"model": model_name, "dataset": dataset_id, "best_config": best_so_far["config"], "trials": trials}


def main():
    p = argparse.ArgumentParser(parents=[_parser_gpu])
    p.add_argument("--pairs", nargs="+", required=True, help="model:dataset pairs, e.g. fastopic:20ng hicot:20ng")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--suffix", required=True)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(run_dir, args.suffix)

    pairs = []
    for spec in args.pairs:
        model_name, dataset_id = spec.split(":", 1)
        assert model_name in ALL_MODELS, model_name
        assert dataset_id in ALL_DATASETS, dataset_id
        pairs.append((model_name, dataset_id))

    logger.log(f"=== search start suffix={args.suffix} gpu={_gpu_args.gpu} pairs={pairs} ===")

    final_rows = []
    done_pairs = []
    for i, (model_name, dataset_id) in enumerate(pairs):
        result = search_one_pair(model_name, dataset_id, logger, i + 1, len(pairs))
        done_pairs.append(f"{model_name}:{dataset_id}")
        logger.checkpoint(done_pairs)

        if result["best_config"] is None:
            final_rows.append({
                "dataset": dataset_id, "model": model_name, "k": None, "status": "no_valid_trial",
                **{m: None for m in ALL_METRIC_IDS},
            })
            continue

        # Clean re-run of the selected best config at seed=42 (canonical final result).
        logger.log(f"RERUN model={model_name} dataset={dataset_id} config={result['best_config']} seed=42")
        final_trial = run_trial(model_name, dataset_id, result["best_config"], TIME_BUDGET_S)
        final_rows.append({
            "dataset": dataset_id, "model": model_name, "k": final_trial["k"], "status": final_trial["status"],
            "config": final_trial["config"],
            **{m: final_trial.get(m) for m in ALL_METRIC_IDS},
        })

    final_csv = run_dir / f"final_results_{args.suffix}.csv"
    final_json = run_dir / f"final_results_{args.suffix}.json"
    fieldnames = ["dataset", "model", "k", "status", "config"] + ALL_METRIC_IDS
    with open(final_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in final_rows:
            row = dict(row)
            row["config"] = json.dumps(row.get("config"))
            for k in fieldnames:
                row.setdefault(k, None)
            w.writerow(row)
    final_json.write_text(json.dumps(final_rows, indent=2, default=str))
    logger.log(f"=== search done suffix={args.suffix}, final -> {final_csv} ===")


if __name__ == "__main__":
    main()
