#!/usr/bin/env python
"""RQ3: does VAE-BM generalize across heterogeneous domains, text lengths,
and label granularities under one consistent clustering protocol?

Models: LDA, BERTopic, FASTopic, HiCOT, VAE-BM.
Datasets (generic simple_registry ids, NOT the HiCOT/FASTopic
protocol-pinned variants - confirmed via repo inspection that "20ng",
"bbc_news", "m10", "stack_overflow", "biomedical", "tweet", "banking77"
have no other name-colliding path except 20ng, which ALSO exists as
hicot_20ng/fastopic_20ng/20ng_s2wtm - none of those are used here):
    20ng, bbc_news, m10, stack_overflow, biomedical, tweet, banking77
K = number of ground-truth classes per dataset. Labels used ONLY for K and
final evaluation - never for hyperparameter selection.

LDA/BERTopic/VAE-BM: a complete, valid, single-seed=42, non-oracle 7-dataset
grid for these three already exists in paper_data/{labuai,futurelab_new}/*
(verified: checkpoint_selection="none" throughout, requested_k == num
classes, no PoE/DEC/oracle variants mixed in) - REUSED here verbatim, no
rerun, per "first inspect existing results and reuse valid completed runs."

FASTopic/HiCOT: the existing manifest rows used a single fixed config, not
a search - methodologically inconsistent with this RQ's new requirement to
run a small (4-8 config) unsupervised hyperparameter search first. These
two are therefore rerun here with a real search step, grounded in each
paper's own stated hyperparameters/ranges (see FASTOPIC_SEARCH_GRID/
HICOT_SEARCH_GRID docstrings below) and selected ONLY by unsupervised C_V
(gensim CoherenceModel, local corpus) - never by ACC/NMI/ARI/Purity/labels.
Each search config is fit ONCE and its full 11-metric suite computed in the
same pass (no separate "final" refit) - the winning config's own row IS
the reported RQ3 result, keeping search and reporting fully consistent.

Usage:
    python scripts/run_rq3.py --models fastopic hicot --datasets 20ng bbc_news m10 stack_overflow biomedical tweet banking77
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

import rq4_common as common  # reused for compute_full_metric_suite/get_doc_embeddings/normalize_topic_words

RESULTS_DIR = REPO_ROOT / "results" / "rq3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = RESULTS_DIR / "rq3_results.csv"
SEARCH_CSV = RESULTS_DIR / "rq3_hyperparameter_search.csv"
PROTOCOL_JSON = RESULTS_DIR / "rq3_protocol.json"
RUN_LOG = RESULTS_DIR / "rq3_run.log"

SEED = 42
VOC_SIZE = 5000
DATASETS = ["20ng", "bbc_news", "m10", "stack_overflow", "biomedical", "tweet", "banking77"]
REUSED_MODELS = ["lda", "bertopic", "vaebm"]
SEARCH_MODELS = ["fastopic", "hicot"]

# Grounded in arXiv:2405.17978 Appendix D (paper fixes epochs=200, lr=0.002
# for ALL experiments - no per-dataset tuning in the paper itself) and
# Appendix F (embedder ablation: all-mpnet-base-v2 improves clustering
# specifically). _build_fastopic's existing epochs=20 override is a
# compute-budget default, not a paper value - search brackets it against
# the paper's own 200-epoch setting (capped at 100 for RQ3 tractability,
# documented) and both paper-tested embedders.
FASTOPIC_SEARCH_GRID = [
    {"epochs": 20, "doc_embed_model": "all-MiniLM-L6-v2"},
    {"epochs": 20, "doc_embed_model": "all-mpnet-base-v2"},
    {"epochs": 50, "doc_embed_model": "all-MiniLM-L6-v2"},
    {"epochs": 50, "doc_embed_model": "all-mpnet-base-v2"},
    {"epochs": 100, "doc_embed_model": "all-MiniLM-L6-v2"},
    {"epochs": 100, "doc_embed_model": "all-mpnet-base-v2"},
]

# Grounded in Findings ACL 2025 (2025.findings-acl.715) Appendix G.1/G.2:
# paper searches weight_loss_ECR in [10,30,40,50,100,200] and weight_loss_DT
# in [0.5,0.7,1,2,5,10] (Table 15 sensitivity: DT=1 "good" on 20NG). The
# adapter's own upstream-argparse-derived defaults (ECR=40, DT=250) sit
# OUTSIDE the paper's own tested DT range, so the grid crosses the current
# adapter default against paper-recommended DT values, keeping weight_loss_
# CLT/CLC/threshold_epoch fixed at adapter defaults to keep the grid small
# (the paper's own clustering-update-interval I is not exposed by
# HiCOTAdapter at all and cannot be swept without a code change - documented
# deviation). HiCOT's max_fit_seconds is reduced to 300s (from the existing
# manifest runs' 1200s) uniformly for EVERY RQ3 HiCOT fit (search included),
# since the 6x search multiplier makes the original budget intractable here
# - a disclosed, uniformly-applied compute deviation, not a per-config choice.
HICOT_SEARCH_GRID = [
    {"weight_loss_ECR": 40.0, "weight_loss_DT": 250.0},  # current adapter default
    {"weight_loss_ECR": 40.0, "weight_loss_DT": 1.0},
    {"weight_loss_ECR": 40.0, "weight_loss_DT": 2.0},
    {"weight_loss_ECR": 50.0, "weight_loss_DT": 250.0},
    {"weight_loss_ECR": 50.0, "weight_loss_DT": 1.0},
    {"weight_loss_ECR": 50.0, "weight_loss_DT": 2.0},
]
HICOT_SEARCH_MAX_FIT_SECONDS = 300


def log(msg: str) -> None:
    with open(RUN_LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def load_done_combos() -> set:
    if not RESULTS_CSV.exists():
        return set()
    with open(RESULTS_CSV) as f:
        return {(row["model"], row["dataset"]) for row in csv.DictReader(f)}


RESULT_FIELDS = (["model", "dataset", "k", "actual_k", "seed", "num_documents", "assignment_source", "source"]
                  + [f"{k}" for k in common.ALL_METRIC_KEYS] + ["runtime_s", "status", "error", "selected_config"])


def append_result_row(row: dict) -> None:
    is_new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in RESULT_FIELDS})


SEARCH_FIELDS = ["model", "dataset", "config_json", "cv", "npmi", "td", "irbo",
                  "acc", "nmi", "ari", "purity", "selected"]


def append_search_row(row: dict) -> None:
    is_new = not SEARCH_CSV.exists()
    with open(SEARCH_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SEARCH_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in SEARCH_FIELDS})


def load_dataset_and_k(dataset_id: str):
    from vaebm_benchmark.datasets.simple_registry import load_dataset

    documents, labels, num_classes = load_dataset(dataset_id)
    return documents, labels, num_classes


def build_fastopic(k: int, seed: int, config: dict):
    from vaebm_benchmark.models.fastopic_adapter import FASTopicAdapter

    return FASTopicAdapter(num_topics=k, vocab_size_cap=VOC_SIZE, learning_rate=0.002, low_memory=False, **config)


def build_hicot(k: int, seed: int, config: dict):
    from vaebm_benchmark.models.hicot_adapter import HiCOTAdapter

    return HiCOTAdapter(n_clusters=k, voc_size=VOC_SIZE, epochs=50, sinkhorn_max_iter=100,
                         max_fit_seconds=HICOT_SEARCH_MAX_FIT_SECONDS, random_state=seed, **config)


def fit_and_evaluate(model_name: str, model, documents, labels, dataset_id: str) -> dict:
    """argmax_theta assignment (fastopic/hicot's own native semantics),
    full 11-metric suite via the same shared function RQ4 uses."""
    model.fit(documents)
    feature_space = np.asarray(model.get_document_topics(documents))
    clusters = [int(i) for i in np.argmax(feature_space, axis=1)]
    topics = common.normalize_topic_words(model.get_topics(top_n=10))
    tokenized_corpus = [d.lower().split() for d in documents]
    doc_emb = common.get_doc_embeddings(dataset_id, documents)
    metrics = common.compute_full_metric_suite(topics, tokenized_corpus, clusters, labels, doc_emb)
    metrics["actual_k"] = len(set(clusters))
    return metrics


def run_search(model_name: str, dataset_id: str, k: int, documents, labels, grid: list[dict]) -> tuple[dict, dict]:
    """Fit every config once, log its full metrics, select the config with
    the HIGHEST unsupervised C_V (never touching acc/nmi/ari/purity for
    selection - those are only logged for transparency). Returns
    (best_config, best_metrics)."""
    best_config, best_metrics, best_cv = None, None, -1e18
    for config in grid:
        start = time.perf_counter()
        try:
            if model_name == "fastopic":
                model = build_fastopic(k, SEED, config)
            else:
                model = build_hicot(k, SEED, config)
            metrics = fit_and_evaluate(model_name, model, documents, labels, dataset_id)
            cv = metrics.get("cv")
            cv_for_selection = cv if isinstance(cv, (int, float)) and not np.isnan(cv) else -1e18
            log(f"  SEARCH {model_name}:{dataset_id} config={config} cv={cv} "
                f"(acc={metrics.get('acc'):.4f} nmi={metrics.get('nmi'):.4f} - logged only, not used for selection) "
                f"runtime={time.perf_counter()-start:.1f}s")
            append_search_row({
                "model": model_name, "dataset": dataset_id, "config_json": json.dumps(config),
                "cv": cv, "npmi": metrics.get("npmi"), "td": metrics.get("td"), "irbo": metrics.get("irbo"),
                "acc": metrics.get("acc"), "nmi": metrics.get("nmi"), "ari": metrics.get("ari"),
                "purity": metrics.get("purity"), "selected": False,
            })
            if cv_for_selection > best_cv:
                best_cv, best_config, best_metrics = cv_for_selection, config, metrics
        except Exception as exc:  # noqa: BLE001
            log(f"  SEARCH-FAIL {model_name}:{dataset_id} config={config} error={exc!r}")
            append_search_row({"model": model_name, "dataset": dataset_id, "config_json": json.dumps(config),
                                "cv": None, "selected": False})

    if best_config is not None:
        # Mark the winning row as selected (rewrite last matching row's flag).
        rows = list(csv.DictReader(open(SEARCH_CSV))) if SEARCH_CSV.exists() else []
        for r in rows:
            if (r["model"] == model_name and r["dataset"] == dataset_id
                    and json.loads(r["config_json"]) == best_config and r["selected"] != "True"):
                r["selected"] = "True"
                break
        with open(SEARCH_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SEARCH_FIELDS)
            w.writeheader()
            w.writerows(rows)

    return best_config, best_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=REUSED_MODELS + SEARCH_MODELS)
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    args = p.parse_args()

    done = load_done_combos()
    log(f"=== RQ3 start models={args.models} datasets={args.datasets} seed={SEED} ===")

    for dataset_id in args.datasets:
        documents, labels, k = load_dataset_and_k(dataset_id)
        log(f"Dataset {dataset_id}: k={k} n_docs={len(documents)}")

        for model_name in args.models:
            if (model_name, dataset_id) in done:
                log(f"SKIP {model_name}:{dataset_id} (already in {RESULTS_CSV.name})")
                continue

            start = time.perf_counter()
            row = {"model": model_name, "dataset": dataset_id, "k": k, "seed": SEED,
                   "num_documents": len(documents), "status": "ok", "error": None}
            try:
                if model_name in SEARCH_MODELS:
                    grid = FASTOPIC_SEARCH_GRID if model_name == "fastopic" else HICOT_SEARCH_GRID
                    log(f"START search {model_name}:{dataset_id} ({len(grid)} configs)")
                    best_config, metrics = run_search(model_name, dataset_id, k, documents, labels, grid)
                    if metrics is None:
                        raise RuntimeError("every search config failed")
                    row["assignment_source"] = "argmax_theta"
                    row["source"] = "rq3_search"
                    row["selected_config"] = json.dumps(best_config)
                else:
                    raise ValueError(f"{model_name} should be reused from the manifest, not fit here")

                row["actual_k"] = metrics.pop("actual_k", None)
                for key in common.ALL_METRIC_KEYS:
                    row[key] = metrics.get(key)
                log(f"OK   {model_name}:{dataset_id} selected_config={best_config} "
                    f"cv={metrics.get('cv')} acc={metrics.get('acc')} nmi={metrics.get('nmi')}")
            except Exception as exc:  # noqa: BLE001
                row["status"], row["error"] = "error", f"{exc}"
                log(f"ERROR {model_name}:{dataset_id} error={exc!r}")
                import traceback
                traceback.print_exc()

            row["runtime_s"] = round(time.perf_counter() - start, 1)
            if row["status"] == "ok":
                append_result_row(row)
                done.add((model_name, dataset_id))
            else:
                log(f"NOT marking {model_name}:{dataset_id} done - will retry on next run")

    log("=== RQ3 done ===")


if __name__ == "__main__":
    main()
