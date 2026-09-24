#!/usr/bin/env python
"""Small hyperparameter search for VAE-BM and HiCOT's topic QUALITY
(C_V via Palmetto-Wikipedia, TD) on all 6 FASTopic-protocol datasets
(20NG, NYT, WoS, NeurIPS, ACL, Wikitext-103) - reproduces the axes of
FASTopic's own Table 1, never rerunning FASTopic or its other baselines.

Unlike run_hicot_vaebm_fastopic_search.py (cluster+classification, needs
labels, only 20NG/NYT/WoS), this is topic-quality only - no labels needed,
so it runs on all 6 datasets. Fits on the full corpus (train+test), gets
top_n=10 topic words, scores via metrics/topic_quality.py's own
compute_topic_metrics(). C_V is OPTIONAL (soft-fails to None if Palmetto's
jar/wiki index aren't present on this machine) - TD never does.

Supports splitting work across parallel processes (e.g. one per GPU):
--datasets restricts which dataset ids to run, --gpu pins this process to
one physical GPU index (see the GPU-split comment below for why both
frameworks still need this even on separate GPUs).

Usage:
    python scripts/run_hicot_vaebm_topic_quality_search.py --gpu 0 --datasets fastopic_20ng fastopic_nyt
    python scripts/run_hicot_vaebm_topic_quality_search.py --gpu 1 --datasets fastopic_wos_reconstructed fastopic_neurips
"""
from __future__ import annotations

import argparse
import os
import sys

# GPU pinning happens BEFORE any torch/tensorflow import - both frameworks
# read CUDA_VISIBLE_DEVICES once at their own first CUDA touch, so this
# must be set from the process's very first lines, not after argparse
# alone would suggest.
_parser_gpu = argparse.ArgumentParser(add_help=False)
_parser_gpu.add_argument("--gpu", type=int, default=None)
_gpu_args, _ = _parser_gpu.parse_known_args()
if _gpu_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_args.gpu)

# MUST import/use torch (via sentence_transformers) BEFORE tensorflow ever
# creates a real GPU device in this process - verified directly on labuai
# (2026-09-22): once TensorFlow has claimed a GPU device, importing
# sentence_transformers afterward segfaults at the import itself. Even
# with CUDA_VISIBLE_DEVICES restricting both to the SAME single physical
# GPU (as seen by this process), this order still matters.
from sentence_transformers import SentenceTransformer as _WarmUpSentenceTransformer  # noqa: F401
import torch as _torch
import tensorflow as tf

_gpus = tf.config.list_physical_devices("GPU")
for _g in _gpus:
    tf.config.experimental.set_memory_growth(_g, True)

import csv
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SEED = 42
ALL_DATASETS = [
    "fastopic_20ng", "fastopic_nyt", "fastopic_wos_reconstructed",
    "fastopic_neurips", "fastopic_acl", "fastopic_wikitext103",
]

VAEBM_GRID = [
    {"name": "mini_baseline", "embedder": "all-MiniLM-L6-v2", "alpha": 0.0, "units": 50, "freeze_embedding_branch": False},
    {"name": "gte_baseline", "embedder": "thenlper/gte-large", "alpha": 0.0, "units": 50, "freeze_embedding_branch": False},
]
HICOT_GRID = [
    {"name": "paper_default", "lr": 0.002, "weight_loss_ECR": 40.0, "weight_loss_DT": 250.0, "en_units": 200, "max_fit_seconds": 1800},
    {"name": "wider_units", "lr": 0.002, "weight_loss_ECR": 40.0, "weight_loss_DT": 250.0, "en_units": 300, "max_fit_seconds": 1800},
]


def parse_args():
    p = argparse.ArgumentParser(parents=[_parser_gpu])
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    p.add_argument("--out", default=None, help="CSV output path (default: results/hicot_vaebm_topic_quality_search/trials_<suffix>.csv)")
    p.add_argument("--suffix", default="", help="Suffix for the default --out path, e.g. gpu0 - lets parallel processes write separate files.")
    return p.parse_args()


def run_topic_quality(protocol, model, dataset_id):
    from vaebm_benchmark.metrics.topic_quality import compute_topic_metrics

    all_texts = protocol.prepare_all_documents(dataset_id)
    model.fit(all_texts)
    topics = model.get_topics(top_n=10)
    values, errors = compute_topic_metrics(
        topics, reference_corpus=[], metric_names=["cv_palmetto_wikipedia", "topic_diversity"], top_n=10,
    )
    return {"cv": values.get("cv_palmetto_wikipedia"), "td": values.get("topic_diversity"), "cv_error": errors.get("cv_palmetto_wikipedia")}


def run_combo(protocol, model_name, dataset_id, config, append_row):
    build_fn = protocol.build_vaebm if model_name == "vaebm" else protocol.build_hicot
    start = time.perf_counter()
    row = {"dataset": dataset_id, "model": model_name, "config_name": config["name"],
           "config": {k: v for k, v in config.items() if k != "name"}}
    try:
        model = build_fn(dataset_id, SEED, {k: v for k, v in config.items() if k != "name"})
        result = run_topic_quality(protocol, model, dataset_id)
        row.update(result)
        row["status"] = "ok"
        row["runtime_s"] = round(time.perf_counter() - start, 1)
        print(f"OK   {dataset_id:24s} {model_name:6s} {config['name']:15s} {result} runtime={row['runtime_s']}s", flush=True)
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["error"] = repr(exc)
        row["runtime_s"] = round(time.perf_counter() - start, 1)
        print(f"ERROR {dataset_id:24s} {model_name:6s} {config['name']:15s} error={exc!r}", flush=True)
        traceback.print_exc()
    append_row(row)


def main():
    args = parse_args()
    from vaebm_benchmark.protocols.fastopic_protocol import FASTopicProtocol

    out_path = Path(args.out) if args.out else REPO_ROOT / "results" / "hicot_vaebm_topic_quality_search" / f"trials{('_' + args.suffix) if args.suffix else ''}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "model", "config_name", "config", "status", "cv", "td", "cv_error", "runtime_s", "error"]

    def append_row(row):
        is_new = not out_path.exists()
        with open(out_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if is_new:
                w.writeheader()
            w.writerow(row)

    protocol = FASTopicProtocol(smoke_test=False)
    print(f"=== Topic-quality (C_V/TD) search: gpu={_gpu_args.gpu} datasets={args.datasets} ===", flush=True)
    print(f"Results -> {out_path}\n", flush=True)

    for dataset_id in args.datasets:
        for config in VAEBM_GRID:
            run_combo(protocol, "vaebm", dataset_id, config, append_row)
        for config in HICOT_GRID:
            run_combo(protocol, "hicot", dataset_id, config, append_row)

    print("\n=== DONE ===")
    print("\n=== PUBLISHED REFERENCE (FASTopic paper Table 1, for comparison only) ===")
    for pr in protocol.published_results:
        if pr.metric in ("cv", "td") and pr.dataset_id in args.datasets:
            print(f"  {pr.dataset_id:24s} {pr.metric:4s} = {pr.value}  ({pr.source})")


if __name__ == "__main__":
    main()
