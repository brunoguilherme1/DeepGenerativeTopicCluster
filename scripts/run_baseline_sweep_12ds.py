#!/usr/bin/env python
"""Generic driver: runs ONE model across the 12 locked-architecture
plain datasets, for --experiment cluster and/or classification.
Parametrized so the same script can drive LDA/GloCOM/ECRTM/S2WTM (the
new baselines added 2026-09-21) without four near-identical copies.

Classification uses a single seed=42, matching the established
baseline convention (see paper_data/futurelab/classification5_final.json's
own hicot/fastopic/bertopic rows - single seed=42, not 5) - these are
baselines, not our own main contribution, so they get the lighter
protocol VAE-BM's own locked-architecture sweep does not.

Usage:
    python scripts/run_baseline_sweep_12ds.py --model lda --experiments cluster classification
    python scripts/run_baseline_sweep_12ds.py --model ecrtm --experiments cluster classification --run-dir results/ecrtm_sweep
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    "20ng", "agnews_short", "google_news_t", "imdb", "search_snippets", "bbc_news",
    "tweet", "stack_overflow", "biomedical", "banking77", "m10", "pascal_flickr",
]

PER_COMBO_TIMEOUT_SECONDS = 3600
CACHE_ROOT = Path(os.environ.get("VAEBM_CACHE_ROOT", REPO_ROOT.parent / ".cache"))


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--experiments", nargs="+", choices=["cluster", "classification"], required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()

    datasets = args.datasets or DATASETS
    run_dir = Path(args.run_dir) if args.run_dir else REPO_ROOT / "results" / f"{args.model}_sweep_20260921"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}

    def log(msg: str) -> None:
        line = f"[{now_iso()}] {msg}"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def save_checkpoint() -> None:
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["VAEBM_RESULTS_DIR"] = str(run_dir)
    env["HF_HOME"] = str(CACHE_ROOT / "huggingface")
    env["TRANSFORMERS_CACHE"] = str(CACHE_ROOT / "huggingface" / "transformers")
    env["SENTENCE_TRANSFORMERS_HOME"] = str(CACHE_ROOT / "sentence_transformers")
    env["TORCH_HOME"] = str(CACHE_ROOT / "torch")
    for p in (env["HF_HOME"], env["SENTENCE_TRANSFORMERS_HOME"], env["TORCH_HOME"]):
        Path(p).mkdir(parents=True, exist_ok=True)

    log(f"SWEEP_START model={args.model} experiments={args.experiments} run_dir={run_dir} datasets={datasets}")

    for experiment in args.experiments:
        for idx, dataset in enumerate(datasets, start=1):
            key = f"{experiment}|{dataset}"
            if checkpoint.get(key) == "ok":
                log(f"SKIP experiment={experiment} dataset={dataset} progress={idx}/{len(datasets)} (already ok)")
                continue

            if experiment == "cluster":
                cmd = [
                    sys.executable, "-u", str(REPO_ROOT / "scripts" / "run_experiment.py"),
                    "--experiment", "cluster", "--models", args.model, "--datasets", dataset, "--seed", "42",
                ]
            else:
                cmd = [
                    sys.executable, "-u", str(REPO_ROOT / "scripts" / "run_experiment.py"),
                    "--experiment", "classification", "--models", args.model, "--datasets", dataset,
                    "--split", "random", "--seeds", "42",
                ]

            log(f"START experiment={experiment} dataset={dataset} progress={idx}/{len(datasets)}")
            start = time.perf_counter()
            try:
                proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True,
                                       text=True, timeout=PER_COMBO_TIMEOUT_SECONDS)
                runtime = time.perf_counter() - start
                log_file = run_dir / "logs" / f"{experiment}_{dataset}.log"
                log_file.parent.mkdir(parents=True, exist_ok=True)
                log_file.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
                if proc.returncode != 0:
                    checkpoint[key] = "error"
                    log(f"ERROR experiment={experiment} dataset={dataset} exit={proc.returncode} runtime={runtime:.0f}s "
                        f"stderr_tail={(proc.stderr or '')[-300:]}")
                else:
                    checkpoint[key] = "ok"
                    log(f"OK experiment={experiment} dataset={dataset} runtime={runtime:.0f}s")
            except subprocess.TimeoutExpired:
                checkpoint[key] = "error"
                log(f"ERROR experiment={experiment} dataset={dataset} timeout after {PER_COMBO_TIMEOUT_SECONDS}s")
            save_checkpoint()

    log("SWEEP_COMPLETE")


if __name__ == "__main__":
    main()
