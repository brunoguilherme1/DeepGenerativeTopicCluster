#!/usr/bin/env python
"""RQ1 FINAL multi-seed VAE-BM sweep - replaces the single-seed, per-dataset-
tuned-lambda numbers currently in main.tex's topic_vaebm_all5.tex with a
methodologically cleaner, disclosed, reproducible result.

Fixes applied vs. the prior "Round 27-35" research sweeps (per the
verified critique of this repo's own RQ1 numbers):
  - GLOBAL lambda_relevance=0.1 for every dataset/K (was: a different lambda
    hand-picked per (dataset, K) cell by the SAME Palmetto C_V being
    reported - a real selection-on-the-test-metric concern). The full
    per-dataset lambda sweep from Rounds 26-35 is preserved as-is and used
    only for an appendix sensitivity table, not to pick this run's config.
  - Multi-seed (5 seeds for 20NG/AGNews/SearchSnippets/GoogleNews, 3 seeds
    for IMDB given its longer runtime) instead of single-seed - enables
    mean+-std reporting matching HiCOT's own convention (Tables 18/19).
  - epochs=1 and frozen embedding-branch encoder are the SAME as the prior
    sweeps (this is VAE-BM's actual locked configuration, not a new
    deviation) but are now EXPLICITLY disclosed here in this script's own
    header and must be stated plainly in the paper text.
  - Palmetto C_V only (no local-corpus row) - this is the FINAL reported
    number, not an exploratory sweep, so no need to pay for both.
  - Per-dataset embedder: gte-large for 20NG/AGNews/SearchSnippets/
    GoogleNews, bge-large for IMDB (matches the appendix config table
    already in main.tex - not a new choice, just made explicit/scriptable
    here instead of living only in a table cell).

Same crash-safety pattern as Rounds 17-35 (scripts/run_vaebm_gte_research_
round34_k100_lambda.py): isolated subprocess per (dataset, K, seed) combo,
JSON checkpoint (resumable), run.log/progress.txt/current_status.txt,
per-combo timeout, single retry.

Usage:
    python scripts/run_vaebm_rq1_final_multiseed.py --datasets hicot_20ng --run-dir results/rq1_final_20ng
    python scripts/run_vaebm_rq1_final_multiseed.py --datasets hicot_imdb --seeds 1 2 3 --run-dir results/rq1_final_imdb
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

K_VALUES = [50, 100]
LAMBDA_RELEVANCE = "0.1"  # GLOBAL, not per-dataset-tuned - see module docstring
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
CV_METHOD = "palmetto"
PER_COMBO_TIMEOUT_SECONDS = 2400
MAX_ATTEMPTS = 2  # one retry, vs Round 34's own MAX_ATTEMPTS=1 - slightly more robust given time pressure

EMBEDDER_BY_DATASET = {
    "hicot_20ng": "thenlper/gte-large",
    "hicot_agnews": "thenlper/gte-large",
    "hicot_search_snippets": "thenlper/gte-large",
    "hicot_google_news": "thenlper/gte-large",
    "hicot_imdb": "BAAI/bge-large-en-v1.5",
}
NORMALIZE_MU_BY_DATASET = {
    "hicot_20ng": "1", "hicot_agnews": "1",
    "hicot_search_snippets": "0", "hicot_google_news": "0", "hicot_imdb": "0",
}

BASE_ENV_TEMPLATE = {
    "VAEBM_UNITS": "1024",
    "VAEBM_DIM_EMB": "",
    "VAEBM_FREEZE_EMB": "1",
    "VAEBM_ALPHA": "0.0",
    "VAEBM_LR": "1e-4",
    "VAEBM_EPOCHS": "1",  # disclosed in the paper text - see module docstring
    "VAEBM_EXCLUDE_STOPWORDS": "1",
    "VAEBM_TOP_WORDS_MODE": "relevance",
    "VAEBM_LAMBDA_RELEVANCE": LAMBDA_RELEVANCE,
    "TF_FORCE_GPU_ALLOW_GROWTH": "true",
}

CACHE_ROOT = Path(os.environ.get("VAEBM_CACHE_ROOT", REPO_ROOT.parent / ".cache"))


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def combo_key(dataset: str, k: int, seed: int) -> str:
    return f"{dataset}|k{k}|seed{seed}"


class Sweep:
    def __init__(self, run_dir: Path, datasets: list[str], k_values: list[int], seeds: list[int]):
        self.run_dir = run_dir
        self.logs_dir = run_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.run_log_path = run_dir / "run.log"
        self.status_path = run_dir / "current_status.txt"
        self.progress_path = run_dir / "progress.txt"
        self.checkpoint_path = run_dir / "checkpoint.json"
        self.results_json_path = run_dir / "experiment_results.json"

        self.checkpoint: dict = {}
        if self.checkpoint_path.exists():
            self.checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))

        self.combos = [(d, k, s) for d in datasets for k in k_values for s in seeds]
        self.total = len(self.combos)
        self.started_at = now_iso()

    def log(self, msg: str) -> None:
        line = f"[{now_iso()}] {msg}"
        with open(self.run_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def save_checkpoint(self) -> None:
        tmp = self.checkpoint_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.checkpoint, indent=2), encoding="utf-8")
        os.replace(tmp, self.checkpoint_path)

    def counts(self):
        successful = sum(1 for v in self.checkpoint.values() if v.get("status") == "ok")
        failed = sum(1 for v in self.checkpoint.values() if v.get("status") == "error")
        return successful + failed, successful, failed

    def write_progress(self) -> None:
        completed, successful, failed = self.counts()
        lines = [
            f"Total: {self.total}", f"Completed: {completed}", f"Successful: {successful}",
            f"Failed: {failed}", f"Remaining: {self.total - completed}",
            f"Started at: {self.started_at}", f"Last update: {now_iso()}",
        ]
        self.progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_status(self, dataset: str, k: int, seed: int, attempt: int, progress_idx: int) -> None:
        completed, successful, failed = self.counts()
        text = (
            f"Current: {dataset} k={k} seed={seed}\nAttempt: {attempt}\n"
            f"Completed: {completed}/{self.total}\nSuccessful: {successful}\nFailed: {failed}\n"
            f"Remaining: {self.total - completed}\nStarted at: {self.started_at}\nLast update: {now_iso()}\n"
        )
        self.status_path.write_text(text, encoding="utf-8")

    def _read_result(self, dataset: str, k: int, seed: int):
        if not self.results_json_path.exists():
            return None
        rows = json.loads(self.results_json_path.read_text(encoding="utf-8"))
        for row in reversed(rows):
            if (row.get("model") == "vaebm" and row.get("dataset") == dataset
                    and row.get("k") == k and row.get("seed") == seed):
                return row
        return None

    def run_one(self, dataset: str, k: int, seed: int, progress_idx: int) -> None:
        key = combo_key(dataset, k, seed)
        existing = self.checkpoint.get(key)
        if existing and existing.get("status") == "ok":
            self.log(f"SKIP {key} progress={progress_idx}/{self.total} (already completed at {existing.get('timestamp')})")
            return

        log_file = self.logs_dir / f"{dataset}__k{k}__seed{seed}.log"
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["VAEBM_RESULTS_DIR"] = str(self.run_dir)
        env["HF_HOME"] = str(CACHE_ROOT / "huggingface")
        env["TRANSFORMERS_CACHE"] = str(CACHE_ROOT / "huggingface" / "transformers")
        env["SENTENCE_TRANSFORMERS_HOME"] = str(CACHE_ROOT / "sentence_transformers")
        env["TORCH_HOME"] = str(CACHE_ROOT / "torch")
        env["OPENBLAS_NUM_THREADS"] = "8"
        env["OMP_NUM_THREADS"] = "8"
        env["MKL_NUM_THREADS"] = "8"
        for p in (env["HF_HOME"], env["SENTENCE_TRANSFORMERS_HOME"], env["TORCH_HOME"]):
            Path(p).mkdir(parents=True, exist_ok=True)
        env.update(BASE_ENV_TEMPLATE)
        env["VAEBM_EMBEDDER"] = EMBEDDER_BY_DATASET[dataset]
        env["VAEBM_NORMALIZE_MU"] = NORMALIZE_MU_BY_DATASET[dataset]

        cmd = [
            sys.executable, "-u", str(REPO_ROOT / "scripts" / "run_experiment.py"),
            "--experiment", "topic", "--models", "vaebm", "--datasets", dataset,
            "--k", str(k), "--seed", str(seed), "--protocol", "ecrtm_hicot",
            "--cv-method", CV_METHOD,
        ]

        last_error = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.write_status(dataset, k, seed, attempt, progress_idx)
            self.log(f"START {key} attempt={attempt} progress={progress_idx}/{self.total} embedder={env['VAEBM_EMBEDDER']}")
            start = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd, cwd=str(REPO_ROOT), env=env,
                    capture_output=True, text=True, timeout=PER_COMBO_TIMEOUT_SECONDS,
                )
                runtime = time.perf_counter() - start
                with open(log_file, "a", encoding="utf-8") as lf:
                    lf.write(f"\n=== attempt {attempt} (exit={proc.returncode}, {runtime:.1f}s) ===\n")
                    lf.write(proc.stdout or "")
                    lf.write(proc.stderr or "")

                if proc.returncode != 0:
                    last_error = f"subprocess exited {proc.returncode}: {(proc.stderr or '')[-500:]}"
                else:
                    outcome = self._read_result(dataset, k, seed)
                    if outcome is None:
                        last_error = "subprocess exited 0 but no matching result row found"
                    elif outcome.get("status") == "ok":
                        self.checkpoint[key] = {
                            "status": "ok", "attempts": attempt, "runtime": outcome.get("runtime_seconds"),
                            "cv": outcome.get("cv"), "purity": outcome.get("purity"),
                            "nmi": outcome.get("nmi"), "td": outcome.get("td"),
                            "embedder": env["VAEBM_EMBEDDER"], "lambda_relevance": LAMBDA_RELEVANCE,
                            "timestamp": now_iso(),
                        }
                        self.save_checkpoint()
                        self.write_progress()
                        self.log(f"OK {key} runtime={runtime:.0f}s cv={outcome.get('cv')} "
                                 f"purity={outcome.get('purity')} nmi={outcome.get('nmi')} td={outcome.get('td')} "
                                 f"progress={progress_idx}/{self.total}")
                        return
                    else:
                        last_error = (outcome.get("error") or "")[:500]
            except subprocess.TimeoutExpired:
                last_error = f"timeout after {PER_COMBO_TIMEOUT_SECONDS}s"
                with open(log_file, "a", encoding="utf-8") as lf:
                    lf.write(f"\n=== attempt {attempt} TIMEOUT after {PER_COMBO_TIMEOUT_SECONDS}s ===\n")
            except Exception as exc:  # noqa: BLE001
                last_error = f"driver exception: {exc!r}\n{traceback.format_exc(limit=3)}"

            self.log(f"ATTEMPT-FAIL {key} attempt={attempt} error={last_error[:200]!r}")

        self.checkpoint[key] = {"status": "error", "attempts": MAX_ATTEMPTS, "error": last_error, "timestamp": now_iso()}
        self.save_checkpoint()
        self.write_progress()
        self.log(f"FINAL-FAIL {key} error={last_error[:300]!r}")

    def run_all(self) -> None:
        for idx, (dataset, k, seed) in enumerate(self.combos, start=1):
            self.run_one(dataset, k, seed, idx)
        self.log("=== SWEEP DONE ===")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True, choices=list(EMBEDDER_BY_DATASET.keys()))
    p.add_argument("--k", nargs="+", type=int, default=K_VALUES)
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="Default: 5 seeds (1-5), or 3 seeds (1-3) if the only dataset is hicot_imdb.")
    p.add_argument("--run-dir", type=str, required=True)
    args = p.parse_args()

    seeds = args.seeds
    if seeds is None:
        seeds = [1, 2, 3] if args.datasets == ["hicot_imdb"] else DEFAULT_SEEDS

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    sweep = Sweep(run_dir, args.datasets, args.k, seeds)
    sweep.log(f"=== RQ1 FINAL multi-seed sweep start datasets={args.datasets} k={args.k} seeds={seeds} "
              f"lambda_relevance={LAMBDA_RELEVANCE} (GLOBAL, not per-dataset-tuned) cv_method={CV_METHOD} ===")
    sweep.run_all()


if __name__ == "__main__":
    main()
