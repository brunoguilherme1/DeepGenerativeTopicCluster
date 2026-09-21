#!/usr/bin/env python
"""Cluster experiment (K=n_classes, unsupervised) for the locked VAE-BM
architecture across 12 plain (non-hicot_*) datasets (2026-09-21,
user-authorized) - see docs/vaebm_leaderboard.md's own "Cluster/
Classification (locked architecture)" section and the user's own
to-do list: no hicot_* datasets here (discarded per instruction - the
topic experiment is the only one that uses hicot_*/plain pairs), no
_ts/_full/large-dataset variants (dbpedia_14/yahoo_answers_topics
excluded for size, agnews_full excluded in favor of agnews_short, per
instruction), no per-dataset re-tuning - the SAME architecture the
topic experiment's own sweep established, reused as-is:
  - alpha=0, frozen+identity-initialized gte-large embedding branch
    (bge-large for imdb specifically), K = each dataset's own number of
    ground-truth classes (transductive, no --k flag - cluster's own
    established convention).
  - normalize_mu=1 for the 20ng/agnews-family datasets (matching the
    topic experiment's own hicot_20ng/hicot_agnews exception), off
    elsewhere.
  - epochs=1: under alpha=0 with a frozen+identity-initialized embedding
    branch, mu = mu_emb exactly, independent of epoch count (the BoW
    branch's own training never reaches mu) - epochs=1 is provably
    equivalent to any other value here, not a shortcut (see the
    cluster_runner.py/scientific_models.py commit adding
    VAEBM_FREEZE_EMB/VAEBM_NORMALIZE_MU support, 2026-09-21).
  - Single seed (42): under this architecture, mu and KMeans's own
    default random_state are both independent of --seed, so cluster
    results are already fully deterministic across seeds (confirmed by
    the topic experiment's own Round 28 KMeans-seed sensitivity check) -
    multi-seed would not add information here, unlike classification's
    own genuine train/test-split variance.

Usage:
    python scripts/run_vaebm_cluster_locked_12ds.py
    python scripts/run_vaebm_cluster_locked_12ds.py --run-dir results/vaebm_cluster_locked_20260921_190000
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

SEED = 42

BASE_ENV = {
    "VAEBM_EMBEDDER": "thenlper/gte-large",
    "VAEBM_UNITS": "1024",
    "VAEBM_DIM_EMB": "",
    "VAEBM_FREEZE_EMB": "1",
    "VAEBM_ALPHA": "0.0",
    "VAEBM_LR": "1e-4",
    "VAEBM_EPOCHS": "1",
    "TF_FORCE_GPU_ALLOW_GROWTH": "true",
}

# Each entry: (dataset, env overrides layered on top of BASE_ENV).
DATASETS = [
    ("20ng", {"VAEBM_NORMALIZE_MU": "1"}),
    ("agnews_short", {"VAEBM_NORMALIZE_MU": "1"}),
    ("google_news_t", {}),
    ("imdb", {"VAEBM_EMBEDDER": "BAAI/bge-large-en-v1.5"}),
    ("search_snippets", {}),
    ("bbc_news", {}),
    ("tweet", {}),
    ("stack_overflow", {}),
    ("biomedical", {}),
    ("banking77", {}),
    ("m10", {}),
    ("pascal_flickr", {}),
]

MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 30
PER_COMBO_TIMEOUT_SECONDS = 2400

CACHE_ROOT = Path(os.environ.get("VAEBM_CACHE_ROOT", REPO_ROOT.parent / ".cache"))


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def combo_key(dataset: str) -> str:
    return f"vaebm|{dataset}"


class Sweep:
    def __init__(self, run_dir: Path):
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

        self.total = len(DATASETS)
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

    def write_status(self, dataset: str, attempt: int) -> None:
        completed, successful, failed = self.counts()
        text = (
            f"Dataset: {dataset}\nAttempt: {attempt}\n"
            f"Completed: {completed}/{self.total}\nSuccessful: {successful}\nFailed: {failed}\n"
            f"Remaining: {self.total - completed}\nStarted at: {self.started_at}\nLast update: {now_iso()}\n"
        )
        self.status_path.write_text(text, encoding="utf-8")

    def run_one(self, dataset: str, overrides: dict, progress_idx: int) -> None:
        key = combo_key(dataset)
        existing = self.checkpoint.get(key)
        if existing and existing.get("status") == "ok":
            self.log(f"SKIP dataset={dataset} progress={progress_idx}/{self.total} "
                     f"(already completed at {existing.get('timestamp')})")
            return

        log_file = self.logs_dir / f"{dataset}.log"
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["VAEBM_RESULTS_DIR"] = str(self.run_dir)
        env["HF_HOME"] = str(CACHE_ROOT / "huggingface")
        env["TRANSFORMERS_CACHE"] = str(CACHE_ROOT / "huggingface" / "transformers")
        env["SENTENCE_TRANSFORMERS_HOME"] = str(CACHE_ROOT / "sentence_transformers")
        env["TORCH_HOME"] = str(CACHE_ROOT / "torch")
        for p in (env["HF_HOME"], env["SENTENCE_TRANSFORMERS_HOME"], env["TORCH_HOME"]):
            Path(p).mkdir(parents=True, exist_ok=True)
        env.update(BASE_ENV)
        env.update(overrides)

        cmd = [
            sys.executable, "-u", str(REPO_ROOT / "scripts" / "run_experiment.py"),
            "--experiment", "cluster", "--models", "vaebm", "--datasets", dataset,
            "--seed", str(SEED),
        ]

        last_error = ""
        final_attempt = MAX_ATTEMPTS
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.write_status(dataset, attempt)
            self.log(f"START dataset={dataset} attempt={attempt} progress={progress_idx}/{self.total} overrides={overrides}")
            start = time.perf_counter()
            timed_out = False
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
                    outcome = self._read_result(dataset)
                    if outcome is None:
                        last_error = "subprocess exited 0 but no matching result row found in experiment_results.json"
                    elif outcome.get("status") == "ok":
                        metrics = {k2: outcome.get(k2) for k2 in
                                   ("acc", "nmi", "ari", "ami", "homogeneity", "completeness",
                                    "v_measure", "purity", "silhouette", "davies_bouldin", "calinski_harabasz")}
                        self.checkpoint[key] = {
                            "status": "ok", "attempts": attempt, "runtime": outcome.get("runtime_seconds"),
                            **metrics, "timestamp": now_iso(),
                        }
                        self.save_checkpoint()
                        self.write_progress()
                        self.log(f"OK dataset={dataset} runtime={runtime:.0f}s "
                                 f"acc={metrics.get('acc')} nmi={metrics.get('nmi')} purity={metrics.get('purity')} "
                                 f"progress={progress_idx}/{self.total}")
                        return
                    else:
                        last_error = (outcome.get("error") or "")[:500]
            except subprocess.TimeoutExpired:
                timed_out = True
                last_error = f"timeout after {PER_COMBO_TIMEOUT_SECONDS}s"
                with open(log_file, "a", encoding="utf-8") as lf:
                    lf.write(f"\n=== attempt {attempt} TIMEOUT after {PER_COMBO_TIMEOUT_SECONDS}s ===\n")
            except Exception as exc:  # noqa: BLE001
                last_error = f"driver exception: {exc!r}\n{traceback.format_exc(limit=3)}"

            is_oom = "out of memory" in last_error.lower()
            tag = '"CUDA out of memory"' if is_oom else f'"{last_error[:200].splitlines()[0] if last_error else last_error}"'
            self.log(f"ERROR dataset={dataset} attempt={attempt} error={tag}")

            if timed_out:
                final_attempt = attempt
                break
            if attempt < MAX_ATTEMPTS:
                backoff = RETRY_BACKOFF_SECONDS * (3 if is_oom else 1)
                self.log(f"RETRY dataset={dataset} attempt={attempt + 1} (backoff={backoff}s)")
                time.sleep(backoff)
                final_attempt = attempt + 1

        self.checkpoint[key] = {"status": "error", "attempts": final_attempt, "error": last_error[:2000], "timestamp": now_iso()}
        self.save_checkpoint()
        self.write_progress()
        self.log(f"ERROR dataset={dataset} FINAL after {final_attempt} attempt(s), giving up progress={progress_idx}/{self.total}")

    def _read_result(self, dataset: str) -> dict | None:
        if not self.results_json_path.exists():
            return None
        rows = json.loads(self.results_json_path.read_text(encoding="utf-8"))
        for row in reversed(rows):
            if row.get("model") == "vaebm" and row.get("dataset") == dataset:
                return row
        return None

    def run(self) -> None:
        self.log(f"SWEEP_START run_dir={self.run_dir} datasets={[d for d, _ in DATASETS]} base_env={BASE_ENV}")
        for idx, (dataset, overrides) in enumerate(DATASETS, start=1):
            self.run_one(dataset, overrides, idx)
        self.write_progress()
        completed, successful, failed = self.counts()
        self.log(f"SWEEP_COMPLETE total={self.total} successful={successful} failed={failed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        if not run_dir.exists():
            raise SystemExit(f"--run-dir {run_dir} does not exist - omit --run-dir to start a new run")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = REPO_ROOT / "results" / f"vaebm_cluster_locked_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)

    Sweep(run_dir).run()


if __name__ == "__main__":
    main()
