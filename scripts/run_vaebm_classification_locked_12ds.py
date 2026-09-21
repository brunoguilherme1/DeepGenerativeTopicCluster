#!/usr/bin/env python
"""Classification experiment (K=n_classes, supervised, multi-seed) for
the locked VAE-BM architecture across the same 12 plain (non-hicot_*)
datasets as run_vaebm_cluster_locked_12ds.py (2026-09-21,
user-authorized) - see that script's own docstring for the shared
architecture rationale (alpha=0, frozen+identity-initialized gte-large/
bge-large embedding branch, normalize_mu for the 20ng/agnews-family,
epochs=1 provably equivalent to any other value here).

Unlike cluster, classification IS genuinely seed-dependent under this
architecture: mu itself is deterministic, but the stratified random
80/20 train/test split (`--split random`, since these are plain, not
hicot_*, dataset ids - no official split exists/matters here) is
re-drawn per seed, so which documents land in train vs. test varies,
and so does the resulting SVM accuracy/F1. 5 seeds per dataset, mean/std
computed at report time from the raw per-seed rows this driver stores.

K is never passed explicitly - `--split random` auto-derives it from
each dataset's own number of classes (this experiment's own
`run_single_random_split` behavior), matching cluster's own convention.

Usage:
    python scripts/run_vaebm_classification_locked_12ds.py
    python scripts/run_vaebm_classification_locked_12ds.py --run-dir results/vaebm_classif_locked_20260921_190000
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

SEEDS = [1, 2, 3, 4, 5]

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
PER_COMBO_TIMEOUT_SECONDS = 3600  # 5 seeds in one subprocess call

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
            "--experiment", "classification", "--models", "vaebm", "--datasets", dataset,
            "--split", "random", "--seeds", *[str(s) for s in SEEDS],
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
                    rows = self._read_results(dataset)
                    ok_rows = [r for r in rows if r.get("status") == "ok"]
                    if not rows:
                        last_error = "subprocess exited 0 but no matching result rows found in experiment_results.json"
                    elif len(ok_rows) < len(SEEDS):
                        errs = [r for r in rows if r.get("status") != "ok"]
                        last_error = f"only {len(ok_rows)}/{len(SEEDS)} seeds ok: {(errs[0].get('error') or '')[:400] if errs else ''}"
                    else:
                        accs = [r["accuracy"] for r in ok_rows]
                        f1s = [r["f1"] for r in ok_rows]
                        mean_acc = sum(accs) / len(accs)
                        mean_f1 = sum(f1s) / len(f1s)
                        std_acc = (sum((a - mean_acc) ** 2 for a in accs) / len(accs)) ** 0.5
                        std_f1 = (sum((f - mean_f1) ** 2 for f in f1s) / len(f1s)) ** 0.5
                        self.checkpoint[key] = {
                            "status": "ok", "attempts": attempt, "runtime": runtime,
                            "seeds": SEEDS, "accuracy_per_seed": accs, "f1_per_seed": f1s,
                            "accuracy_mean": mean_acc, "accuracy_std": std_acc,
                            "f1_mean": mean_f1, "f1_std": std_f1,
                            "timestamp": now_iso(),
                        }
                        self.save_checkpoint()
                        self.write_progress()
                        self.log(f"OK dataset={dataset} runtime={runtime:.0f}s "
                                 f"accuracy={mean_acc:.4f}+/-{std_acc:.4f} f1={mean_f1:.4f}+/-{std_f1:.4f} "
                                 f"progress={progress_idx}/{self.total}")
                        return
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

    def _read_results(self, dataset: str) -> list[dict]:
        if not self.results_json_path.exists():
            return []
        rows = json.loads(self.results_json_path.read_text(encoding="utf-8"))
        matches = {}
        for row in rows:
            if row.get("model") == "vaebm" and row.get("dataset") == dataset and row.get("seed") in SEEDS:
                matches[row["seed"]] = row  # last one per seed wins (resume-safe)
        return list(matches.values())

    def run(self) -> None:
        self.log(f"SWEEP_START run_dir={self.run_dir} datasets={[d for d, _ in DATASETS]} seeds={SEEDS} base_env={BASE_ENV}")
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
        run_dir = REPO_ROOT / "results" / f"vaebm_classif_locked_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)

    Sweep(run_dir).run()


if __name__ == "__main__":
    main()
