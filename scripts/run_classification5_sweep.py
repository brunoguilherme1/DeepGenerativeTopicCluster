#!/usr/bin/env python
"""Durable, resumable driver for the 5-model x 26-dataset downstream
classification sweep (models: sbert_minilm, sbert_gte, hicot, fastopic,
bertopic) - `--experiment classification --split random` (see
experiment/classification_runner.py's own module docstring): document ->
model representation (theta for hicot/fastopic, embeddings for
sbert_minilm/sbert_gte/bertopic) -> SVM -> labels, over a stratified
80/20 split we draw ourselves, K auto-derived per dataset from its own
num_classes (--k omitted).

Mirrors run_cluster7_sweep.py's architecture exactly (isolated
subprocess per (model, dataset) combination for real GPU-memory release
+ crash isolation, JSON checkpoint for safe resume, run.log/
current_status.txt/progress.txt for live monitoring, uniform 30-minute
per-combo timeout with skip-not-retry-on-timeout - user-authorized
policy, 2026-09-15, same reasoning as cluster7's own) - adapted for the
classification experiment's own result shape (accuracy/f1 instead of
acc/nmi/purity).

Usage:
    python scripts/run_classification5_sweep.py                      # start a new run
    python scripts/run_classification5_sweep.py --run-dir results/classification5_all_datasets_20260915_120000  # resume
"""

from __future__ import annotations

import argparse
import csv
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

MODELS = ["sbert_minilm", "sbert_gte", "hicot", "fastopic", "bertopic"]

# Same 26-dataset list as run_cluster7_sweep.py's own DATASETS - "agnews"
# deliberately omitted (pre-existing alias of "agnews_short", see that
# script's own comment).
DATASETS = [
    "20ng", "imdb", "agnews_short", "search_snippets", "stack_overflow",
    "biomedical", "google_news_ts", "google_news_t", "google_news_s", "tweet",
    "hicot_20ng", "hicot_imdb", "hicot_agnews", "hicot_search_snippets", "hicot_google_news",
    "agnews_full", "banking77", "bbc_news", "dblp", "dbpedia_14",
    "m10", "pascal_flickr", "tweet_eval_emotion", "tweet_eval_sentiment",
    "yahoo_answers_topics", "20ng_s2wtm",
]
SEED = 42

# User-authorized (2026-09-15), same policy as run_cluster7_sweep.py's
# own: one attempt only, uniform 30-minute cap for every model including
# hicot (whose own internal max_fit_seconds early-stop - see
# experiment/classification_runner.py::_build_model_for_classification -
# keeps this subprocess-level timeout a pure safety net, not hicot's
# primary stopping mechanism). A timeout is not retried - the same
# computation under the same budget won't finish faster a second time.
MAX_ATTEMPTS = 1
RETRY_BACKOFF_SECONDS = 20
PER_COMBO_TIMEOUT_SECONDS = 1800

CACHE_ROOT = Path(os.environ.get("VAEBM_CACHE_ROOT", REPO_ROOT.parent / ".cache"))


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def combo_key(model: str, dataset: str) -> str:
    return f"{model}|{dataset}"


class Sweep:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.logs_dir = run_dir / "logs"
        self.tables_dir = run_dir / "tables"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.run_log_path = run_dir / "run.log"
        self.status_path = run_dir / "current_status.txt"
        self.progress_path = run_dir / "progress.txt"
        self.checkpoint_path = run_dir / "checkpoint.json"
        # run_experiment.py's classification experiment writes under
        # <VAEBM_RESULTS_DIR>/classification/classification_results.json
        # (_run_classification's own convention).
        self.results_json_path = run_dir / "classification" / "classification_results.json"

        self.checkpoint: dict = {}
        if self.checkpoint_path.exists():
            self.checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))

        self.total = len(MODELS) * len(DATASETS)
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

    def write_status(self, model: str, dataset: str, attempt: int) -> None:
        completed, successful, failed = self.counts()
        text = (
            f"Current model: {model}\nDataset: {dataset}\nAttempt: {attempt}\n"
            f"Completed: {completed}/{self.total}\nSuccessful: {successful}\nFailed: {failed}\n"
            f"Remaining: {self.total - completed}\nStarted at: {self.started_at}\nLast update: {now_iso()}\n"
        )
        self.status_path.write_text(text, encoding="utf-8")

    def run_one(self, model: str, dataset: str, progress_idx: int) -> None:
        key = combo_key(model, dataset)
        existing = self.checkpoint.get(key)
        if existing and existing.get("status") == "ok":
            self.log(f"SKIP model={model} dataset={dataset} progress={progress_idx}/{self.total} "
                     f"(already completed at {existing.get('timestamp')})")
            return

        log_file = self.logs_dir / f"{model}__{dataset}.log"
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["VAEBM_RESULTS_DIR"] = str(self.run_dir)
        env["HF_HOME"] = str(CACHE_ROOT / "huggingface")
        env["TRANSFORMERS_CACHE"] = str(CACHE_ROOT / "huggingface" / "transformers")
        env["SENTENCE_TRANSFORMERS_HOME"] = str(CACHE_ROOT / "sentence_transformers")
        env["TORCH_HOME"] = str(CACHE_ROOT / "torch")
        for p in (env["HF_HOME"], env["SENTENCE_TRANSFORMERS_HOME"], env["TORCH_HOME"]):
            Path(p).mkdir(parents=True, exist_ok=True)
        # 20min - leaves a 10-minute margin under this sweep's own
        # 30-minute external subprocess timeout, same margin
        # run_cluster7_sweep.py's own hicot builds use.
        env.setdefault("VAEBM_HICOT_MAX_FIT_SECONDS", "1200")

        cmd = [
            sys.executable, "-u", str(REPO_ROOT / "scripts" / "run_experiment.py"),
            "--experiment", "classification", "--split", "random",
            "--models", model, "--datasets", dataset, "--seed", str(SEED),
        ]

        last_error = ""
        final_attempt = MAX_ATTEMPTS
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.write_status(model, dataset, attempt)
            self.log(f"START model={model} dataset={dataset} attempt={attempt} progress={progress_idx}/{self.total}")
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
                    outcome = self._read_result(model, dataset)
                    if outcome is None:
                        last_error = "subprocess exited 0 but no matching result row found in classification_results.json"
                    elif outcome["status"] == "ok":
                        self.checkpoint[key] = {
                            "status": "ok", "attempts": attempt, "runtime": outcome["runtime_seconds"],
                            "k": outcome["k"], "accuracy": outcome["accuracy"], "f1": outcome["f1"],
                            "num_train_docs": outcome["num_train_docs"], "num_test_docs": outcome["num_test_docs"],
                            "timestamp": now_iso(),
                        }
                        self.save_checkpoint()
                        self.write_progress()
                        self.log(f"OK model={model} dataset={dataset} k={outcome['k']} runtime={runtime:.0f}s "
                                 f"accuracy={outcome['accuracy']} f1={outcome['f1']} progress={progress_idx}/{self.total}")
                        return
                    else:
                        last_error = (outcome.get("error") or "")[:500]
            except subprocess.TimeoutExpired:
                timed_out = True
                last_error = f"timeout after {PER_COMBO_TIMEOUT_SECONDS}s"
                with open(log_file, "a", encoding="utf-8") as lf:
                    lf.write(f"\n=== attempt {attempt} TIMEOUT after {PER_COMBO_TIMEOUT_SECONDS}s ===\n")
            except Exception as exc:  # noqa: BLE001 - the driver itself must never die on one combo
                last_error = f"driver exception: {exc!r}\n{traceback.format_exc(limit=3)}"

            is_oom = "out of memory" in last_error.lower()
            tag = '"CUDA out of memory"' if is_oom else f'"{last_error[:200].splitlines()[0] if last_error else last_error}"'
            self.log(f"ERROR model={model} dataset={dataset} attempt={attempt} error={tag}")

            if timed_out:
                final_attempt = attempt
                break
            if attempt < MAX_ATTEMPTS:
                self.log(f"RETRY model={model} dataset={dataset} attempt={attempt + 1}")
                time.sleep(RETRY_BACKOFF_SECONDS)
                final_attempt = attempt + 1

        self.checkpoint[key] = {"status": "error", "attempts": final_attempt, "error": last_error[:2000], "timestamp": now_iso()}
        self.save_checkpoint()
        self.write_progress()
        self.log(f"ERROR model={model} dataset={dataset} FINAL after {final_attempt} attempt(s), giving up "
                 f"progress={progress_idx}/{self.total}")

    def _read_result(self, model: str, dataset: str) -> dict | None:
        if not self.results_json_path.exists():
            return None
        rows = json.loads(self.results_json_path.read_text(encoding="utf-8"))
        for row in reversed(rows):
            if row.get("model") == model and row.get("dataset") == dataset and row.get("seed") == SEED:
                return row
        return None

    def run(self) -> None:
        run_dir_head_node = str(self.run_dir).replace("/raid/brunoguilherme", "/shared/dgx-raid/brunoguilherme")
        self.log(f"SWEEP_START run_dir={self.run_dir}")
        self.log(f"MONITOR tail -f {run_dir_head_node}/run.log")
        self.log(f"MONITOR cat {run_dir_head_node}/current_status.txt")
        self.log(f"MONITOR cat {run_dir_head_node}/progress.txt")

        idx = 0
        for model in MODELS:
            for dataset in DATASETS:
                idx += 1
                self.run_one(model, dataset, idx)

        self.write_progress()
        self.log("SWEEP_LOOP_DONE all combinations attempted - building final tables")
        self.build_final_outputs()
        completed, successful, failed = self.counts()
        self.log(f"SWEEP_COMPLETE total={self.total} successful={successful} failed={failed}")

    def build_final_outputs(self) -> None:
        """K varies per dataset here (auto-derived from num_classes), so
        classification_report.py's own render_classification_table()
        (built around ONE shared K across every dataset, the official
        ECRTM-reproduction protocol's convention) doesn't fit this
        sweep's shape - write a simple one-row-per-(model,dataset) table
        directly from the checkpoint instead, same information
        run_cluster7_sweep.py's own tables carry."""
        header = ["Model", "Dataset", "K", "Accuracy", "F1", "Runtime(s)"]
        rows_by_dataset: dict[str, list[list[str]]] = {d: [] for d in DATASETS}
        flat_rows = []
        for model in MODELS:
            for dataset in DATASETS:
                ck = self.checkpoint.get(combo_key(model, dataset))
                if ck is None:
                    continue
                if ck.get("status") == "ok":
                    row = [model, dataset, str(ck["k"]), f"{ck['accuracy']:.4f}", f"{ck['f1']:.4f}", f"{ck['runtime']:.0f}"]
                    flat_rows.append({
                        "experiment_type": "classification", "model": model, "dataset": dataset, "k": ck["k"],
                        "seed": SEED, "accuracy": ck["accuracy"], "f1": ck["f1"],
                        "num_train_docs": ck.get("num_train_docs"), "num_test_docs": ck.get("num_test_docs"),
                        "runtime_seconds": ck["runtime"], "status": "ok", "error": "",
                    })
                else:
                    row = [model, dataset, "N/A", "ERROR", "ERROR", "N/A"]
                    flat_rows.append({
                        "experiment_type": "classification", "model": model, "dataset": dataset, "k": None,
                        "seed": SEED, "accuracy": None, "f1": None,
                        "num_train_docs": None, "num_test_docs": None,
                        "runtime_seconds": None, "status": "error", "error": ck.get("error", "not run"),
                    })
                rows_by_dataset[dataset].append(row)

        def render_table(rows: list[list[str]]) -> str:
            widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i]) for i in range(len(header))]
            lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(header))]
            lines.append("  ".join("-" * w for w in widths))
            for r in rows:
                lines.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)))
            return "\n".join(lines)

        for dataset, rows in rows_by_dataset.items():
            if not rows:
                continue
            (self.tables_dir / f"{dataset}.txt").write_text(render_table(rows) + "\n", encoding="utf-8")

        all_rows = [r for rows in rows_by_dataset.values() for r in rows]
        (self.run_dir / "final_all_datasets.txt").write_text(render_table(all_rows) + "\n", encoding="utf-8")
        self.log(f"WROTE tables/<dataset>.txt ({len([d for d in DATASETS if rows_by_dataset[d]])} datasets) and final_all_datasets.txt")

        (self.run_dir / "final_classification_results.json").write_text(json.dumps(flat_rows, indent=2), encoding="utf-8")
        if flat_rows:
            with open(self.run_dir / "final_classification_results.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
                writer.writeheader()
                writer.writerows(flat_rows)
        self.log("WROTE final_classification_results.json / final_classification_results.csv")


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
        run_dir = REPO_ROOT / "results" / f"classification5_all_datasets_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)

    Sweep(run_dir).run()


if __name__ == "__main__":
    main()
