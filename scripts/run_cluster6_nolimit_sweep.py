#!/usr/bin/env python
"""Durable, resumable driver for the 6-model x 26-dataset cluster-
experiment sweep on this project's SECOND compute environment (SMART
SENSE lab / viper08, 2026-09-14) - unlike scripts/run_cluster7_sweep.py
(FutureLab, 30-min-per-combo cap), this environment has NO wall-clock
budget: every combination runs to natural completion, however long that
takes. HiCOT itself is also configured for full convergence here (see
launch script's own VAEBM_HICOT_SINKHORN_MAX_ITER/
VAEBM_HICOT_MAX_FIT_SECONDS env vars), not the FutureLab-tuned speed
defaults.

Mirrors run_cluster7_sweep.py's checkpoint/resume/live-log architecture
(isolated subprocess per (model, dataset), JSON checkpoint, run.log/
current_status.txt/progress.txt) since this environment isn't actively
monitored the way FutureLab was - the checkpoint is what makes it safe
to check back on later (or resume after any interruption) without
losing or recomputing anything.

Usage:
    python scripts/run_cluster6_nolimit_sweep.py                      # start a new run
    python scripts/run_cluster6_nolimit_sweep.py --run-dir results/cluster6_all_datasets_20260914_190000  # resume
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

# All 4 named SBERT variants (no bertopic - never requested for this run).
MODELS = ["sbert_gte", "sbert_bge", "sbert_mpnet", "sbert_minilm", "fastopic", "hicot"]

DATASETS = [
    "20ng", "imdb", "agnews_short", "search_snippets", "stack_overflow",
    "biomedical", "google_news_ts", "google_news_t", "google_news_s", "tweet",
    "hicot_20ng", "hicot_imdb", "hicot_agnews", "hicot_search_snippets", "hicot_google_news",
    "agnews_full", "banking77", "bbc_news", "dblp", "dbpedia_14",
    "m10", "pascal_flickr", "tweet_eval_emotion", "tweet_eval_sentiment",
    "yahoo_answers_topics", "20ng_s2wtm",
]
SEED = 42

MAX_ATTEMPTS = 3  # genuine transient failures only (crash, flaky download) - no timeout concept here
RETRY_BACKOFF_SECONDS = 20

CACHE_ROOT = Path(os.environ.get("VAEBM_CACHE_ROOT", "/ssd/bruno.gomes/.cache"))


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
        self.results_json_path = run_dir / "cluster" / "cluster_results.json"

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
        # HiCOT full-convergence config for this environment - see this
        # module's own docstring. Harmless for every other model (unread).
        env.setdefault("VAEBM_HICOT_SINKHORN_MAX_ITER", "500")
        env.setdefault("VAEBM_HICOT_MAX_FIT_SECONDS", "0")  # "0" -> early-stop disabled entirely

        cmd = [
            sys.executable, "-u", str(REPO_ROOT / "scripts" / "run_experiment.py"),
            "--experiment", "cluster", "--models", model, "--datasets", dataset, "--seed", str(SEED),
        ]

        last_error = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.write_status(model, dataset, attempt)
            self.log(f"START model={model} dataset={dataset} attempt={attempt} progress={progress_idx}/{self.total}")
            start = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd, cwd=str(REPO_ROOT), env=env,
                    capture_output=True, text=True, timeout=None,  # no wall-clock limit in this environment
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
                        last_error = "subprocess exited 0 but no matching result row found in cluster_results.json"
                    elif outcome["status"] == "ok":
                        self.checkpoint[key] = {
                            "status": "ok", "attempts": attempt, "runtime": outcome["runtime_seconds"],
                            "acc": outcome["acc"], "nmi": outcome["nmi"], "purity": outcome["purity"],
                            "timestamp": now_iso(),
                        }
                        self.save_checkpoint()
                        self.write_progress()
                        self.log(f"OK model={model} dataset={dataset} runtime={runtime:.0f}s "
                                 f"acc={outcome['acc']} nmi={outcome['nmi']} progress={progress_idx}/{self.total}")
                        return
                    else:
                        last_error = (outcome.get("error") or "")[:500]
            except Exception as exc:  # noqa: BLE001 - the driver itself must never die on one combo
                last_error = f"driver exception: {exc!r}\n{traceback.format_exc(limit=3)}"

            is_oom = "out of memory" in last_error.lower()
            tag = '"CUDA out of memory"' if is_oom else f'"{last_error[:200].splitlines()[0] if last_error else last_error}"'
            self.log(f"ERROR model={model} dataset={dataset} attempt={attempt} error={tag}")
            if attempt < MAX_ATTEMPTS:
                self.log(f"RETRY model={model} dataset={dataset} attempt={attempt + 1}")
                time.sleep(RETRY_BACKOFF_SECONDS)

        self.checkpoint[key] = {"status": "error", "attempts": MAX_ATTEMPTS, "error": last_error[:2000], "timestamp": now_iso()}
        self.save_checkpoint()
        self.write_progress()
        self.log(f"ERROR model={model} dataset={dataset} FINAL after {MAX_ATTEMPTS} attempts, giving up "
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
        self.log(f"SWEEP_START run_dir={self.run_dir}")
        self.log(f"MONITOR tail -f {self.run_dir}/run.log")
        self.log(f"MONITOR cat {self.run_dir}/current_status.txt")
        self.log(f"MONITOR cat {self.run_dir}/progress.txt")

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
        from vaebm_benchmark.experiment.cluster_report import (
            cluster_results_to_rows, render_cluster_latex_table, render_cluster_table,
        )
        from vaebm_benchmark.experiment.cluster_runner import ClusterResult

        rows_by_key = {}
        if self.results_json_path.exists():
            for row in json.loads(self.results_json_path.read_text(encoding="utf-8")):
                rows_by_key[(row["model"], row["dataset"], row["seed"])] = row

        results: list[ClusterResult] = []
        for model in MODELS:
            for dataset in DATASETS:
                ck = self.checkpoint.get(combo_key(model, dataset))
                if ck is None:
                    continue
                row = rows_by_key.get((model, dataset, SEED))
                if row is not None:
                    fields = {k: v for k, v in row.items() if k != "experiment_type"}
                    results.append(ClusterResult(experiment=row["experiment_type"], **fields))
                else:
                    results.append(ClusterResult(
                        experiment="cluster", model=model, dataset=dataset, seed=SEED,
                        requested_k=0, actual_k=None, num_classes=0,
                        representation_source="", assignment_source="",
                        runtime_seconds=0.0, status="error", error=ck.get("error", "not run"),
                    ))

        for dataset in DATASETS:
            per_dataset = [r for r in results if r.dataset == dataset]
            if not per_dataset:
                continue
            txt = render_cluster_table(per_dataset, datasets_per_block=1)
            (self.tables_dir / f"{dataset}.txt").write_text(txt + "\n", encoding="utf-8")
            tex = render_cluster_latex_table(per_dataset)
            (self.tables_dir / f"{dataset}.tex").write_text(tex + "\n", encoding="utf-8")

        combined_txt = render_cluster_table(results, datasets_per_block=4)
        (self.run_dir / "final_all_datasets.txt").write_text(combined_txt + "\n", encoding="utf-8")
        self.log(f"WROTE tables/<dataset>.txt/.tex ({len(DATASETS)} datasets) and final_all_datasets.txt")

        rows = cluster_results_to_rows(results)
        for row in rows:
            row["k"] = row.pop("requested_k")
        (self.run_dir / "final_cluster_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        import csv
        with open(self.run_dir / "final_cluster_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        self.log("WROTE final_cluster_results.json / final_cluster_results.csv")


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
        run_dir = REPO_ROOT / "results" / f"cluster6_all_datasets_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)

    Sweep(run_dir).run()


if __name__ == "__main__":
    main()
