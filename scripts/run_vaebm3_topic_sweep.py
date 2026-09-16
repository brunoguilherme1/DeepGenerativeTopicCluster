#!/usr/bin/env python
"""Durable, resumable driver for the 3-VAE-BM-variant x 10-dataset x
2-K topic-quality sweep (models: vaebm, vaebm_poe, vaebm_dec - see
models/vaebm_poe.py, models/vaebm_dec.py). Default embedder
(all-MiniLM-L6-v2 - GTE-large was tried and reverted, 2026-09-16: ~15x
more parameters made per-combo embedding time impractical under this
sweep's own 30-minute cap).

Datasets: HiCOT's own official 5-dataset benchmark suite plus each
one's "normal" (non-HiCOT-artifact) counterpart - user-specified
pairing, 2026-09-16: hicot_20ng/20ng, hicot_imdb/imdb,
hicot_agnews/agnews_short, hicot_search_snippets/search_snippets,
hicot_google_news/google_news_ts. K in {50, 100} (ECRTM/HiCOT's own
Table 2/3 convention, not dataset-derived - see experiment/runner.py's
own docstring on why `--k` here differs from the cluster experiment's
num_classes-derived K).

Combo grain is (model, dataset, k) - NOT (model, dataset) like
run_cluster7_sweep.py/run_vaebm3_sweep.py, since the topic experiment's
own K is a real, independent sweep axis here. Otherwise mirrors those
scripts' architecture exactly (isolated subprocess per combo for real
GPU-memory release + crash isolation, JSON checkpoint for safe resume,
run.log/current_status.txt/progress.txt for live monitoring, uniform
30-minute per-combo timeout with skip-not-retry-on-timeout).

Usage:
    python scripts/run_vaebm3_topic_sweep.py                      # start a new run
    python scripts/run_vaebm3_topic_sweep.py --run-dir results/vaebm3_topic_20260916_120000  # resume
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

MODELS = ["vaebm", "vaebm_poe", "vaebm_dec"]
DATASETS = [
    "hicot_20ng", "20ng",
    "hicot_imdb", "imdb",
    "hicot_agnews", "agnews_short",
    "hicot_search_snippets", "search_snippets",
    "hicot_google_news", "google_news_ts",
]
KS = [50, 100]
SEED = 42

MAX_ATTEMPTS = 1  # same policy as run_cluster7_sweep.py - a timeout is not retried
RETRY_BACKOFF_SECONDS = 20
PER_COMBO_TIMEOUT_SECONDS = 1800  # 30 min external safety net; vaebm_poe/vaebm_dec also
                                   # internally early-stop at 1200s (20min) via their own
                                   # VAEBM_POE_MAX_FIT_SECONDS/VAEBM_DEC_MAX_FIT_SECONDS env vars

CACHE_ROOT = Path(os.environ.get("VAEBM_CACHE_ROOT", REPO_ROOT.parent / ".cache"))


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def combo_key(model: str, dataset: str, k: int) -> str:
    return f"{model}|{dataset}|{k}"


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
        # run_experiment.py's topic experiment writes directly under
        # <VAEBM_RESULTS_DIR>/experiment_results.json (RESULTS_DIR root,
        # unlike cluster/classification's own subdirectories).
        self.results_json_path = run_dir / "experiment_results.json"

        self.checkpoint: dict = {}
        if self.checkpoint_path.exists():
            self.checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))

        self.total = len(MODELS) * len(DATASETS) * len(KS)
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

    def write_status(self, model: str, dataset: str, k: int, attempt: int) -> None:
        completed, successful, failed = self.counts()
        text = (
            f"Current model: {model}\nDataset: {dataset}\nK: {k}\nAttempt: {attempt}\n"
            f"Completed: {completed}/{self.total}\nSuccessful: {successful}\nFailed: {failed}\n"
            f"Remaining: {self.total - completed}\nStarted at: {self.started_at}\nLast update: {now_iso()}\n"
        )
        self.status_path.write_text(text, encoding="utf-8")

    def run_one(self, model: str, dataset: str, k: int, progress_idx: int) -> None:
        key = combo_key(model, dataset, k)
        existing = self.checkpoint.get(key)
        if existing and existing.get("status") == "ok":
            self.log(f"SKIP model={model} dataset={dataset} k={k} progress={progress_idx}/{self.total} "
                     f"(already completed at {existing.get('timestamp')})")
            return

        log_file = self.logs_dir / f"{model}__{dataset}__k{k}.log"
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["VAEBM_RESULTS_DIR"] = str(self.run_dir)
        env["HF_HOME"] = str(CACHE_ROOT / "huggingface")
        env["TRANSFORMERS_CACHE"] = str(CACHE_ROOT / "huggingface" / "transformers")
        env["SENTENCE_TRANSFORMERS_HOME"] = str(CACHE_ROOT / "sentence_transformers")
        env["TORCH_HOME"] = str(CACHE_ROOT / "torch")
        for p in (env["HF_HOME"], env["SENTENCE_TRANSFORMERS_HOME"], env["TORCH_HOME"]):
            Path(p).mkdir(parents=True, exist_ok=True)
        env.setdefault("VAEBM_POE_MAX_FIT_SECONDS", "1200")
        env.setdefault("VAEBM_DEC_MAX_FIT_SECONDS", "1200")
        # VAEBM_EMBEDDER left unset - all-MiniLM-L6-v2 default (see
        # experiment/scientific_models.py::_vaebm_embedder)

        cmd = [
            sys.executable, "-u", str(REPO_ROOT / "scripts" / "run_experiment.py"),
            "--experiment", "topic", "--models", model, "--datasets", dataset,
            "--k", str(k), "--seed", str(SEED),
        ]

        last_error = ""
        final_attempt = MAX_ATTEMPTS
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.write_status(model, dataset, k, attempt)
            self.log(f"START model={model} dataset={dataset} k={k} attempt={attempt} progress={progress_idx}/{self.total}")
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
                    outcome = self._read_result(model, dataset, k)
                    if outcome is None:
                        last_error = "subprocess exited 0 but no matching result row found in experiment_results.json"
                    elif outcome["status"] == "ok":
                        self.checkpoint[key] = {
                            "status": "ok", "attempts": attempt, "runtime": outcome["runtime_seconds"],
                            "cv": outcome["cv"], "purity": outcome["purity"], "nmi": outcome["nmi"], "td": outcome["td"],
                            "timestamp": now_iso(),
                        }
                        self.save_checkpoint()
                        self.write_progress()
                        self.log(f"OK model={model} dataset={dataset} k={k} runtime={runtime:.0f}s "
                                 f"cv={outcome['cv']} purity={outcome['purity']} nmi={outcome['nmi']} td={outcome['td']} "
                                 f"progress={progress_idx}/{self.total}")
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
            self.log(f"ERROR model={model} dataset={dataset} k={k} attempt={attempt} error={tag}")

            if timed_out:
                final_attempt = attempt
                break
            if attempt < MAX_ATTEMPTS:
                self.log(f"RETRY model={model} dataset={dataset} k={k} attempt={attempt + 1}")
                time.sleep(RETRY_BACKOFF_SECONDS)
                final_attempt = attempt + 1

        self.checkpoint[key] = {"status": "error", "attempts": final_attempt, "error": last_error[:2000], "timestamp": now_iso()}
        self.save_checkpoint()
        self.write_progress()
        self.log(f"ERROR model={model} dataset={dataset} k={k} FINAL after {final_attempt} attempt(s), giving up "
                 f"progress={progress_idx}/{self.total}")

    def _read_result(self, model: str, dataset: str, k: int) -> dict | None:
        if not self.results_json_path.exists():
            return None
        rows = json.loads(self.results_json_path.read_text(encoding="utf-8"))
        for row in reversed(rows):
            if row.get("model") == model and row.get("dataset") == dataset and row.get("k") == k and row.get("seed") == SEED:
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
                for k in KS:
                    idx += 1
                    self.run_one(model, dataset, k, idx)

        self.write_progress()
        self.log("SWEEP_LOOP_DONE all combinations attempted - building final tables")
        self.build_final_outputs()
        completed, successful, failed = self.counts()
        self.log(f"SWEEP_COMPLETE total={self.total} successful={successful} failed={failed}")

    def build_final_outputs(self) -> None:
        from vaebm_benchmark.experiment.report import render_latex_table_for_k, render_table_for_k, results_to_rows
        from vaebm_benchmark.experiment.runner import ExperimentResult

        rows_by_key = {}
        if self.results_json_path.exists():
            for row in json.loads(self.results_json_path.read_text(encoding="utf-8")):
                rows_by_key[(row["model"], row["dataset"], row["k"], row["seed"])] = row

        results: list[ExperimentResult] = []
        for model in MODELS:
            for dataset in DATASETS:
                for k in KS:
                    ck = self.checkpoint.get(combo_key(model, dataset, k))
                    if ck is None:
                        continue
                    row = rows_by_key.get((model, dataset, k, SEED))
                    if row is not None:
                        fields = {kk: v for kk, v in row.items() if kk not in ("topics_energy", "topics_freq")}
                        results.append(ExperimentResult(**fields))
                    else:
                        results.append(ExperimentResult(
                            model=model, dataset=dataset, k=k, cv=None, purity=None, nmi=None, td=None,
                            seed=SEED, runtime_seconds=0.0, status="error", error=ck.get("error", "not run"),
                        ))

        for k in KS:
            txt = render_table_for_k(results, k)
            (self.tables_dir / f"k{k}.txt").write_text(txt + "\n", encoding="utf-8")
            tex = render_latex_table_for_k(results, k)
            (self.tables_dir / f"k{k}.tex").write_text(tex + "\n", encoding="utf-8")
        self.log(f"WROTE tables/k<k>.txt/.tex ({len(KS)} K values)")

        rows = results_to_rows(results)
        (self.run_dir / "final_experiment_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        with open(self.run_dir / "final_experiment_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        self.log("WROTE final_experiment_results.json / final_experiment_results.csv")


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
        run_dir = REPO_ROOT / "results" / f"vaebm3_topic_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)

    Sweep(run_dir).run()


if __name__ == "__main__":
    main()
