#!/usr/bin/env python
"""labuai counterpart of run_vaebm3_topic_sweep.py (FutureLab): same
3-VAE-BM-variant x 10-dataset x 2-K topic-quality sweep, but forces the
GTE-large embedder, has NO wall-clock limit (internal or external), and
runs several (model, dataset, k) combos concurrently across this
workstation's GPUs. See run_vaebm3_cluster_sweep_labuai.py's own
docstring for the full fault-tolerance/backoff/subprocess-isolation
rationale - identical policy here, just applied to the topic experiment's
(model, dataset, k) combo grain (K is a real, independent sweep axis
here, same reason run_vaebm3_topic_sweep.py's own docstring gives).

Usage:
    python scripts/run_vaebm3_topic_sweep_labuai.py
    python scripts/run_vaebm3_topic_sweep_labuai.py --run-dir results/vaebm3_topic_gte_labuai_20260917_010000
    python scripts/run_vaebm3_topic_sweep_labuai.py --workers-per-gpu 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
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
EMBEDDER = "thenlper/gte-large"

MAX_ATTEMPTS = 5
RETRY_BACKOFF_SECONDS = 30
OOM_RETRY_BACKOFF_SECONDS = 90

CACHE_ROOT = Path(os.environ.get("VAEBM_CACHE_ROOT", "/ssd/bruno.gomes/.cache"))


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def combo_key(model: str, dataset: str, k: int) -> str:
    return f"{model}|{dataset}|{k}"


def detect_num_gpus() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15)
        return max(1, len([line for line in out.stdout.splitlines() if line.startswith("GPU ")]))
    except Exception:  # noqa: BLE001
        return 1


class Sweep:
    def __init__(self, run_dir: Path, num_gpus: int, workers_per_gpu: int):
        self.run_dir = run_dir
        self.num_gpus = num_gpus
        self.workers_per_gpu = workers_per_gpu
        self.logs_dir = run_dir / "logs"
        self.tables_dir = run_dir / "tables"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.run_log_path = run_dir / "run.log"
        self.status_path = run_dir / "current_status.txt"
        self.progress_path = run_dir / "progress.txt"
        self.checkpoint_path = run_dir / "checkpoint.json"
        self.results_json_path = run_dir / "experiment_results.json"

        self.checkpoint: dict = {}
        if self.checkpoint_path.exists():
            self.checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))

        self.total = len(MODELS) * len(DATASETS) * len(KS)
        self.started_at = now_iso()

        self._lock = threading.Lock()
        self._active: dict[int, str] = {}

    def log(self, msg: str) -> None:
        line = f"[{now_iso()}] {msg}"
        with self._lock:
            with open(self.run_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        print(line, flush=True)

    def save_checkpoint(self) -> None:
        with self._lock:
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
        with self._lock:
            self.progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_status(self, slot: int, gpu_index: int, model: str | None, dataset: str | None, k: int | None, attempt: int | None) -> None:
        with self._lock:
            if model is None:
                self._active.pop(slot, None)
            else:
                self._active[slot] = f"GPU {gpu_index} (slot {slot}): model={model} dataset={dataset} k={k} attempt={attempt}"
            completed, successful, failed = self.counts()
            lines = [self._active[s] for s in sorted(self._active)]
            text = (
                "\n".join(lines) + ("\n\n" if lines else "") +
                f"Completed: {completed}/{self.total}\nSuccessful: {successful}\nFailed: {failed}\n"
                f"Remaining: {self.total - completed}\nStarted at: {self.started_at}\nLast update: {now_iso()}\n"
            )
            self.status_path.write_text(text, encoding="utf-8")

    def run_one(self, model: str, dataset: str, k: int, progress_idx: int, slot: int, gpu_index: int) -> None:
        key = combo_key(model, dataset, k)
        with self._lock:
            existing = self.checkpoint.get(key)
        if existing and existing.get("status") == "ok":
            self.log(f"SKIP model={model} dataset={dataset} k={k} progress={progress_idx}/{self.total} "
                     f"(already completed at {existing.get('timestamp')})")
            return

        log_file = self.logs_dir / f"{model}__{dataset}__k{k}.log"
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["VAEBM_RESULTS_DIR"] = str(self.run_dir)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        env["HF_HOME"] = str(CACHE_ROOT / "huggingface")
        env["TRANSFORMERS_CACHE"] = str(CACHE_ROOT / "huggingface" / "transformers")
        env["SENTENCE_TRANSFORMERS_HOME"] = str(CACHE_ROOT / "sentence_transformers")
        env["TORCH_HOME"] = str(CACHE_ROOT / "torch")
        for p in (env["HF_HOME"], env["SENTENCE_TRANSFORMERS_HOME"], env["TORCH_HOME"]):
            Path(p).mkdir(parents=True, exist_ok=True)
        env["VAEBM_EMBEDDER"] = EMBEDDER
        # See run_vaebm3_cluster_sweep_labuai.py's own comment: TF's default
        # greedy allocator grabs ~9GB/process on an 11GB card regardless of
        # actual need, guaranteeing any 2 co-located workers collide.
        env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        env["VAEBM_POE_MAX_FIT_SECONDS"] = "0"
        env["VAEBM_DEC_MAX_FIT_SECONDS"] = "0"

        cmd = [
            sys.executable, "-u", str(REPO_ROOT / "scripts" / "run_experiment.py"),
            "--experiment", "topic", "--models", model, "--datasets", dataset,
            "--k", str(k), "--seed", str(SEED),
        ]

        last_error = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.write_status(slot, gpu_index, model, dataset, k, attempt)
            self.log(f"START model={model} dataset={dataset} k={k} attempt={attempt} gpu={gpu_index} progress={progress_idx}/{self.total}")
            start = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd, cwd=str(REPO_ROOT), env=env,
                    capture_output=True, text=True, timeout=None,
                )
                runtime = time.perf_counter() - start
                with self._lock:
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
                        with self._lock:
                            self.checkpoint[key] = {
                                "status": "ok", "attempts": attempt, "runtime": outcome["runtime_seconds"],
                                "cv": outcome["cv"], "purity": outcome["purity"], "nmi": outcome["nmi"], "td": outcome["td"],
                                "embedder": EMBEDDER, "timestamp": now_iso(),
                            }
                        self.save_checkpoint()
                        self.write_progress()
                        self.log(f"OK model={model} dataset={dataset} k={k} runtime={runtime:.0f}s gpu={gpu_index} "
                                 f"cv={outcome['cv']} purity={outcome['purity']} nmi={outcome['nmi']} td={outcome['td']} "
                                 f"progress={progress_idx}/{self.total}")
                        self.write_status(slot, gpu_index, None, None, None, None)
                        return
                    else:
                        last_error = (outcome.get("error") or "")[:500]
            except Exception as exc:  # noqa: BLE001
                last_error = f"driver exception: {exc!r}\n{traceback.format_exc(limit=3)}"

            is_oom = any(s in last_error.lower() for s in ("out of memory", "cuda error", "failed to allocate memory", "exited -9"))
            tag = '"CUDA out of memory"' if is_oom else f'"{last_error[:200].splitlines()[0] if last_error else last_error}"'
            self.log(f"ERROR model={model} dataset={dataset} k={k} attempt={attempt} gpu={gpu_index} error={tag}")
            if attempt < MAX_ATTEMPTS:
                backoff = OOM_RETRY_BACKOFF_SECONDS if is_oom else RETRY_BACKOFF_SECONDS
                self.log(f"RETRY model={model} dataset={dataset} k={k} attempt={attempt + 1} (backoff={backoff}s, oom={is_oom})")
                time.sleep(backoff)

        with self._lock:
            self.checkpoint[key] = {"status": "error", "attempts": MAX_ATTEMPTS, "error": last_error[:2000],
                                     "embedder": EMBEDDER, "timestamp": now_iso()}
        self.save_checkpoint()
        self.write_progress()
        self.log(f"ERROR model={model} dataset={dataset} k={k} FINAL after {MAX_ATTEMPTS} attempts, giving up "
                 f"progress={progress_idx}/{self.total}")
        self.write_status(slot, gpu_index, None, None, None, None)

    def _read_result(self, model: str, dataset: str, k: int) -> dict | None:
        if not self.results_json_path.exists():
            return None
        rows = json.loads(self.results_json_path.read_text(encoding="utf-8"))
        for row in reversed(rows):
            if row.get("model") == model and row.get("dataset") == dataset and row.get("k") == k and row.get("seed") == SEED:
                return row
        return None

    def run(self) -> None:
        total_workers = self.num_gpus * self.workers_per_gpu
        self.log(f"SWEEP_START run_dir={self.run_dir} embedder={EMBEDDER} num_gpus={self.num_gpus} "
                 f"workers_per_gpu={self.workers_per_gpu} total_workers={total_workers}")
        self.log(f"MONITOR tail -f {self.run_dir}/run.log")
        self.log(f"MONITOR cat {self.run_dir}/current_status.txt")
        self.log(f"MONITOR cat {self.run_dir}/progress.txt")

        work_q: queue.Queue = queue.Queue()
        idx = 0
        for model in MODELS:
            for dataset in DATASETS:
                for k in KS:
                    idx += 1
                    key = combo_key(model, dataset, k)
                    if self.checkpoint.get(key, {}).get("status") == "ok":
                        self.log(f"SKIP model={model} dataset={dataset} k={k} progress={idx}/{self.total} "
                                 f"(already completed at {self.checkpoint[key].get('timestamp')})")
                        continue
                    work_q.put((model, dataset, k, idx))

        def worker(slot: int, gpu_index: int) -> None:
            while True:
                try:
                    model, dataset, k, progress_idx = work_q.get_nowait()
                except queue.Empty:
                    return
                self.run_one(model, dataset, k, progress_idx, slot, gpu_index)
                work_q.task_done()

        threads = [
            threading.Thread(target=worker, args=(slot, slot % self.num_gpus), name=f"slot{slot}")
            for slot in range(total_workers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

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
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    args = parser.parse_args()

    num_gpus = args.num_gpus if args.num_gpus is not None else detect_num_gpus()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        if not run_dir.exists():
            raise SystemExit(f"--run-dir {run_dir} does not exist - omit --run-dir to start a new run")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = REPO_ROOT / "results" / f"vaebm3_topic_gte_labuai_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)

    Sweep(run_dir, num_gpus, args.workers_per_gpu).run()


if __name__ == "__main__":
    main()
