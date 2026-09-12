#!/usr/bin/env python
"""Durable, resumable driver for the 11-SBERT-variant x 15-dataset x 2-K
topic experiment sweep (--protocol ecrtm_hicot --cv-method palmetto).

Each (model, dataset, k) combination runs in its OWN isolated subprocess
(a fresh `python scripts/run_experiment.py --experiment topic ...`
invocation) - this is what actually releases GPU/CUDA memory between
combinations (a dead process releases its CUDA context unconditionally;
no in-process cleanup can guarantee that the way process exit does) and
lets one crashing/hanging combination never take down the other 329.

Resumable by design: `checkpoint.json` under the run directory records
one entry per (model, dataset, k) key with status "ok"/"error". Re-running
this script with the SAME --run-dir skips every key already marked "ok"
and re-attempts anything missing or marked "error" - safe to interrupt
and restart at any point, never recomputes a completed combination.

Usage:
    python scripts/run_sbert11_sweep.py                      # start a new run
    python scripts/run_sbert11_sweep.py --run-dir results/sbert11_ecrtm_hicot_20260912_181000  # resume

See docs/methodological_notes.md for the scientific-integrity constraints
this driver is built to respect (never substitutes a different HF model,
never silently falls back from Palmetto to local-corpus C_V, never merges
hicot_* datasets with the ordinary ones).
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

# --- the 11 named SBERT+KMeans variants, exactly as specified ---
SBERT_VARIANTS = {
    "sbert_minilm": {"embedder": "sentence-transformers/all-MiniLM-L6-v2"},
    "sbert_mpnet": {"embedder": "sentence-transformers/all-mpnet-base-v2"},
    "sbert_t5": {"embedder": "sentence-transformers/sentence-t5-large"},
    "sbert_e5": {"embedder": "intfloat/e5-large-v2"},
    "sbert_bge": {"embedder": "BAAI/bge-large-en-v1.5"},
    "sbert_gte": {"embedder": "thenlper/gte-large"},
    "sbert_msmpnet": {"embedder": "sentence-transformers/msmarco-mpnet-base-v4"},
    "sbert_msdistilbert": {"embedder": "sentence-transformers/msmarco-distilbert-base-v4"},
    "sbert_distilroberta": {"embedder": "sentence-transformers/all-distilroberta-v1"},
    "sbert_paraphrase": {"embedder": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"},
    "sbert_distilbert": {"embedder": "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"},
}

DATASETS = [
    "20ng", "imdb", "agnews_short", "search_snippets", "stack_overflow",
    "biomedical", "google_news_ts", "google_news_t", "google_news_s", "tweet",
    "hicot_20ng", "hicot_imdb", "hicot_agnews", "hicot_search_snippets", "hicot_google_news",
]
KS = [50, 100]
SEED = 42
PROTOCOL = "ecrtm_hicot"
CV_METHOD = "palmetto"

MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 20
PER_COMBO_TIMEOUT_SECONDS = 3600  # generous: covers a cold HF model download + Palmetto CV for k=100

# All caches/artifacts on RAID - never /home (full disk). REPO_ROOT already
# lives under /raid/.../DeepGenerativeTopicCluster, so its parent is RAID too.
CACHE_ROOT = Path(os.environ.get("VAEBM_CACHE_ROOT", REPO_ROOT.parent / ".cache"))


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def combo_key(model: str, dataset: str, k: int) -> str:
    return f"{model}|{dataset}|{k}"


class Sweep:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.logs_dir = run_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.run_log_path = run_dir / "run.log"
        self.status_path = run_dir / "current_status.txt"
        self.progress_path = run_dir / "progress.txt"
        self.checkpoint_path = run_dir / "checkpoint.json"
        self.sbert_configs_path = run_dir / "sbert11_configs.json"
        self.results_json_path = run_dir / "experiment_results.json"

        if not self.sbert_configs_path.exists():
            self.sbert_configs_path.write_text(json.dumps(SBERT_VARIANTS, indent=2), encoding="utf-8")

        self.checkpoint: dict = {}
        if self.checkpoint_path.exists():
            self.checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))

        self.total = len(SBERT_VARIANTS) * len(DATASETS) * len(KS)
        self.started_at = now_iso()

    # ---------- logging / status files ----------

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
        completed = successful + failed
        return completed, successful, failed

    def write_progress(self, current: dict | None = None) -> None:
        completed, successful, failed = self.counts()
        lines = [
            f"Total: {self.total}",
            f"Completed: {completed}",
            f"Successful: {successful}",
            f"Failed: {failed}",
            f"Remaining: {self.total - completed}",
            f"Started at: {self.started_at}",
            f"Last update: {now_iso()}",
        ]
        if current:
            lines.append(f"Current model: {current.get('model', '')}")
            lines.append(f"Current dataset: {current.get('dataset', '')}")
            lines.append(f"Current K: {current.get('k', '')}")
        self.progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_status(self, model: str, dataset: str, k: int, attempt: int) -> None:
        completed, successful, failed = self.counts()
        text = (
            f"Current model: {model}\n"
            f"Dataset: {dataset}\n"
            f"K: {k}\n"
            f"Attempt: {attempt}\n"
            f"Completed: {completed}/{self.total}\n"
            f"Successful: {successful}\n"
            f"Failed: {failed}\n"
            f"Remaining: {self.total - completed}\n"
            f"Started at: {self.started_at}\n"
            f"Last update: {now_iso()}\n"
        )
        self.status_path.write_text(text, encoding="utf-8")

    # ---------- Palmetto setup ----------

    def ensure_palmetto(self) -> bool:
        self.log("SETUP_START palmetto/wikipedia index check")
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from setup_palmetto import ensure_palmetto_ready  # noqa: E402

        for attempt in (1, 2):
            try:
                ready = ensure_palmetto_ready()
            except Exception as exc:  # noqa: BLE001
                self.log(f"SETUP_ERROR palmetto attempt={attempt} error={exc!r}")
                ready = False
            if ready:
                self.log("SETUP_OK palmetto/wikipedia index ready")
                return True
            if attempt == 1:
                self.log("SETUP_RETRY palmetto setup failed, retrying once")
                time.sleep(10)
        self.log("SETUP_FAILED palmetto/wikipedia index could not be prepared after 2 attempts - "
                  "this is a genuine infrastructure blocker for --cv-method palmetto, stopping sweep launch")
        return False

    # ---------- one (model, dataset, k) combination ----------

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
        env["PIP_CACHE_DIR"] = str(CACHE_ROOT / "pip")
        for p in (env["HF_HOME"], env["SENTENCE_TRANSFORMERS_HOME"], env["TORCH_HOME"], env["PIP_CACHE_DIR"]):
            Path(p).mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "-u", str(REPO_ROOT / "scripts" / "run_experiment.py"),
            "--experiment", "topic",
            "--models", model,
            "--datasets", dataset,
            "--k", str(k),
            "--seed", str(SEED),
            "--protocol", PROTOCOL,
            "--cv-method", CV_METHOD,
            "--sbert-configs", str(self.sbert_configs_path),
        ]

        last_error = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.write_status(model, dataset, k, attempt)
            self.log(f"START model={model} dataset={dataset} k={k} attempt={attempt} progress={progress_idx}/{self.total}")
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
                    outcome = self._read_result(model, dataset, k)
                    if outcome is None:
                        last_error = "subprocess exited 0 but no matching result row found in experiment_results.json"
                    elif outcome["status"] == "ok":
                        self.checkpoint[key] = {
                            "status": "ok", "attempts": attempt, "runtime": outcome["runtime_seconds"],
                            "cv": outcome["cv"], "purity": outcome["purity"], "nmi": outcome["nmi"], "td": outcome["td"],
                            "cv_source": outcome["cv_source"], "timestamp": now_iso(),
                        }
                        self.save_checkpoint()
                        self.write_progress()
                        self.log(f"OK model={model} dataset={dataset} k={k} runtime={runtime:.0f}s "
                                 f"cv_source={outcome['cv_source']} progress={progress_idx}/{self.total}")
                        return
                    else:
                        last_error = (outcome.get("error") or "")[:500]
            except subprocess.TimeoutExpired:
                runtime = time.perf_counter() - start
                last_error = f"timeout after {PER_COMBO_TIMEOUT_SECONDS}s"
                with open(log_file, "a", encoding="utf-8") as lf:
                    lf.write(f"\n=== attempt {attempt} TIMEOUT after {PER_COMBO_TIMEOUT_SECONDS}s ===\n")
            except Exception as exc:  # noqa: BLE001 - the driver itself must never die on one combo
                last_error = f"driver exception: {exc!r}\n{traceback.format_exc(limit=3)}"

            is_oom = "out of memory" in last_error.lower() or "cuda out of memory" in last_error.lower()
            if is_oom:
                self.log(f"ERROR model={model} dataset={dataset} k={k} attempt={attempt} error=\"CUDA out of memory\"")
            else:
                self.log(f"ERROR model={model} dataset={dataset} k={k} attempt={attempt} "
                         f"error=\"{last_error[:200].splitlines()[0] if last_error else last_error}\"")

            if attempt < MAX_ATTEMPTS:
                self.log(f"RETRY model={model} dataset={dataset} k={k} attempt={attempt + 1}")
                time.sleep(RETRY_BACKOFF_SECONDS)

        # exhausted all attempts
        self.checkpoint[key] = {
            "status": "error", "attempts": MAX_ATTEMPTS, "error": last_error[:2000], "timestamp": now_iso(),
        }
        self.save_checkpoint()
        self.write_progress()
        self.log(f"ERROR model={model} dataset={dataset} k={k} FINAL after {MAX_ATTEMPTS} attempts, giving up "
                 f"progress={progress_idx}/{self.total}")

    def _read_result(self, model: str, dataset: str, k: int) -> dict | None:
        if not self.results_json_path.exists():
            return None
        rows = json.loads(self.results_json_path.read_text(encoding="utf-8"))
        for row in reversed(rows):  # last-written match for this exact combo
            if row.get("model") == model and row.get("dataset") == dataset and row.get("k") == k and row.get("seed") == SEED:
                return row
        return None

    # ---------- main loop ----------

    def run(self) -> None:
        run_dir_head_node = str(self.run_dir).replace("/raid/brunoguilherme", "/shared/dgx-raid/brunoguilherme")
        self.log(f"SWEEP_START run_dir={self.run_dir}")
        self.log(f"MONITOR tail -f {run_dir_head_node}/run.log")
        self.log(f"MONITOR cat {run_dir_head_node}/current_status.txt")
        self.log(f"MONITOR cat {run_dir_head_node}/progress.txt")

        if not self.ensure_palmetto():
            self.write_progress()
            self.log("SWEEP_ABORTED palmetto/wikipedia setup failed - see SETUP_FAILED above")
            return

        idx = 0
        for model, cfg in SBERT_VARIANTS.items():
            for dataset in DATASETS:
                for k in KS:
                    idx += 1
                    self.run_one(model, dataset, k, idx)

        self.write_progress()
        self.log("SWEEP_LOOP_DONE all 330 combinations attempted - building final tables")
        self.build_final_tables()
        completed, successful, failed = self.counts()
        self.log(f"SWEEP_COMPLETE total={self.total} successful={successful} failed={failed}")

    def build_final_tables(self) -> None:
        from vaebm_benchmark.experiment.report import (
            render_latex_table_for_k, render_table_for_k, results_to_rows,
        )
        from vaebm_benchmark.experiment.runner import ExperimentResult

        rows_by_key = {}
        if self.results_json_path.exists():
            for row in json.loads(self.results_json_path.read_text(encoding="utf-8")):
                rows_by_key[(row["model"], row["dataset"], row["k"], row["seed"])] = row

        results: list[ExperimentResult] = []
        for model in SBERT_VARIANTS:
            for dataset in DATASETS:
                for k in KS:
                    ck = self.checkpoint.get(combo_key(model, dataset, k))
                    if ck is None:
                        continue  # never attempted (shouldn't happen after a full run) -> renders as N/A
                    row = rows_by_key.get((model, dataset, k, SEED))
                    if row is not None:
                        results.append(ExperimentResult(**{k2: v for k2, v in row.items() if k2 not in ("topics_energy", "topics_freq")}))
                    else:
                        # subprocess crashed/timed out before writing a result row - synthesize
                        # an ERROR row from the checkpoint so it still renders as ERROR, not N/A.
                        results.append(ExperimentResult(
                            model=model, dataset=dataset, k=k, cv=None, purity=None, nmi=None, td=None,
                            seed=SEED, runtime_seconds=0.0, status="error", error=ck.get("error", "not run"),
                            evaluation_protocol=PROTOCOL, cv_source="palmetto_wikipedia",
                        ))

        for k in KS:
            txt = render_table_for_k(results, k)
            (self.run_dir / f"final_table_k{k}.txt").write_text(txt + "\n", encoding="utf-8")
            tex = render_latex_table_for_k(results, k)
            (self.run_dir / f"final_table_k{k}.tex").write_text(tex + "\n", encoding="utf-8")
            self.log(f"WROTE final_table_k{k}.txt / .tex")

        rows = results_to_rows(results)
        (self.run_dir / "final_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        import csv
        with open(self.run_dir / "final_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        self.log("WROTE final_results.json / final_results.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None,
                         help="Resume an existing run directory. Omit to start a new timestamped one under results/.")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        if not run_dir.exists():
            raise SystemExit(f"--run-dir {run_dir} does not exist - omit --run-dir to start a new run")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = REPO_ROOT / "results" / f"sbert11_ecrtm_hicot_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)

    sweep = Sweep(run_dir)
    sweep.run()


if __name__ == "__main__":
    main()
