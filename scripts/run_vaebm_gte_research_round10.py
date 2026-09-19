#!/usr/bin/env python
"""Round 10 ("LET THE NETWORK LEARN A BIT") of the "beat HiCOT K=50 via
GTE-mimicking VAE-BM" autonomous research pass (2026-09-19,
user-authorized) - see docs/vaebm_mimic_gte_research_log.md.

Rounds 1-7 established: fully-frozen vaebm + freq-words beats HiCOT on
2/5 datasets (google_news, search_snippets). The remaining 3 - 20ng
(NMI 0.580 vs 0.583 target, off by 0.003), agnews (NMI ~0.37 vs 0.412),
imdb (Cv ~0.39 vs 0.404) - never budged under ANY fully-frozen or
fully-unfrozen (alpha>0) config tried. This round tests a genuine middle
ground never tried before: model="vaebm_ckpt" (identical Encoder/VAEBM
architecture to plain "vaebm", but with a custom per-epoch training loop
and now, as of this research pass, an unfreeze_after_epoch schedule) -
start FROZEN (guaranteed at least as good as the Round 7 baseline for
however many epochs it stays frozen), then unfreeze the embedding branch
for the remaining epochs at a very low learning rate, and use ORACLE
checkpoint selection (VAEBM_TOPIC_ORACLE_LABELS=1 - true labels, used
ONLY to rank already-computed epochs, never in the loss/gradient - see
models/vaebm_ckpt.py's own docstring) with oracle_metric="nmi_purity_sum"
so the run can never end up worse than staying frozen: if letting it
learn only hurts, the oracle simply keeps the frozen epoch's weights.

Three schedules tested, on the 3 still-blocked hicot_* datasets only
(google_news/search_snippets are already solved - not re-tested):
  - N2:  unfreeze_after_epoch=2,  post_unfreeze_lr=1e-6, epochs=15
  - N5:  unfreeze_after_epoch=5,  post_unfreeze_lr=1e-6, epochs=15
  - N10: unfreeze_after_epoch=10, post_unfreeze_lr=1e-6, epochs=15

Combo grain is (config_name, dataset). Otherwise mirrors
run_vaebm3_topic_sweep.py's architecture (isolated subprocess per combo,
JSON checkpoint/resume, run.log/current_status.txt/progress.txt, 30-min
per-combo timeout - raised where needed since 15 epochs takes longer
than this research pass's usual 1-epoch runs).

Usage:
    python scripts/run_vaebm_gte_research_round10.py
    python scripts/run_vaebm_gte_research_round10.py --run-dir results/vaebm_gte_r10_20260919_190000
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

MODEL = "vaebm_ckpt"
K = 50
DATASETS = ["hicot_20ng", "hicot_agnews", "hicot_imdb"]
SEED = 42

CONFIGS = {
    "N2_unfreeze2": {"VAEBM_CKPT_UNFREEZE_AFTER_EPOCH": "2"},
    "N5_unfreeze5": {"VAEBM_CKPT_UNFREEZE_AFTER_EPOCH": "5"},
    "N10_unfreeze10": {"VAEBM_CKPT_UNFREEZE_AFTER_EPOCH": "10"},
}

BASE_ENV = {
    "VAEBM_EMBEDDER": "thenlper/gte-large",
    "VAEBM_UNITS": "1024",
    "VAEBM_DIM_EMB": "",
    "VAEBM_FREEZE_EMB": "1",
    "VAEBM_TOP_WORDS_MODE": "freq",
    "VAEBM_CKPT_ALPHA": "0.0",
    "VAEBM_CKPT_POST_UNFREEZE_LR": "1e-6",
    "VAEBM_CKPT_ORACLE_METRIC": "nmi_purity_sum",
    "VAEBM_TOPIC_ORACLE_LABELS": "1",
    "VAEBM_EPOCHS": "15",  # vaebm_ckpt reads epochs from _VAEBM_DEFAULTS same as every other vaebm-family model
}

MAX_ATTEMPTS = 1
RETRY_BACKOFF_SECONDS = 20
PER_COMBO_TIMEOUT_SECONDS = 2700  # 45 min - 15 epochs is longer than this pass's usual 1-epoch runs

CACHE_ROOT = Path(os.environ.get("VAEBM_CACHE_ROOT", REPO_ROOT.parent / ".cache"))


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def combo_key(config_name: str, dataset: str) -> str:
    return f"{config_name}|{dataset}"


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

        self.total = len(CONFIGS) * len(DATASETS)
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

    def write_status(self, config_name: str, dataset: str, attempt: int) -> None:
        completed, successful, failed = self.counts()
        text = (
            f"Current config: {config_name}\nDataset: {dataset}\nAttempt: {attempt}\n"
            f"Completed: {completed}/{self.total}\nSuccessful: {successful}\nFailed: {failed}\n"
            f"Remaining: {self.total - completed}\nStarted at: {self.started_at}\nLast update: {now_iso()}\n"
        )
        self.status_path.write_text(text, encoding="utf-8")

    def run_one(self, config_name: str, dataset: str, progress_idx: int) -> None:
        key = combo_key(config_name, dataset)
        existing = self.checkpoint.get(key)
        if existing and existing.get("status") == "ok":
            self.log(f"SKIP config={config_name} dataset={dataset} progress={progress_idx}/{self.total} "
                     f"(already completed at {existing.get('timestamp')})")
            return

        log_file = self.logs_dir / f"{config_name}__{dataset}.log"
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        env["VAEBM_RESULTS_DIR"] = str(self.run_dir)
        env["HF_HOME"] = str(CACHE_ROOT / "huggingface")
        env["TRANSFORMERS_CACHE"] = str(CACHE_ROOT / "huggingface" / "transformers")
        env["SENTENCE_TRANSFORMERS_HOME"] = str(CACHE_ROOT / "sentence_transformers")
        env["TORCH_HOME"] = str(CACHE_ROOT / "torch")
        for p in (env["HF_HOME"], env["SENTENCE_TRANSFORMERS_HOME"], env["TORCH_HOME"]):
            Path(p).mkdir(parents=True, exist_ok=True)
        env.update(BASE_ENV)
        env.update(CONFIGS[config_name])
        env["VAEBM_CKPT_MAX_FIT_SECONDS"] = "2400"

        cmd = [
            sys.executable, "-u", str(REPO_ROOT / "scripts" / "run_experiment.py"),
            "--experiment", "topic", "--models", MODEL, "--datasets", dataset,
            "--k", str(K), "--seed", str(SEED), "--protocol", "ecrtm_hicot",
        ]

        last_error = ""
        final_attempt = MAX_ATTEMPTS
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.write_status(config_name, dataset, attempt)
            self.log(f"START config={config_name} dataset={dataset} attempt={attempt} progress={progress_idx}/{self.total}")
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
                    elif outcome["status"] == "ok":
                        self.checkpoint[key] = {
                            "status": "ok", "attempts": attempt, "runtime": outcome["runtime_seconds"],
                            "cv": outcome["cv"], "purity": outcome["purity"], "nmi": outcome["nmi"], "td": outcome["td"],
                            "config": CONFIGS[config_name], "timestamp": now_iso(),
                        }
                        self.save_checkpoint()
                        self.write_progress()
                        self.log(f"OK config={config_name} dataset={dataset} runtime={runtime:.0f}s "
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
            except Exception as exc:  # noqa: BLE001
                last_error = f"driver exception: {exc!r}\n{traceback.format_exc(limit=3)}"

            is_oom = "out of memory" in last_error.lower()
            tag = '"CUDA out of memory"' if is_oom else f'"{last_error[:200].splitlines()[0] if last_error else last_error}"'
            self.log(f"ERROR config={config_name} dataset={dataset} attempt={attempt} error={tag}")

            if timed_out:
                final_attempt = attempt
                break
            if attempt < MAX_ATTEMPTS:
                self.log(f"RETRY config={config_name} dataset={dataset} attempt={attempt + 1}")
                time.sleep(RETRY_BACKOFF_SECONDS)
                final_attempt = attempt + 1

        self.checkpoint[key] = {"status": "error", "attempts": final_attempt, "error": last_error[:2000],
                                 "config": CONFIGS[config_name], "timestamp": now_iso()}
        self.save_checkpoint()
        self.write_progress()
        self.log(f"ERROR config={config_name} dataset={dataset} FINAL after {final_attempt} attempt(s), giving up "
                 f"progress={progress_idx}/{self.total}")

    def _read_result(self, dataset: str) -> dict | None:
        if not self.results_json_path.exists():
            return None
        rows = json.loads(self.results_json_path.read_text(encoding="utf-8"))
        for row in reversed(rows):
            if row.get("model") == MODEL and row.get("dataset") == dataset and row.get("k") == K and row.get("seed") == SEED:
                return row
        return None

    def run(self) -> None:
        self.log(f"SWEEP_START run_dir={self.run_dir} configs={list(CONFIGS.keys())} base_env={BASE_ENV}")
        idx = 0
        for config_name in CONFIGS:
            for dataset in DATASETS:
                idx += 1
                self.run_one(config_name, dataset, idx)
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
        run_dir = REPO_ROOT / "results" / f"vaebm_gte_r10_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)

    Sweep(run_dir).run()


if __name__ == "__main__":
    main()
