#!/usr/bin/env python
"""Round 19 (COMPOSE ROUND 17/18's INDEPENDENTLY-POSITIVE FINDINGS +
one genuinely new idea for IMDB) of the systematic K=50 VAE-BM search
(2026-09-20, user-authorized) - see docs/vaebm_leaderboard.md's own
"Round 19 (next)" section.

Round 18 found: (A) GloVe-hybrid topic words + stopword exclusion
recovers search_snippets' prior best (0.450 Palmetto Cv, -0.010 from
target) but doesn't improve past it; (D) VAEBM_NORMALIZE_MU=1 gives
small Cv-safe Purity/NMI gains on 20NG and AGNews (20NG's NMI gap
narrowed to -0.001, AGNews improved on all 3 metrics) but was never
tested on search_snippets combined with (A), nor re-verified on the
PLAIN 20ng/agnews_short counterparts for diagnostic completeness.

Also: IMDB's Cv gap (-0.054) is the largest unresolved one and no
experiment since the stopword fix has moved it further. The GloVe-hybrid
static-embedding topic-word mode (load_hicot_word_embeddings) works for
ANY hicot_* dataset, not just search_snippets - Rounds 14-16 tried it on
IMDB WITHOUT the stopword fix and found it didn't help there, but never
tried it WITH the fix (the same combination that recovered
search_snippets' best in Round 18's Finding A). Worth one genuinely new
combination before concluding IMDB's gap needs an architectural change
(Stage E) rather than another topic-word-extraction tweak.

Four combos (8 with cv_local+Palmetto):
  1. hicot_search_snippets: GloVe-hybrid + stopword-exclude + normalize_mu
     (Finding A + Finding D combined - untested).
  2. hicot_imdb: GloVe-hybrid (pool=40) + stopword-exclude (never tried
     together - Rounds 14-16 tested the hybrid alone, pre-fix).
  3. 20ng (plain): + normalize_mu - diagnostic parity check for Finding D.
  4. agnews_short (plain): + normalize_mu - diagnostic parity check for
     Finding D.

Otherwise mirrors Round 17/18's own architecture (isolated subprocess
per combo, JSON checkpoint/resume, run.log/current_status.txt/
progress.txt).

Usage:
    python scripts/run_vaebm_gte_research_round19.py
    python scripts/run_vaebm_gte_research_round19.py --run-dir results/vaebm_gte_r19_20260920_190000
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

K = 50
SEED = 42
CV_METHODS = ["local", "palmetto"]

BASE_ENV = {
    "VAEBM_EMBEDDER": "thenlper/gte-large",
    "VAEBM_UNITS": "1024",
    "VAEBM_DIM_EMB": "",
    "VAEBM_FREEZE_EMB": "1",
    "VAEBM_ALPHA": "0.0",
    "VAEBM_LR": "1e-4",
    "VAEBM_EPOCHS": "1",
    "VAEBM_TOP_WORDS_MODE": "freq",
    "VAEBM_EXCLUDE_STOPWORDS": "1",
    "TF_FORCE_GPU_ALLOW_GROWTH": "true",
}

COMBOS = [
    # 1. search_snippets: Finding A (GloVe-hybrid+stopword) + Finding D
    # (normalize_mu) combined.
    ("E_glove_hybrid_stopword_normmu", "hicot_search_snippets", {
        "VAEBM_TOP_WORDS_MODE": "static",
        "VAEBM_USE_HICOT_VOCAB": "1",
        "VAEBM_USE_HICOT_STATIC_EMB": "1",
        "VAEBM_STATIC_CANDIDATE_POOL": "40",
        "VAEBM_NORMALIZE_MU": "1",
    }),
    # 2. hicot_imdb: GloVe-hybrid + stopword-exclude, never combined
    # before (Rounds 14-16 tried the hybrid alone, pre-stopword-fix).
    ("F_imdb_glove_hybrid_stopword", "hicot_imdb", {
        "VAEBM_TOP_WORDS_MODE": "static",
        "VAEBM_USE_HICOT_VOCAB": "1",
        "VAEBM_USE_HICOT_STATIC_EMB": "1",
        "VAEBM_STATIC_CANDIDATE_POOL": "40",
    }),
    # 3-4. plain-dataset normalize_mu diagnostic parity checks.
    ("G_normalize_mu_plain", "20ng", {"VAEBM_NORMALIZE_MU": "1"}),
    ("G_normalize_mu_plain", "agnews_short", {"VAEBM_NORMALIZE_MU": "1"}),
]

MAX_ATTEMPTS = 1
RETRY_BACKOFF_SECONDS = 20
PER_COMBO_TIMEOUT_SECONDS = 2400

CACHE_ROOT = Path(os.environ.get("VAEBM_CACHE_ROOT", REPO_ROOT.parent / ".cache"))


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def combo_key(label: str, dataset: str, cv_method: str) -> str:
    return f"{label}|{dataset}|{cv_method}"


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

        self.total = len(COMBOS) * len(CV_METHODS)
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

    def write_status(self, label: str, dataset: str, cv_method: str, attempt: int) -> None:
        completed, successful, failed = self.counts()
        text = (
            f"Current label: {label}\nDataset: {dataset}\nCV method: {cv_method}\nAttempt: {attempt}\n"
            f"Completed: {completed}/{self.total}\nSuccessful: {successful}\nFailed: {failed}\n"
            f"Remaining: {self.total - completed}\nStarted at: {self.started_at}\nLast update: {now_iso()}\n"
        )
        self.status_path.write_text(text, encoding="utf-8")

    def run_one(self, label: str, dataset: str, overrides: dict, cv_method: str, progress_idx: int) -> None:
        key = combo_key(label, dataset, cv_method)
        existing = self.checkpoint.get(key)
        if existing and existing.get("status") == "ok":
            self.log(f"SKIP label={label} dataset={dataset} cv={cv_method} progress={progress_idx}/{self.total} "
                     f"(already completed at {existing.get('timestamp')})")
            return

        log_file = self.logs_dir / f"{label}__{dataset}__{cv_method}.log"
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
            "--experiment", "topic", "--models", "vaebm", "--datasets", dataset,
            "--k", str(K), "--seed", str(SEED), "--protocol", "ecrtm_hicot",
            "--cv-method", cv_method,
        ]

        last_error = ""
        final_attempt = MAX_ATTEMPTS
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.write_status(label, dataset, cv_method, attempt)
            self.log(f"START label={label} dataset={dataset} cv={cv_method} attempt={attempt} progress={progress_idx}/{self.total} "
                     f"overrides={overrides}")
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
                            "timestamp": now_iso(),
                        }
                        self.save_checkpoint()
                        self.write_progress()
                        self.log(f"OK label={label} dataset={dataset} cv_method={cv_method} runtime={runtime:.0f}s "
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
            self.log(f"ERROR label={label} dataset={dataset} cv={cv_method} attempt={attempt} error={tag}")

            if timed_out:
                final_attempt = attempt
                break
            if attempt < MAX_ATTEMPTS:
                self.log(f"RETRY label={label} dataset={dataset} cv={cv_method} attempt={attempt + 1}")
                time.sleep(RETRY_BACKOFF_SECONDS)
                final_attempt = attempt + 1

        self.checkpoint[key] = {"status": "error", "attempts": final_attempt, "error": last_error[:2000], "timestamp": now_iso()}
        self.save_checkpoint()
        self.write_progress()
        self.log(f"ERROR label={label} dataset={dataset} cv={cv_method} FINAL after {final_attempt} attempt(s), giving up "
                 f"progress={progress_idx}/{self.total}")

    def _read_result(self, dataset: str) -> dict | None:
        if not self.results_json_path.exists():
            return None
        rows = json.loads(self.results_json_path.read_text(encoding="utf-8"))
        for row in reversed(rows):
            if row.get("model") == "vaebm" and row.get("dataset") == dataset and row.get("k") == K and row.get("seed") == SEED:
                return row
        return None

    def run(self) -> None:
        self.log(f"SWEEP_START run_dir={self.run_dir} combos={[c[0] + '/' + c[1] for c in COMBOS]} "
                 f"cv_methods={CV_METHODS} base_env={BASE_ENV}")
        idx = 0
        for label, dataset, overrides in COMBOS:
            for cv_method in CV_METHODS:
                idx += 1
                self.run_one(label, dataset, overrides, cv_method, idx)
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
        run_dir = REPO_ROOT / "results" / f"vaebm_gte_r19_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)

    Sweep(run_dir).run()


if __name__ == "__main__":
    main()
