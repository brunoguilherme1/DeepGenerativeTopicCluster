#!/usr/bin/env python
"""Re-run HiCOT's classification baseline on the 12 plain datasets used by
the locked-architecture VAE-BM sweep, now that models/hicot_adapter.py's
stopword-removal gap is fixed (2026-09-21, user-specified policy: stopword
removal is mandatory for a baseline like HiCOT, not a tunable knob the
way VAEBM_EXCLUDE_STOPWORDS is for VAE-BM). Single seed=42, matching the
existing paper_data/futurelab/classification5_final.json HiCOT rows'
own convention (that file used seed=42 only, not 5 seeds) - this is a
baseline correction, not a new protocol.

Usage:
    python scripts/run_hicot_classification_stopword_fix_12ds.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    "20ng", "agnews_short", "google_news_t", "imdb", "search_snippets", "bbc_news",
    "tweet", "stack_overflow", "biomedical", "banking77", "m10", "pascal_flickr",
]

PER_COMBO_TIMEOUT_SECONDS = 3600
CACHE_ROOT = Path(os.environ.get("VAEBM_CACHE_ROOT", REPO_ROOT.parent / ".cache"))


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    run_dir = REPO_ROOT / "results" / "hicot_classif_stopword_fix_20260921"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    def log(msg: str) -> None:
        line = f"[{now_iso()}] {msg}"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["VAEBM_RESULTS_DIR"] = str(run_dir)
    env["HF_HOME"] = str(CACHE_ROOT / "huggingface")
    env["TRANSFORMERS_CACHE"] = str(CACHE_ROOT / "huggingface" / "transformers")
    env["SENTENCE_TRANSFORMERS_HOME"] = str(CACHE_ROOT / "sentence_transformers")
    env["TORCH_HOME"] = str(CACHE_ROOT / "torch")
    for p in (env["HF_HOME"], env["SENTENCE_TRANSFORMERS_HOME"], env["TORCH_HOME"]):
        Path(p).mkdir(parents=True, exist_ok=True)

    log(f"SWEEP_START run_dir={run_dir} datasets={DATASETS}")
    for idx, dataset in enumerate(DATASETS, start=1):
        cmd = [
            sys.executable, "-u", str(REPO_ROOT / "scripts" / "run_experiment.py"),
            "--experiment", "classification", "--models", "hicot", "--datasets", dataset,
            "--split", "random", "--seeds", "42",
        ]
        log(f"START dataset={dataset} progress={idx}/{len(DATASETS)}")
        start = time.perf_counter()
        try:
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True,
                                   text=True, timeout=PER_COMBO_TIMEOUT_SECONDS)
            runtime = time.perf_counter() - start
            log_file = run_dir / "logs" / f"{dataset}.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
            if proc.returncode != 0:
                log(f"ERROR dataset={dataset} exit={proc.returncode} runtime={runtime:.0f}s "
                    f"stderr_tail={(proc.stderr or '')[-300:]}")
            else:
                log(f"OK dataset={dataset} runtime={runtime:.0f}s")
        except subprocess.TimeoutExpired:
            log(f"ERROR dataset={dataset} timeout after {PER_COMBO_TIMEOUT_SECONDS}s")

    log("SWEEP_COMPLETE")

    # Consolidate into a single JSON matching classification_runner.py's
    # own per-seed row schema, for paper_data ingestion.
    results_path = run_dir / "classification" / "classification_results.json"
    if results_path.exists():
        rows = json.loads(results_path.read_text(encoding="utf-8"))
        out_path = REPO_ROOT / "results" / "hicot_classification_stopword_fix.json"
        out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"Wrote {len(rows)} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
