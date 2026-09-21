#!/usr/bin/env python
"""VAE-BM classification re-run under LogisticRegression (fairness fix,
2026-09-21) for the 9 datasets that do NOT need the fit-once-reuse-mu
trick - 20ng/imdb/biomedical are handled separately by
run_vaebm_classification_fast_slow_datasets.py (already fixed to use
the same shared _make_svm()). Fresh run-dir - NOT resuming the old
run_vaebm_classification_locked_12ds.py checkpoint, whose "ok" markers
are from the old LinearSVC classifier and would wrongly skip a re-run.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = [1, 2, 3, 4, 5]
BASE_ENV = {
    "VAEBM_EMBEDDER": "thenlper/gte-large", "VAEBM_UNITS": "1024", "VAEBM_DIM_EMB": "",
    "VAEBM_FREEZE_EMB": "1", "VAEBM_ALPHA": "0.0", "VAEBM_LR": "1e-4", "VAEBM_EPOCHS": "1",
    "TF_FORCE_GPU_ALLOW_GROWTH": "true",
}
DATASETS = [
    ("agnews_short", {"VAEBM_NORMALIZE_MU": "1"}),
    ("google_news_t", {}), ("search_snippets", {}), ("bbc_news", {}), ("tweet", {}),
    ("stack_overflow", {}), ("banking77", {}), ("m10", {}), ("pascal_flickr", {}),
]
CACHE_ROOT = Path(os.environ.get("VAEBM_CACHE_ROOT", REPO_ROOT.parent / ".cache"))


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    run_dir = REPO_ROOT / "results" / "vaebm_classif_relock_logreg_20260921"
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
    env.update(BASE_ENV)

    log(f"SWEEP_START datasets={[d for d, _ in DATASETS]}")
    for idx, (dataset, overrides) in enumerate(DATASETS, start=1):
        e = dict(env)
        e.update(overrides)
        cmd = [
            sys.executable, "-u", str(REPO_ROOT / "scripts" / "run_experiment.py"),
            "--experiment", "classification", "--models", "vaebm", "--datasets", dataset,
            "--split", "random", "--seeds", *[str(s) for s in SEEDS],
        ]
        log(f"START dataset={dataset} progress={idx}/{len(DATASETS)}")
        start = time.perf_counter()
        try:
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=e, capture_output=True, text=True, timeout=3600)
            runtime = time.perf_counter() - start
            log_file = run_dir / "logs" / f"{dataset}.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
            if proc.returncode != 0:
                stderr_tail = (proc.stderr or "")[-300:]
                log(f"ERROR dataset={dataset} exit={proc.returncode} runtime={runtime:.0f}s stderr_tail={stderr_tail}")
            else:
                log(f"OK dataset={dataset} runtime={runtime:.0f}s")
        except subprocess.TimeoutExpired:
            log(f"ERROR dataset={dataset} timeout after 3600s")
    log("SWEEP_COMPLETE")


if __name__ == "__main__":
    main()
