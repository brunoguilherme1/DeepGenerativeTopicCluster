#!/usr/bin/env python
"""ECRTM Topic-experiment sweep (2026-09-21): 5 official hicot_*
datasets x K in {50, 100}, Palmetto Cv (the only metric valid for a
"beats HiCOT" claim - see docs/vaebm_leaderboard.md's own rule).
Replaces the Topic tables' copied-from-HiCOT's-paper ECRTM row with a
genuine same-environment run. protocol=ecrtm_hicot matches ECRTM's own
top_n=15/TD definition (Dieng et al.), the same protocol VAE-BM's own
headline sweep used for this exact comparison.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["hicot_20ng", "hicot_agnews", "hicot_search_snippets", "hicot_google_news", "hicot_imdb"]
KS = [50, 100]
PER_COMBO_TIMEOUT_SECONDS = 3600
CACHE_ROOT = Path(os.environ.get("VAEBM_CACHE_ROOT", REPO_ROOT.parent / ".cache"))


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    run_dir = REPO_ROOT / "results" / "ecrtm_topic_sweep_20260921"
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

    combos = [(d, k) for k in KS for d in DATASETS]
    log(f"SWEEP_START combos={combos}")
    for idx, (dataset, k) in enumerate(combos, start=1):
        cmd = [
            sys.executable, "-u", str(REPO_ROOT / "scripts" / "run_experiment.py"),
            "--experiment", "topic", "--models", "ecrtm", "--datasets", dataset,
            "--k", str(k), "--seed", "42", "--protocol", "ecrtm_hicot", "--cv-method", "palmetto",
        ]
        log(f"START dataset={dataset} k={k} progress={idx}/{len(combos)}")
        start = time.perf_counter()
        try:
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=PER_COMBO_TIMEOUT_SECONDS)
            runtime = time.perf_counter() - start
            log_file = run_dir / "logs" / f"{dataset}_k{k}.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
            if proc.returncode != 0:
                stderr_tail = (proc.stderr or "")[-300:]
                log(f"ERROR dataset={dataset} k={k} exit={proc.returncode} runtime={runtime:.0f}s stderr_tail={stderr_tail}")
            else:
                log(f"OK dataset={dataset} k={k} runtime={runtime:.0f}s")
        except subprocess.TimeoutExpired:
            log(f"ERROR dataset={dataset} k={k} timeout after {PER_COMBO_TIMEOUT_SECONDS}s")
    log("SWEEP_COMPLETE")


if __name__ == "__main__":
    main()
