"""Incremental progress persistence for `--resume` (Colab sessions can
disconnect mid-run). After every `checkpoint_every` LLM decisions, the
CURRENT iteration's partial refinement state (which document indices
have already been decided, and what their decision was) is flushed to
disk under `results/llm_cache/progress/<run_key>.json`, alongside the
LLM decision cache itself (llm/cache.py) - so a disconnected session can
restart without losing already-completed LLM work, not merely re-hit a
cache built during a fully-finished prior run.

`run_key` (see utils/run_identity.py) must uniquely identify the exact
configuration being resumed (model, dataset, seed, edge-detection
config, LLM model, etc.) - resuming with even one different setting uses
a different progress file, never a stale/mismatched one.
"""

from __future__ import annotations

import json
from pathlib import Path


def progress_path(cache_dir: Path, run_key: str) -> Path:
    return Path(cache_dir) / "progress" / f"{run_key}.json"


def load_progress(cache_dir: Path, run_key: str) -> dict:
    """Returns `{"iteration": int, "decisions": {"<doc_index>": {...}}}`.
    An empty/fresh state (iteration 0, no decisions) if no progress file
    exists yet for this exact run_key."""
    path = progress_path(cache_dir, run_key)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"iteration": 0, "decisions": {}}


def save_progress(cache_dir: Path, run_key: str, state: dict) -> None:
    path = progress_path(cache_dir, run_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def clear_progress(cache_dir: Path, run_key: str) -> None:
    """Called on successful completion of a run - a finished run's
    progress file is no longer meaningful to resume from."""
    path = progress_path(cache_dir, run_key)
    if path.exists():
        path.unlink()
