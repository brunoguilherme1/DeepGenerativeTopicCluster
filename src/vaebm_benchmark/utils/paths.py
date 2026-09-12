"""Repo-relative paths, shared by every module so nothing hardcodes an
absolute path or assumes a particular working directory."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIG_DIR = REPO_ROOT / "configs"
MODEL_CONFIG_DIR = CONFIG_DIR / "models"
DATASET_CONFIG_DIR = CONFIG_DIR / "datasets"
EXPERIMENT_CONFIG_DIR = CONFIG_DIR / "experiments"

# VAEBM_RESULTS_DIR lets a caller redirect every experiment output (CSV/
# JSON/topics/tables) into a dedicated run directory - e.g. a large sweep
# script isolating each subprocess's results under
# results/<sweep_name>_<timestamp>/ instead of the shared default
# results/experiment_results.csv, without touching any other run's data.
# Unset -> unchanged default behavior.
RESULTS_DIR = Path(os.environ["VAEBM_RESULTS_DIR"]) if os.environ.get("VAEBM_RESULTS_DIR") else REPO_ROOT / "results"
