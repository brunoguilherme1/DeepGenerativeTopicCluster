"""Repo-relative paths, shared by every module so nothing hardcodes an
absolute path or assumes a particular working directory."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIG_DIR = REPO_ROOT / "configs"
MODEL_CONFIG_DIR = CONFIG_DIR / "models"
DATASET_CONFIG_DIR = CONFIG_DIR / "datasets"
EXPERIMENT_CONFIG_DIR = CONFIG_DIR / "experiments"
RESULTS_DIR = REPO_ROOT / "results"
