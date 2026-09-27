#!/usr/bin/env python
"""One-off: populate results/rq3/rq3_results.csv with the already-valid
lda/bertopic/vaebm rows found in paper_data/{labuai,futurelab_new}/*
(single seed=42, non-oracle, checkpoint_selection="none", requested_k ==
num_classes for all 7 RQ3 datasets) - per "first inspect existing results
and reuse valid completed runs," these are NOT rerun.

Source files (confirmed via repo investigation):
  paper_data/labuai/vaebm_cluster_locked_20260921.json
  paper_data/labuai/lda_locked_cluster_20260921.json
  paper_data/futurelab_new/bertopic_cluster_fullmetrics_20260921.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq4_common as common  # noqa: E402  (for ALL_METRIC_KEYS)

DATASETS = {"20ng", "bbc_news", "m10", "stack_overflow", "biomedical", "tweet", "banking77"}
SOURCE_FILES = [
    ("paper_data/labuai/vaebm_cluster_locked_20260921.json", "vaebm"),
    ("paper_data/labuai/lda_locked_cluster_20260921.json", "lda"),
    ("paper_data/futurelab_new/bertopic_cluster_fullmetrics_20260921.json", "bertopic"),
]

RESULTS_DIR = REPO_ROOT / "results" / "rq3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = RESULTS_DIR / "rq3_results.csv"
RESULT_FIELDS = (["model", "dataset", "k", "actual_k", "seed", "num_documents", "assignment_source", "source"]
                  + list(common.ALL_METRIC_KEYS) + ["runtime_s", "status", "error", "selected_config"])


def main():
    rows = []
    seen = set()
    for rel_path, expected_model in SOURCE_FILES:
        data = json.loads((REPO_ROOT / rel_path).read_text())
        for r in data:
            if r.get("model") != expected_model or r.get("dataset") not in DATASETS or r.get("status") != "ok":
                continue
            key = (r["model"], r["dataset"])
            if key in seen:
                print(f"SKIP duplicate {key} in {rel_path}")
                continue
            seen.add(key)
            if r.get("checkpoint_selection") not in (None, "none"):
                print(f"WARNING {key}: checkpoint_selection={r.get('checkpoint_selection')!r} - not a plain run, check manually")
            row = {
                "model": r["model"], "dataset": r["dataset"], "k": r.get("requested_k"),
                "actual_k": r.get("actual_k"), "seed": r.get("seed"), "num_documents": r.get("num_documents"),
                "assignment_source": r.get("assignment_source"), "source": rel_path,
                "runtime_s": r.get("runtime_seconds"), "status": "ok", "error": None,
                "selected_config": None,
            }
            for m in common.TOPIC_METRIC_KEYS:
                row[m] = None  # topic-quality metrics are not part of RQ3's required 11-metric set
            for m in common.EXTERNAL_METRIC_KEYS + common.INTERNAL_METRIC_KEYS:
                row[m] = r.get(m)
            rows.append(row)

    assert len(rows) == 21, f"expected 21 reused rows (3 models x 7 datasets), got {len(rows)}"
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} reused rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
