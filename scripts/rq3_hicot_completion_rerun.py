#!/usr/bin/env python
"""RQ3 quality pass: complete HiCOT's training for the ALREADY-SELECTED
config from the unsupervised search (no re-selection, no label-based
tuning) - the search phase capped max_fit_seconds at 300s (uniformly, to
keep the 6-config search tractable), which left HiCOT badly undertrained
on 6/7 datasets (actual_k collapsing to 1-14 vs requested K). This reruns
the SAME selected {weight_loss_ECR, weight_loss_DT} config with the
codebase's own standard max_fit_seconds=1200 (matching cluster_runner.py's
established default, not a hand-picked new number), uniformly for ALL 7
datasets - not cherry-picked to only the worst ones.

Both the original (300s) and completed (1200s) results are preserved:
this script never overwrites rq3_hyperparameter_search.csv (the original
search log stays exactly as recorded), and logs old-vs-new side by side to
rq3_hicot_completion_log.csv before touching rq3_results.csv. The main
results table is only updated if the completion run succeeds (status=ok);
a failed completion leaves the original 300s row in place untouched
(negative-result preservation).

Usage:
    python scripts/rq3_hicot_completion_rerun.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import rq4_common as common
from run_rq3 import fit_and_evaluate, load_dataset_and_k, RESULT_FIELDS, VOC_SIZE

RESULTS_CSV = REPO_ROOT / "results" / "rq3" / "rq3_results.csv"
COMPLETION_LOG_CSV = REPO_ROOT / "results" / "rq3" / "rq3_hicot_completion_log.csv"
COMPLETION_MAX_FIT_SECONDS = 1200  # cluster_runner.py's own established HiCOT default, not new
LOG_FIELDS = ["dataset", "config_json", "old_nmi", "old_acc", "old_actual_k",
              "new_nmi", "new_acc", "new_actual_k", "runtime_s", "status", "error", "replaced"]


def log_completion(row: dict) -> None:
    is_new = not COMPLETION_LOG_CSV.exists()
    with open(COMPLETION_LOG_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in LOG_FIELDS})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=None,
                    help="Limit to specific dataset(s) (default: all HiCOT rows). Same selected config, "
                         "only the training budget changes - not a re-selection.")
    p.add_argument("--max-fit-seconds", type=int, default=COMPLETION_MAX_FIT_SECONDS,
                    help="Override the completion budget (default: %(default)s, the codebase's standard).")
    args = p.parse_args()
    max_fit_seconds = args.max_fit_seconds

    rows = list(csv.DictReader(open(RESULTS_CSV)))
    hicot_rows = {r["dataset"]: r for r in rows if r["model"] == "hicot"}
    if args.datasets:
        hicot_rows = {k: v for k, v in hicot_rows.items() if k in args.datasets}
    print(f"Found {len(hicot_rows)} HiCOT rows to complete (uniform {max_fit_seconds}s budget, same selected config).")

    for dataset_id, old_row in hicot_rows.items():
        config = json.loads(old_row["selected_config"])
        k = int(old_row["k"])
        print(f"\n=== Completing hicot:{dataset_id} config={config} (was actual_k={old_row['actual_k']}, nmi={old_row['nmi']}) "
              f"budget={max_fit_seconds}s ===", flush=True)

        documents, labels, k_check = load_dataset_and_k(dataset_id)
        assert k_check == k, f"k mismatch for {dataset_id}: {k} vs {k_check}"

        start = time.perf_counter()
        log_row = {"dataset": dataset_id, "config_json": json.dumps(config),
                   "old_nmi": old_row["nmi"], "old_acc": old_row["acc"], "old_actual_k": old_row["actual_k"],
                   "replaced": False}
        try:
            from vaebm_benchmark.models.hicot_adapter import HiCOTAdapter
            # Build with the completion budget instead of the search's 300s cap.
            model = HiCOTAdapter(
                n_clusters=k, voc_size=VOC_SIZE, epochs=50, sinkhorn_max_iter=100,
                max_fit_seconds=max_fit_seconds, random_state=42, **config,
            )
            metrics = fit_and_evaluate("hicot", model, documents, labels, dataset_id)
            runtime_s = time.perf_counter() - start
            log_row.update({
                "new_nmi": metrics.get("nmi"), "new_acc": metrics.get("acc"), "new_actual_k": metrics.get("actual_k"),
                "runtime_s": round(runtime_s, 1), "status": "ok", "error": None,
            })

            # Replace the row in rq3_results.csv (provenance: selected_config unchanged,
            # only training budget/actual_k/metrics change).
            for r in rows:
                if r["model"] == "hicot" and r["dataset"] == dataset_id:
                    r["actual_k"] = metrics.pop("actual_k", None)
                    for key in common.ALL_METRIC_KEYS:
                        r[key] = metrics.get(key)
                    r["runtime_s"] = round(runtime_s, 1)
                    r["source"] = "rq3_hicot_completion_rerun_1200s (same config as rq3_hyperparameter_search.csv, longer budget)"
                    break
            log_row["replaced"] = True
            print(f"OK completed hicot:{dataset_id} nmi {old_row['nmi']}->{metrics.get('nmi')} "
                  f"actual_k {old_row['actual_k']}->{log_row['new_actual_k']} runtime={runtime_s:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            log_row.update({"new_nmi": None, "new_acc": None, "new_actual_k": None,
                             "runtime_s": round(time.perf_counter() - start, 1), "status": "error", "error": f"{exc}"})
            print(f"ERROR completing hicot:{dataset_id}: {exc!r} - keeping original 300s row unchanged", flush=True)
            import traceback
            traceback.print_exc()

        log_completion(log_row)

        # Persist rq3_results.csv incrementally after every dataset.
        with open(RESULTS_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
            w.writeheader()
            w.writerows(rows)

    print("\n=== HiCOT completion rerun done ===")


if __name__ == "__main__":
    main()
