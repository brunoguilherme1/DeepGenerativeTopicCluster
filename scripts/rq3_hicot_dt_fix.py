#!/usr/bin/env python
"""RQ3 quality pass, part 2: test whether HiCOT's own repo-default
weight_loss_DT=250.0 (25-500x outside HiCOT's paper-documented range
[0.5, 0.7, 1, 2, 5, 10], same root cause as the RQ2 fix in
rq2_hicot_quality_fix.py) is the actual cause of m10/tweet remaining
severely undertrained (actual_k collapsing far below the requested K) even
after rq3_hicot_completion_rerun.py's 1200s completion budget fix.

Unlike the completion rerun (which kept the search's originally-selected
config and only raised the training budget), this reruns m10/tweet with a
small weight_loss_DT grid ({1, 2, 5}, HiCOT's own paper range) at the SAME
1200s completion budget, weight_loss_ECR held at the value the original
6-config search selected for that dataset (read from rq3_results.csv's
selected_config, not re-swept - avoids conflating two hyperparameters in
one small search). Selection is via unsupervised C_V only (already computed
by fit_and_evaluate as part of the full metric suite) - never NMI/ACC.

Only replaces the dataset's row in rq3_results.csv if the new run succeeds
AND its actual_k/NMI are not worse than the completion-rerun row (a
DT-driven regression is a valid negative result and must not silently
overwrite a working row) - logged either way to rq3_hicot_dt_fix_log.csv.

Usage:
    python scripts/rq3_hicot_dt_fix.py --datasets m10 tweet
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

import rq4_common as common
from run_rq3 import fit_and_evaluate, load_dataset_and_k, RESULT_FIELDS, VOC_SIZE

RESULTS_CSV = REPO_ROOT / "results" / "rq3" / "rq3_results.csv"
LOG_CSV = REPO_ROOT / "results" / "rq3" / "rq3_hicot_dt_fix_log.csv"
MAX_FIT_SECONDS = 1200  # same budget as the completion rerun - isolates the DT effect
DT_GRID = [1.0, 2.0, 5.0]
LOG_FIELDS = ["dataset", "weight_loss_dt", "cv", "nmi", "acc", "actual_k", "runtime_s", "status", "error", "selected", "replaced"]


def log_row(row: dict) -> None:
    is_new = not LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in LOG_FIELDS})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True, help="HiCOT dataset(s) to rerun, e.g. m10 tweet")
    args = p.parse_args()

    rows = list(csv.DictReader(open(RESULTS_CSV)))
    hicot_rows = {r["dataset"]: r for r in rows if r["model"] == "hicot"}

    for dataset_id in args.datasets:
        old_row = hicot_rows.get(dataset_id)
        if old_row is None:
            print(f"SKIP {dataset_id}: no existing hicot row in rq3_results.csv")
            continue
        base_config = json.loads(old_row["selected_config"])
        ecr = base_config.get("weight_loss_ECR", 40.0)
        k = int(old_row["k"])
        print(f"\n=== DT-grid rerun hicot:{dataset_id} k={k} ecr(fixed)={ecr} "
              f"dt_grid={DT_GRID} (was dt={base_config.get('weight_loss_DT')}, "
              f"old actual_k={old_row['actual_k']}, nmi={old_row['nmi']}) ===", flush=True)

        documents, labels, k_check = load_dataset_and_k(dataset_id)
        assert k_check == k, f"k mismatch for {dataset_id}: {k} vs {k_check}"

        best_dt, best_metrics, best_cv, best_runtime = None, None, -1e18, None
        for dt in DT_GRID:
            config = {**base_config, "weight_loss_DT": dt}
            start = time.perf_counter()
            try:
                from vaebm_benchmark.models.hicot_adapter import HiCOTAdapter
                model = HiCOTAdapter(n_clusters=k, voc_size=VOC_SIZE, epochs=50, sinkhorn_max_iter=100,
                                      max_fit_seconds=MAX_FIT_SECONDS, random_state=42, **config)
                metrics = fit_and_evaluate("hicot", model, documents, labels, dataset_id)
                runtime_s = time.perf_counter() - start
                cv = metrics.get("cv")
                cv_for_selection = cv if isinstance(cv, (int, float)) else -1e18
                print(f"  SEARCH dt={dt} cv={cv} nmi={metrics.get('nmi')} actual_k={metrics.get('actual_k')} "
                      f"runtime={runtime_s:.1f}s", flush=True)
                log_row({"dataset": dataset_id, "weight_loss_dt": dt, "cv": cv, "nmi": metrics.get("nmi"),
                          "acc": metrics.get("acc"), "actual_k": metrics.get("actual_k"),
                          "runtime_s": round(runtime_s, 1), "status": "ok", "error": None, "selected": False, "replaced": False})
                if cv_for_selection > best_cv:
                    best_cv, best_dt, best_metrics, best_runtime = cv_for_selection, dt, metrics, runtime_s
            except Exception as exc:  # noqa: BLE001
                runtime_s = time.perf_counter() - start
                print(f"  SEARCH-FAIL dt={dt} error={exc!r}", flush=True)
                log_row({"dataset": dataset_id, "weight_loss_dt": dt, "status": "error", "error": f"{exc}",
                          "runtime_s": round(runtime_s, 1), "selected": False, "replaced": False})

        if best_dt is None:
            print(f"ERROR {dataset_id}: every DT value failed, leaving existing row unchanged")
            continue

        print(f"SELECTED dt={best_dt} cv={best_cv} (was dt={base_config.get('weight_loss_DT')})")
        old_actual_k = int(old_row["actual_k"]) if old_row.get("actual_k") not in (None, "", "None") else -1
        new_actual_k = best_metrics.get("actual_k") or -1
        old_nmi = float(old_row["nmi"]) if old_row.get("nmi") not in (None, "", "None") else -1.0
        new_nmi = best_metrics.get("nmi") if best_metrics.get("nmi") is not None else -1.0
        is_improvement = (new_actual_k >= old_actual_k) and (new_nmi >= old_nmi)

        if is_improvement:
            selected_config = {**base_config, "weight_loss_DT": best_dt}
            for r in rows:
                if r["model"] == "hicot" and r["dataset"] == dataset_id:
                    r["actual_k"] = best_metrics.pop("actual_k", None)
                    for key in common.ALL_METRIC_KEYS:
                        r[key] = best_metrics.get(key)
                    r["runtime_s"] = round(best_runtime, 1)
                    r["selected_config"] = json.dumps(selected_config)
                    r["source"] = f"rq3_hicot_dt_fix (DT grid {DT_GRID}, ECR fixed at {ecr}, C_V-selected dt={best_dt})"
                    break
            with open(RESULTS_CSV, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
                w.writeheader()
                w.writerows(rows)
            print(f"REPLACED row for {dataset_id}: actual_k {old_actual_k}->{new_actual_k}, nmi {old_nmi:.4f}->{new_nmi:.4f}")
        else:
            print(f"NOT REPLACED for {dataset_id}: best DT candidate (actual_k={new_actual_k}, nmi={new_nmi:.4f}) "
                  f"did not beat the existing row (actual_k={old_actual_k}, nmi={old_nmi:.4f}) - "
                  f"this IS the answer to the DT=250 causal hypothesis: DT was not the (sole) cause here.")

    print("\n=== RQ3 HiCOT DT-fix done ===")


if __name__ == "__main__":
    main()
