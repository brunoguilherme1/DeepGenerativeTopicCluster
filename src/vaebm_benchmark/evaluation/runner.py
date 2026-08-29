"""Ties a BaselineProtocol, a model (baseline or VAE-BM), and the metrics
module together, and persists results in the results/<baseline>/ layout
this project's README documents: protocol.json (provenance +
MATCH/DIFFERENCE/UNKNOWN verdicts), baseline_results.csv, vaebm_results.csv,
comparison.csv (published vs. reproduced vs. vaebm, side by side - never
collapsed into one number, so a reader can see published != reproduced
before any claim of "VAE-BM beats X.")
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from vaebm_benchmark.protocols.base import BaselineProtocol
from vaebm_benchmark.utils.paths import RESULTS_DIR
from vaebm_benchmark.utils.provenance import capture_environment
from vaebm_benchmark.utils.seeding import set_all_seeds


def _record_top_n(protocol: BaselineProtocol) -> int:
    return protocol.metric_specs[0].top_n if protocol.metric_specs else 10


def run_baseline(protocol: BaselineProtocol, dataset_id: str, seed: int) -> dict:
    set_all_seeds(seed)
    documents = protocol.prepare_dataset(dataset_id)
    model = protocol.build_baseline(dataset_id, seed)
    model.fit(documents)
    eval_documents = protocol.prepare_eval_documents(dataset_id)
    true_labels = protocol.prepare_labels(dataset_id)
    metrics = protocol.evaluate(model, documents, eval_documents=eval_documents, true_labels=true_labels)
    return {
        "protocol": protocol.name,
        "system": "baseline",
        "dataset_id": dataset_id,
        "seed": seed,
        "k": protocol.topic_count.get(dataset_id),
        "num_documents": len(documents),
        "metrics": metrics,
        "topics": model.get_topics(top_n=_record_top_n(protocol)),
    }


def run_vaebm(protocol: BaselineProtocol, dataset_id: str, seed: int) -> dict:
    set_all_seeds(seed)
    documents = protocol.prepare_dataset(dataset_id)
    model = protocol.build_vaebm(dataset_id, seed)
    model.fit(documents)
    eval_documents = protocol.prepare_eval_documents(dataset_id)
    true_labels = protocol.prepare_labels(dataset_id)
    metrics = protocol.evaluate(model, documents, eval_documents=eval_documents, true_labels=true_labels)
    top_n = _record_top_n(protocol)
    result = {
        "protocol": protocol.name,
        "system": "vaebm",
        "dataset_id": dataset_id,
        "seed": seed,
        "k": protocol.topic_count.get(dataset_id),
        "num_documents": len(documents),
        "metrics": metrics,
        "topics_energy": model.get_topics_both_views(top_n=top_n)["energy"] if hasattr(model, "get_topics_both_views") else model.get_topics(top_n=top_n),
    }
    if hasattr(model, "get_topics_both_views"):
        result["topics_freq"] = model.get_topics_both_views(top_n=top_n)["freq"]
    return result


def results_dir_for(protocol_name: str) -> Path:
    d = RESULTS_DIR / protocol_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_protocol_json(protocol: BaselineProtocol) -> Path:
    path = results_dir_for(protocol.name) / "protocol.json"
    payload = protocol.verify()
    payload["environment"] = capture_environment()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def _flatten_row(record: dict) -> dict:
    row = {k: v for k, v in record.items() if k not in ("metrics", "topics", "topics_energy", "topics_freq")}
    row.update({f"metric_{name}": value for name, value in record["metrics"].items()})
    return row


def append_csv_row(path: Path, row: dict) -> None:
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_baseline_result(protocol_name: str, record: dict) -> Path:
    path = results_dir_for(protocol_name) / "baseline_results.csv"
    append_csv_row(path, _flatten_row(record))
    with open(results_dir_for(protocol_name) / f"baseline_{record['dataset_id']}_topics.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    return path


def write_vaebm_result(protocol_name: str, record: dict) -> Path:
    path = results_dir_for(protocol_name) / "vaebm_results.csv"
    append_csv_row(path, _flatten_row(record))
    with open(results_dir_for(protocol_name) / f"vaebm_{record['dataset_id']}_topics.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    return path


def write_comparison(protocol: BaselineProtocol, dataset_id: str) -> Path:
    """Builds comparison.csv: one row per metric, columns
    published/reproduced_baseline/vaebm - never silently dropping the
    published column just because it might be missing (a "not reported by
    the paper" note is written instead)."""
    base_dir = results_dir_for(protocol.name)
    baseline_path = base_dir / f"baseline_{dataset_id}_topics.json"
    vaebm_path = base_dir / f"vaebm_{dataset_id}_topics.json"
    if not baseline_path.exists() or not vaebm_path.exists():
        raise FileNotFoundError(
            f"Run both run_baseline.py and run_vaebm.py for dataset '{dataset_id}' before comparing."
        )
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_record = json.load(f)
    with open(vaebm_path, "r", encoding="utf-8") as f:
        vaebm_record = json.load(f)

    published = {
        (p.dataset_id, p.metric): p for p in protocol.published_results if p.dataset_id == dataset_id
    }

    metric_names = sorted(set(baseline_record["metrics"]) | set(vaebm_record["metrics"]))
    comparison_path = base_dir / "comparison.csv"
    with open(comparison_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset_id", "metric", "published", "published_source", "reproduced_baseline", "vaebm"])
        for metric in metric_names:
            pub = published.get((dataset_id, metric))
            writer.writerow([
                dataset_id,
                metric,
                pub.value if pub else "not_reported",
                pub.source if pub else "",
                baseline_record["metrics"].get(metric, ""),
                vaebm_record["metrics"].get(metric, ""),
            ])
    return comparison_path
