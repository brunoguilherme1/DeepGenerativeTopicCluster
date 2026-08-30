"""Ties a BaselineProtocol, a model (baseline or VAE-BM), and the metrics
module together, and persists results as IMMUTABLE per-run records under
`results/<protocol>/runs/<run_id>/` - never a shared, overwritable
protocol.json/topics-JSON that silently loses history on the next run
(see docs/repository_comparison_report.md for why this replaced the
previous shared-file design).

Every run's `run_id` is a deterministic hash (utils/run_identity.py) of
everything that determines whether two runs are actually comparable:
protocol, dataset, artifact checksum, preprocessing version, vocabulary
checksum, K, model(+variant), seed, metric set, and mode. Two runs with
the same `pairing_key` are the baseline and VAE-BM sides of one
legitimate comparison; `compare.py`/`write_comparison()` below pairs
runs THIS way, never by "whichever row happened to be last in a CSV."
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from vaebm_benchmark.protocols.base import BaselineProtocol
from vaebm_benchmark.utils.paths import RESULTS_DIR
from vaebm_benchmark.utils.provenance import capture_environment
from vaebm_benchmark.utils.run_identity import (
    pairing_key as compute_pairing_key,
)
from vaebm_benchmark.utils.run_identity import (
    run_components,
    run_id as compute_run_id,
)
from vaebm_benchmark.utils.run_identity import (
    run_key as compute_run_key,
)
from vaebm_benchmark.utils.run_identity import vocabulary_checksum
from vaebm_benchmark.utils.seeding import set_all_seeds


def runs_dir_for(protocol_name: str) -> Path:
    d = RESULTS_DIR / protocol_name / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _metric_set(protocol: BaselineProtocol) -> list[str]:
    return sorted(m.name for m in protocol.metric_specs)


def _components(protocol: BaselineProtocol, dataset_id: str, model_label: str, seed: int) -> dict:
    return run_components(
        protocol=protocol.name,
        dataset=dataset_id,
        artifact_checksum=protocol.artifact_checksum(dataset_id),
        preprocessing_version=protocol.preprocessing_version(dataset_id),
        vocabulary_checksum=vocabulary_checksum(protocol.vocabulary_for(dataset_id)),
        k=protocol.topic_count[dataset_id],
        model=model_label,
        seed=seed,
        metric_set=_metric_set(protocol),
        mode=protocol.mode,
    )


def _persist_run(
    protocol: BaselineProtocol,
    components: dict,
    *,
    topics: dict,
    metrics: dict,
    metric_errors: dict,
    num_documents: int,
) -> Path:
    key = compute_run_key(components)
    pairing = compute_pairing_key(components)
    rid = compute_run_id(components, key)
    run_dir = runs_dir_for(protocol.name) / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    protocol_verification = protocol.verify()

    metadata = {
        "run_id": rid,
        "run_key": key,
        "pairing_key": pairing,
        "components": components,
        "num_documents": num_documents,
        "fair_comparison": protocol_verification["fair_comparison"],
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    (run_dir / "topics.json").write_text(json.dumps(topics, indent=2, default=str), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "metric_errors": metric_errors}, indent=2, default=str), encoding="utf-8"
    )
    (run_dir / "protocol_verification.json").write_text(
        json.dumps(protocol_verification, indent=2, default=str), encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(json.dumps(capture_environment(), indent=2, default=str), encoding="utf-8")
    return run_dir


def run_baseline(protocol: BaselineProtocol, dataset_id: str, seed: int) -> Path:
    set_all_seeds(seed)
    documents = protocol.prepare_dataset(dataset_id)
    model = protocol.build_baseline(dataset_id, seed)
    model.fit(documents)
    eval_documents = protocol.prepare_eval_documents(dataset_id)
    true_labels = protocol.prepare_labels(dataset_id)
    metrics, metric_errors = protocol.evaluate(model, documents, eval_documents=eval_documents, true_labels=true_labels)

    top_n = protocol.metric_specs[0].top_n if protocol.metric_specs else 10
    topics = {"topics": model.get_topics(top_n=top_n)}

    components = _components(protocol, dataset_id, "baseline", seed)
    return _persist_run(protocol, components, topics=topics, metrics=metrics, metric_errors=metric_errors,
                         num_documents=len(documents))


def run_vaebm(protocol: BaselineProtocol, dataset_id: str, seed: int, variant: str = "stability_adjusted") -> Path:
    if variant not in protocol.vaebm_variants():
        raise ValueError(f"Unknown VAE-BM variant '{variant}'; available: {protocol.vaebm_variants()}")

    set_all_seeds(seed)
    documents = protocol.prepare_dataset(dataset_id)
    model = protocol.build_vaebm(dataset_id, seed, variant=variant)
    model.fit(documents)
    eval_documents = protocol.prepare_eval_documents(dataset_id)
    true_labels = protocol.prepare_labels(dataset_id)
    metrics, metric_errors = protocol.evaluate(model, documents, eval_documents=eval_documents, true_labels=true_labels)

    top_n = protocol.metric_specs[0].top_n if protocol.metric_specs else 10
    topics: dict = {}
    if hasattr(model, "get_topics_both_views"):
        views = model.get_topics_both_views(top_n=top_n)
        topics["topics_energy"] = views["energy"]
        topics["topics_freq"] = views["freq"]
    else:
        topics["topics"] = model.get_topics(top_n=top_n)

    components = _components(protocol, dataset_id, f"vaebm:{variant}", seed)
    return _persist_run(protocol, components, topics=topics, metrics=metrics, metric_errors=metric_errors,
                         num_documents=len(documents))


def write_protocol_json(protocol: BaselineProtocol) -> Path:
    """A convenience, standalone verification snapshot (scripts/
    verify_protocol.py) - NOT a run record. Every actual run ALSO
    snapshots protocol.verify() into its own immutable
    protocol_verification.json at the moment it ran (see _persist_run
    above), which this file does not replace."""
    path = RESULTS_DIR / protocol.name
    path.mkdir(parents=True, exist_ok=True)
    payload = protocol.verify()
    payload["environment"] = capture_environment()
    out = path / "protocol.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


def _load_runs(protocol_name: str) -> list[dict]:
    runs = []
    for run_dir in sorted(runs_dir_for(protocol_name).iterdir()):
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metrics_path = run_dir / "metrics.json"
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        runs.append({"dir": run_dir, "metadata": metadata, **metrics_payload})
    return runs


def write_comparison(protocol: BaselineProtocol, dataset_id: str) -> Path:
    """Builds comparison.csv: one row per metric, per VAE-BM variant -
    published / reproduced_baseline / vaebm(variant) - paired by
    `pairing_key`, NEVER by "the last row in a CSV." Requires a baseline
    run and at least one VAE-BM variant run sharing the current protocol
    configuration's pairing_key to already exist under runs/."""
    runs = _load_runs(protocol.name)
    matching = [r for r in runs if r["metadata"]["components"]["dataset"] == dataset_id]
    if not matching:
        raise FileNotFoundError(
            f"No runs found for dataset '{dataset_id}' under results/{protocol.name}/runs/ - "
            "run run_baseline.py and run_vaebm.py first."
        )

    by_pairing: dict[str, dict[str, dict]] = {}
    for run in matching:
        pairing = run["metadata"]["pairing_key"]
        model_label = run["metadata"]["components"]["model"]
        by_pairing.setdefault(pairing, {})[model_label] = run

    published = {p.metric: p for p in protocol.published_results if p.dataset_id == dataset_id}

    result_dir = RESULTS_DIR / protocol.name
    result_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = result_dir / "comparison.csv"
    rows: list[dict] = []
    for pairing, models in by_pairing.items():
        baseline_run = models.get("baseline")
        vaebm_runs = {label: run for label, run in models.items() if label.startswith("vaebm:")}
        if baseline_run is None or not vaebm_runs:
            continue  # incomplete pairing - only compare fully-paired run groups
        metric_names = sorted(set(baseline_run.get("metrics", {})) | {
            name for run in vaebm_runs.values() for name in run.get("metrics", {})
        })
        for metric in metric_names:
            pub = published.get(metric)
            row = {
                "dataset_id": dataset_id,
                "pairing_key": pairing,
                "metric": metric,
                "published": pub.value if pub else "not_reported",
                "published_source": pub.source if pub else "",
                "reproduced_baseline": baseline_run.get("metrics", {}).get(metric),
                "reproduced_baseline_error": baseline_run.get("metric_errors", {}).get(metric, ""),
            }
            for label, run in vaebm_runs.items():
                variant = label.split(":", 1)[1]
                row[f"vaebm_{variant}"] = run.get("metrics", {}).get(metric)
                row[f"vaebm_{variant}_error"] = run.get("metric_errors", {}).get(metric, "")
            row["fair_comparison"] = baseline_run["metadata"]["fair_comparison"]
            rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No fully-paired baseline+VAE-BM run group found for dataset '{dataset_id}' - "
            "run both run_baseline.py and at least one run_vaebm.py variant with matching protocol configuration."
        )

    import csv

    fieldnames = list(rows[0].keys())
    with open(comparison_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return comparison_path
