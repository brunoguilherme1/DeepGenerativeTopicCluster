"""Aggregation (mean/std/95% CI across seeds) and table rendering for the
`classification` experiment - see classification_runner.py's own module
docstring for the experiment itself.

Expected table, per this project's own requested layout:
    Model | Dataset | Accuracy mean +/- std | Accuracy 95% CI | F1 mean +/- std | F1 95% CI
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

from vaebm_benchmark.experiment.classification_runner import ClassificationRunResult
from vaebm_benchmark.experiment.report import DATASET_DISPLAY_NAMES, MODEL_DISPLAY_NAMES
from vaebm_benchmark.utils.stats import summarize


@dataclass
class ClassificationAggregate:
    model: str
    dataset: str
    k: int
    n_runs: int
    representation_source: str
    accuracy_mean: float
    accuracy_std: float
    accuracy_ci_lower: float
    accuracy_ci_upper: float
    f1_mean: float
    f1_std: float
    f1_ci_lower: float
    f1_ci_upper: float
    status: str  # "ok" if at least one seed succeeded, else "error"
    seeds_ok: int
    seeds_total: int

    def as_dict(self) -> dict:
        return asdict(self)


def aggregate_classification_results(
    results: list[ClassificationRunResult], confidence: float = 0.95
) -> list[ClassificationAggregate]:
    """Groups per-seed results by (model, dataset, k) and summarizes
    accuracy/f1 across the seeds that succeeded - "ok" if AT LEAST ONE
    seed succeeded (mean over the successful ones, matching
    cluster_report.py::aggregate_cluster_results's own convention); only
    "error" if EVERY seed for that combination failed."""
    groups: dict[tuple[str, str, int], list[ClassificationRunResult]] = defaultdict(list)
    for r in results:
        groups[(r.model, r.dataset, r.k)].append(r)

    aggregated = []
    for (model, dataset, k), group in groups.items():
        ok_results = [r for r in group if r.status == "ok"]
        representation_source = group[0].representation_source
        if not ok_results:
            aggregated.append(
                ClassificationAggregate(
                    model=model, dataset=dataset, k=k, n_runs=0, representation_source=representation_source,
                    accuracy_mean=0.0, accuracy_std=0.0, accuracy_ci_lower=0.0, accuracy_ci_upper=0.0,
                    f1_mean=0.0, f1_std=0.0, f1_ci_lower=0.0, f1_ci_upper=0.0,
                    status="error", seeds_ok=0, seeds_total=len(group),
                )
            )
            continue

        accuracy_summary = summarize([r.accuracy for r in ok_results], confidence=confidence)
        f1_summary = summarize([r.f1 for r in ok_results], confidence=confidence)
        aggregated.append(
            ClassificationAggregate(
                model=model, dataset=dataset, k=k, n_runs=accuracy_summary["n_runs"],
                representation_source=representation_source,
                accuracy_mean=accuracy_summary["mean"], accuracy_std=accuracy_summary["std"],
                accuracy_ci_lower=accuracy_summary["ci_lower"], accuracy_ci_upper=accuracy_summary["ci_upper"],
                f1_mean=f1_summary["mean"], f1_std=f1_summary["std"],
                f1_ci_lower=f1_summary["ci_lower"], f1_ci_upper=f1_summary["ci_upper"],
                status="ok", seeds_ok=len(ok_results), seeds_total=len(group),
            )
        )
    return aggregated


def _fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.3f} +/- {std:.3f}"


def _fmt_ci(lower: float, upper: float) -> str:
    return f"[{lower:.3f}, {upper:.3f}]"


def render_classification_table(aggregated: list[ClassificationAggregate], k: int) -> str:
    """Model | Dataset | Accuracy mean+/-std | Accuracy 95% CI | F1 mean+/-std | F1 95% CI - one table per K."""
    subset = [a for a in aggregated if a.k == k]
    models = list(dict.fromkeys(a.model for a in subset))
    datasets = list(dict.fromkeys(a.dataset for a in subset))
    by_key = {(a.model, a.dataset): a for a in subset}

    headers = ["Model", "Dataset", "Accuracy mean+/-std", "Accuracy 95% CI", "F1 mean+/-std", "F1 95% CI"]
    rows = [headers]
    for model in models:
        model_label = MODEL_DISPLAY_NAMES.get(model, model)
        for dataset in datasets:
            a = by_key.get((model, dataset))
            dataset_label = DATASET_DISPLAY_NAMES.get(dataset, dataset)
            if a is None:
                rows.append([model_label, dataset_label, "N/A", "N/A", "N/A", "N/A"])
            elif a.status != "ok":
                rows.append([model_label, dataset_label, "ERROR", "ERROR", "ERROR", "ERROR"])
            else:
                rows.append(
                    [
                        model_label, dataset_label,
                        _fmt_mean_std(a.accuracy_mean, a.accuracy_std), _fmt_ci(a.accuracy_ci_lower, a.accuracy_ci_upper),
                        _fmt_mean_std(a.f1_mean, a.f1_std), _fmt_ci(a.f1_ci_lower, a.f1_ci_upper),
                    ]
                )

    col_widths = [max(len(row[i]) for row in rows) + 2 for i in range(len(headers))]
    lines = [f"K = {k} Topics (SVM classification)", ""]
    for row in rows:
        lines.append("".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def render_all_classification_tables(aggregated: list[ClassificationAggregate]) -> str:
    ks = list(dict.fromkeys(a.k for a in aggregated))
    return "\n\n".join(render_classification_table(aggregated, k) for k in ks)


def per_run_rows(results: list[ClassificationRunResult]) -> list[dict]:
    return [
        {
            "experiment_type": "classification",
            "model": r.model, "dataset": r.dataset, "k": r.k, "seed": r.seed,
            "accuracy": r.accuracy, "f1": r.f1,
            "representation_source": r.representation_source,
            "num_train_docs": r.num_train_docs, "num_test_docs": r.num_test_docs,
            "runtime_seconds": round(r.runtime_seconds, 3), "status": r.status, "error": r.error,
        }
        for r in results
    ]


def aggregated_rows(aggregated: list[ClassificationAggregate]) -> list[dict]:
    return [
        {
            "experiment_type": "classification_aggregated",
            "model": a.model, "dataset": a.dataset, "k": a.k,
            "n_runs": a.n_runs, "seeds_ok": a.seeds_ok, "seeds_total": a.seeds_total,
            "representation_source": a.representation_source,
            "accuracy_mean": a.accuracy_mean, "accuracy_std": a.accuracy_std,
            "accuracy_ci_lower": a.accuracy_ci_lower, "accuracy_ci_upper": a.accuracy_ci_upper,
            "f1_mean": a.f1_mean, "f1_std": a.f1_std,
            "f1_ci_lower": a.f1_ci_lower, "f1_ci_upper": a.f1_ci_upper,
            "status": a.status,
        }
        for a in aggregated
    ]
