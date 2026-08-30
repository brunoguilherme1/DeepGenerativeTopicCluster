"""Output tables for the llm_cluster_refinement experiment:

  - a detailed before/after/delta table, one block per dataset, columns
    grouped by metric (ACC, NMI, Silhouette, DBI by default) each with
    Before/After/Delta sub-columns, rows = models.
  - a compact table, two rows per model (`Base`, `+LLM`), one column per
    metric - easier to scan when comparing many metrics at once.

Both read directly from LLMRefinementResult - no aggregation across
seeds is implemented here (unlike experiment/cluster_report.py); each
(model, dataset, seed) combination is its own row/before-after-pair.
"""

from __future__ import annotations

from typing import Optional

from vaebm_benchmark.experiment.llm_refinement_runner import LLMRefinementResult
from vaebm_benchmark.experiment.report import DATASET_DISPLAY_NAMES, MODEL_DISPLAY_NAMES

# davies_bouldin is LOWER-is-better - see metrics/geometry_quality.py.
LOWER_IS_BETTER = {"davies_bouldin"}

DETAILED_METRICS = ["acc", "nmi", "silhouette", "davies_bouldin"]
COMPACT_METRICS = ["acc", "nmi", "ari", "silhouette", "davies_bouldin", "calinski_harabasz"]
METRIC_DISPLAY_NAMES = {
    "acc": "ACC", "nmi": "NMI", "ari": "ARI", "ami": "AMI", "purity": "Purity",
    "silhouette": "Silhouette", "davies_bouldin": "DBI", "calinski_harabasz": "CH",
}


def _fmt(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _fmt_delta(value: Optional[float], metric: str) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value >= 0 else ""
    arrow = ""
    if metric in LOWER_IS_BETTER:
        arrow = " (worse)" if value > 0 else (" (better)" if value < 0 else "")
    else:
        arrow = " (better)" if value > 0 else (" (worse)" if value < 0 else "")
    return f"{sign}{value:.3f}{arrow}"


def render_detailed_table(results: list[LLMRefinementResult], metrics: list[str] = None) -> str:
    metrics = metrics or DETAILED_METRICS
    datasets = list(dict.fromkeys(r.dataset for r in results))

    col_width = 9
    blocks = []
    for dataset in datasets:
        subset = [r for r in results if r.dataset == dataset]
        models = list(dict.fromkeys(r.model for r in subset))
        model_col_width = max(len("Model"), *(len(MODEL_DISPLAY_NAMES.get(m, m)) for m in models)) + 2

        lines = [f"Dataset: {DATASET_DISPLAY_NAMES.get(dataset, dataset)}", ""]

        header1 = " " * model_col_width
        for metric in metrics:
            header1 += METRIC_DISPLAY_NAMES[metric].center(col_width * 3)
        lines.append(header1.rstrip())

        header2 = "Model".ljust(model_col_width)
        for _metric in metrics:
            header2 += "Before".center(col_width) + "After".center(col_width) + "Delta".center(col_width)
        lines.append(header2.rstrip())

        lines.append("-" * (model_col_width + col_width * 3 * len(metrics)))

        by_model = {r.model: r for r in subset}
        for model in models:
            r = by_model[model]
            row = MODEL_DISPLAY_NAMES.get(model, model).ljust(model_col_width)
            if r.status != "ok":
                row += "ERROR".center(col_width * 3 * len(metrics))
                lines.append(row.rstrip())
                continue
            for metric in metrics:
                before = getattr(r, f"{metric}_before")
                after = getattr(r, f"{metric}_after")
                delta = getattr(r, f"delta_{metric}")
                row += _fmt(before).center(col_width) + _fmt(after).center(col_width) + _fmt(delta).center(col_width)
            lines.append(row.rstrip())

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def render_compact_table(results: list[LLMRefinementResult], metrics: list[str] = None) -> str:
    metrics = metrics or COMPACT_METRICS
    datasets = list(dict.fromkeys(r.dataset for r in results))

    model_col_width = max(len("Model"), *(len(MODEL_DISPLAY_NAMES.get(r.model, r.model)) for r in results)) + 2
    variant_col_width = max(len("Variant"), len("+LLM")) + 2
    metric_col_width = 9

    blocks = []
    for dataset in datasets:
        subset = [r for r in results if r.dataset == dataset]

        lines = [f"Dataset: {DATASET_DISPLAY_NAMES.get(dataset, dataset)}", ""]
        header = "Model".ljust(model_col_width) + "Variant".ljust(variant_col_width)
        for metric in metrics:
            header += METRIC_DISPLAY_NAMES[metric].center(metric_col_width)
        lines.append(header.rstrip())
        lines.append("-" * (model_col_width + variant_col_width + metric_col_width * len(metrics)))

        for r in subset:
            model_label = MODEL_DISPLAY_NAMES.get(r.model, r.model)
            if r.status != "ok":
                lines.append(f"{model_label.ljust(model_col_width)}{'ERROR':<{variant_col_width}}")
                continue
            base_row = model_label.ljust(model_col_width) + "Base".ljust(variant_col_width)
            llm_row = " ".ljust(model_col_width) + "+LLM".ljust(variant_col_width)
            for metric in metrics:
                base_row += _fmt(getattr(r, f"{metric}_before")).center(metric_col_width)
                llm_row += _fmt(getattr(r, f"{metric}_after")).center(metric_col_width)
            lines.append(base_row.rstrip())
            lines.append(llm_row.rstrip())

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def llm_refinement_results_to_rows(results: list[LLMRefinementResult]) -> list[dict]:
    return [r.as_dict() for r in results]
