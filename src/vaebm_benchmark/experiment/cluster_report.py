"""Paper-style result table for the cluster experiment: rows = models,
columns grouped by dataset (ACC, NMI each), blocks of 4 datasets each -
matching the reference table's own layout (e.g. Zhang et al.-style short-
text-clustering tables: AgNews/SearchSnippets/StackOverflow/Biomedical,
then GoogleNews-TS/-T/-S/Tweet). K differs per dataset (the benchmark's
own num_classes, not a single value swept like the topic experiment's
K) - each dataset header is annotated with the K actually used.

Formatting matches the reference table exactly:
  - only the single BEST value per column is marked (bold in LaTeX,
    `*value*` in the plain-text table) - no second-best marking.
  - a (model, dataset) combination that was never run at all renders as
    "-" (e.g. a baseline the reference table itself marks "-" for
    AgNews/GoogleNews); a combination that WAS attempted but failed
    renders as "ERROR" instead, so the two cases stay visually distinct.
  - a caption states how many seeds the table was averaged over, so a
    single-seed smoke-test table is never silently presented as if it
    were "averaged over 5 random runs" the way the reference table is.

Values are stored as raw decimals everywhere (CSV/JSON, ClusterResult
itself) - only the PRINTED/exported table applies `--format percent`
(the source paper's own convention, e.g. "88.2" for 0.882) vs.
`--format decimal` (e.g. "0.882"). See scripts/run_experiment.py.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Optional

from vaebm_benchmark.experiment.cluster_runner import GEOMETRY_METRIC_IDS, LABEL_METRIC_IDS, ClusterResult
from vaebm_benchmark.experiment.report import DATASET_DISPLAY_NAMES, MODEL_DISPLAY_NAMES

# Table B's own requested column order: ACC | NMI | ARI | AMI |
# Homogeneity | Completeness | V-measure | Purity | Silhouette | DB | CH.
CLUSTER_METRIC_ORDER = LABEL_METRIC_IDS + GEOMETRY_METRIC_IDS
CLUSTER_METRIC_DISPLAY_NAMES = {
    "acc": "ACC", "nmi": "NMI", "ari": "ARI", "ami": "AMI",
    "homogeneity": "Homog.", "completeness": "Complet.", "v_measure": "V-meas.", "purity": "Purity",
    "silhouette": "Silh.", "davies_bouldin": "DB", "calinski_harabasz": "CH",
}


def aggregate_cluster_results(results: list[ClusterResult]) -> tuple[list[ClusterResult], int, Optional[int]]:
    """Groups per-seed results by (model, dataset) and averages acc/nmi
    across the seeds that succeeded. A combination is "ok" if AT LEAST
    ONE seed succeeded (mean over the successful ones); only "error" if
    EVERY seed for that combination failed.

    Returns (aggregated_results, num_seeds, single_seed_value).
    `num_seeds` = number of distinct seeds observed across the WHOLE
    input (used for the "averaged over N random runs" caption).
    `single_seed_value` is that one seed's value when num_seeds == 1
    (so the caption can say "seed=42" instead of a meaningless
    aggregate marker), else None."""
    groups: dict[tuple[str, str], list[ClusterResult]] = defaultdict(list)
    all_seeds = set()
    for r in results:
        groups[(r.model, r.dataset)].append(r)
        all_seeds.add(r.seed)

    aggregated = []
    for (model, dataset), group in groups.items():
        ok_results = [r for r in group if r.status == "ok"]
        total_runtime = sum(r.runtime_seconds for r in group)
        if not ok_results:
            example = next((r.error.splitlines()[0] for r in group if r.error), "")
            aggregated.append(ClusterResult(
                experiment="cluster", model=model, dataset=dataset, seed=-1,
                requested_k=0, actual_k=None, num_classes=0,
                representation_source=group[0].representation_source, assignment_source=group[0].assignment_source,
                runtime_seconds=total_runtime, status="error",
                error=f"all {len(group)} seed(s) failed" + (f": {example}" if example else ""),
                **{name: None for name in CLUSTER_METRIC_ORDER},
            ))
            continue
        averaged_metrics = {
            name: statistics.mean(v for r in ok_results if (v := getattr(r, name)) is not None)
            if any(getattr(r, name) is not None for r in ok_results) else None
            for name in CLUSTER_METRIC_ORDER
        }
        aggregated.append(ClusterResult(
            experiment="cluster", model=model, dataset=dataset,
            seed=-1,  # -1 = "aggregated across seeds", not a real single seed
            requested_k=ok_results[0].requested_k, actual_k=ok_results[0].actual_k,
            num_classes=ok_results[0].num_classes,
            representation_source=ok_results[0].representation_source, assignment_source=ok_results[0].assignment_source,
            runtime_seconds=total_runtime, status="ok",
            **averaged_metrics,
        ))

    num_seeds = len(all_seeds)
    single_seed_value = next(iter(all_seeds)) if num_seeds == 1 else None
    return aggregated, num_seeds, single_seed_value


def _caption(num_seeds: int, single_seed_value: Optional[int]) -> str:
    if num_seeds <= 1:
        return f"Results from a single run (seed={single_seed_value})."
    return f"Results averaged over {num_seeds} random run{'s' if num_seeds != 1 else ''}."


def _fmt(value: Optional[float], fmt: str) -> Optional[str]:
    if value is None:
        return None
    return f"{value * 100:.1f}" if fmt == "percent" else f"{value:.3f}"


def _bold_best_markers(values: list[Optional[float]], fmt: str) -> list[str]:
    """`*value*` for the single best (highest) value in the column - no
    second-best marking, matching the reference table exactly. A
    combination never run at all (`None`) renders as "-"."""
    formatted = [_fmt(v, fmt) for v in values]
    numeric = {v for v in values if v is not None}
    best = max(numeric) if numeric else None
    out = []
    for v, f in zip(values, formatted):
        if f is None:
            out.append("-")
        elif best is not None and v == best and len(numeric) > 1:
            out.append(f"*{f}*")
        else:
            out.append(f)
    return out


def _dataset_header_label(dataset: str, k_by_dataset: dict[str, int]) -> str:
    label = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    k = k_by_dataset.get(dataset)
    return f"{label} (K={k})" if k is not None else label


def _column_values(by_key, models, dataset, metric):
    return [
        getattr(by_key.get((m, dataset)), metric, None)
        if by_key.get((m, dataset)) and by_key[(m, dataset)].status == "ok" else None
        for m in models
    ]


def render_cluster_table(
    results: list[ClusterResult],
    fmt: str = "percent",
    datasets_per_block: int = 4,
) -> str:
    """One or more blocks (datasets_per_block datasets each, matching the
    reference table's own two-block layout for 8 datasets), each with
    models as rows and ACC/NMI grouped per dataset as columns. `results`
    is the FLAT per-seed list (see cluster_runner.run_sweep) - averaging
    across seeds happens here, not in the runner."""
    aggregated, num_seeds, single_seed_value = aggregate_cluster_results(results)

    datasets = list(dict.fromkeys(r.dataset for r in aggregated))
    models = list(dict.fromkeys(r.model for r in aggregated))
    model_labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]
    model_col_width = max(len("Model"), *(len(label) for label in model_labels)) + 2
    metric_col_width = 9 if fmt == "percent" else 10

    by_key = {(r.model, r.dataset): r for r in aggregated}
    k_by_dataset: dict[str, int] = {}
    for r in aggregated:
        if r.dataset not in k_by_dataset and r.status == "ok":
            k_by_dataset[r.dataset] = r.requested_k

    blocks = []
    for start in range(0, len(datasets), datasets_per_block):
        block_datasets = datasets[start:start + datasets_per_block]
        dataset_block_width = len(CLUSTER_METRIC_ORDER) * metric_col_width

        lines = []
        header1 = " " * model_col_width
        for dataset in block_datasets:
            header1 += _dataset_header_label(dataset, k_by_dataset).center(dataset_block_width)
        lines.append(header1.rstrip())

        header2 = "Model".ljust(model_col_width)
        for _dataset in block_datasets:
            for metric in CLUSTER_METRIC_ORDER:
                header2 += CLUSTER_METRIC_DISPLAY_NAMES[metric].center(metric_col_width)
        lines.append(header2.rstrip())

        total_width = model_col_width + dataset_block_width * len(block_datasets)
        lines.append("-" * total_width)

        column_markers: dict[tuple[str, str], list[str]] = {}
        for dataset in block_datasets:
            for metric in CLUSTER_METRIC_ORDER:
                values = _column_values(by_key, models, dataset, metric)
                column_markers[(dataset, metric)] = _bold_best_markers(values, fmt)

        for model_idx, model in enumerate(models):
            row = model_labels[model_idx].ljust(model_col_width)
            for dataset in block_datasets:
                result = by_key.get((model, dataset))
                if result is not None and result.status != "ok":
                    for _metric in CLUSTER_METRIC_ORDER:
                        row += "ERROR".center(metric_col_width)
                    continue
                for metric in CLUSTER_METRIC_ORDER:
                    row += column_markers[(dataset, metric)][model_idx].center(metric_col_width)
            lines.append(row.rstrip())

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) + "\n\n" + _caption(num_seeds, single_seed_value)


def render_cluster_latex_table(results: list[ClusterResult], fmt: str = "percent") -> str:
    """A booktabs-style LaTeX table (bold=best only, matching the
    reference table), all datasets in one table (LaTeX pagination/
    blocking is left to the caller). `results` is the flat per-seed list;
    averaging happens here."""
    aggregated, num_seeds, single_seed_value = aggregate_cluster_results(results)

    datasets = list(dict.fromkeys(r.dataset for r in aggregated))
    models = list(dict.fromkeys(r.model for r in aggregated))
    by_key = {(r.model, r.dataset): r for r in aggregated}
    k_by_dataset: dict[str, int] = {}
    for r in aggregated:
        if r.dataset not in k_by_dataset and r.status == "ok":
            k_by_dataset[r.dataset] = r.requested_k

    def _latex_cell(marked: str) -> str:
        if marked in ("-", "ERROR"):
            return marked
        if marked.startswith("*") and marked.endswith("*"):
            return f"\\textbf{{{marked.strip('*')}}}"
        return marked

    column_markers: dict[tuple[str, str], list[str]] = {}
    for dataset in datasets:
        for metric in CLUSTER_METRIC_ORDER:
            values = _column_values(by_key, models, dataset, metric)
            column_markers[(dataset, metric)] = _bold_best_markers(values, fmt)

    n_metrics = len(CLUSTER_METRIC_ORDER)
    col_spec = "l" + "c" * n_metrics * len(datasets)
    lines = [
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
        "Model & " + " & ".join(
            f"\\multicolumn{{{n_metrics}}}{{c}}{{{_dataset_header_label(d, k_by_dataset)}}}" for d in datasets
        ) + " \\\\",
        " & " + " & ".join(" & ".join(CLUSTER_METRIC_DISPLAY_NAMES[m] for m in CLUSTER_METRIC_ORDER) for _d in datasets) + " \\\\",
        "\\midrule",
    ]
    for model in models:
        model_label = MODEL_DISPLAY_NAMES.get(model, model)
        cells = [model_label]
        for dataset in datasets:
            result = by_key.get((model, dataset))
            if result is not None and result.status != "ok":
                cells.extend(["ERROR"] * n_metrics)
                continue
            for metric in CLUSTER_METRIC_ORDER:
                cells.append(_latex_cell(column_markers[(dataset, metric)][models.index(model)]))
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"% {_caption(num_seeds, single_seed_value)}")
    return "\n".join(lines)


def cluster_results_to_rows(results: list[ClusterResult]) -> list[dict]:
    return [
        {
            "experiment_type": "cluster",
            "model": r.model, "dataset": r.dataset, "seed": r.seed,
            "requested_k": r.requested_k, "actual_k": r.actual_k, "num_classes": r.num_classes,
            "representation_source": r.representation_source, "assignment_source": r.assignment_source,
            **{name: getattr(r, name) for name in CLUSTER_METRIC_ORDER},
            "runtime_seconds": round(r.runtime_seconds, 3), "status": r.status, "error": r.error,
        }
        for r in results
    ]
