"""Paper-style result table: rows = models, columns grouped by dataset
(C_V, Purity, NMI, TD each), one table printed per K - matching the
visual organization of the reference paper table as closely as practical
in a terminal.
"""

from __future__ import annotations

from typing import Optional

from vaebm_benchmark.experiment.runner import ExperimentResult

DATASET_DISPLAY_NAMES = {
    "search_snippets": "SearchSnippets",
    "agnews_short": "AGNews",
    "20ng": "20NG",
    "imdb": "IMDB",
}
MODEL_DISPLAY_NAMES = {
    "vaebm": "VAE-BM",
    "bertopic": "BERTopic",
}
METRIC_ORDER = ["cv", "purity", "nmi", "td"]
METRIC_DISPLAY_NAMES = {"cv": "C_V", "purity": "Purity", "nmi": "NMI", "td": "TD"}


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def build_table(results: list[ExperimentResult], k: int):
    """Returns a pandas DataFrame with a MultiIndex column (dataset,
    metric) and one row per model, for a single K."""
    import pandas as pd

    subset = [r for r in results if r.k == k]
    datasets = list(dict.fromkeys(r.dataset for r in subset))  # first-seen order, de-duplicated
    models = list(dict.fromkeys(r.model for r in subset))

    columns = pd.MultiIndex.from_product(
        [[DATASET_DISPLAY_NAMES.get(d, d) for d in datasets], [METRIC_DISPLAY_NAMES[m] for m in METRIC_ORDER]]
    )
    df = pd.DataFrame(index=[MODEL_DISPLAY_NAMES.get(m, m) for m in models], columns=columns, dtype=object)

    by_key = {(r.model, r.dataset): r for r in subset}
    for model in models:
        model_label = MODEL_DISPLAY_NAMES.get(model, model)
        for dataset in datasets:
            dataset_label = DATASET_DISPLAY_NAMES.get(dataset, dataset)
            result = by_key.get((model, dataset))
            for metric in METRIC_ORDER:
                metric_label = METRIC_DISPLAY_NAMES[metric]
                if result is None:
                    df.loc[model_label, (dataset_label, metric_label)] = "N/A"
                elif result.status != "ok":
                    df.loc[model_label, (dataset_label, metric_label)] = "ERROR"
                else:
                    df.loc[model_label, (dataset_label, metric_label)] = _fmt(getattr(result, metric))
    return df


def print_table(results: list[ExperimentResult], k: int) -> None:
    df = build_table(results, k)
    print(f"K = {k}\n")
    print(df.to_string())
    print()


def print_all_tables(results: list[ExperimentResult]) -> None:
    ks = list(dict.fromkeys(r.k for r in results))  # first-seen order
    for k in ks:
        print_table(results, k)


def _rank_markers(values: list[Optional[float]]) -> list[str]:
    """`*value*` for the best (highest) value in the column, `_value_`
    for the second-best, plain otherwise - a plain-text stand-in for the
    reference table's bold/underline convention that survives piping to
    a log file or a non-ANSI terminal. Ties are marked once, at the
    higher rank. N<2 comparable values -> no marking (nothing to rank)."""
    formatted = [None if v is None else f"{v:.3f}" for v in values]
    numeric = sorted({v for v in values if v is not None}, reverse=True)
    if len(numeric) < 2:
        return [f if f is not None else "N/A" for f in formatted]
    best, second = numeric[0], numeric[1]
    out = []
    for v, f in zip(values, formatted):
        if f is None:
            out.append("N/A")
        elif v == best:
            out.append(f"*{f}*")
        elif v == second:
            out.append(f"_{f}_")
        else:
            out.append(f)
    return out


def _column_markers_for_k(results: list[ExperimentResult], k: int, models: list[str], datasets: list[str]):
    by_key = {(r.model, r.dataset): r for r in results if r.k == k}
    column_markers: dict[tuple[str, str], list[str]] = {}
    for dataset in datasets:
        for metric in METRIC_ORDER:
            values = []
            for model in models:
                result = by_key.get((model, dataset))
                values.append(getattr(result, metric) if result and result.status == "ok" else None)
            column_markers[(dataset, metric)] = _rank_markers(values)
    return by_key, column_markers


def render_table_for_k(results: list[ExperimentResult], k: int, header: bool = True) -> str:
    """One table for a single K: datasets grouped side-by-side,
    CV/Purity/NMI/TD sub-columns per dataset - matching the reference
    paper table's layout. Best value per (dataset, metric) column is
    wrapped `*like this*`, second-best `_like this_` (see
    _rank_markers). ERROR/status!=ok cells print as `ERROR`, a missing
    (model, dataset) combination as `N/A`."""
    datasets = list(dict.fromkeys(r.dataset for r in results if r.k == k))
    models = list(dict.fromkeys(r.model for r in results if r.k == k))
    model_labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]
    model_col_width = max(len("Model"), *(len(label) for label in model_labels)) + 2
    metric_col_width = 8
    dataset_block_width = len(METRIC_ORDER) * metric_col_width

    lines = [f"K = {k} Topics", ""] if header else []

    header1 = " " * model_col_width
    for dataset in datasets:
        header1 += DATASET_DISPLAY_NAMES.get(dataset, dataset).center(dataset_block_width)
    lines.append(header1.rstrip())

    header2 = "Model".ljust(model_col_width)
    for _dataset in datasets:
        for metric in METRIC_ORDER:
            header2 += METRIC_DISPLAY_NAMES[metric].center(metric_col_width)
    lines.append(header2.rstrip())

    total_width = model_col_width + dataset_block_width * len(datasets)
    lines.append("-" * total_width)

    by_key, column_markers = _column_markers_for_k(results, k, models, datasets)
    for model_idx, model in enumerate(models):
        row = model_labels[model_idx].ljust(model_col_width)
        for dataset in datasets:
            result = by_key.get((model, dataset))
            if result is not None and result.status != "ok":
                for _metric in METRIC_ORDER:
                    row += "ERROR".center(metric_col_width)
                continue
            for metric in METRIC_ORDER:
                row += column_markers[(dataset, metric)][model_idx].center(metric_col_width)
        lines.append(row.rstrip())

    return "\n".join(lines)


def render_stacked_report(results: list[ExperimentResult]) -> str:
    """One block per K (in first-seen order) - see render_table_for_k."""
    ks = list(dict.fromkeys(r.k for r in results))
    return "\n\n".join(render_table_for_k(results, k) for k in ks)


def render_latex_table_for_k(results: list[ExperimentResult], k: int) -> str:
    """A booktabs-style LaTeX table for a single K, matching the
    reference table's convention: \\textbf{} for the best value per
    column, \\underline{} for second-best."""
    datasets = list(dict.fromkeys(r.dataset for r in results if r.k == k))
    models = list(dict.fromkeys(r.model for r in results if r.k == k))
    by_key, column_markers = _column_markers_for_k(results, k, models, datasets)

    def _latex_cell(marked: str) -> str:
        if marked in ("N/A", "ERROR"):
            return marked
        if marked.startswith("*") and marked.endswith("*"):
            return f"\\textbf{{{marked.strip('*')}}}"
        if marked.startswith("_") and marked.endswith("_"):
            return f"\\underline{{{marked.strip('_')}}}"
        return marked

    n_metrics = len(METRIC_ORDER)
    col_spec = "l" + "".join("c" * n_metrics for _ in datasets)
    lines = [
        f"% K = {k} topics",
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
    ]
    header1 = "Model & " + " & ".join(
        f"\\multicolumn{{{n_metrics}}}{{c}}{{{DATASET_DISPLAY_NAMES.get(d, d)}}}" for d in datasets
    ) + " \\\\"
    lines.append(header1)
    latex_metric_names = {"cv": "$C_V$", "purity": "Purity", "nmi": "NMI", "td": "TD"}
    header2 = " & " + " & ".join(
        " & ".join(latex_metric_names[m] for m in METRIC_ORDER) for _d in datasets
    ) + " \\\\"
    lines.append(header2)
    lines.append("\\midrule")

    for model in models:
        model_label = MODEL_DISPLAY_NAMES.get(model, model)
        cells = [model_label]
        for dataset in datasets:
            result = by_key.get((model, dataset))
            if result is not None and result.status != "ok":
                cells.extend(["ERROR"] * n_metrics)
                continue
            for metric in METRIC_ORDER:
                cells.append(_latex_cell(column_markers[(dataset, metric)][models.index(model)]))
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def results_to_rows(results: list[ExperimentResult]) -> list[dict]:
    return [
        {
            "model": r.model, "dataset": r.dataset, "k": r.k,
            "cv": r.cv, "purity": r.purity, "nmi": r.nmi, "td": r.td,
            "seed": r.seed, "runtime_seconds": round(r.runtime_seconds, 3), "status": r.status, "error": r.error,
            # Paper-alignment metadata - see experiment/runner.py's own
            # docstring and docs/methodological_notes.md #10.
            "evaluation_protocol": r.evaluation_protocol, "top_n": r.top_n,
            "cv_source": r.cv_source, "td_definition": r.td_definition,
            "assignment_source": r.assignment_source, "topic_source": r.topic_source,
        }
        for r in results
    ]
