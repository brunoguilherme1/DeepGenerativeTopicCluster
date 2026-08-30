from vaebm_benchmark.experiment.cluster_report import (
    aggregate_cluster_results,
    cluster_results_to_rows,
    render_cluster_latex_table,
    render_cluster_table,
)
from vaebm_benchmark.experiment.cluster_runner import ClusterResult


def _result(model, dataset, seed, acc, nmi, k=8, status="ok", error=""):
    return ClusterResult(
        experiment="cluster", model=model, dataset=dataset, seed=seed,
        requested_k=k, actual_k=k, num_classes=k, acc=acc, nmi=nmi,
        runtime_seconds=1.0, status=status, error=error,
    )


def test_aggregate_averages_across_seeds():
    results = [
        _result("vaebm", "search_snippets", 1, acc=0.60, nmi=0.40),
        _result("vaebm", "search_snippets", 2, acc=0.70, nmi=0.50),
        _result("vaebm", "search_snippets", 3, acc=0.80, nmi=0.60),
    ]
    aggregated, num_seeds, single_seed = aggregate_cluster_results(results)
    assert num_seeds == 3
    assert single_seed is None
    assert len(aggregated) == 1
    row = aggregated[0]
    assert row.acc == 0.70  # mean(0.60, 0.70, 0.80)
    assert row.nmi == 0.50  # mean(0.40, 0.50, 0.60)
    assert row.status == "ok"


def test_aggregate_single_seed_reduces_to_identity():
    results = [_result("vaebm", "search_snippets", 42, acc=0.64, nmi=0.44)]
    aggregated, num_seeds, single_seed = aggregate_cluster_results(results)
    assert num_seeds == 1
    assert single_seed == 42
    assert aggregated[0].acc == 0.64
    assert aggregated[0].nmi == 0.44


def test_aggregate_averages_only_over_successful_seeds():
    results = [
        _result("vaebm", "search_snippets", 1, acc=0.80, nmi=0.60),
        _result("vaebm", "search_snippets", 2, acc=None, nmi=None, status="error", error="OOM"),
    ]
    aggregated, num_seeds, _ = aggregate_cluster_results(results)
    assert num_seeds == 2
    assert aggregated[0].status == "ok"
    assert aggregated[0].acc == 0.80  # only the successful seed counted


def test_aggregate_marks_error_only_if_every_seed_failed():
    results = [
        _result("vaebm", "search_snippets", 1, acc=None, nmi=None, status="error", error="OOM"),
        _result("vaebm", "search_snippets", 2, acc=None, nmi=None, status="error", error="OOM"),
    ]
    aggregated, _, _ = aggregate_cluster_results(results)
    assert aggregated[0].status == "error"
    assert "all 2 seed(s) failed" in aggregated[0].error


def test_never_run_combination_renders_as_dash_not_error():
    results = [
        _result("vaebm", "agnews_short", 42, acc=0.80, nmi=0.60, k=4),
        # "bertopic" never run on agnews_short at all
    ]
    table = render_cluster_table(results, fmt="percent")
    assert "-" in table
    assert "ERROR" not in table


def test_best_value_marked_bold_no_second_best_marking():
    results = [
        _result("vaebm", "search_snippets", 42, acc=0.60, nmi=0.40),
        _result("bertopic", "search_snippets", 42, acc=0.80, nmi=0.65),
    ]
    table = render_cluster_table(results, fmt="percent")
    assert "*80.0*" in table  # best marked
    assert "_60.0_" not in table  # no underline/second-best marking in this convention
    assert "60.0" in table  # second value present, unmarked


def test_percent_vs_decimal_formatting():
    results = [_result("vaebm", "search_snippets", 42, acc=0.6404, nmi=0.4403)]
    percent_table = render_cluster_table(results, fmt="percent")
    decimal_table = render_cluster_table(results, fmt="decimal")
    assert "64.0" in percent_table
    assert "0.640" in decimal_table


def test_caption_reflects_actual_seed_count():
    single = [_result("vaebm", "search_snippets", 42, acc=0.6, nmi=0.4)]
    multi = [
        _result("vaebm", "search_snippets", 1, acc=0.6, nmi=0.4),
        _result("vaebm", "search_snippets", 2, acc=0.7, nmi=0.5),
    ]
    assert "single run (seed=42)" in render_cluster_table(single)
    assert "averaged over 2 random runs" in render_cluster_table(multi)


def test_latex_table_uses_textbf_only_no_underline():
    results = [
        _result("vaebm", "search_snippets", 42, acc=0.60, nmi=0.40),
        _result("bertopic", "search_snippets", 42, acc=0.80, nmi=0.65),
    ]
    tex = render_cluster_latex_table(results)
    assert "\\textbf{80.0}" in tex
    assert "\\underline" not in tex


def test_dataset_header_annotated_with_k():
    results = [_result("vaebm", "agnews_short", 42, acc=0.5, nmi=0.3, k=4)]
    table = render_cluster_table(results)
    assert "K=4" in table


def test_results_to_rows_includes_required_fields():
    rows = cluster_results_to_rows([_result("vaebm", "search_snippets", 42, acc=0.6, nmi=0.4)])
    row = rows[0]
    for field in ("experiment", "model", "dataset", "seed", "requested_k", "actual_k",
                  "num_classes", "acc", "nmi", "runtime_seconds", "status"):
        assert field in row
