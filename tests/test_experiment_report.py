from vaebm_benchmark.experiment.report import build_table, results_to_rows
from vaebm_benchmark.experiment.runner import ExperimentResult


def _result(model, dataset, k, **overrides):
    defaults = dict(model=model, dataset=dataset, k=k, cv=0.5, purity=0.6, nmi=0.4, td=0.7,
                     seed=42, runtime_seconds=1.23, status="ok")
    defaults.update(overrides)
    return ExperimentResult(**defaults)


def test_build_table_shape_and_values():
    results = [
        _result("vaebm", "search_snippets", 50, cv=0.616, purity=0.747),
        _result("bertopic", "search_snippets", 50, cv=0.455, purity=0.884),
    ]
    df = build_table(results, k=50)
    assert list(df.index) == ["VAE-BM", "BERTopic"]
    assert ("SearchSnippets", "C_V") in df.columns
    assert df.loc["VAE-BM", ("SearchSnippets", "C_V")] == "0.616"
    assert df.loc["BERTopic", ("SearchSnippets", "Purity")] == "0.884"


def test_build_table_only_includes_requested_k():
    results = [
        _result("vaebm", "search_snippets", 50),
        _result("vaebm", "search_snippets", 100),
    ]
    df_50 = build_table(results, k=50)
    df_100 = build_table(results, k=100)
    assert len(df_50) == 1
    assert len(df_100) == 1


def test_build_table_marks_errors_distinctly_from_missing():
    ok = _result("vaebm", "search_snippets", 50)
    failed = _result("bertopic", "search_snippets", 50, status="error", cv=None, purity=None, nmi=None, td=None)
    df = build_table([ok, failed], k=50)
    assert df.loc["BERTopic", ("SearchSnippets", "C_V")] == "ERROR"
    assert df.loc["VAE-BM", ("SearchSnippets", "C_V")] != "ERROR"


def test_results_to_rows_preserves_required_fields():
    rows = results_to_rows([_result("vaebm", "search_snippets", 50)])
    row = rows[0]
    for field in ("model", "dataset", "k", "cv", "purity", "nmi", "td", "seed", "runtime_seconds", "status"):
        assert field in row


def test_multiple_datasets_grouped_side_by_side():
    results = [
        _result("vaebm", "search_snippets", 50),
        _result("vaebm", "20ng", 50),
    ]
    df = build_table(results, k=50)
    assert ("SearchSnippets", "C_V") in df.columns
    assert ("20NG", "C_V") in df.columns
