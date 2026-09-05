"""Multi-run aggregation: mean/std/95% CI across repeated runs (e.g.
different seeds) of the same (model, dataset) configuration - used by
experiment/classification_runner.py's SVM accuracy/F1 aggregation. A
single model's own run-to-run uncertainty, not a two-model comparison
(there is no paired/bootstrap comparison logic here, deliberately - this
project's classification table reports each model's own mean/std/CI,
never a "is A significantly better than B" verdict)."""

from __future__ import annotations

import math
import statistics


def confidence_interval_of_mean(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Single-sample Student's t confidence interval on the mean.
    n=0 -> (0.0, 0.0); n=1 -> a degenerate (value, value) interval (no
    spread is measurable from one run) - both cases are edge guards, not
    meaningful confidence statements, and callers should treat n<2 runs
    as "not enough seeds for a CI" rather than reading the interval at
    face value."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (values[0], values[0])
    from scipy import stats as scipy_stats

    mean = statistics.mean(values)
    sem = statistics.stdev(values) / math.sqrt(n)
    t_critical = scipy_stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_critical * sem
    return (mean - margin, mean + margin)


def summarize(values: list[float], confidence: float = 0.95) -> dict:
    """{n_runs, mean, std, ci_lower, ci_upper} - the exact aggregate
    fields experiment/classification_runner.py persists per (model,
    dataset, metric). `std` is the sample standard deviation
    (ddof=1, statistics.stdev's own convention) - 0.0 for a single run,
    not undefined."""
    n = len(values)
    mean = statistics.mean(values) if n else 0.0
    std = statistics.stdev(values) if n > 1 else 0.0
    ci_lower, ci_upper = confidence_interval_of_mean(values, confidence=confidence)
    return {"n_runs": n, "mean": mean, "std": std, "ci_lower": ci_lower, "ci_upper": ci_upper}
