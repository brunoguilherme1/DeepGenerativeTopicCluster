#!/usr/bin/env python
"""Fits and evaluates a baseline model under its own protocol. Persists
an immutable run record under results/<baseline>/runs/<run_id>/.

Usage:
    python scripts/run_baseline.py --baseline fastopic --dataset nyt
    python scripts/run_baseline.py --baseline glocom --dataset search_snippets
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=["fastopic", "glocom"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true", help="Use paper-scale settings (e.g. 200 epochs) instead of the smoke-test default.")
    args = parser.parse_args()

    from vaebm_benchmark.evaluation.registry import get_protocol
    from vaebm_benchmark.evaluation.runner import run_baseline

    protocol = get_protocol(args.baseline, smoke_test=not args.full)
    if args.dataset not in protocol.topic_count:
        raise SystemExit(
            f"Dataset '{args.dataset}' is not part of the '{args.baseline}' protocol. "
            f"Available: {sorted(protocol.topic_count)}"
        )

    print(f"Running {protocol.name} baseline on '{args.dataset}' (seed={args.seed}, mode={protocol.mode})...")
    run_dir = run_baseline(protocol, args.dataset, args.seed)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    print(f"Metrics: {metrics['metrics']}")
    if metrics["metric_errors"]:
        print(f"Metric errors (unavailable, not substituted): {metrics['metric_errors']}")
    print(f"Written: {run_dir}")


if __name__ == "__main__":
    sys.exit(main())
