#!/usr/bin/env python
"""Fits and evaluates a baseline model under its own protocol.

Usage:
    python scripts/run_baseline.py --baseline fastopic --dataset 20ng
    python scripts/run_baseline.py --baseline glocom --dataset stack_overflow
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=["fastopic", "glocom"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true", help="Use paper-scale settings (e.g. 200 epochs) instead of the smoke-test default.")
    args = parser.parse_args()

    from vaebm_benchmark.evaluation.registry import get_protocol
    from vaebm_benchmark.evaluation.runner import run_baseline, write_baseline_result

    protocol = get_protocol(args.baseline, smoke_test=not args.full)
    if args.dataset not in protocol.topic_count:
        raise SystemExit(
            f"Dataset '{args.dataset}' is not part of the '{args.baseline}' protocol. "
            f"Available: {sorted(protocol.topic_count)}"
        )

    print(f"Running {protocol.name} baseline on '{args.dataset}' (seed={args.seed})...")
    record = run_baseline(protocol, args.dataset, args.seed)
    path = write_baseline_result(protocol.name, record)
    print(f"Metrics: {record['metrics']}")
    print(f"Written: {path}")


if __name__ == "__main__":
    sys.exit(main())
