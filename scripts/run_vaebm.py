#!/usr/bin/env python
"""Fits and evaluates VAE-BM under a baseline's protocol (same dataset
artifact, preprocessing, vocab, K, and metrics as that baseline).

Usage:
    python scripts/run_vaebm.py --protocol fastopic --dataset 20ng
    python scripts/run_vaebm.py --protocol glocom --dataset stack_overflow
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, choices=["fastopic", "glocom"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true", help="Use paper-scale settings (e.g. 200 epochs) instead of the smoke-test default.")
    args = parser.parse_args()

    from vaebm_benchmark.evaluation.registry import get_protocol
    from vaebm_benchmark.evaluation.runner import run_vaebm, write_vaebm_result

    protocol = get_protocol(args.protocol, smoke_test=not args.full)
    if args.dataset not in protocol.topic_count:
        raise SystemExit(
            f"Dataset '{args.dataset}' is not part of the '{args.protocol}' protocol. "
            f"Available: {sorted(protocol.topic_count)}"
        )

    print(f"Running VAE-BM under the {protocol.name} protocol on '{args.dataset}' (seed={args.seed})...")
    record = run_vaebm(protocol, args.dataset, args.seed)
    path = write_vaebm_result(protocol.name, record)
    print(f"Metrics: {record['metrics']}")
    print(f"Written: {path}")


if __name__ == "__main__":
    sys.exit(main())
