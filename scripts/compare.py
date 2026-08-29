#!/usr/bin/env python
"""Builds results/<baseline>/comparison.csv: published vs. reproduced
baseline vs. VAE-BM, side by side, for one dataset. Requires
run_baseline.py and run_vaebm.py to have already been run for that
dataset under the same protocol.

Usage:
    python scripts/compare.py --baseline fastopic --dataset 20ng
"""

from __future__ import annotations

import argparse
import csv
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=["fastopic", "glocom"])
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    from vaebm_benchmark.evaluation.registry import get_protocol
    from vaebm_benchmark.evaluation.runner import write_comparison

    protocol = get_protocol(args.baseline)
    path = write_comparison(protocol, args.dataset)

    print(f"{'metric':<20}{'published':<14}{'reproduced':<14}{'vaebm':<14}")
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(
                f"{row['metric']:<20}{str(row['published']):<14}"
                f"{str(row['reproduced_baseline']):<14}{str(row['vaebm']):<14}"
            )
    print(f"\nWritten: {path}")


if __name__ == "__main__":
    sys.exit(main())
