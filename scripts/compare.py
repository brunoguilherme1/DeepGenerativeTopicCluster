#!/usr/bin/env python
"""Builds results/<baseline>/comparison.csv: published vs. reproduced
baseline vs. VAE-BM (per variant), for one dataset - paired by
`pairing_key` (protocol+dataset+artifact checksum+preprocessing+
vocabulary+K+seed+metric set+mode), never by "the last row in a CSV."
Requires run_baseline.py and at least one run_vaebm.py variant to have
already been run with matching protocol configuration.

Usage:
    python scripts/compare.py --baseline fastopic --dataset nyt
"""

from __future__ import annotations

import argparse
import csv
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=["fastopic", "glocom"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--full", action="store_true", help="Match runs produced with --full (paper-scale settings).")
    args = parser.parse_args()

    from vaebm_benchmark.evaluation.registry import get_protocol
    from vaebm_benchmark.evaluation.runner import write_comparison

    protocol = get_protocol(args.baseline, smoke_test=not args.full)
    path = write_comparison(protocol, args.dataset)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No comparison rows.")
        return

    variant_columns = [key for key in rows[0] if key.startswith("vaebm_") and not key.endswith("_error")]
    header = ["metric", "published", "reproduced_baseline"] + variant_columns
    widths = {col: max(len(col), 14) for col in header}
    print("".join(f"{col:<{widths[col]}}" for col in header))
    for row in rows:
        print("".join(f"{str(row[col]):<{widths[col]}}" for col in header))

    fair = rows[0]["fair_comparison"]
    print(f"\nfair_comparison = {fair}")
    print(f"Written: {path}")


if __name__ == "__main__":
    sys.exit(main())
