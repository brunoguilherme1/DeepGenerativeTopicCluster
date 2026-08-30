#!/usr/bin/env python
"""Prints a protocol's own MATCH/DIFFERENCE/UNKNOWN provenance report -
dataset artifact, checksum, upstream commit, preprocessing, vocabulary,
K, split, seeds, metrics, baseline/VAE-BM implementation, mode - and the
aggregate `fair_comparison` boolean (true only if every single field is
MATCH). Writes the same payload to results/<protocol>/protocol.json.

Exit code is 1 whenever fair_comparison is false (a DIFFERENCE or UNKNOWN
exists anywhere), 0 otherwise - suitable for CI gating.

Usage:
    python scripts/verify_protocol.py --baseline fastopic
    python scripts/verify_protocol.py --baseline glocom
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=["fastopic", "glocom"])
    parser.add_argument("--full", action="store_true", help="Use paper-scale settings (e.g. 200 epochs) instead of the smoke-test default.")
    args = parser.parse_args()

    from vaebm_benchmark.evaluation.registry import get_protocol
    from vaebm_benchmark.evaluation.runner import write_protocol_json

    protocol = get_protocol(args.baseline, smoke_test=not args.full)
    report = protocol.verify()

    print(f"Protocol: {protocol.name} (mode={protocol.mode})")
    print(f"Paper: {protocol.paper}")
    print(f"Official repository: {protocol.official_repository}")
    print(f"Upstream commit: {protocol.upstream_commit}\n")

    for field, check in report["checks"].items():
        note = f" ({check['note']})" if check["note"] else ""
        print(f"{check['status']:12} {field}: {check['detail']}{note}")

    print(f"\nSummary: " + ", ".join(f"{status}={count}" for status, count in report["summary"].items()))
    print(f"fair_comparison = {report['fair_comparison']}")

    path = write_protocol_json(protocol)
    print(f"\nWritten: {path}")

    return 0 if report["fair_comparison"] else 1


if __name__ == "__main__":
    sys.exit(main())
