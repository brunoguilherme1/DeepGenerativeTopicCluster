#!/usr/bin/env python
"""Prints a protocol's own provenance report - paper, official repo,
dataset, checksum, preprocessing, vocabulary, K, seeds, metrics, and
baseline/VAE-BM implementation - each with a MATCH/DIFFERENCE/UNKNOWN
verdict, and writes the same payload to results/<protocol>/protocol.json.

Usage:
    python scripts/verify_protocol.py --baseline fastopic
    python scripts/verify_protocol.py --baseline glocom
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=["fastopic", "glocom"])
    parser.add_argument("--full", action="store_true", help="Use paper-scale settings (e.g. 200 epochs) instead of the smoke-test default.")
    args = parser.parse_args()

    from vaebm_benchmark.evaluation.registry import get_protocol
    from vaebm_benchmark.evaluation.runner import write_protocol_json

    protocol = get_protocol(args.baseline, smoke_test=not args.full)
    report = protocol.verify()

    print(f"Protocol: {protocol.name}")
    print(f"Paper: {protocol.paper}")
    print(f"Official repository: {protocol.official_repository}\n")

    for section, fields in report.items():
        if section in ("environment",):
            continue
        print(f"[{section}]")
        if isinstance(fields, dict):
            for key, value in fields.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {fields}")
        print()

    path = write_protocol_json(protocol)
    print(f"Written: {path}")


if __name__ == "__main__":
    sys.exit(main())
