#!/usr/bin/env python
"""Fits and evaluates VAE-BM under a baseline's protocol (same dataset
artifact, preprocessing, vocab, K, and metrics as that baseline).
Persists an immutable run record under results/<protocol>/runs/<run_id>/.

`--variant` distinguishes the AS-SUPPLIED hyperparameters
("protocol_faithful") from this project's documented stability
substitution ("stability_adjusted" - see docs/methodological_notes.md
#8/#9). Both are real, runnable configurations; running
`--variant protocol_faithful` is expected to reproduce the training
divergence documented there, not a bug in this script.

Usage:
    python scripts/run_vaebm.py --protocol fastopic --dataset nyt
    python scripts/run_vaebm.py --protocol fastopic --dataset nyt --variant protocol_faithful
    python scripts/run_vaebm.py --protocol glocom --dataset search_snippets
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, choices=["fastopic", "glocom"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", default="stability_adjusted", choices=["protocol_faithful", "stability_adjusted"])
    parser.add_argument("--full", action="store_true", help="Use paper-scale settings (e.g. 200 epochs) instead of the smoke-test default.")
    args = parser.parse_args()

    from vaebm_benchmark.evaluation.registry import get_protocol
    from vaebm_benchmark.evaluation.runner import run_vaebm

    protocol = get_protocol(args.protocol, smoke_test=not args.full)
    if args.dataset not in protocol.topic_count:
        raise SystemExit(
            f"Dataset '{args.dataset}' is not part of the '{args.protocol}' protocol. "
            f"Available: {sorted(protocol.topic_count)}"
        )

    print(f"Running VAE-BM ({args.variant}) under the {protocol.name} protocol on '{args.dataset}' "
          f"(seed={args.seed}, mode={protocol.mode})...")
    if args.variant == "protocol_faithful":
        print("NOTE: protocol_faithful uses the AS-SUPPLIED lr - this is known to diverge to NaN "
              "at this vocab scale for both protocols (see docs/methodological_notes.md #8). "
              "That outcome, if it occurs, is the expected result being recorded, not a script bug.")

    run_dir = run_vaebm(protocol, args.dataset, args.seed, variant=args.variant)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    print(f"Metrics: {metrics['metrics']}")
    if metrics["metric_errors"]:
        print(f"Metric errors (unavailable, not substituted): {metrics['metric_errors']}")
    print(f"Written: {run_dir}")


if __name__ == "__main__":
    sys.exit(main())
