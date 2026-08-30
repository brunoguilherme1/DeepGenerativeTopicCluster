"""Deterministic run identity: a run key computed from every field that
determines whether two runs are actually comparable - protocol, dataset,
artifact checksum, preprocessing version, vocabulary checksum, K, model
(+ variant), seed, metric set, and mode (smoke/full).

Two runs with the SAME `pairing_key()` are the baseline and VAE-BM sides
of one legitimate comparison (same everything except which model
produced the result). Two runs with the same `run_key()` are the exact
same run (safe to treat as idempotent/overwritable); the `runs/` layout
in evaluation/runner.py uses this to avoid ever silently pairing a
baseline result against an unrelated "whatever VAE-BM row happened to be
last" - the failure mode this module exists to rule out.
"""

from __future__ import annotations

import hashlib
import json


def _digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def vocabulary_checksum(vocabulary: list[str]) -> str:
    """Stable hash of a vocabulary list, order-sensitive (vocab order
    determines column index in every BoW matrix built against it - two
    vocabularies with the same words in a different order are NOT the
    same vocabulary for this project's purposes)."""
    return hashlib.sha256("\n".join(vocabulary).encode("utf-8")).hexdigest()


def run_components(
    *,
    protocol: str,
    dataset: str,
    artifact_checksum: str,
    preprocessing_version: str,
    vocabulary_checksum: str,
    k: int,
    model: str,
    seed: int,
    metric_set: list[str],
    mode: str,
) -> dict:
    return {
        "protocol": protocol,
        "dataset": dataset,
        "artifact_checksum": artifact_checksum,
        "preprocessing_version": preprocessing_version,
        "vocabulary_checksum": vocabulary_checksum,
        "k": k,
        "model": model,
        "seed": seed,
        "metric_set": sorted(metric_set),
        "mode": mode,
    }


def run_key(components: dict) -> str:
    """Identity of THIS exact run - includes `model`, so a baseline run
    and its paired VAE-BM run have DIFFERENT run_keys but the same
    pairing_key()."""
    return _digest(components)


def pairing_key(components: dict) -> str:
    """Identity of the COMPARISON this run belongs to - everything except
    `model`. Use this, never "the last row in a CSV," to find the
    baseline result a given VAE-BM result should be compared against (or
    vice versa)."""
    reduced = {key: value for key, value in components.items() if key != "model"}
    return _digest(reduced)


def run_id(components: dict, run_key_value: str) -> str:
    """Human-readable + collision-proof run directory name."""
    short = run_key_value[:12]
    return f"{components['dataset']}-{components['model']}-seed{components['seed']}-{components['mode']}-{short}"
