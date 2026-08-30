"""Conservative reassignment policy (LLMEdgeRefine-style): a document
changes cluster ONLY if the LLM's parsed decision is valid, proposes a
DIFFERENT cluster than its current one, AND the LLM's own confidence
meets the configured `--min-confidence` threshold. Otherwise the
document's ORIGINAL cluster is kept unchanged.

Extracted into its own pure function (rather than left inline in
experiment/llm_refinement_runner.py) so this policy - the one point in
the whole pipeline where an LLM's suggestion can actually change a
result - is independently unit-testable without a loaded LLM, a fitted
baseline model, or any dataset at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ReassignmentDecision:
    original_cluster: int
    suggested_cluster: int
    final_cluster: int
    confidence: float
    changed: bool


def decide_reassignment(
    valid: bool,
    llm_cluster_id: Optional[int],
    llm_confidence: Optional[float],
    current_cluster: int,
    min_confidence: float,
) -> ReassignmentDecision:
    """Takes the plain (valid, cluster_id, confidence) fields directly -
    NOT a `ParsedDecision` object - so this works identically whether the
    decision came from a fresh `llm/parsing.py::parse_llm_response()`
    call or from a decision reloaded from the persistent cache
    (llm/cache.py), which stores plain JSON-serializable dicts, not
    `ParsedDecision` instances.

    `has_confidence` guards a subtle edge case: `parse_llm_response()`
    never actually returns `valid=True` with `confidence=None` (a missing
    confidence always makes a decision invalid), but this function
    doesn't assume that invariant holds for whatever `valid`/`confidence`
    pair it's handed. Without this guard, a missing confidence would
    default to `0.0` and - at `--min-confidence 0.0` - `0.0 >= 0.0` would
    incorrectly satisfy the threshold and reassign despite the confidence
    being genuinely UNKNOWN, not a real zero score."""
    has_confidence = valid and llm_confidence is not None
    suggested = llm_cluster_id if valid else current_cluster
    confidence = llm_confidence if has_confidence else 0.0
    changed = bool(has_confidence and suggested != current_cluster and confidence >= min_confidence)
    final_cluster = suggested if changed else current_cluster
    return ReassignmentDecision(
        original_cluster=current_cluster,
        suggested_cluster=suggested,
        final_cluster=final_cluster,
        confidence=confidence,
        changed=changed,
    )
