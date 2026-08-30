"""Robust parsing/validation of the LLM's JSON response - free-form
output is NEVER trusted directly. On any failure (no JSON object found,
malformed JSON, `cluster_id` missing/not an integer/outside the
candidate set, `confidence` missing/out of `[0, 1]`) the decision is
marked invalid; callers (experiment/llm_refinement_runner.py) keep the
document's ORIGINAL cluster in that case and record the failure - never
silently guess a cluster or clamp an out-of-range confidence into range.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


@dataclass
class ParsedDecision:
    cluster_id: Optional[int]
    confidence: Optional[float]
    valid: bool
    raw_text: str
    parse_error: str = ""


def parse_llm_response(raw_text: str, candidate_cluster_ids: list[int]) -> ParsedDecision:
    match = _JSON_OBJECT_RE.search(raw_text)
    if not match:
        return ParsedDecision(None, None, False, raw_text, "no JSON object found in response")

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return ParsedDecision(None, None, False, raw_text, f"JSON decode error: {exc}")

    if not isinstance(payload, dict):
        return ParsedDecision(None, None, False, raw_text, "parsed JSON is not an object")

    cluster_id = payload.get("cluster_id")
    confidence = payload.get("confidence")

    if isinstance(cluster_id, bool) or not isinstance(cluster_id, int):
        return ParsedDecision(None, confidence, False, raw_text, "cluster_id missing or not an integer")
    if cluster_id not in candidate_cluster_ids:
        return ParsedDecision(cluster_id, confidence, False, raw_text,
                               f"cluster_id {cluster_id} not in candidate set {sorted(candidate_cluster_ids)}")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return ParsedDecision(cluster_id, None, False, raw_text, "confidence missing or not a number")
    confidence = float(confidence)
    if not (0.0 <= confidence <= 1.0):
        return ParsedDecision(cluster_id, confidence, False, raw_text, f"confidence {confidence} outside [0, 1]")

    return ParsedDecision(cluster_id, confidence, True, raw_text, "")
