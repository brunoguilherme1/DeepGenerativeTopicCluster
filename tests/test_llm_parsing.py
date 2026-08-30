"""Unit tests for llm/parsing.py - written per this project's own
instructions but NOT executed in this session (no LLM, no dataset, no
model involved - pure string/JSON logic)."""

from vaebm_benchmark.llm.parsing import parse_llm_response


def test_valid_response_parses_correctly():
    parsed = parse_llm_response('{"cluster_id": 3, "confidence": 0.87}', candidate_cluster_ids=[1, 3, 5])
    assert parsed.valid
    assert parsed.cluster_id == 3
    assert parsed.confidence == 0.87


def test_response_with_surrounding_text_still_parses():
    raw = 'Sure, here is my answer:\n{"cluster_id": 5, "confidence": 0.6}\nHope that helps!'
    parsed = parse_llm_response(raw, candidate_cluster_ids=[1, 3, 5])
    assert parsed.valid
    assert parsed.cluster_id == 5


def test_no_json_object_is_invalid():
    parsed = parse_llm_response("I choose cluster 3.", candidate_cluster_ids=[1, 3, 5])
    assert not parsed.valid
    assert "no JSON object" in parsed.parse_error


def test_malformed_json_is_invalid():
    parsed = parse_llm_response('{"cluster_id": 3, "confidence": }', candidate_cluster_ids=[1, 3, 5])
    assert not parsed.valid
    assert "JSON decode error" in parsed.parse_error


def test_cluster_id_outside_candidate_set_is_invalid():
    parsed = parse_llm_response('{"cluster_id": 99, "confidence": 0.9}', candidate_cluster_ids=[1, 3, 5])
    assert not parsed.valid
    assert parsed.cluster_id == 99  # recorded even though invalid, for debugging
    assert "not in candidate set" in parsed.parse_error


def test_confidence_out_of_range_is_invalid():
    parsed = parse_llm_response('{"cluster_id": 3, "confidence": 1.5}', candidate_cluster_ids=[1, 3, 5])
    assert not parsed.valid
    assert "outside [0, 1]" in parsed.parse_error


def test_confidence_negative_is_invalid():
    parsed = parse_llm_response('{"cluster_id": 3, "confidence": -0.1}', candidate_cluster_ids=[1, 3, 5])
    assert not parsed.valid


def test_missing_cluster_id_is_invalid():
    parsed = parse_llm_response('{"confidence": 0.9}', candidate_cluster_ids=[1, 3, 5])
    assert not parsed.valid
    assert "cluster_id" in parsed.parse_error


def test_missing_confidence_is_invalid():
    parsed = parse_llm_response('{"cluster_id": 3}', candidate_cluster_ids=[1, 3, 5])
    assert not parsed.valid
    assert "confidence" in parsed.parse_error


def test_cluster_id_as_float_is_invalid_not_coerced():
    """cluster_id must be an integer - a float like 3.0 is NOT silently
    coerced, since a genuinely malformed/ambiguous response should be
    treated as a parse failure, not guessed at."""
    parsed = parse_llm_response('{"cluster_id": 3.0, "confidence": 0.9}', candidate_cluster_ids=[1, 3, 5])
    assert not parsed.valid


def test_boolean_cluster_id_is_invalid():
    """Python's bool is a subclass of int - explicitly rejected so
    {"cluster_id": true, ...} can never accidentally validate."""
    parsed = parse_llm_response('{"cluster_id": true, "confidence": 0.9}', candidate_cluster_ids=[1, 3, 5])
    assert not parsed.valid


def test_confidence_boundary_values_are_valid():
    assert parse_llm_response('{"cluster_id": 1, "confidence": 0.0}', [1]).valid
    assert parse_llm_response('{"cluster_id": 1, "confidence": 1.0}', [1]).valid
