"""Unit tests for llm/reassignment.py - the conservative reassignment
policy (item 10 of this feature's spec). Written but NOT executed in
this session, per this task's own instructions."""

from vaebm_benchmark.llm.reassignment import decide_reassignment


def test_no_change_when_llm_agrees_with_current_cluster():
    decision = decide_reassignment(valid=True, llm_cluster_id=3, llm_confidence=0.95, current_cluster=3, min_confidence=0.70)
    assert not decision.changed
    assert decision.final_cluster == 3


def test_changes_when_different_cluster_and_confidence_above_threshold():
    decision = decide_reassignment(valid=True, llm_cluster_id=5, llm_confidence=0.80, current_cluster=3, min_confidence=0.70)
    assert decision.changed
    assert decision.final_cluster == 5
    assert decision.suggested_cluster == 5
    assert decision.original_cluster == 3


def test_no_change_when_confidence_below_threshold():
    decision = decide_reassignment(valid=True, llm_cluster_id=5, llm_confidence=0.50, current_cluster=3, min_confidence=0.70)
    assert not decision.changed
    assert decision.final_cluster == 3  # original cluster kept


def test_no_change_when_confidence_exactly_at_threshold_boundary():
    """>= threshold, not > - the boundary value itself must be accepted."""
    decision = decide_reassignment(valid=True, llm_cluster_id=5, llm_confidence=0.70, current_cluster=3, min_confidence=0.70)
    assert decision.changed


def test_invalid_decision_keeps_original_cluster_regardless_of_fields():
    """An invalid (failed-to-parse) decision must ALWAYS keep the
    original cluster, even if stray cluster_id/confidence values are
    present in the payload."""
    decision = decide_reassignment(valid=False, llm_cluster_id=5, llm_confidence=0.99, current_cluster=3, min_confidence=0.70)
    assert not decision.changed
    assert decision.final_cluster == 3
    assert decision.confidence == 0.0  # invalid decisions report 0 confidence, not the untrusted parsed value


def test_invalid_decision_with_no_cluster_id_still_keeps_original():
    decision = decide_reassignment(valid=False, llm_cluster_id=None, llm_confidence=None, current_cluster=7, min_confidence=0.70)
    assert not decision.changed
    assert decision.final_cluster == 7
    assert decision.suggested_cluster == 7


def test_missing_confidence_on_valid_decision_defaults_to_zero_never_reassigns():
    decision = decide_reassignment(valid=True, llm_cluster_id=5, llm_confidence=None, current_cluster=3, min_confidence=0.0)
    # Even with min_confidence=0.0, a None confidence must not be treated as satisfying the threshold silently.
    assert decision.confidence == 0.0
    assert not decision.changed
