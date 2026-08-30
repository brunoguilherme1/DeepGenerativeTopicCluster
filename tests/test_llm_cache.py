"""Unit tests for llm/cache.py - written but NOT executed in this
session, per this task's own instructions."""

from vaebm_benchmark.llm.cache import LLMDecisionCache, make_cache_key


def test_cache_key_is_deterministic():
    key1 = make_cache_key("mistralai/Mistral-7B-Instruct-v0.3", "some document text", [1, 3, 5], "context-hash-abc")
    key2 = make_cache_key("mistralai/Mistral-7B-Instruct-v0.3", "some document text", [1, 3, 5], "context-hash-abc")
    assert key1 == key2


def test_cache_key_differs_by_document():
    key1 = make_cache_key("model", "document A", [1, 3, 5], "ctx")
    key2 = make_cache_key("model", "document B", [1, 3, 5], "ctx")
    assert key1 != key2


def test_cache_key_differs_by_llm_model():
    key1 = make_cache_key("model-a", "same document", [1, 3, 5], "ctx")
    key2 = make_cache_key("model-b", "same document", [1, 3, 5], "ctx")
    assert key1 != key2


def test_cache_key_is_invariant_to_candidate_cluster_order():
    """The candidate SET matters, not the order it was computed in."""
    key1 = make_cache_key("model", "document", [5, 1, 3], "ctx")
    key2 = make_cache_key("model", "document", [1, 3, 5], "ctx")
    assert key1 == key2


def test_cache_key_differs_by_candidate_cluster_set():
    key1 = make_cache_key("model", "document", [1, 3, 5], "ctx")
    key2 = make_cache_key("model", "document", [1, 3, 7], "ctx")
    assert key1 != key2


def test_cache_key_differs_by_context_hash():
    """Same document/candidates but the candidate clusters' CONTEXT
    changed (e.g. across refinement iterations, after centroids moved) -
    must be a different key, not reused stale."""
    key1 = make_cache_key("model", "document", [1, 3, 5], "context-v1")
    key2 = make_cache_key("model", "document", [1, 3, 5], "context-v2")
    assert key1 != key2


def test_cache_roundtrip_via_disk(tmp_path):
    cache = LLMDecisionCache(tmp_path / "llm_cache")
    key = make_cache_key("model", "doc", [1, 2], "ctx")
    assert cache.get(key) is None

    decision = {"cluster_id": 2, "confidence": 0.9, "valid": True, "parse_error": ""}
    cache.set(key, decision)
    assert cache.get(key) == decision
    cache.flush()

    reloaded = LLMDecisionCache(tmp_path / "llm_cache")
    assert reloaded.get(key) == decision
    assert len(reloaded) == 1


def test_cache_flush_is_a_noop_when_nothing_changed(tmp_path):
    cache_dir = tmp_path / "llm_cache"
    cache = LLMDecisionCache(cache_dir)
    cache.flush()  # nothing set yet - must not create decisions.json unnecessarily
    assert not (cache_dir / "decisions.json").exists()
