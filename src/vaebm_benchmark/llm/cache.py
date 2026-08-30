"""Persistent LLM-decision cache (results/llm_cache/decisions.json by
default) - if the SAME prompt-determining inputs are encountered again
(even across separate Colab sessions/runs), the stored decision is
reused and the LLM is never called again for it.

The cache key includes every input that could affect the LLM's answer:
the LLM model name (a different model may answer differently), the
document's own content, the exact set of candidate cluster ids offered,
and a hash of the candidate clusters' CONTEXT (top words + representative
examples) - not just their ids. This last part matters across refinement
ITERATIONS: the same document might be offered the same candidate
cluster ids again in iteration 2, but if centroids/representative
documents shifted after iteration 1's reassignments, the context has
genuinely changed and the decision must be recomputed, not reused stale.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def make_cache_key(
    llm_model: str,
    document: str,
    candidate_clusters: list[int],
    cluster_context_hash: str,
) -> str:
    payload = {
        "llm_model": llm_model,
        "document_hash": hashlib.sha256(document.encode("utf-8")).hexdigest(),
        "candidate_clusters": sorted(candidate_clusters),
        "cluster_context_hash": cluster_context_hash,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class LLMDecisionCache:
    """A flat, content-addressed JSON store. Loaded once at construction,
    held in memory, and explicitly `.flush()`ed to disk (never on every
    single `.set()` - that would make a large refinement run I/O-bound on
    Colab's often-slow disk); callers flush periodically (see
    experiment/llm_refinement_runner.py's `checkpoint_every`) and always
    at the end of a run."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "decisions.json"
        self._cache: dict = self._load()
        self._dirty = False

    def _load(self) -> dict:
        if self.cache_file.exists():
            return json.loads(self.cache_file.read_text(encoding="utf-8"))
        return {}

    def get(self, key: str) -> dict | None:
        return self._cache.get(key)

    def set(self, key: str, decision: dict) -> None:
        self._cache[key] = decision
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        self.cache_file.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")
        self._dirty = False

    def __len__(self) -> int:
        return len(self._cache)
