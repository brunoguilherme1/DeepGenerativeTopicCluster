"""Deterministic prompt template for LLM-based cluster refinement
(LLMEdgeRefine-style, Feng et al., EMNLP 2024).

The LLM is shown ONLY unsupervised cluster context - the document text,
its current cluster id, and each candidate cluster's own representative
words/examples (derived from documents and/or the baseline model's own
topic words, never from ground-truth labels). No ground-truth class
names, dataset labels, or any other evaluation-only information is ever
included in the prompt - this is checked structurally by construction
(the functions below have no parameter through which a label could even
be passed in).
"""

from __future__ import annotations

PROMPT_TEMPLATE = """You are refining an unsupervised document clustering result.

Document:
{document}

The document is currently assigned to cluster {current_cluster}.

Candidate clusters:
{candidate_block}
Choose the single cluster that best matches the semantic meaning of the document.

Return JSON only, with no other text, in exactly this format:
{{"cluster_id": <integer>, "confidence": <number between 0 and 1>}}
"""

CLUSTER_BLOCK_TEMPLATE = """Cluster {cluster_id}
Representative words: {words}
Representative examples:
{examples}
"""


def build_cluster_block(cluster_id: int, words: list[str], examples: list[str]) -> str:
    example_lines = "\n".join(f"- {ex}" for ex in examples) if examples else "- (none available)"
    return CLUSTER_BLOCK_TEMPLATE.format(cluster_id=cluster_id, words=", ".join(words) or "(none available)", examples=example_lines)


def build_prompt(document: str, current_cluster: int, candidate_contexts: dict[int, dict]) -> str:
    """`candidate_contexts`: {cluster_id: {"words": [...], "examples": [...]}}
    (see llm/cluster_context.py::build_cluster_contexts) - MUST include an
    entry for `current_cluster` itself (it is always one of the
    candidates - see llm/candidate_clusters.py)."""
    blocks = "\n".join(
        build_cluster_block(cid, ctx["words"], ctx["examples"])
        for cid, ctx in sorted(candidate_contexts.items())
    )
    return PROMPT_TEMPLATE.format(document=document, current_cluster=current_cluster, candidate_block=blocks)
