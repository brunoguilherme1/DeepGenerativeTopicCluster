#!/usr/bin/env python
"""RQ4, Experiment A: LLM-guided topic-word refinement (model-agnostic),
per Chang, Wang, Ren, Wang & Huang 2025 "A Large Language Model Guided Topic
Refinement Mechanism for Short Text Modeling" (DASFAA 2025 / arXiv:2403.17706).

Implements the paper's Algorithm 1 faithfully: for each topic, words are
visited in REVERSE relevance order (least important first); for each word
w_ij the LLM is shown the OTHER words in the (evolving, partially refined)
topic and asked whether w_ij shares their common theme (Fig. 3's exact
prompt/JSON response format). On "No", the word is replaced by the first
of the LLM's 10 suggested alternatives that exists in a fixed vocabulary V;
if none do, by the V word most similar on average to the 10 alternatives
(paper's own fallback rule) - see build_vocabulary()/build_word2vec()'s
docstrings for the one documented deviation (definition of V and of the
embedding space for that fallback, neither of which the paper pins down).

Model-agnostic: operates only on each topic model's own already-extracted
top words, never its internals - applied identically (same LLM, same
prompt, same N, same per-topic call budget) to FASTopic, HiCOT, and VAE-BM.

Usage:
    python scripts/run_rq4_topic_refinement.py --models fastopic hicot vaebm --datasets 20ng agnews_short --seed 42
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rq4_common as common

METHOD = "2025_topic_refinement"
TOP_N = 10  # N in the paper: words per topic shown to the LLM and used for every metric

# Verbatim Fig. 3 prompt template from arXiv:2403.17706 (topic-words-minus-word
# framing, per-word Yes/No + 10-alternative-word JSON response format).
PROMPT_TEMPLATE = """Please analyze the following tasks and provide your answer in the specified format.

1. Determine the common topic shared by these words: [{topic_words}].
2. Assess whether the word "{word}" aligns with the same common topic as the words listed above.

Respond with:
- "Yes", if the given word shares the common topic.
- If "No", suggest 10 single-word alternatives that are commonly used and closely related to this topic. These words should be easily recognizable and distinct from the ones in the provided list.

Format your response in JSON, including the fields "Topic", "Answer", and "Alternative words" (only if the answer is "No")."""


def parse_json_response(response: str) -> dict | None:
    """The paper's prompt asks for a JSON object with Topic/Answer/Alternative words.
    Mistral-Instruct output isn't guaranteed to be pure JSON, so extract the first
    {...} block and parse leniently."""
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return None
    blob = match.group(0)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        blob = re.sub(r",\s*([}\]])", r"\1", blob)  # common LLM JSON slip: trailing commas
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            return None


def build_vocabulary(tokenized_corpus: list[list[str]], vocab_size: int = 5000) -> list[str]:
    """V in the paper. Not exposed uniformly by every model adapter, so we define
    it identically for every backbone as the vocab_size most frequent corpus
    tokens (same vocab_size passed to CLUSTER_MODEL_BUILDERS) - deterministic,
    fair across backbones, and documented here as a deviation from any
    model-internal vocabulary the paper may have assumed."""
    from collections import Counter

    counts = Counter(w for doc in tokenized_corpus for w in doc)
    return [w for w, _ in counts.most_common(vocab_size)]


def build_word2vec(tokenized_corpus: list[list[str]], seed: int):
    """Fallback embedding space for Algorithm 1's 'most semantically similar on
    average to the alternative words' step, used only when none of the LLM's 10
    suggested alternatives are found in V. The paper does not specify this
    embedding space; gensim Word2Vec on the same reference corpus is our
    documented, deterministic choice."""
    from gensim.models import Word2Vec

    return Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, seed=seed, workers=1)


def pick_replacement(alternatives: list[str], vocabulary: set[str], current_topic: list[str], w2v) -> str | None:
    for alt in alternatives:
        if alt.lower() in vocabulary and alt.lower() not in {w.lower() for w in current_topic}:
            return alt
    valid_alts = [a for a in alternatives if a in w2v.wv]
    if not valid_alts:
        return None
    best_word, best_score = None, -2.0
    excluded = {w.lower() for w in current_topic}
    for v_word in vocabulary:
        if v_word.lower() in excluded or v_word not in w2v.wv:
            continue
        score = sum(w2v.wv.similarity(v_word, a) for a in valid_alts) / len(valid_alts)
        if score > best_score:
            best_score, best_word = score, v_word
    return best_word


def refine_topics(topics: list[list[str]], vocabulary: list[str], w2v, llm_client, cache: dict, run_stats: dict) -> list[list[str]]:
    """Algorithm 1: for each topic, iterate words in reverse relevance order
    (least important first), operating on the evolving refined topic t'_i."""
    vocab_set = {w.lower() for w in vocabulary}
    refined = []
    for topic_words in topics:
        t_prime = list(topic_words[:TOP_N])
        for idx in range(len(t_prime) - 1, -1, -1):
            word = t_prime[idx]
            remaining = [w for j, w in enumerate(t_prime) if j != idx]
            prompt = PROMPT_TEMPLATE.format(topic_words=", ".join(remaining), word=word)
            response = common.llm_generate_cached(llm_client, prompt, cache, run_stats)

            parsed = parse_json_response(response)
            answer = str(parsed.get("Answer", "Yes")).strip().lower() if parsed else "yes"
            if answer.startswith("no") and parsed:
                alternatives = parsed.get("Alternative words") or []
                alternatives = [a for a in alternatives if isinstance(a, str)]
                replacement = pick_replacement(alternatives, vocab_set, t_prime, w2v)
                if replacement:
                    t_prime[idx] = replacement
            # "yes" (or unparsable response, treated as retain): word stays as-is.
        refined.append(t_prime)
    return refined


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--llm-model", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", "none"])
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--max-docs", type=int, default=None, help="Smoke-test only: truncate each dataset to the first N docs.")
    args = p.parse_args()

    from vaebm_benchmark.llm.client import LLMClient

    llm_client = LLMClient(model_name=args.llm_model, quantization=args.quantization, max_new_tokens=args.max_new_tokens)
    cache = common.load_cache()
    done = common.load_checkpoint()

    common.log(f"=== RQ4-A ({METHOD}) start models={args.models} datasets={args.datasets} k={args.k} seed={args.seed} ===")
    common.log(f"LLM: {args.llm_model} quantization={args.quantization}")

    for dataset_id in args.datasets:
        for model_name in args.models:
            combo_key = common.build_combo_key(METHOD, model_name, dataset_id, args.k, args.seed, args.max_docs)
            if combo_key in done:
                common.log(f"SKIP {combo_key} (already in checkpoint)")
                continue

            start = time.perf_counter()
            run_stats = common.new_run_stats()
            common.log(f"START {combo_key}")
            base_metrics, refined_metrics, extra = None, None, {}
            status, error = "ok", None
            try:
                documents, labels, clusters, topics, _ = common.build_model_and_assignment(
                    model_name, dataset_id, args.seed, args.k, top_n_words=TOP_N, max_docs=args.max_docs)
                tokenized_corpus = [d.split() for d in documents]
                doc_emb = common.get_doc_embeddings(dataset_id, documents)

                # Document assignments are untouched by design in Experiment A -
                # clustering metrics are computed on the SAME `clusters` both
                # times, as a consistency/sanity check (should come out identical).
                base_metrics = common.compute_full_metric_suite(topics, tokenized_corpus, clusters, labels, doc_emb)

                vocabulary = build_vocabulary(tokenized_corpus)
                w2v = build_word2vec(tokenized_corpus, args.seed)
                refined_topics = refine_topics(topics, vocabulary, w2v, llm_client, cache, run_stats)
                refined_metrics = common.compute_full_metric_suite(refined_topics, tokenized_corpus, clusters, labels, doc_emb)

                unexpected_cluster_drift = {
                    k_: (base_metrics.get(k_), refined_metrics.get(k_))
                    for k_ in common.EXTERNAL_METRIC_KEYS + common.INTERNAL_METRIC_KEYS
                    if base_metrics.get(k_) != refined_metrics.get(k_)
                }
                if unexpected_cluster_drift:
                    common.log(f"WARNING {combo_key} clustering metrics changed despite unchanged assignments "
                               f"(should be impossible - investigate): {unexpected_cluster_drift}")
                extra = {"base_topics": topics, "refined_topics": refined_topics,
                         "unexpected_cluster_drift": unexpected_cluster_drift}

                common.log(f"OK   {combo_key} cv {base_metrics['cv']:.4f}->{refined_metrics['cv']:.4f} "
                           f"td {base_metrics['td']:.4f}->{refined_metrics['td']:.4f} "
                           f"llm_calls={run_stats['llm_calls']} cache_hits={run_stats['cache_hits']}")
            except Exception as exc:  # noqa: BLE001
                status, error = "error", f"{exc}"
                common.log(f"ERROR {combo_key} error={exc!r}")
                import traceback
                traceback.print_exc()

            common.append_result(METHOD, model_name, dataset_id, args.k, args.seed, args.llm_model,
                                  run_stats, time.perf_counter() - start, status, error,
                                  base_metrics, refined_metrics, extra)
            done.add(combo_key)
            common.save_checkpoint(done)

    common.log(f"=== RQ4-A ({METHOD}) done ===")


if __name__ == "__main__":
    main()
