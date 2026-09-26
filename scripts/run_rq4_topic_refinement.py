#!/usr/bin/env python
"""RQ4, Experiment A: LLM-guided topic-word refinement (model-agnostic),
per Pham & Le et al. 2025 "A Large Language Model Guided Topic Refinement
Mechanism for Short Text Modeling" (DASFAA 2025 / arXiv:2403.17706) -
for each topic's own top-N words, the LLM identifies semantically
"intruder" word(s) and suggests coherent replacement(s); the refined
topic-word list is evaluated against the same coherence/diversity metrics
as the base list. Model-agnostic: operates only on each topic model's own
already-extracted top words, never its internals - applied identically to
FASTopic, HiCOT, and VAE-BM (ours).

Reuses this repo's own existing infrastructure rather than reimplementing
it: cluster_runner.CLUSTER_MODEL_BUILDERS for building/fitting each model,
simple_registry.load_dataset for data, metrics/topic_quality.py's
coherence()/topic_diversity()/irbo() for C_V/NPMI/TD/IRBO (local-corpus,
not Palmetto - avoids the Palmetto performance cliff found earlier this
session, and this experiment's own metric is inherently a BEFORE/AFTER
delta on the same reference corpus, not a cross-paper absolute number),
and llm.client.LLMClient for the actual Mistral-7B-Instruct-v0.3 4-bit
generation (identical LLM/settings for every backbone, per this task's
own requirement).

Usage:
    python scripts/run_rq4_topic_refinement.py --models fastopic hicot vaebm --datasets 20ng agnews_short --seed 42
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

RESULTS_DIR = REPO_ROOT / "results" / "rq4_llm"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PARTIAL_JSON = RESULTS_DIR / "partial_results.json"
PARTIAL_CSV = RESULTS_DIR / "partial_results.csv"
CHECKPOINT_JSON = RESULTS_DIR / "checkpoint.json"
LLM_CACHE_JSONL = RESULTS_DIR / "llm_cache.jsonl"
RUN_LOG = RESULTS_DIR / "run.log"

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


def log(msg: str) -> None:
    with open(RUN_LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def load_cache() -> dict:
    cache = {}
    if LLM_CACHE_JSONL.exists():
        with open(LLM_CACHE_JSONL) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cache[rec["key"]] = rec["response"]
    return cache


def append_cache(key: str, prompt: str, response: str) -> None:
    with open(LLM_CACHE_JSONL, "a") as f:
        f.write(json.dumps({"key": key, "prompt": prompt, "response": response}) + "\n")


def cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


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
        # Common LLM JSON slip: trailing commas
        blob = re.sub(r",\s*([}\]])", r"\1", blob)
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
    # None of the 10 alternatives are in V: fall back to the V word most similar
    # on average to the alternatives (paper's own fallback rule), excluding
    # words already in the topic.
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
            key = cache_key(prompt)
            if key in cache:
                response = cache[key]
                run_stats["cache_hits"] += 1
            else:
                result = llm_client.generate(prompt)
                response = result.text
                append_cache(key, prompt, response)
                cache[key] = response
                run_stats["llm_calls"] += 1
                run_stats["total_prompt_tokens"] += result.prompt_tokens
                run_stats["total_completion_tokens"] += result.completion_tokens

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


def build_model_and_topics(model_name: str, dataset_id: str, seed: int, k: int):
    from vaebm_benchmark.datasets.simple_registry import load_dataset
    from vaebm_benchmark.experiment.cluster_runner import CLUSTER_MODEL_BUILDERS
    from vaebm_benchmark.experiment.scientific_models import assignment_source_for_model
    from vaebm_benchmark.utils.seeding import set_all_seeds
    import numpy as np

    set_all_seeds(seed)
    documents, labels, num_classes = load_dataset(dataset_id)
    model = CLUSTER_MODEL_BUILDERS[model_name](k, seed, 5000)
    model.fit(documents)

    assignment_source = assignment_source_for_model(model_name)
    if assignment_source == "argmax_theta":
        feature_space = model.get_document_topics(documents)
        clusters = [int(i) for i in np.argmax(np.asarray(feature_space), axis=1)]
    else:
        clusters = model.get_document_clusters(documents)

    topics = model.get_topics(top_n=TOP_N)
    tokenized_corpus = [d.split() for d in documents]
    return topics, tokenized_corpus, clusters, labels


def compute_metrics(topics: list[list[str]], reference_corpus: list[list[str]]) -> dict:
    from vaebm_benchmark.metrics.topic_quality import coherence, topic_diversity, irbo

    cv, _ = coherence(topics, reference_corpus, top_n=TOP_N, measure="c_v")
    npmi, _ = coherence(topics, reference_corpus, top_n=TOP_N, measure="c_npmi")
    td = topic_diversity(topics, top_n=TOP_N)
    irbo_score = irbo(topics, top_n=TOP_N)
    return {"cv": cv, "npmi": npmi, "td": td, "irbo": irbo_score}


def compute_cluster_metrics(clusters, labels) -> dict:
    from vaebm_benchmark.metrics.clustering_quality import compute_clustering_metrics

    m = compute_clustering_metrics(clusters, labels, ["purity", "nmi"])
    return m


def load_checkpoint() -> set:
    if CHECKPOINT_JSON.exists():
        return set(json.load(open(CHECKPOINT_JSON)).get("done", []))
    return set()


def save_checkpoint(done: set) -> None:
    CHECKPOINT_JSON.write_text(json.dumps({"done": sorted(done), "ts": time.time()}, indent=2))


def append_result(row: dict) -> None:
    is_new = not PARTIAL_CSV.exists()
    fields = ["model", "dataset", "k", "seed", "cv_base", "npmi_base", "td_base", "irbo_base",
              "cv_llm", "npmi_llm", "td_llm", "irbo_llm", "purity", "nmi",
              "llm_calls", "cache_hits", "runtime_s", "status", "error"]
    with open(PARTIAL_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in fields})

    all_rows = []
    if PARTIAL_JSON.exists():
        all_rows = json.loads(PARTIAL_JSON.read_text())
    all_rows.append(row)
    PARTIAL_JSON.write_text(json.dumps(all_rows, indent=2, default=str))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--llm-model", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--quantization", default="4bit", choices=["4bit", "none"])
    p.add_argument("--max-new-tokens", type=int, default=200)
    args = p.parse_args()

    from vaebm_benchmark.llm.client import LLMClient

    llm_client = LLMClient(model_name=args.llm_model, quantization=args.quantization, max_new_tokens=args.max_new_tokens)
    cache = load_cache()
    done = load_checkpoint()

    log(f"=== RQ4-A start models={args.models} datasets={args.datasets} k={args.k} seed={args.seed} ===")
    log(f"LLM: {args.llm_model} quantization={args.quantization}")

    for dataset_id in args.datasets:
        for model_name in args.models:
            combo_key = f"{model_name}:{dataset_id}"
            if combo_key in done:
                log(f"SKIP {combo_key} (already in checkpoint)")
                continue

            start = time.perf_counter()
            row = {"model": model_name, "dataset": dataset_id, "k": args.k, "seed": args.seed}
            run_stats = {"llm_calls": 0, "cache_hits": 0, "total_prompt_tokens": 0, "total_completion_tokens": 0}
            log(f"START model={model_name} dataset={dataset_id}")
            try:
                topics, ref_corpus, clusters, labels = build_model_and_topics(model_name, dataset_id, args.seed, args.k)
                base_metrics = compute_metrics(topics, ref_corpus)
                cluster_metrics = compute_cluster_metrics(clusters, labels)

                vocabulary = build_vocabulary(ref_corpus)
                w2v = build_word2vec(ref_corpus, args.seed)
                refined_topics = refine_topics(topics, vocabulary, w2v, llm_client, cache, run_stats)
                llm_metrics = compute_metrics(refined_topics, ref_corpus)

                row.update({
                    "cv_base": base_metrics["cv"], "npmi_base": base_metrics["npmi"],
                    "td_base": base_metrics["td"], "irbo_base": base_metrics["irbo"],
                    "cv_llm": llm_metrics["cv"], "npmi_llm": llm_metrics["npmi"],
                    "td_llm": llm_metrics["td"], "irbo_llm": llm_metrics["irbo"],
                    "purity": cluster_metrics["purity"], "nmi": cluster_metrics["nmi"],
                    "llm_calls": run_stats["llm_calls"], "cache_hits": run_stats["cache_hits"],
                    "runtime_s": round(time.perf_counter() - start, 1), "status": "ok", "error": None,
                })
                log(f"OK   {combo_key} cv {base_metrics['cv']:.4f}->{llm_metrics['cv']:.4f} "
                    f"td {base_metrics['td']:.4f}->{llm_metrics['td']:.4f} "
                    f"llm_calls={run_stats['llm_calls']} cache_hits={run_stats['cache_hits']}")
            except Exception as exc:  # noqa: BLE001
                row.update({"status": "error", "error": f"{exc}", "runtime_s": round(time.perf_counter() - start, 1)})
                log(f"ERROR {combo_key} error={exc!r}")
                import traceback
                traceback.print_exc()

            append_result(row)
            done.add(combo_key)
            save_checkpoint(done)

    log("=== RQ4-A done ===")


if __name__ == "__main__":
    main()
