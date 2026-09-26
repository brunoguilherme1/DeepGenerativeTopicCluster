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

TOP_N = 10  # words per topic shown to the LLM and used for every metric

PROMPT_TEMPLATE = """You are refining a topic model's word list. Below are the top words for one topic, in order of importance.

Topic words: {words}

One or more of these words may be a semantic "intruder" - a word that does not fit the coherent theme the other words share. Identify at most 2 intruder words and suggest one coherent replacement word for each, a word that fits the theme implied by the remaining words and is not already in the list.

Respond in exactly this format, one line per intruder (or the single line "NONE" if every word already fits):
REPLACE <intruder_word> WITH <replacement_word>
"""


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


def parse_refinement(response: str, topic_words: list[str]) -> dict[str, str]:
    """Returns {intruder_word: replacement_word}, empty if NONE or unparsable."""
    if "NONE" in response.upper() and "REPLACE" not in response.upper():
        return {}
    out = {}
    for line in response.splitlines():
        m = re.search(r"REPLACE\s+(\S+)\s+WITH\s+(\S+)", line, re.IGNORECASE)
        if m:
            intruder, replacement = m.group(1).strip(".,\"'"), m.group(2).strip(".,\"'")
            if intruder.lower() in [w.lower() for w in topic_words]:
                out[intruder] = replacement
    return out


def refine_topics(topics: list[list[str]], llm_client, cache: dict, run_stats: dict) -> list[list[str]]:
    refined = []
    for topic_words in topics:
        words = topic_words[:TOP_N]
        prompt = PROMPT_TEMPLATE.format(words=", ".join(words))
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

        replacements = parse_refinement(response, words)
        new_words = [replacements.get(w, w) for w in words]
        # De-dup while preserving order (a suggested replacement might already be in the list)
        seen = set()
        deduped = []
        for w in new_words:
            if w.lower() not in seen:
                seen.add(w.lower())
                deduped.append(w)
        refined.append(deduped)
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
    p.add_argument("--max-new-tokens", type=int, default=64)
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

                refined_topics = refine_topics(topics, llm_client, cache, run_stats)
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
