#!/usr/bin/env python
"""RQ4, Experiment B: reasoning-based cluster refinement, per Islam 2026
"Reasoning-Based Refinement of Unsupervised Text Clusters with LLMs"
(Findings ACL 2026, 2026.findings-acl.482).

Implements the paper's own three stages, treating each backbone's existing
hard cluster assignment (fastopic/hicot/vaebm's argmax-theta or native
cluster labels) as the "proposal step" the paper's framework is designed to
be agnostic to:

  1. Coherence verification - for each cluster, an LLM-written one-sentence
     summary of its 5 centroid-nearest documents is checked by the LLM
     against those same 5 documents; incoherent clusters are discarded.
  2. Redundancy adjudication - surviving clusters' summaries are SBERT-
     embedded; pairs with cosine similarity >= the paper's own selected
     threshold (tau=0.85) are merged, with the LLM asked for one
     consolidated summary per merged group.
  3. Label grounding (two-stage) - each surviving/merged cluster gets an
     LLM-generated short label from its summary; labels are SBERT-embedded
     and semantically similar ones (same tau=0.85) are consolidated by the
     LLM into one label per group.

This repo's existing --experiment llm_cluster_refinement code (edge-based
per-document reassignment near decision boundaries) is a DIFFERENT method
and is intentionally NOT reused here as if it reproduced this paper - only
LLMClient and the cache/checkpoint infrastructure are shared (via
rq4_common), per instruction not to overclaim reproduction.

Two deliberate, documented deviations from the paper (real-compute-budget
driven, applied IDENTICALLY to every backbone so no backbone is favored):

  - Representative-document selection uses a SHARED SBERT embedding space
    (all-MiniLM-L6-v2) for centroid distance, since fastopic/hicot/vaebm's
    own native feature spaces are not mutually comparable and the paper's
    own pipeline (HDBSCAN on UMAP(SVD(TF-IDF))) has no direct analogue here.
  - The paper's final "assign label to individual text" step re-labels
    EVERY document with one LLM call each; at 18k+ documents per dataset
    that is computationally infeasible within the smoke-test/full-run
    budget available here. Instead: documents from a merged/surviving
    cluster keep that cluster's (merged) id; documents whose original
    cluster was discarded (Stage 1) are reassigned, WITHOUT any further
    LLM call, to the nearest surviving cluster's centroid in the same
    shared SBERT space. No ground-truth label is ever used in this step.

Usage:
    python scripts/run_rq4_cluster_refinement.py --models fastopic hicot vaebm --datasets 20ng agnews_short --seed 42
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import rq4_common as common

METHOD = "2026_cluster_refinement"
TOP_K_REPRESENTATIVE = 5   # paper's own choice (App. 4.2.1: "top-5 documents closest to the centroid")
MERGE_THRESHOLD = 0.85     # paper's own selected grid-search value (tau)
DOC_CHAR_LIMIT = 400       # bound prompt length; identical for every backbone
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

SUMMARY_PROMPT = """Write ONE concise sentence describing the shared topic or theme of the following {n} texts.

{numbered}

Respond with only the one-sentence summary, no preamble."""

COHERENCE_PROMPT = """Cluster summary: "{summary}"

Representative texts from this cluster:
{numbered}

Does the summary above accurately and coherently describe a common theme shared by ALL of the representative texts? Respond with exactly one word: Yes or No."""

MERGE_SUMMARY_PROMPT = """The following cluster summaries were found to be semantically redundant (describing overlapping themes):

{numbered}

Write ONE concise consolidated sentence that captures the shared theme of all of them. Respond with only the one-sentence summary, no preamble."""

LABEL_PROMPT = """Cluster summary: "{summary}"

Generate ONE short, human-readable label (1-4 words) for this cluster's topic. Respond with only the label, no preamble, no punctuation."""

LABEL_MERGE_PROMPT = """The following cluster labels were found to be semantically similar:

{numbered}

Generate ONE short, consolidated label (1-4 words) that represents all of them. Respond with only the label, no preamble, no punctuation."""


class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def groups(self):
        out: dict = {}
        for i in self.parent:
            out.setdefault(self.find(i), []).append(i)
        return list(out.values())


def numbered_texts(texts: list[str]) -> str:
    return "\n".join(f"{i + 1}. {t[:DOC_CHAR_LIMIT]}" for i, t in enumerate(texts))


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return a @ b.T


def stage1_coherence_verification(cluster_ids, clusters, documents, doc_emb, llm_client, cache, run_stats):
    """Returns {cluster_id: {"summary": str, "coherent": bool, "rep_idx": [...]}}"""
    clusters = np.asarray(clusters)
    out = {}
    for cid in cluster_ids:
        idx = np.where(clusters == cid)[0]
        if len(idx) == 0:
            continue
        centroid = doc_emb[idx].mean(axis=0, keepdims=True)
        sims = cosine_sim_matrix(doc_emb[idx], centroid).ravel()
        order = idx[np.argsort(-sims)][:TOP_K_REPRESENTATIVE]
        rep_texts = [documents[i] for i in order]

        summary = common.llm_generate_cached(
            llm_client, SUMMARY_PROMPT.format(n=len(rep_texts), numbered=numbered_texts(rep_texts)),
            cache, run_stats).strip()
        coherence_resp = common.llm_generate_cached(
            llm_client, COHERENCE_PROMPT.format(summary=summary, numbered=numbered_texts(rep_texts)),
            cache, run_stats).strip().lower()
        coherent = coherence_resp.startswith("yes")
        out[cid] = {"summary": summary, "coherent": coherent, "rep_idx": order.tolist()}
    return out


def stage2_redundancy_adjudication(coherent_clusters: dict, llm_client, cache, run_stats):
    """Returns (group_id -> consolidated_summary, original_cluster_id -> group_id)."""
    ids = list(coherent_clusters.keys())
    if not ids:
        return {}, {}
    summaries = [coherent_clusters[c]["summary"] for c in ids]
    sbert = common.get_sbert()
    emb = sbert.encode(summaries, show_progress_bar=False, normalize_embeddings=True)
    sim = cosine_sim_matrix(emb, emb)

    uf = UnionFind(ids)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if sim[i, j] >= MERGE_THRESHOLD:
                uf.union(ids[i], ids[j])

    group_summary: dict = {}
    cluster_to_group: dict = {}
    for group in uf.groups():
        group_id = min(group)
        for cid in group:
            cluster_to_group[cid] = group_id
        if len(group) == 1:
            group_summary[group_id] = coherent_clusters[group[0]]["summary"]
        else:
            member_summaries = [coherent_clusters[c]["summary"] for c in sorted(group)]
            numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(member_summaries))
            group_summary[group_id] = common.llm_generate_cached(
                llm_client, MERGE_SUMMARY_PROMPT.format(numbered=numbered), cache, run_stats).strip()
    return group_summary, cluster_to_group


def stage3_label_grounding(group_summary: dict, llm_client, cache, run_stats):
    """Returns group_id -> final_label (two-stage: generate then consolidate)."""
    group_ids = list(group_summary.keys())
    if not group_ids:
        return {}
    raw_labels = {
        gid: common.llm_generate_cached(llm_client, LABEL_PROMPT.format(summary=group_summary[gid]), cache, run_stats)
        .strip().strip(".").splitlines()[0][:60]
        for gid in group_ids
    }
    sbert = common.get_sbert()
    label_texts = [raw_labels[g] for g in group_ids]
    emb = sbert.encode(label_texts, show_progress_bar=False, normalize_embeddings=True)
    sim = cosine_sim_matrix(emb, emb)

    uf = UnionFind(group_ids)
    for i in range(len(group_ids)):
        for j in range(i + 1, len(group_ids)):
            if sim[i, j] >= MERGE_THRESHOLD:
                uf.union(group_ids[i], group_ids[j])

    final_label: dict = {}
    for label_group in uf.groups():
        if len(label_group) == 1:
            final_label[label_group[0]] = raw_labels[label_group[0]]
        else:
            member_labels = [raw_labels[g] for g in sorted(label_group)]
            numbered = "\n".join(f"{i + 1}. {lbl}" for i, lbl in enumerate(member_labels))
            consolidated = common.llm_generate_cached(
                llm_client, LABEL_MERGE_PROMPT.format(numbered=numbered), cache, run_stats).strip().splitlines()[0][:60]
            for g in label_group:
                final_label[g] = consolidated
    return final_label


def reassign_discarded_documents(clusters, coherent_clusters, cluster_to_group, doc_emb):
    """Documents in an incoherent (Stage-1-discarded) cluster are moved to the
    nearest surviving/merged cluster's centroid in SBERT space - no LLM call,
    no ground-truth label used. See module docstring for why this replaces
    the paper's own per-document LLM relabeling step."""
    clusters = np.asarray(clusters)
    final = np.full(len(clusters), -1, dtype=int)
    surviving_group_ids = sorted(set(cluster_to_group.values()))
    group_index = {g: i for i, g in enumerate(surviving_group_ids)}

    for original_cid, group_id in cluster_to_group.items():
        idx = np.where(clusters == original_cid)[0]
        final[idx] = group_index[group_id]

    unassigned = np.where(final == -1)[0]
    if len(unassigned) == 0:
        return final, group_index

    if not surviving_group_ids:
        # Degenerate case: every cluster was flagged incoherent. Fall back to
        # the original assignment untouched rather than losing all documents.
        return clusters, {cid: i for i, cid in enumerate(sorted(set(clusters.tolist())))}

    centroids = np.stack([
        doc_emb[final == group_index[g]].mean(axis=0) for g in surviving_group_ids
    ])
    sims = cosine_sim_matrix(doc_emb[unassigned], centroids)
    nearest = np.argmax(sims, axis=1)
    final[unassigned] = nearest
    return final, group_index


def cluster_top_words(tokenized_docs: list[list[str]], cluster_ids: list[int], top_n: int = 10) -> list[list[str]]:
    """c-TF-IDF-style representative words per final cluster, used only for the
    'topic quality where meaningful' metrics (CV/NPMI/TD/IRBO) - cluster
    refinement itself never touches topic-word lists."""
    cluster_counts: dict = {}
    word_cluster_presence = Counter()
    for tokens, cid in zip(tokenized_docs, cluster_ids):
        c = cluster_counts.setdefault(cid, Counter())
        c.update(tokens)
    for c in cluster_counts.values():
        word_cluster_presence.update(c.keys())
    num_clusters = max(len(cluster_counts), 1)
    topics = []
    for cid in sorted(cluster_counts):
        counter = cluster_counts[cid]
        scores = {w: cnt * np.log(1 + num_clusters / (1 + word_cluster_presence[w])) for w, cnt in counter.items()}
        top = sorted(scores, key=scores.get, reverse=True)[:top_n]
        topics.append(top)
    return topics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--llm-model", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", "none"])
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--max-docs", type=int, default=None, help="Smoke-test only: truncate each dataset to the first N docs.")
    args = p.parse_args()

    from vaebm_benchmark.llm.client import LLMClient

    llm_client = LLMClient(model_name=args.llm_model, quantization=args.quantization, max_new_tokens=args.max_new_tokens)
    cache = common.load_cache()
    done = common.load_checkpoint()

    common.log(f"=== RQ4-B ({METHOD}) start models={args.models} datasets={args.datasets} k={args.k} seed={args.seed} ===")
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
                documents, labels, base_clusters, topics, _ = common.build_model_and_assignment(
                    model_name, dataset_id, args.seed, args.k, top_n_words=10, max_docs=args.max_docs)
                tokenized_corpus = [d.split() for d in documents]
                doc_emb = common.get_doc_embeddings(dataset_id, documents)

                base_topic_words = cluster_top_words(tokenized_corpus, base_clusters)
                base_metrics = common.compute_full_metric_suite(base_topic_words, tokenized_corpus, base_clusters, labels, doc_emb)

                cluster_ids = sorted(set(base_clusters))
                stage1 = stage1_coherence_verification(cluster_ids, base_clusters, documents, doc_emb, llm_client, cache, run_stats)
                coherent_clusters = {c: v for c, v in stage1.items() if v["coherent"]}
                n_discarded = len(stage1) - len(coherent_clusters)

                group_summary, cluster_to_group = stage2_redundancy_adjudication(coherent_clusters, llm_client, cache, run_stats)
                final_labels = stage3_label_grounding(group_summary, llm_client, cache, run_stats)
                final_clusters, group_index = reassign_discarded_documents(base_clusters, coherent_clusters, cluster_to_group, doc_emb)

                final_topic_words = cluster_top_words(tokenized_corpus, final_clusters.tolist())
                refined_metrics = common.compute_full_metric_suite(final_topic_words, tokenized_corpus, final_clusters.tolist(), labels, doc_emb)

                extra = {
                    "n_base_clusters": len(cluster_ids), "n_discarded_incoherent": n_discarded,
                    "n_final_clusters": len(group_index),
                    "final_labels": {str(k_): v for k_, v in final_labels.items()},
                    "group_summaries": {str(k_): v for k_, v in group_summary.items()},
                }
                common.log(f"OK   {combo_key} base_clusters={len(cluster_ids)} discarded={n_discarded} "
                           f"final_clusters={len(group_index)} acc {base_metrics.get('acc'):.4f}->{refined_metrics.get('acc'):.4f} "
                           f"nmi {base_metrics.get('nmi'):.4f}->{refined_metrics.get('nmi'):.4f} "
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

    common.log(f"=== RQ4-B ({METHOD}) done ===")


if __name__ == "__main__":
    main()
