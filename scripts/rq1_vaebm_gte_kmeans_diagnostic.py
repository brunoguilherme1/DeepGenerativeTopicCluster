#!/usr/bin/env python
"""RQ1 diagnostic (appendix-level, NOT a main topic-model baseline): fit
VAE-BM ONCE per dataset with a stronger documented embedder (GTE-large,
per arXiv:2405.17978 Appendix F's own finding that a higher-quality
embedder helps clustering specifically), then sweep K externally via
KMeans on the frozen latent mu (confirmed by direct code inspection:
`model.get_mu(documents)` returns a stable, already-`.numpy()`-converted
array unaffected by the `n_clusters` passed at construction, so this
needs exactly one VAE fit per dataset, not one per K).

This is a K-sensitivity ANALYSIS across the 5 official RQ1 datasets, not
a hyperparameter search - no config is "selected"; every K's full metrics
are reported and preserved (including negative results, e.g. if a larger
K doesn't help).

Topic words per swept-K cluster are derived via c-TF-IDF (since the
model's own get_topics() reflects its internally-fixed n_clusters, not
the externally swept K) - the same technique already used in
run_rq4_cluster_refinement.py's cluster_top_words(), reused here for
consistency. C_V/NPMI use the local-corpus gensim measure (not Palmetto)
given this sweeps 8 K values x 5 datasets = 40 conditions from a single
fit each - a documented compute-budget choice, not a claim of exact
Palmetto-C_V equivalence.

Usage:
    python scripts/rq1_vaebm_gte_kmeans_diagnostic.py --datasets hicot_20ng hicot_imdb hicot_agnews hicot_search_snippets hicot_google_news
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.cluster import KMeans

import rq4_common as common

RESULTS_DIR = REPO_ROOT / "results" / "rq1_diagnostics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = RESULTS_DIR / "rq1_vaebm_gte_kmeans_diagnostic.csv"
RUN_LOG = RESULTS_DIR / "rq1_vaebm_gte_kmeans_diagnostic.log"

SEED = 42
K_VALUES = [20, 40, 50, 60, 70, 80, 90, 100]
EMBEDDER = "thenlper/gte-large"
RESULT_FIELDS = (["dataset", "embedder", "k", "seed"] + list(common.ALL_METRIC_KEYS) + ["runtime_s", "status", "error"])


def log(msg: str) -> None:
    with open(RUN_LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def load_done() -> set:
    if not RESULTS_CSV.exists():
        return set()
    with open(RESULTS_CSV) as f:
        return {(row["dataset"], row["k"]) for row in csv.DictReader(f)}


def append_result(row: dict) -> None:
    is_new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in RESULT_FIELDS})


def cluster_top_words(tokenized_docs: list[list[str]], cluster_ids: list[int], top_n: int = 10) -> list[list[str]]:
    cluster_counts: dict = {}
    word_cluster_presence = Counter()
    for tokens, cid in zip(tokenized_docs, cluster_ids):
        cluster_counts.setdefault(cid, Counter()).update(tokens)
    for c in cluster_counts.values():
        word_cluster_presence.update(c.keys())
    num_clusters = max(len(cluster_counts), 1)
    topics = []
    for cid in sorted(cluster_counts):
        counter = cluster_counts[cid]
        scores = {w: cnt * np.log(1 + num_clusters / (1 + word_cluster_presence[w])) for w, cnt in counter.items()}
        topics.append(sorted(scores, key=scores.get, reverse=True)[:top_n])
    return topics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True)
    args = p.parse_args()

    from vaebm_benchmark.datasets.simple_registry import load_dataset
    from vaebm_benchmark.models.vaebm_adapter import VAEBMAdapter

    done = load_done()
    log(f"=== RQ1 VAE-BM GTE+KMeans K-sensitivity diagnostic start datasets={args.datasets} k_values={K_VALUES} embedder={EMBEDDER} ===")

    for dataset_id in args.datasets:
        documents, labels, num_classes = load_dataset(dataset_id)
        tokenized_corpus = [d.lower().split() for d in documents]
        log(f"Dataset {dataset_id}: n_docs={len(documents)} num_classes={num_classes}")

        start_fit = time.perf_counter()
        model = VAEBMAdapter(n_clusters=num_classes, voc_size=5000, embedder=EMBEDDER, alpha=0.0, random_state=SEED)
        model.fit(documents)
        mu = model.get_mu(documents)
        fit_runtime = time.perf_counter() - start_fit
        log(f"  fit done in {fit_runtime:.1f}s, mu shape={mu.shape}")

        for k in K_VALUES:
            if (dataset_id, str(k)) in done:
                log(f"  SKIP {dataset_id} k={k} (already done)")
                continue
            start = time.perf_counter()
            status, error = "ok", None
            metrics = {}
            try:
                clusters = KMeans(n_clusters=k, random_state=SEED, n_init="auto").fit_predict(mu)
                topics = cluster_top_words(tokenized_corpus, clusters.tolist())
                metrics = common.compute_full_metric_suite(topics, tokenized_corpus, clusters.tolist(), labels, mu)
                log(f"  OK {dataset_id} k={k} acc={metrics.get('acc'):.4f} nmi={metrics.get('nmi'):.4f} "
                    f"cv={metrics.get('cv')} runtime={time.perf_counter()-start:.1f}s")
            except Exception as exc:  # noqa: BLE001
                status, error = "error", f"{exc}"
                log(f"  ERROR {dataset_id} k={k} error={exc!r}")
                import traceback
                traceback.print_exc()

            row = {"dataset": dataset_id, "embedder": EMBEDDER, "k": k, "seed": SEED,
                   "runtime_s": round(time.perf_counter() - start, 1), "status": status, "error": error}
            row.update(metrics)
            append_result(row)

    log("=== RQ1 VAE-BM GTE+KMeans K-sensitivity diagnostic done ===")


if __name__ == "__main__":
    main()
