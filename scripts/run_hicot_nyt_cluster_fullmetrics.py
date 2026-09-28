#!/usr/bin/env python
"""HiCOT cluster experiment on NYT (fastopic_nyt artifact - the only NYT
dataset this project has; not registered in simple_registry.py, so
--experiment cluster can't reach it directly), reporting the SAME full
11-metric set --experiment cluster does (label-based: ACC/NMI/ARI/AMI/
Homogeneity/Completeness/V-measure/Purity via compute_clustering_metrics;
label-free geometry: Silhouette/Davies-Bouldin/Calinski-Harabasz via
compute_geometry_metrics), transductive full-corpus fit, argmax(theta) hard
assignment - mirroring cluster_runner.py::run_single's own logic exactly,
just pointed at FASTopicProtocol's NYT loader instead of simple_registry.

Usage:
    python scripts/run_hicot_nyt_cluster_fullmetrics.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SEED = 42


def main():
    import numpy as np
    from vaebm_benchmark.metrics.clustering_quality import compute_clustering_metrics, compute_geometry_metrics
    from vaebm_benchmark.protocols.fastopic_protocol import FASTopicProtocol
    from vaebm_benchmark.utils.seeding import set_all_seeds

    set_all_seeds(SEED)

    protocol = FASTopicProtocol(smoke_test=False)  # full epochs (hicot=200)
    dataset_id = "fastopic_nyt"

    print(f"Loading {dataset_id}...", flush=True)
    all_texts, all_labels = protocol.prepare_all_documents_and_labels(dataset_id)
    print(f"docs={len(all_texts)} K={protocol.topic_count[dataset_id]}", flush=True)

    start = time.perf_counter()
    model = protocol.build_hicot(dataset_id, SEED, config={"max_fit_seconds": 600})
    print("model built, fitting...", flush=True)
    model.fit(all_texts)
    print(f"fit done in {time.perf_counter()-start:.1f}s", flush=True)

    feature_space = model.get_document_topics(all_texts)
    clusters = [int(i) for i in np.argmax(np.asarray(feature_space), axis=1)]
    actual_k = len(set(clusters))
    print(f"actual_k={actual_k}", flush=True)

    label_metric_ids = ["acc", "nmi", "ari", "ami", "homogeneity", "completeness", "v_measure", "purity"]
    geometry_metric_ids = ["silhouette", "davies_bouldin", "calinski_harabasz"]

    label_metrics = compute_clustering_metrics(clusters, all_labels, label_metric_ids)
    try:
        geometry_metrics = compute_geometry_metrics(feature_space, clusters, geometry_metric_ids)
    except Exception as exc:  # noqa: BLE001
        print(f"geometry metrics failed: {exc}")
        geometry_metrics = {name: None for name in geometry_metric_ids}

    print("\n=== HiCOT on NYT (fastopic_nyt), K={}, seed={} ===".format(protocol.topic_count[dataset_id], SEED))
    print(f"{'Metric':20s} {'Value':>12s}")
    print("-" * 34)
    for name in label_metric_ids:
        v = label_metrics.get(name)
        print(f"{name:20s} {v*100:11.1f}%" if v is not None else f"{name:20s} {'N/A':>12s}")
    for name in geometry_metric_ids:
        v = geometry_metrics.get(name)
        print(f"{name:20s} {v:12.4f}" if v is not None else f"{name:20s} {'N/A':>12s}")
    print(f"\nactual_k={actual_k} num_classes={len(set(all_labels))} runtime_s={time.perf_counter()-start:.1f}")


if __name__ == "__main__":
    main()
