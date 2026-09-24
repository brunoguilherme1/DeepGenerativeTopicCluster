#!/usr/bin/env python
"""SBERT+KMeans (all-MiniLM-L6-v2) transductive clustering (Purity/NMI) on the
3 FASTopic-protocol datasets (fastopic_20ng, fastopic_nyt,
fastopic_wos_reconstructed), for comparison against FASTopic's own published
Table 2 numbers. Not part of FASTopicProtocol.build_model() (which is
deliberately vaebm/hicot only) - built directly here since sbert_kmeans is
this project's own lightweight baseline, not one of FASTopic's rerun-forbidden
reference rows.

Usage:
    python scripts/run_sbert_kmeans_fastopic_datasets.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SEED = 42
DATASETS = ["fastopic_20ng", "fastopic_nyt", "fastopic_wos_reconstructed"]


def run_cluster(protocol, dataset_id):
    from sklearn.metrics import normalized_mutual_info_score
    from sklearn.metrics.cluster import contingency_matrix
    import numpy as np
    from vaebm_benchmark.models.sbert_kmeans_adapter import SBERTKMeansAdapter

    all_texts, all_labels = protocol.prepare_all_documents_and_labels(dataset_id)
    model = SBERTKMeansAdapter(
        n_clusters=protocol.topic_count[dataset_id],
        embedder="all-MiniLM-L6-v2",
        random_state=SEED,
    )
    model.fit(all_texts)
    preds = np.asarray(model.get_document_clusters(all_texts))
    labels_arr = np.asarray(all_labels)

    cm = contingency_matrix(labels_arr, preds)
    purity = float(np.sum(np.amax(cm, axis=0)) / np.sum(cm))
    nmi = float(normalized_mutual_info_score(labels_arr, preds))
    return {"purity": purity, "nmi": nmi}


def main():
    from vaebm_benchmark.protocols.fastopic_protocol import FASTopicProtocol

    protocol = FASTopicProtocol(smoke_test=True)
    print("=== SBERT+KMeans (all-MiniLM-L6-v2) clustering on FASTopic datasets ===")

    results = []
    for dataset_id in DATASETS:
        start = time.perf_counter()
        print(f"START dataset={dataset_id} ...", flush=True)
        try:
            r = run_cluster(protocol, dataset_id)
            runtime = time.perf_counter() - start
            print(f"OK   dataset={dataset_id} result={r} runtime={runtime:.1f}s")
            results.append({"dataset": dataset_id, "status": "ok", "result": r, "runtime_s": runtime})
        except Exception as exc:  # noqa: BLE001
            runtime = time.perf_counter() - start
            print(f"ERROR dataset={dataset_id} error={exc!r}")
            import traceback
            traceback.print_exc()
            results.append({"dataset": dataset_id, "status": "error", "error": repr(exc), "runtime_s": runtime})

    print("\n=== SUMMARY ===")
    for r in results:
        tag = "OK  " if r["status"] == "ok" else "FAIL"
        detail = r.get("result", r.get("error"))
        print(f"  {tag} {r['dataset']:28s} {detail}")

    print("\n=== PUBLISHED REFERENCE (FASTopic paper, for comparison only) ===")
    for pr in protocol.published_results:
        if pr.metric in ("purity", "nmi"):
            print(f"  {pr.dataset_id:28s} {pr.metric:10s} = {pr.value}  ({pr.source})")


if __name__ == "__main__":
    main()
