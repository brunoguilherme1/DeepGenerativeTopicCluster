#!/usr/bin/env python
"""Experiment runner with three modes:

  --experiment topic (default): VAE-BM vs. BERTopic topic-quality
      comparison (C_V, Purity, NMI, TD) at requested K values. See
      src/vaebm_benchmark/experiment/{runner,report}.py.

  --experiment cluster: pure document-clustering-quality comparison
      (ACC via Hungarian-matched accuracy, NMI) across any registered
      model that can produce hard document clusters (vaebm, bertopic,
      fastopic, glocom). K is NOT requested via --k here - it is always
      the benchmark dataset's own number of ground-truth classes,
      inspected only to size K, never used during fit(). See
      src/vaebm_benchmark/experiment/{cluster_runner,cluster_report}.py.

  --experiment llm_cluster_refinement: an LLM-based post-clustering
      refinement stage (LLMEdgeRefine-style) applied on top of the SAME
      hard clusters --experiment cluster produces (reuses its model
      registry) - before/after comparison of ACC/NMI/ARI/AMI/Purity plus
      Silhouette/Davies-Bouldin/Calinski-Harabasz. See
      src/vaebm_benchmark/experiment/{llm_refinement_runner,llm_refinement_report}.py
      and src/vaebm_benchmark/llm/*.py.

Usage (topic):
    python scripts/run_experiment.py --models vaebm --datasets search_snippets --k 50
    python scripts/run_experiment.py --models vaebm bertopic --datasets search_snippets --k 50 100

Usage (cluster):
    python scripts/run_experiment.py --experiment cluster --models vaebm bertopic --datasets search_snippets
    python scripts/run_experiment.py --experiment cluster --models all --datasets all-short
    python scripts/run_experiment.py --experiment cluster --models vaebm bertopic fastopic glocom \\
        --datasets agnews_short search_snippets stack_overflow biomedical google_news_ts google_news_t google_news_s tweet

Usage (llm_cluster_refinement):
    python scripts/run_experiment.py \\
        --experiment llm_cluster_refinement \\
        --models vaebm bertopic fastopic glocom \\
        --datasets search_snippets \\
        --llm-model mistralai/Mistral-7B-Instruct-v0.3 \\
        --edge-fraction 0.10 --candidate-clusters 3 --min-confidence 0.70 \\
        --refinement-iterations 1 --quantization 4bit

`--models all` and `--datasets all-short` expand to every model/dataset
registered for the experiment being run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys


def _parse_ks(raw_values: list[str]) -> list[int]:
    """Accepts either `--k 50 100` or `--k 50,100` (or a mix)."""
    ks: list[int] = []
    for value in raw_values:
        for part in value.split(","):
            part = part.strip()
            if part:
                ks.append(int(part))
    return ks


def _append_csv(csv_path, rows: list[dict]) -> None:
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _merge_json(json_path, rows: list[dict]) -> None:
    existing = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else []
    existing.extend(rows)
    json_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _load_vaebm_configs(raw: str) -> dict:
    """Accepts either a literal JSON object, or a path to a .json/.yaml/
    .yml file containing one - see --vaebm-configs' own help text."""
    import json
    import os

    if os.path.isfile(raw):
        with open(raw, encoding="utf-8") as f:
            text = f.read()
        if raw.endswith((".yaml", ".yml")):
            import yaml

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
    else:
        data = json.loads(raw)

    if not isinstance(data, dict):
        raise SystemExit(
            f"--vaebm-configs must be a JSON/YAML object mapping variant name -> override dict, got {type(data).__name__}"
        )
    return data


def _run_topic(args) -> None:
    from vaebm_benchmark.datasets.simple_registry import list_datasets
    from vaebm_benchmark.experiment.report import (
        render_latex_table_for_k,
        render_table_for_k,
        results_to_rows,
    )
    from vaebm_benchmark.experiment.runner import KNOWN_MODELS, run_sweep
    from vaebm_benchmark.utils.paths import RESULTS_DIR

    models = KNOWN_MODELS if args.models == ["all"] else args.models
    unknown_models = [m for m in models if m not in KNOWN_MODELS]
    if unknown_models:
        raise SystemExit(f"Unknown model(s) for --experiment topic: {unknown_models}. Available: {KNOWN_MODELS}")

    datasets = list_datasets() if args.datasets == ["all"] else args.datasets
    unknown = [d for d in datasets if d not in list_datasets()]
    if unknown:
        raise SystemExit(f"Unknown dataset(s) {unknown}. Available: {list_datasets()}")

    if not args.k:
        raise SystemExit("--experiment topic requires --k (e.g. --k 50 or --k 50 100)")
    ks = _parse_ks(args.k)

    print(f"Running [topic]: models={models} datasets={datasets} k={ks} seed={args.seed}\n")
    results = run_sweep(models, datasets, ks, seed=args.seed)

    for result in results:
        if result.status != "ok":
            print(f"[ERROR] model={result.model} dataset={result.dataset} k={result.k}: {result.error.splitlines()[0]}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    topics_dir = RESULTS_DIR / "experiment" / "topics"
    topics_dir.mkdir(parents=True, exist_ok=True)

    rows = results_to_rows(results)
    csv_path = RESULTS_DIR / "experiment_results.csv"
    json_path = RESULTS_DIR / "experiment_results.json"
    _append_csv(csv_path, rows)
    _merge_json(json_path, rows)

    for result in results:
        topics_path = topics_dir / f"{result.model}_{result.dataset}_k{result.k}_seed{result.seed}.json"
        topics_path.write_text(
            json.dumps({"topics_energy": result.topics_energy, "topics_freq": result.topics_freq}, indent=2),
            encoding="utf-8",
        )

    print()
    written_tables = []
    for k in ks:
        table_text = render_table_for_k(results, k)
        print(table_text)
        print()

        txt_path = RESULTS_DIR / f"table_k{k}.txt"
        txt_path.write_text(table_text + "\n", encoding="utf-8")
        written_tables.append(txt_path)

        tex_path = RESULTS_DIR / f"table_k{k}.tex"
        tex_path.write_text(render_latex_table_for_k(results, k) + "\n", encoding="utf-8")
        written_tables.append(tex_path)

    print(f"Results appended to: {csv_path}")
    print(f"Results merged into: {json_path}")
    print(f"Topics written to: {topics_dir}/")
    for path in written_tables:
        print(f"Table written to: {path}")


def _run_cluster(args) -> None:
    from vaebm_benchmark.datasets.simple_registry import list_short_text_datasets
    from vaebm_benchmark.experiment.cluster_report import (
        cluster_results_to_rows,
        render_cluster_latex_table,
        render_cluster_table,
    )
    from vaebm_benchmark.experiment.cluster_runner import list_cluster_models, run_sweep
    from vaebm_benchmark.utils.paths import RESULTS_DIR

    all_models = list_cluster_models()
    models = all_models if args.models == ["all"] else args.models
    unknown_models = [m for m in models if m not in all_models]
    if unknown_models:
        raise SystemExit(f"Unknown model(s) for --experiment cluster: {unknown_models}. Available: {all_models}")

    datasets = list_short_text_datasets() if args.datasets == ["all-short"] else args.datasets

    if args.k:
        print("Note: --k is ignored for --experiment cluster - K is always each dataset's own num_classes.")

    seed_desc = f"seeds={args.seeds}" if args.seeds else f"seed={args.seed}"
    print(f"Running [cluster]: models={models} datasets={datasets} {seed_desc}\n")
    results = run_sweep(models, datasets, seed=args.seed, voc_size=args.voc_size, seeds=args.seeds)

    for result in results:
        if result.status != "ok":
            print(f"[ERROR] model={result.model} dataset={result.dataset}: {result.error.splitlines()[0]}")
        else:
            print(f"[ok] model={result.model} dataset={result.dataset} "
                  f"requested_k={result.requested_k} actual_k={result.actual_k} num_classes={result.num_classes}")

    cluster_dir = RESULTS_DIR / "cluster"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    rows = cluster_results_to_rows(results)
    csv_path = cluster_dir / "cluster_results.csv"
    json_path = cluster_dir / "cluster_results.json"
    _append_csv(csv_path, rows)
    _merge_json(json_path, rows)

    print()
    table_text = render_cluster_table(results, fmt=args.format)
    print(table_text)

    txt_path = cluster_dir / "table.txt"
    txt_path.write_text(table_text + "\n", encoding="utf-8")
    tex_path = cluster_dir / "table.tex"
    tex_path.write_text(render_cluster_latex_table(results, fmt=args.format) + "\n", encoding="utf-8")

    print(f"\nResults appended to: {csv_path}")
    print(f"Results merged into: {json_path}")
    print(f"Table written to: {txt_path}")
    print(f"Table written to: {tex_path}")


def _run_llm_cluster_refinement(args) -> None:
    from vaebm_benchmark.datasets.simple_registry import list_short_text_datasets
    from vaebm_benchmark.experiment.cluster_runner import list_cluster_models
    from vaebm_benchmark.experiment.llm_refinement_report import (
        llm_refinement_results_to_rows,
        render_compact_table,
        render_detailed_table,
    )
    from vaebm_benchmark.experiment.llm_refinement_runner import RefinementConfig, run_sweep
    from vaebm_benchmark.utils.paths import RESULTS_DIR

    all_models = list_cluster_models()  # same registry the `cluster` experiment uses - no separate LLM-refinement registry to keep in sync
    models = all_models if args.models == ["all"] else args.models
    unknown_models = [m for m in models if m not in all_models]
    if unknown_models:
        raise SystemExit(f"Unknown model(s) for --experiment llm_cluster_refinement: {unknown_models}. Available: {all_models}")

    datasets = list_short_text_datasets() if args.datasets == ["all-short"] else args.datasets

    config = RefinementConfig(
        llm_model=args.llm_model,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        quantization=args.quantization,
        edge_fraction=args.edge_fraction,
        edge_representation=args.edge_representation,
        candidate_clusters=args.candidate_clusters,
        min_confidence=args.min_confidence,
        refinement_iterations=args.refinement_iterations,
        top_words=args.top_words,
        top_docs=args.top_docs,
        voc_size=args.voc_size,
        compute_topic_metrics=args.topic_metrics,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        cache_dir=RESULTS_DIR / "llm_cache",
    )

    print(f"Running [llm_cluster_refinement]: models={models} datasets={datasets} seed={args.seed}")
    print(f"  llm_model={config.llm_model} quantization={config.quantization} "
          f"edge_fraction={config.edge_fraction} edge_representation={config.edge_representation} "
          f"candidate_clusters={config.candidate_clusters} min_confidence={config.min_confidence} "
          f"refinement_iterations={config.refinement_iterations} resume={config.resume}\n")

    results = run_sweep(models, datasets, seed=args.seed, config=config)

    for result in results:
        if result.status != "ok":
            print(f"[ERROR] model={result.model} dataset={result.dataset}: {result.error.splitlines()[0]}")
            continue
        acc_summary = f"acc {result.acc_before:.3f}->{result.acc_after:.3f}" if result.acc_before is not None else "acc N/A"
        print(f"[ok] model={result.model} dataset={result.dataset} "
              f"edge_points={result.number_of_edge_points} ({result.percentage_documents_sent_to_llm:.1f}% of docs) "
              f"llm_calls={result.number_of_llm_calls} cache_hits={result.number_of_cache_hits} "
              f"reassignments={result.number_of_reassignments} {acc_summary}")

    refinement_dir = RESULTS_DIR / "llm_refinement"
    refinement_dir.mkdir(parents=True, exist_ok=True)

    rows = llm_refinement_results_to_rows(results)
    # Nested dict fields (native_topic_metrics, cluster_derived_topic_metrics_*) are
    # JSON-stringified for the CSV only - the JSON export keeps them as real nested objects.
    dict_fields = ["native_topic_metrics", "cluster_derived_topic_metrics_before", "cluster_derived_topic_metrics_after"]
    csv_rows = [
        {**row, **{field: json.dumps(row[field]) for field in dict_fields}}
        for row in rows
    ]
    csv_path = refinement_dir / "llm_refinement_results.csv"
    json_path = refinement_dir / "llm_refinement_results.json"
    _append_csv(csv_path, csv_rows)
    _merge_json(json_path, rows)

    print()
    print(render_detailed_table(results))
    print()
    print(render_compact_table(results))

    print(f"\nResults appended to: {csv_path}")
    print(f"Results merged into: {json_path}")
    print(f"LLM decision cache: {config.cache_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="topic", choices=["topic", "cluster", "llm_cluster_refinement"])
    parser.add_argument("--models", nargs="+", required=True, help="Model names, or 'all'")
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset ids, or 'all' (topic) / 'all-short' (cluster)")
    parser.add_argument("--k", nargs="+", default=None, help="Topic experiment only: one or more topic/cluster counts, e.g. --k 50 100 or --k 50,100")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                         help="Cluster experiment only: run every (model, dataset) once per seed and average "
                              "ACC/NMI across them for the printed/exported table (e.g. --seeds 1 2 3 4 5 for a "
                              "reference-style 'averaged over N random runs' table). Every individual seed's "
                              "result is still persisted to cluster_results.csv/json. Overrides --seed if given.")
    parser.add_argument("--voc-size", type=int, default=5000, help="Vectorizer vocabulary cap (VAE-BM/FASTopic/GloCOM)")
    parser.add_argument("--format", default="percent", choices=["percent", "decimal"], help="Cluster table print format (CSV/JSON always store decimals)")
    parser.add_argument(
        "--vaebm-embedder", default=None,
        help="Topic experiment only: VAE-BM's embedding branch input - any SentenceTransformer model name (default: "
             "VAEBMAdapter's own, all-MiniLM-L6-v2), or 'bag'/'bow'/'tfidf' to use a classical vectorizer there "
             "instead of a neural sentence embedding. Applies to EVERY vaebm-family model (bare 'vaebm' and any "
             "--vaebm-configs variant that doesn't itself override 'embedder') - for a per-variant override instead, "
             "set 'embedder' inside that variant's --vaebm-configs entry.",
    )
    parser.add_argument(
        "--vaebm-vectorizer-type", default=None, choices=["tfidf", "count"],
        help="Topic experiment only: bag-of-words vectorizer for VAE-BM's BoW branch (default: VAEBMAdapter's own, "
             "tfidf). Applies to EVERY vaebm-family model unless a --vaebm-configs variant overrides "
             "'vectorizer_type' itself.",
    )
    parser.add_argument(
        "--vaebm-configs", default=None,
        help="Topic experiment only: define any number of named VAE-BM configurations, each an arbitrary set of "
             "VAEBMAdapter overrides (alpha, units, dim, dim_emb, epochs, batch_size, lr, vectorizer_type, embedder, "
             "top_words_mode - anything VAEBMAdapter's own __init__ accepts, except n_clusters/voc_size/random_state, "
             "which stay controlled by --k/--voc-size/--seed for every model). Accepts either a literal JSON object "
             "or a path to a .json/.yaml/.yml file, e.g. "
             '\'{"vaebm_a05": {"alpha": 0.5}, "vaebm_deep": {"alpha": 0.9, "units": 100}}\' '
             "- include the names you define here in --models to run them. A key here always wins over "
             "--vaebm-embedder/--vaebm-vectorizer-type for that same variant. See "
             "experiment/runner.py's register_vaebm_variants().",
    )

    # --- llm_cluster_refinement only ---
    llm_group = parser.add_argument_group("llm_cluster_refinement")
    llm_group.add_argument("--llm-model", default="mistralai/Mistral-7B-Instruct-v0.3",
                            help="Hugging Face model id. Not loaded unless the experiment actually needs it "
                                 "(e.g. every edge point already cache-hit) - see llm/client.py.")
    llm_group.add_argument("--max-new-tokens", type=int, default=64)
    llm_group.add_argument("--device", default="auto", help="Passed to transformers' device_map.")
    llm_group.add_argument("--quantization", default="4bit", choices=["4bit", "none"],
                            help="4bit requires a CUDA device (bitsandbytes); 'none' runs full precision "
                                 "(correct but very slow for a 7B model on CPU).")
    llm_group.add_argument("--edge-fraction", type=float, default=0.10,
                            help="Per-cluster fraction of documents (ranked by distance to their own cluster's "
                                 "centroid) sent to the LLM for refinement.")
    llm_group.add_argument("--edge-representation", default="shared", choices=["native", "shared"],
                            help="'native' = each model's own representation (VAE-BM's latent mu, etc.); "
                                 "'shared' (default) = the same sentence-embedding model for every baseline, "
                                 "making edge detection/candidate selection comparable across backbones.")
    llm_group.add_argument("--candidate-clusters", type=int, default=3,
                            help="Current cluster + this many total candidates offered to the LLM per edge document.")
    llm_group.add_argument("--min-confidence", type=float, default=0.70,
                            help="A document only changes cluster if the LLM's confidence is >= this threshold.")
    llm_group.add_argument("--refinement-iterations", type=int, default=1,
                            help="Recompute centroids/edge points and repeat this many times.")
    llm_group.add_argument("--top-words", type=int, default=15, help="Representative words per cluster in the LLM prompt.")
    llm_group.add_argument("--top-docs", type=int, default=3, help="Representative example documents per cluster in the LLM prompt.")
    llm_group.add_argument("--topic-metrics", action="store_true",
                            help="Also compute C_V/NPMI/TD/IRBO for native and cluster-derived (c-TF-IDF) topics, "
                                 "before and after refinement (kept as separate metric families - never described "
                                 "as native FASTopic/GloCOM topics).")
    llm_group.add_argument("--checkpoint-every", type=int, default=20,
                            help="Flush the LLM decision cache (and, with --resume, progress state) after this many decisions.")
    llm_group.add_argument("--resume", action="store_true",
                            help="Reload any partial refinement progress for this exact configuration and continue, "
                                 "rather than starting over - see llm/resume.py.")

    args = parser.parse_args()

    if args.vaebm_embedder or args.vaebm_vectorizer_type or args.vaebm_configs:
        if args.experiment != "topic":
            raise SystemExit(
                "--vaebm-embedder/--vaebm-vectorizer-type/--vaebm-configs are topic-experiment only "
                "(cluster/llm_cluster_refinement have their own VAE-BM builder)"
            )
        from vaebm_benchmark.experiment.runner import register_vaebm_variants, set_vaebm_defaults

        defaults = {}
        if args.vaebm_embedder:
            defaults["embedder"] = args.vaebm_embedder
        if args.vaebm_vectorizer_type:
            defaults["vectorizer_type"] = args.vaebm_vectorizer_type
        if defaults:
            set_vaebm_defaults(**defaults)

        if args.vaebm_configs:
            register_vaebm_variants(_load_vaebm_configs(args.vaebm_configs))

    if args.experiment == "topic":
        _run_topic(args)
    elif args.experiment == "cluster":
        _run_cluster(args)
    else:
        _run_llm_cluster_refinement(args)


if __name__ == "__main__":
    sys.exit(main())
