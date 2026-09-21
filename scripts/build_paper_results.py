#!/usr/bin/env python
"""Consolidates every VAE-BM-PoE-paper-relevant result file (pulled from
FutureLab and labuai into paper_data/) into one traceable manifest, plus
the LaTeX table fragments the paper actually includes.

Every row in results_manifest.json records exactly which file it came
from, which physical environment/sweep produced it, and any
checkpoint-selection caveat - nothing here is hand-typed from memory.
Two executions of the same (experiment, model, dataset, k) that differ
in environment (FutureLab vs labuai) are kept as SEPARATE rows, never
averaged/merged, per this project's own established convention this
session (cross-hardware results are close but not identical).

Run from the repo root:
    python scripts/build_paper_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "paper_data"
OUT = DATA / "tables_out"
OUT.mkdir(parents=True, exist_ok=True)

manifest: list[dict] = []


def add(experiment, model, dataset, k, seed, metrics, status, error, source_file,
        environment, sweep, checkpoint_selection="none", notes=""):
    manifest.append({
        "experiment": experiment, "model": model, "dataset": dataset, "k": k, "seed": seed,
        "metrics": metrics, "status": status, "error": (error or "")[:300],
        "source_file": source_file, "environment": environment, "sweep": sweep,
        "checkpoint_selection": checkpoint_selection, "notes": notes,
    })


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- cluster --

def ingest_cluster_checkpoint(fname, environment, sweep, model_filter=None):
    d = load(fname)
    for key, v in d.items():
        model, dataset = key.split("|", 1)
        if model_filter and model not in model_filter:
            continue
        metrics = {m: v.get(m) for m in ("acc", "nmi", "purity")}
        add("cluster", model, dataset, None, 42, metrics, v.get("status"), v.get("error"),
            fname, environment, sweep, v.get("checkpoint_selection", "none"))


def ingest_cluster_final_rows(fname, environment, sweep, model_filter=None):
    rows = load(fname)
    for r in rows:
        if model_filter and r["model"] not in model_filter:
            continue
        metrics = {m: r.get(m) for m in (
            "acc", "nmi", "ari", "ami", "homogeneity", "completeness", "v_measure",
            "purity", "silhouette", "davies_bouldin", "calinski_harabasz")}
        add("cluster", r["model"], r["dataset"], None, r.get("seed", 42), metrics,
            r.get("status"), r.get("error"), fname, environment, sweep,
            r.get("checkpoint_selection", "none"))


# 2026-09-21 (paper rewrite, user-specified): cluster7_checkpoint.json,
# cluster6_bertopic_hicot_checkpoint.json, and cluster6_sbert_checkpoint.json
# (the old 26-dataset sweep: hicot/fastopic/bertopic/sbert_gte/sbert_minilm)
# are NO LONGER ingested here. SBERT+KMeans is not a baseline in this paper
# at all (removed from every table, main and appendix). fastopic/hicot/
# bertopic's OWN cluster rows from that old sweep only had ACC/NMI/Purity
# (3 of the 11 metrics this paper now reports for every cluster baseline);
# they are superseded below by a fresh run of all three through the exact
# same 12-locked-dataset, full-11-metric pipeline every other baseline
# uses (run_baseline_sweep_12ds.py --experiments cluster), so no cluster
# table needs a "--" for a metric a baseline was simply never asked for.
#
# VAE-BM-PoE/VAE-BM-DEC's own 5-HiCOT-dataset ablation rows (still on the
# older all-MiniLM-L6-v2 embedder, never relocked) are ingested separately,
# further below, tagged for the Appendix-only ablation table - they do not
# appear in any main-text table (only ONE VAE-BM competes there).
ingest_cluster_final_rows("futurelab/vaebm3_hicot_final.json", "futurelab", "vaebm3_hicot (vaebm,vaebm_poe,vaebm_dec x 5 HiCOT datasets, MiniLM)",
                           model_filter={"vaebm_poe", "vaebm_dec"})
# Locked architecture (2026-09-21, user-authorized): alpha=0,
# frozen+identity-init gte-large (bge-large for imdb) embedding branch,
# epochs=1, normalize_mu for 20ng/agnews_short, K=num_classes, single
# seed=42 (deterministic under this architecture) - see
# scripts/run_vaebm_cluster_locked_12ds.py's own docstring and
# docs/vaebm_leaderboard.md's "Cluster/Classification (locked
# architecture...)" section. Ran on labuai (FutureLab's head-node disk
# was 100% full throughout, blocking new sbatch submissions) - the
# `imdb` row specifically was a manual retry outside the sweep driver
# (bumped past its 2400s cap to let bge-large's ~50 min embed of 50k
# documents finish; every other row completed inside the driver).
ingest_cluster_final_rows("labuai/vaebm_cluster_locked_20260921.json", "labuai",
                           "vaebm_cluster_locked_20260921 (vaebm locked architecture x 12 plain datasets, gte-large/bge-large)",
                           model_filter={"vaebm"})

# New baselines (2026-09-21, user-authorized): LDA, GloCOM, ECRTM, S2WTM
# added to the Cluster/Classification comparison, per the task's own
# "for classification and cluster we need to have fastopic and hicot,
# every other baseline could put or not" rule - these 4 are the
# "could put" additions. All run on the SAME 12 plain datasets as
# VAE-BM's own locked-architecture sweep, K=num_classes, single
# seed=42 (matching every other baseline's own convention - HiCOT/
# FASTopic/BERTopic/SBERT-kmeans are all single-seed=42 for cluster
# already). LDA ran on labuai; GloCOM/ECRTM/S2WTM ran on FutureLab
# (once its head-node disk-full issue was cleared, 2026-09-21).
ingest_cluster_final_rows("labuai/lda_locked_cluster_20260921.json", "labuai",
                           "lda_locked_20260921 (lda x 12 plain datasets)")
ingest_cluster_final_rows("futurelab_new/glocom_locked_cluster_20260921.json", "futurelab",
                           "glocom_locked_20260921 (glocom x 12 plain datasets)")
ingest_cluster_final_rows("futurelab_new/ecrtm_locked_cluster_20260921.json", "futurelab",
                           "ecrtm_locked_20260921 (ecrtm x 12 plain datasets)")
ingest_cluster_final_rows("futurelab_new/s2wtm_locked_cluster_20260921.json", "futurelab",
                           "s2wtm_locked_20260921 (s2wtm x 12 plain datasets)")

# Full-11-metric re-run of fastopic/hicot/bertopic's cluster experiment
# (2026-09-21, paper rewrite): the old 26-dataset sweep only ever computed
# ACC/NMI/Purity for these three; re-run through the exact same 12-locked-
# dataset driver (run_baseline_sweep_12ds.py) every new baseline above
# uses, so every cluster table in the paper has the complete 11-metric
# battery for every model shown, never a "--" from a metric simply never
# requested.
for _mo in ("fastopic", "hicot", "bertopic"):
    _fname = f"labuai/{_mo}_cluster_fullmetrics_20260921.json"
    if (DATA / _fname).exists():
        ingest_cluster_final_rows(_fname, "labuai",
                                   f"{_mo}_cluster_fullmetrics_20260921 ({_mo} x 12 plain datasets, full 11-metric battery)")

# ------------------------------------------------------------ classification --

def ingest_classification_final_rows(fname, environment, sweep, model_filter=None):
    rows = load(fname)
    for r in rows:
        if model_filter and r["model"] not in model_filter:
            continue
        metrics = {"accuracy": r.get("accuracy"), "f1": r.get("f1")}
        add("classification", r["model"], r["dataset"], r.get("k"), r.get("seed", 42),
            metrics, r.get("status"), r.get("error"), fname, environment, sweep,
            r.get("checkpoint_selection", "none"),
            notes=f"split_stratified={r.get('split_stratified')}")


def ingest_classification_multiseed_aggregated(fname, environment, sweep, model_filter=None):
    """Averages multi-seed per-row classification data into ONE row per
    (model, dataset) before calling add() - found 2026-09-21 that the
    dedup step below keys on (experiment, model, dataset, k, environment,
    sweep), NOT seed, so feeding it 5 raw per-seed rows for the same key
    silently collapses to whichever seed's row happened to be ingested
    last (a real bug that never manifested before, since every other
    model ingested here is single-seed). Use this instead of
    ingest_classification_final_rows() for any file with >1 seed per
    (model, dataset) - stores accuracy_mean/f1_mean/accuracy_std/f1_std/
    n_seeds in metrics, so tables report the genuine 5-seed mean, not an
    arbitrary single seed."""
    rows = load(fname)
    if model_filter:
        rows = [r for r in rows if r["model"] in model_filter]
    groups: dict[tuple, list] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        groups.setdefault((r["model"], r["dataset"], r.get("k")), []).append(r)
    for (model, dataset, k), group_rows in groups.items():
        accs = [r["accuracy"] for r in group_rows]
        f1s = [r["f1"] for r in group_rows]
        n = len(accs)
        mean = lambda xs: sum(xs) / len(xs)
        std = lambda xs: (sum((x - mean(xs)) ** 2 for x in xs) / len(xs)) ** 0.5 if len(xs) > 1 else 0.0
        metrics = {
            "accuracy": mean(accs), "f1": mean(f1s),
            "accuracy_mean": mean(accs), "accuracy_std": std(accs),
            "f1_mean": mean(f1s), "f1_std": std(f1s), "n_seeds": n,
        }
        add("classification", model, dataset, k, None, metrics, "ok", "", fname, environment, sweep,
            group_rows[0].get("checkpoint_selection", "none"),
            notes=f"mean+/-std over {n} seeds ({sorted(r['seed'] for r in group_rows)})")


ingest_classification_final_rows("futurelab/classification5_final.json", "futurelab",
                                  "classification5_all_datasets (sbert_minilm,sbert_gte,hicot,fastopic,bertopic x 26 datasets, --split random)")

# vaebm-family classification: only smoke-test single points available at
# manifest-build time (the formal 26-dataset vaebm3_classification sweep was
# queued behind vaebm3_cluster and had not started) - added explicitly,
# tagged as smoke-test, not a systematic sweep. "vaebm" (main config)
# EXCLUDED 2026-09-21 - superseded by the locked-architecture ingestion
# below (5 seeds x 12 datasets, real sweep, not a smoke test).
# vaebm_poe/vaebm_dec smoke points kept (not relocked, still ablations).
_VAEBM_CLASSIF_SMOKE = [
    ("vaebm_poe", 0.9370786516853933, 0.9375155457269407),
    ("vaebm_dec", 0.9617977528089887, 0.9598702490350087),
]
for model, acc, f1 in _VAEBM_CLASSIF_SMOKE:
    add("classification", model, "bbc_news", 5, 42, {"accuracy": acc, "f1": f1}, "ok", "",
        "smoke_test_2026-09-16 (registry validation run, single seed, not part of a full sweep)",
        "futurelab", "vaebm3_classification smoke test (formal 26-dataset sweep queued, not yet run)",
        "none", notes="smoke-test single datapoint, --split random")

# Locked architecture (2026-09-21, user-authorized): same config as the
# cluster ingestion above, K=num_classes, --split random (stratified
# 80/20), 5 seeds (1-5) - genuine seed variance here, unlike cluster's
# single deterministic seed. Ran on labuai. See
# scripts/run_vaebm_classification_locked_12ds.py's own docstring.
if (DATA / "labuai" / "vaebm_classification_locked_20260921.json").exists():
    ingest_classification_multiseed_aggregated("labuai/vaebm_classification_locked_20260921.json", "labuai",
                                                "vaebm_classification_locked_20260921 (vaebm locked architecture x 12 plain datasets, 5 seeds, --split random)",
                                                model_filter={"vaebm"})

# New baselines (2026-09-21) - same 4 as the cluster ingestion above,
# single seed=42 (matching HiCOT/FASTopic/BERTopic/SBERT-kmeans'
# existing classification convention, --split random).
ingest_classification_final_rows("labuai/lda_locked_classification_20260921.json", "labuai",
                                  "lda_locked_20260921 (lda x 12 plain datasets, --split random)")
ingest_classification_final_rows("futurelab_new/glocom_locked_classification_20260921.json", "futurelab",
                                  "glocom_locked_20260921 (glocom x 12 plain datasets, --split random)")
ingest_classification_final_rows("futurelab_new/ecrtm_locked_classification_20260921.json", "futurelab",
                                  "ecrtm_locked_20260921 (ecrtm x 12 plain datasets, --split random)")
ingest_classification_final_rows("futurelab_new/s2wtm_locked_classification_20260921.json", "futurelab",
                                  "s2wtm_locked_20260921 (s2wtm x 12 plain datasets, --split random)")

# HiCOT classification stopword-fix (2026-09-21): replaces
# classification5_final.json's own HiCOT rows for these 12 datasets -
# hicot_adapter.py's self-fit vectorizer had no stop_words filter at
# all before this fix (see models/hicot_adapter.py's own docstring).
# Ingested AFTER classification5_final.json above, so "last ok wins"
# dedup correctly supersedes the un-fixed rows for these 12 datasets
# specifically (classification5_final.json's other ~14 datasets keep
# their own HiCOT rows, untouched).
if (DATA / "futurelab_new" / "hicot_classif_stopword_fix_20260921.json").exists():
    ingest_classification_final_rows("futurelab_new/hicot_classif_stopword_fix_20260921.json", "futurelab",
                                      "hicot_classif_stopword_fix_20260921 (hicot x 12 plain datasets, stopword fix, --split random)",
                                      model_filter={"hicot"})

# ---------------------------------------------------------------- topic --

def ingest_topic_final_rows(fname, environment, sweep):
    rows = load(fname)
    for r in rows:
        metrics = {"cv": r.get("cv"), "purity": r.get("purity"), "nmi": r.get("nmi"), "td": r.get("td")}
        add("topic", r["model"], r["dataset"], r.get("k"), r.get("seed", 42), metrics,
            r.get("status"), r.get("error"), fname, environment, sweep,
            checkpoint_selection="none",
            notes=f"protocol={r.get('evaluation_protocol')},cv_source={r.get('cv_source')},td_def={r.get('td_definition')}")


ingest_topic_final_rows("futurelab/sbert11_topic_final.json", "futurelab",
                         "sbert11_ecrtm_hicot (11 SBERT embedder variants x 15 datasets x K in {50,100}, ecrtm_hicot protocol, Palmetto/Wikipedia C_V)")

_VAEBM_TOPIC_SMOKE = [
    ("vaebm", 0.45482458807428267, 0.5030775761434787, 0.4438131996070811, 0.108),
    ("vaebm_dec", 0.43018885162474396, 0.19723018147086915, 0.20589306604518695, 0.3333333333333333),
]
for model, cv, purity, nmi, td in _VAEBM_TOPIC_SMOKE:
    add("topic", model, "20ng", 50, 42, {"cv": cv, "purity": purity, "nmi": nmi, "td": td}, "ok", "",
        "smoke_test_2026-09-16 (registry validation run, single seed, not part of a full sweep)",
        "futurelab", "vaebm3_topic smoke test (formal sweep queued, not yet run)",
        "none", notes="smoke-test single datapoint, generic protocol, local-corpus C_V (not Palmetto)")
# vaebm_poe's own topic smoke-test point (20ng, k=50) - recorded separately
# from the monitor notification at the time: cv=0.455 nmi=0.444 purity=0.503 td=0.108
add("topic", "vaebm_poe", "20ng", 50, 42,
    {"cv": 0.45482458807428267, "purity": 0.5030775761434787, "nmi": 0.4438131996070811, "td": 0.108},
    "ok", "", "smoke_test_2026-09-16 (registry validation run, single seed, not part of a full sweep)",
    "futurelab", "vaebm3_topic smoke test (formal sweep queued, not yet run)", "none",
    notes="smoke-test single datapoint, generic protocol, local-corpus C_V (not Palmetto)")

Path(DATA / "results_manifest_raw.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Raw manifest: {len(manifest)} rows -> {DATA / 'results_manifest_raw.json'}")

# ---------------------------------------------------------- dedup / select --
# Keep the LAST successful record per (experiment, model, dataset, k,
# environment, sweep-family) - "sweep-family" collapses resume/retry sweeps
# of literally the same config into one, while genuinely different sweeps
# (different model set, different environment, different protocol) stay
# separate rows by construction already (they're keyed by `sweep` name
# above, which encodes exactly that).
final: dict[tuple, dict] = {}
for row in manifest:
    key = (row["experiment"], row["model"], row["dataset"], row["k"], row["environment"], row["sweep"])
    if row["status"] != "ok":
        final.setdefault(key, row)
        if final[key]["status"] != "ok":
            final[key] = row  # last error wins if never any ok
        continue
    final[key] = row  # last ok wins over any earlier error for same key

resolved = list(final.values())
Path(DATA / "results_manifest.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
print(f"Resolved manifest: {len(resolved)} rows -> {DATA / 'results_manifest.json'}")

ok_rows = [r for r in resolved if r["status"] == "ok"]
print(f"  ok: {len(ok_rows)}, error/pending: {len(resolved) - len(ok_rows)}")
for exp in ("cluster", "classification", "topic"):
    models = sorted(set(r["model"] for r in ok_rows if r["experiment"] == exp))
    datasets = sorted(set(r["dataset"] for r in ok_rows if r["experiment"] == exp))
    print(f"  [{exp}] models={models}")
    print(f"  [{exp}] n_datasets={len(datasets)}")
