#!/usr/bin/env python
"""Generates every LaTeX table file the paper includes, directly from
paper_data/results_manifest.json (build_paper_results.py must run first).
No number in any .tex file here is hand-typed.

Table philosophy (2026-09-21 rewrite, user-specified): ONE VAE-BM ("VAE-BM
(ours)") is the proposed model everywhere in the main paper - no PoE/DEC
rows, no alpha variants, no SBERT+KMeans embedding-clustering baselines
anywhere in the paper (main or appendix). Cluster/Classification restrict
to the 12-dataset suite where every baseline (LDA, GloCOM, ECRTM, S2WTM,
FASTopic, HiCOT, BERTopic) plus VAE-BM has a genuinely completed result,
so no table is dominated by "--". Cluster reports the full 11-metric
battery, one table per dataset (never truncated to ACC/NMI only).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "paper_data"
OUT = DATA / "tables_out"
OUT.mkdir(parents=True, exist_ok=True)

manifest = json.loads((DATA / "results_manifest.json").read_text(encoding="utf-8"))

MODEL_LABEL = {
    "vaebm": "VAE-BM (ours)",
    "hicot": "HiCOT", "fastopic": "FASTopic", "bertopic": "BERTopic", "lda": "LDA",
    "glocom": "GloCOM", "ecrtm": "ECRTM", "s2wtm": "S2WTM",
}
DATASET_LABEL = {
    "20ng": "20NG", "hicot_20ng": "20NG-H", "imdb": "IMDB", "hicot_imdb": "IMDB-H",
    "agnews_short": "AGNews", "hicot_agnews": "AGNews-H",
    "search_snippets": "SearchSnip.", "hicot_search_snippets": "SearchSnip.-H",
    "google_news_t": "GNews-T", "hicot_google_news": "GNews-H",
    "stack_overflow": "StackOv.", "biomedical": "Biomed.", "tweet": "Tweet",
    "banking77": "Banking77", "bbc_news": "BBCNews", "m10": "M10", "pascal_flickr": "Pascal-Fl.",
}

HIGHER_BETTER = {
    "acc": True, "nmi": True, "ari": True, "ami": True, "homogeneity": True, "completeness": True,
    "v_measure": True, "purity": True, "silhouette": True, "davies_bouldin": False, "calinski_harabasz": True,
    "cv": True, "td": True, "accuracy": True, "f1": True,
}

# The one proposed model - always listed last, separated from the
# baselines above it by a \midrule, its row label bolded.
PROPOSED_MODELS = {"vaebm"}

# The 12-dataset suite used for both Document Clustering and
# Classification - deliberately excludes the HiCOT-artifact datasets
# (Topic Modeling only, \S5.1), and excludes every dataset from the
# earlier, superseded 25/26-dataset registry (dbpedia_14, agnews_full,
# yahoo_answers_topics, dblp, google_news_s/ts, tweet_eval_*,
# 20ng_s2wtm) that this paper no longer reports on.
LOCKED_DATASET_ORDER = [
    "20ng", "agnews_short", "google_news_t", "imdb", "search_snippets",
    "bbc_news", "tweet", "stack_overflow", "biomedical", "banking77", "m10", "pascal_flickr",
]

BASELINE_ORDER = ["lda", "glocom", "ecrtm", "s2wtm", "fastopic", "hicot", "bertopic"]


def esc(s):
    return str(s).replace("_", "\\_")


def lbl(name, table):
    return table.get(name, esc(name))


def row_label(model):
    base = lbl(model, MODEL_LABEL)
    return f"\\textbf{{{base}}}" if model in PROPOSED_MODELS else base


def fmt(x, nd=3):
    if x is None:
        return None
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return None


def get(experiment, model=None, dataset=None, k=None):
    rows = [r for r in manifest if r["experiment"] == experiment and r["status"] == "ok"]
    if model is not None:
        rows = [r for r in rows if r["model"] == model]
    if dataset is not None:
        rows = [r for r in rows if r["dataset"] == dataset]
    if k is not None:
        rows = [r for r in rows if r["k"] == k]
    return rows


def best_row(experiment, model, dataset, k=None):
    """Prefers FutureLab when the same model/dataset exists on both
    environments - keeps exactly one row per model in every table
    (never two rows for the same model just because two environments
    ran it), with an LU superscript when only labuai has it."""
    rows = get(experiment, model, dataset, k)
    if not rows:
        return None, None
    fl = [r for r in rows if r["environment"] == "futurelab"]
    if fl:
        return fl[0], ""
    return rows[0], "$^{\\mathrm{LU}}$"


def mark_cell(s, mark):
    if s is None or s == "--":
        return "--"
    if mark == "best":
        return f"\\textbf{{{s}}}"
    if mark == "second":
        return f"\\underline{{{s}}}"
    return s


def first_proposed_index(models):
    for i, m in enumerate(models):
        if m in PROPOSED_MODELS:
            return i if i > 0 else None
    return None


# ================================================================= TOPIC ====
# Topic Modeling keeps HiCOT's own published K=50/K=100 table on its five
# official benchmark datasets (ETM, NTM+CL, ECRTM, FASTopic, NeuroMax,
# HiCOT, hand-authored in tables/topic_vaebm_*.tex with our VAE-BM row
# appended) - not regenerated here. No SBERT+KMeans ablation is produced
# or referenced anywhere in this paper.

# ============================================================= CLUSTER =====
# Every baseline below has a genuinely completed 11-metric cluster result
# on all 12 locked-suite datasets (verified against the manifest before
# writing this list) - no baseline is included "for completeness" with
# mostly-missing cells.
CLUSTER_ALL_MODELS = [m for m in BASELINE_ORDER + ["vaebm"] if get("cluster", model=m)]
CLUSTER_DATASETS = [d for d in LOCKED_DATASET_ORDER
                     if d in set(r["dataset"] for r in manifest if r["experiment"] == "cluster" and r["status"] == "ok")]

CLUSTER_FULL_METRICS = ["acc", "nmi", "ari", "ami", "homogeneity", "completeness", "v_measure",
                          "purity", "silhouette", "davies_bouldin", "calinski_harabasz"]
FULL_HEAD = {"acc": "ACC", "nmi": "NMI", "ari": "ARI", "ami": "AMI", "homogeneity": "Hom.",
              "completeness": "Comp.", "v_measure": "V-m.", "purity": "Pur.",
              "silhouette": "Sil.", "davies_bouldin": "DB", "calinski_harabasz": "CH"}


def build_cluster_full_table(ds):
    """One table per dataset, every model that has a completed result,
    all 11 clustering metrics - the ONLY cluster table format this paper
    uses (main text and appendix alike differ only in which datasets'
    tables are \\input where, decided in main.tex, not here)."""
    split_at = first_proposed_index(CLUSTER_ALL_MODELS)
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
              r"\caption{Document clustering, full metric set, " + lbl(ds, DATASET_LABEL) + r". "
              r"Higher is better for every metric except Davies-Bouldin (DB), where lower is better. "
              r"$^{\mathrm{LU}}$ marks a labuai value (shown only when no FutureLab value exists for that "
              r"model/dataset). \textbf{Bold}/\underline{underline} = best/second-best per metric.}",
              r"\label{tab:cluster-full-" + ds + "}",
              r"\adjustbox{max width=\textwidth}{",
              r"\begin{tabular}{l" + "r" * len(CLUSTER_FULL_METRICS) + "}", r"\toprule",
              "Model & " + " & ".join(FULL_HEAD[m] for m in CLUSTER_FULL_METRICS) + r" \\", r"\midrule"]
    col_raws = {m: [] for m in CLUSTER_FULL_METRICS}
    model_cells = {}
    for model in CLUSTER_ALL_MODELS:
        r, tag = best_row("cluster", model, ds)
        cells = []
        for m in CLUSTER_FULL_METRICS:
            if r is None:
                cells.append((None, tag))
                continue
            v = r["metrics"].get(m)
            s = fmt(v, 2 if m == "calinski_harabasz" else 3)
            cells.append((s, tag))
            if v is not None:
                col_raws[m].append((model, float(v)))
        model_cells[model] = cells
    marks = {}
    for m, vals in col_raws.items():
        ordered = sorted(vals, key=lambda t: t[1], reverse=HIGHER_BETTER[m])
        if ordered:
            marks[(m, ordered[0][0])] = "best"
        if len(ordered) > 1 and ordered[1][1] != ordered[0][1]:
            marks[(m, ordered[1][0])] = "second"
    emitted = 0
    for mi, model in enumerate(CLUSTER_ALL_MODELS):
        cells = model_cells[model]
        if all(c[0] is None for c in cells):
            continue
        if split_at is not None and mi == split_at and emitted:
            lines.append(r"\midrule")
        emitted += 1
        rendered = []
        for m, (s, tag) in zip(CLUSTER_FULL_METRICS, cells):
            mark = marks.get((m, model))
            rendered.append(mark_cell(s, mark) + (tag if s else ""))
        lines.append(f"{row_label(model)} & " + " & ".join(rendered) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}"]
    (OUT / f"cluster_full_{ds}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return emitted


n_written = 0
for ds in CLUSTER_DATASETS:
    n = build_cluster_full_table(ds)
    if n:
        n_written += 1
print(f"Cluster: wrote {n_written} full-metric tables (one per dataset), models={CLUSTER_ALL_MODELS}")

# ======================================================= CLASSIFICATION ====
CLASSIF_ALL_MODELS = [m for m in BASELINE_ORDER + ["vaebm"] if get("classification", model=m)]
CLASSIF_DATASETS = [d for d in LOCKED_DATASET_ORDER
                     if d in set(r["dataset"] for r in manifest if r["experiment"] == "classification" and r["status"] == "ok")]
CLASSIF_METRICS = ["accuracy", "f1"]
CLASSIF_HEAD = {"accuracy": "Acc.", "f1": "F1"}


def build_classif_table(models, datasets, path, caption, label):
    n = len(datasets)
    split_at = first_proposed_index(models)
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
              r"\caption{" + caption + "}", r"\label{" + label + "}",
              r"\adjustbox{max width=\textwidth}{",
              r"\begin{tabular}{l" + "cc" * n + "}", r"\toprule",
              " & " + " & ".join(r"\multicolumn{2}{c}{" + lbl(d, DATASET_LABEL) + "}" for d in datasets) + r" \\",
              "".join(f"\\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(n)),
              " & " + " & ".join(" & ".join(CLASSIF_HEAD[m] for m in CLASSIF_METRICS) for _ in datasets) + r" \\",
              r"\midrule"]
    col_raws = {(d, m): [] for d in datasets for m in CLASSIF_METRICS}
    model_cells = {}
    for model in models:
        row_cells = []
        for d in datasets:
            r, tag = best_row("classification", model, d)
            for m in CLASSIF_METRICS:
                if r is None:
                    row_cells.append((None, tag))
                    continue
                v = r["metrics"].get(m)
                s = fmt(v, 3)
                row_cells.append((s, tag))
                if v is not None:
                    col_raws[(d, m)].append((model, float(v)))
        model_cells[model] = row_cells
    marks = {}
    for (d, m), vals in col_raws.items():
        ordered = sorted(vals, key=lambda t: t[1], reverse=HIGHER_BETTER[m])
        if ordered:
            marks[(d, m, ordered[0][0])] = "best"
        if len(ordered) > 1 and ordered[1][1] != ordered[0][1]:
            marks[(d, m, ordered[1][0])] = "second"
    for mi, model in enumerate(models):
        if split_at is not None and mi == split_at:
            lines.append(r"\midrule")
        cells = []
        for ci, d in enumerate(datasets):
            for m in CLASSIF_METRICS:
                s, tag = model_cells[model][ci * len(CLASSIF_METRICS) + CLASSIF_METRICS.index(m)]
                mark = marks.get((d, m, model))
                cells.append(mark_cell(s, mark) + (tag if s not in (None, "--") and s is not None else ""))
        lines.append(f"{row_label(model)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


DS_PER_BLOCK_CLASSIF = 4
classif_blocks = [CLASSIF_DATASETS[i:i + DS_PER_BLOCK_CLASSIF] for i in range(0, len(CLASSIF_DATASETS), DS_PER_BLOCK_CLASSIF)]
for bi, block in enumerate(classif_blocks, start=1):
    build_classif_table(
        CLASSIF_ALL_MODELS, block,
        OUT / f"classification_block{bi}.tex",
        f"Classification (Accuracy/F1, stratified random 80/20 split), datasets {bi} of {len(classif_blocks)}. "
        r"Every baseline row (LDA/GloCOM/ECRTM/S2WTM/FASTopic/HiCOT/BERTopic) is a single run (seed 42); "
        r"\textbf{VAE-BM (ours)} is the mean over 5 seeds (1--5), since which documents land in train vs.\ "
        r"test genuinely varies by seed (std/CI in the released manifest, Appendix~\ref{sec:appendix}). "
        r"\textbf{Bold}/\underline{underline} = best/second-best per dataset+metric.",
        f"tab:classif-block{bi}")

print(f"Classification: wrote {len(classif_blocks)} tables, models={CLASSIF_ALL_MODELS}, datasets={CLASSIF_DATASETS}")

print(f"\nWrote tables to {OUT}")
for f in sorted(OUT.glob("*.tex")):
    print(" ", f.name)
