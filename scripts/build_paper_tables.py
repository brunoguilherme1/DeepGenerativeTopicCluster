#!/usr/bin/env python
"""Generates every LaTeX table file the paper includes, directly from
paper_data/results_manifest.json (build_paper_results.py must run first).
No number in any .tex file here is hand-typed.

Table philosophy (2026-09-17 rewrite, user-specified): the grouping unit
is the EXPERIMENT, never the model family. Every model with results for
a given experiment appears together, as rows, in every table for that
experiment; tables are split ONLY by dataset when too wide for one page.

Model-selection rewrite (2026-09-17, second user pass): main-paper tables
must tell the VAE-BM-PoE story, not exhaustively benchmark every SBERT
variant. Main tables show core baselines (LDA, FASTopic, HiCOT, BERTopic)
plus at most two embedding-clustering baselines (SBERT-MiniLM, SBERT-GTE),
then a \\midrule, then the proposed family (VAE-BM, VAE-BM-PoE, VAE-BM-DEC)
with VAE-BM-PoE's row label bolded for visual identification. The full
10/11-SBERT-variant topic ablation moves to an appendix table instead.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "paper_data"
OUT = DATA / "tables_out"
OUT.mkdir(parents=True, exist_ok=True)

manifest = json.loads((DATA / "results_manifest.json").read_text(encoding="utf-8"))

# One extra real datapoint found in results/experiment/ (an early ad-hoc
# single-combo topic-experiment test, predating the formal sbert11 sweep) -
# adds a genuine bertopic cell the rest of the manifest has none of.
manifest.append({
    "experiment": "topic", "model": "bertopic", "dataset": "search_snippets", "k": 50, "seed": 42,
    "metrics": {"cv": 0.4554208405666126, "purity": 0.88354943273906, "nmi": 0.5383245418011808, "td": 0.932},
    "status": "ok", "error": "", "source_file": "results/experiment/results.csv (early ad-hoc test, both environments)",
    "environment": "futurelab", "sweep": "ad-hoc single-combo test (generic protocol, local-corpus C_V)",
    "checkpoint_selection": "none", "notes": "protocol=generic,cv_source=local_corpus",
})

MODEL_LABEL = {
    "vaebm": "VAE-BM", "vaebm_poe": "VAE-BM-PoE", "vaebm_dec": "VAE-BM-DEC",
    "hicot": "HiCOT", "fastopic": "FASTopic", "bertopic": "BERTopic", "lda": "LDA",
    "glocom": "GloCOM", "ecrtm": "ECRTM", "s2wtm": "S2WTM",
    "sbert_gte": "SBERT-GTE", "sbert_minilm": "SBERT-MiniLM", "sbert_mpnet": "SBERT-MPNet",
    "sbert_bge": "SBERT-BGE", "sbert_e5": "SBERT-E5", "sbert_t5": "SBERT-T5",
    "sbert_distilbert": "SBERT-DistilBERT", "sbert_distilroberta": "SBERT-DistilRoBERTa",
    "sbert_msdistilbert": "SBERT-MS-DistilBERT", "sbert_msmpnet": "SBERT-MS-MPNet",
    "sbert_paraphrase": "SBERT-Paraphrase-MPNet",
}
DATASET_LABEL = {
    "20ng": "20NG", "hicot_20ng": "20NG-H", "imdb": "IMDB", "hicot_imdb": "IMDB-H",
    "agnews_short": "AGNews", "hicot_agnews": "AGNews-H",
    "search_snippets": "SearchSnip.", "hicot_search_snippets": "SearchSnip.-H",
    "google_news_ts": "GNews-TS", "hicot_google_news": "GNews-H",
    "google_news_t": "GNews-T", "google_news_s": "GNews-S",
    "stack_overflow": "StackOv.", "biomedical": "Biomed.", "tweet": "Tweet",
    "agnews_full": "AGNews-F", "banking77": "Banking77", "bbc_news": "BBCNews",
    "dblp": "DBLP", "dbpedia_14": "DBpedia14", "m10": "M10", "pascal_flickr": "Pascal-Fl.",
    "tweet_eval_emotion": "TwEval-Emo", "tweet_eval_sentiment": "TwEval-Sent",
    "yahoo_answers_topics": "YahooAns", "20ng_s2wtm": "20NG-S2WTM",
}

HIGHER_BETTER = {
    "acc": True, "nmi": True, "ari": True, "ami": True, "homogeneity": True, "completeness": True,
    "v_measure": True, "purity": True, "silhouette": True, "davies_bouldin": False, "calinski_harabasz": True,
    "cv": True, "td": True, "accuracy": True, "f1": True,
}

# The proposed family - always listed last within a model list, separated
# from the baselines above it by a \midrule, with VAE-BM-PoE's row label
# (only the label, never its metric cells) bolded for visual identification.
PROPOSED_MODELS = {"vaebm", "vaebm_poe", "vaebm_dec"}


def esc(s):
    return str(s).replace("_", "\\_")


def lbl(name, table):
    return table.get(name, esc(name))


def row_label(model):
    base = lbl(model, MODEL_LABEL)
    return f"\\textbf{{{base}}}" if model == "vaebm_poe" else base


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


def bold_group(experiment, model, metric):
    """Comparability group for best/second-best highlighting - default
    is one global group, EXCEPT topic C_V, which uses a different C_V
    source for the sbert-family (Palmetto/Wikipedia) vs. the VAE-BM
    family (local-corpus) - never compared/bolded against each other."""
    if experiment == "topic" and metric == "cv":
        return "sbert" if model.startswith("sbert") else "other"
    return "all"


def mark_cell(s, mark):
    if s is None or s == "--":
        return "--"
    if mark == "best":
        return f"\\textbf{{{s}}}"
    if mark == "second":
        return f"\\underline{{{s}}}"
    return s


def first_proposed_index(models):
    """Index of the first proposed-family model in `models`, or None if
    there isn't one, or if it's already first (no baselines to separate
    it from)."""
    for i, m in enumerate(models):
        if m in PROPOSED_MODELS:
            return i if i > 0 else None
    return None


# ================================================================= TOPIC ====
# Main-paper topic models: core baselines with actual topic-experiment
# results, plus at most two embedding-clustering baselines (MiniLM, GTE),
# then the proposed family. LDA/FASTopic/HiCOT are listed for intent but
# drop out below since none has a completed topic-experiment result.
MAIN_TOPIC_MODELS = ["lda", "fastopic", "hicot", "bertopic", "sbert_minilm", "sbert_gte"] + \
    ["vaebm", "vaebm_poe", "vaebm_dec"]
MAIN_TOPIC_MODELS = [m for m in MAIN_TOPIC_MODELS if get("topic", model=m)]

# Appendix topic models: the full sentence-embedding ablation (all SBERT
# variants), plus BERTopic and the proposed family for context - nothing
# from the main table's evidence is lost, just de-emphasized there.
APPENDIX_TOPIC_MODELS = ["lda", "fastopic", "hicot", "bertopic"] + \
    sorted(m for m in set(r["model"] for r in manifest if r["experiment"] == "topic") if m.startswith("sbert")) + \
    ["vaebm", "vaebm_poe", "vaebm_dec"]
APPENDIX_TOPIC_MODELS = [m for m in APPENDIX_TOPIC_MODELS if get("topic", model=m)]

TOPIC_DATASET_ORDER = ["20ng", "agnews_short", "imdb", "hicot_20ng", "hicot_agnews", "hicot_imdb",
                        "search_snippets", "hicot_search_snippets", "stack_overflow",
                        "biomedical", "tweet", "google_news_ts", "google_news_t", "google_news_s",
                        "hicot_google_news"]
TOPIC_DATASETS = [d for d in TOPIC_DATASET_ORDER if d in set(r["dataset"] for r in manifest if r["experiment"] == "topic" and r["status"] == "ok")]
TOPIC_METRICS = ["cv", "purity", "nmi", "td"]
TOPIC_METRIC_HEAD = {"cv": "$C_V$", "purity": "Pur.", "nmi": "NMI", "td": "TD"}

DS_PER_BLOCK_TOPIC = 3
topic_blocks = [TOPIC_DATASETS[i:i + DS_PER_BLOCK_TOPIC] for i in range(0, len(TOPIC_DATASETS), DS_PER_BLOCK_TOPIC)]


def build_topic_table(models, block, bi, n_blocks, path, caption, label):
    n = len(block)
    ncols = 1 + n * len(TOPIC_METRICS)
    split_at = first_proposed_index(models)
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
              r"\caption{" + caption + "}",
              r"\label{" + label + "}",
              r"\adjustbox{max width=\textwidth}{",
              r"\begin{tabular}{l" + "cccc" * n + "}", r"\toprule",
              " & " + " & ".join(r"\multicolumn{4}{c}{" + lbl(d, DATASET_LABEL) + "}" for d in block) + r" \\",
              "".join(f"\\cmidrule(lr){{{2+4*i}-{5+4*i}}}" for i in range(n)),
              " & " + " & ".join(" & ".join(TOPIC_METRIC_HEAD[m] for m in TOPIC_METRICS) for _ in block) + r" \\",
              r"\midrule"]
    for k in (50, 100):
        lines.append(r"\multicolumn{" + str(ncols) + r"}{l}{\textit{" + str(k) + r" Topics}} \\")
        col_raws = {(d, m): [] for d in block for m in TOPIC_METRICS}
        model_cells = {}
        for model in models:
            row_cells = []
            for d in block:
                r, tag = best_row("topic", model, d, k)
                for m in TOPIC_METRICS:
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
            groups = {}
            for model, v in vals:
                g = bold_group("topic", model, m)
                groups.setdefault(g, []).append((model, v))
            for g, gv in groups.items():
                ordered = sorted(gv, key=lambda t: t[1], reverse=HIGHER_BETTER[m])
                if ordered:
                    marks[(d, m, ordered[0][0])] = "best"
                if len(ordered) > 1 and ordered[1][1] != ordered[0][1]:
                    marks[(d, m, ordered[1][0])] = "second"
        for mi, model in enumerate(models):
            if split_at is not None and mi == split_at:
                lines.append(r"\midrule")
            cells = []
            for ci, d in enumerate(block):
                for m in TOPIC_METRICS:
                    s, tag = model_cells[model][ci * len(TOPIC_METRICS) + TOPIC_METRICS.index(m)]
                    mark = marks.get((d, m, model))
                    cells.append(mark_cell(s, mark) + (tag if s not in (None, "--") and s is not None else ""))
            lines.append(f"{row_label(model)} & " + " & ".join(cells) + r" \\")
        if k == 50:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


for bi, block in enumerate(topic_blocks, start=1):
    build_topic_table(
        MAIN_TOPIC_MODELS, block, bi, len(topic_blocks),
        OUT / f"topic_block{bi}.tex",
        f"Topic modeling ({bi}/{len(topic_blocks)}), $K=50$ and $K=100$ panels. "
        r"Core baselines and at most two embedding-clustering baselines (SBERT-MiniLM, SBERT-GTE) above the "
        r"horizontal rule; our proposed family below it, with \textbf{VAE-BM-PoE} bolded for identification. The full "
        r"10-encoder sentence-embedding ablation is in Appendix~\ref{sec:appendix-topic-full} (Tables~"
        r"\ref{tab:topic-appendix-block1}--\ref{tab:topic-appendix-block" + str(len(topic_blocks)) + r"}). "
        r"$C_V$/Pur./NMI/TD; \textbf{bold}=best, \underline{underline}=second-best, computed separately "
        r"within each dataset+metric (and, for $C_V$ only, separately for the SBERT-family rows, which use "
        r"Palmetto/Wikipedia $C_V$, versus the VAE-BM-family rows, which use local-corpus $C_V$ - the two "
        r"are never compared or bolded against each other). -- marks a combination not yet available.",
        f"tab:topic-block{bi}")

for bi, block in enumerate(topic_blocks, start=1):
    build_topic_table(
        APPENDIX_TOPIC_MODELS, block, bi, len(topic_blocks),
        OUT / f"topic_appendix_block{bi}.tex",
        f"Sentence-embedding ablation for topic modeling ({bi}/{len(topic_blocks)}), $K=50$ and $K=100$ panels: "
        r"all 10 SBERT encoder variants, plus BERTopic and our proposed family for context (\textbf{VAE-BM-PoE} "
        r"bolded). This is the full ablation summarized by the two-baseline main-text Table~\ref{tab:topic-block" + str(bi) + r"}. "
        r"$C_V$/Pur./NMI/TD; \textbf{bold}=best, \underline{underline}=second-best, computed separately "
        r"within each dataset+metric (and, for $C_V$ only, separately for the SBERT-family rows, which use "
        r"Palmetto/Wikipedia $C_V$, versus the VAE-BM-family rows, which use local-corpus $C_V$ - the two "
        r"are never compared or bolded against each other). -- marks a combination not yet available.",
        f"tab:topic-appendix-block{bi}")

# ============================================================= CLUSTER =====
# Main-paper cluster models: core baselines, then at most two SBERT
# variants (MiniLM, GTE), then a \midrule, then the proposed family.
CLUSTER_ALL_MODELS = ["lda", "glocom", "ecrtm", "s2wtm", "fastopic", "hicot", "bertopic", "sbert_minilm", "sbert_gte"] + \
    ["vaebm", "vaebm_poe", "vaebm_dec"]
CLUSTER_ALL_MODELS = [m for m in CLUSTER_ALL_MODELS if get("cluster", model=m)]
CLUSTER_DATASET_ORDER = ["hicot_20ng", "hicot_agnews", "hicot_google_news", "hicot_imdb", "hicot_search_snippets",
                          "20ng", "imdb", "agnews_short", "search_snippets", "google_news_ts"]
ALL_CLUSTER_DATASETS = sorted(set(r["dataset"] for r in manifest if r["experiment"] == "cluster" and r["status"] == "ok"))
CLUSTER_DATASETS = [d for d in CLUSTER_DATASET_ORDER if d in ALL_CLUSTER_DATASETS] + \
                    sorted(d for d in ALL_CLUSTER_DATASETS if d not in CLUSTER_DATASET_ORDER)
CLUSTER_MAIN_METRICS = ["acc", "nmi"]
CLUSTER_MAIN_HEAD = {"acc": "ACC", "nmi": "NMI"}

DS_PER_BLOCK_CLUSTER = 4
cluster_blocks = [CLUSTER_DATASETS[i:i + DS_PER_BLOCK_CLUSTER] for i in range(0, len(CLUSTER_DATASETS), DS_PER_BLOCK_CLUSTER)]


def build_grouped_table(experiment, models, datasets, metrics, metric_head, path, caption, label, k=None):
    n = len(datasets)
    split_at = first_proposed_index(models)
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
              r"\caption{" + caption + "}", r"\label{" + label + "}",
              r"\adjustbox{max width=\textwidth}{",
              r"\begin{tabular}{l" + ("c" * len(metrics)) * n + "}", r"\toprule",
              " & " + " & ".join(r"\multicolumn{" + str(len(metrics)) + "}{c}{" + lbl(d, DATASET_LABEL) + "}" for d in datasets) + r" \\",
              "".join(f"\\cmidrule(lr){{{2+len(metrics)*i}-{1+len(metrics)*(i+1)}}}" for i in range(n)),
              " & " + " & ".join(" & ".join(metric_head[m] for m in metrics) for _ in datasets) + r" \\",
              r"\midrule"]
    col_raws = {(d, m): [] for d in datasets for m in metrics}
    model_cells = {}
    for model in models:
        row_cells = []
        for d in datasets:
            r, tag = best_row(experiment, model, d, k)
            for m in metrics:
                if r is None:
                    row_cells.append((None, tag))
                    continue
                v = r["metrics"].get(m)
                s = fmt(v, 2 if m == "calinski_harabasz" else 3)
                row_cells.append((s, tag))
                if v is not None:
                    col_raws[(d, m)].append((model, float(v)))
        model_cells[model] = row_cells
    marks = {}
    for (d, m), vals in col_raws.items():
        groups = {}
        for model, v in vals:
            g = bold_group(experiment, model, m)
            groups.setdefault(g, []).append((model, v))
        for g, gv in groups.items():
            ordered = sorted(gv, key=lambda t: t[1], reverse=HIGHER_BETTER[m])
            if ordered:
                marks[(d, m, ordered[0][0])] = "best"
            if len(ordered) > 1 and ordered[1][1] != ordered[0][1]:
                marks[(d, m, ordered[1][0])] = "second"
    for mi, model in enumerate(models):
        if split_at is not None and mi == split_at:
            lines.append(r"\midrule")
        cells = []
        for ci, d in enumerate(datasets):
            for m in metrics:
                s, tag = model_cells[model][ci * len(metrics) + metrics.index(m)]
                mark = marks.get((d, m, model))
                cells.append(mark_cell(s, mark) + (tag if s not in (None, "--") and s is not None else ""))
        lines.append(f"{row_label(model)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


for bi, block in enumerate(cluster_blocks, start=1):
    build_grouped_table(
        "cluster", CLUSTER_ALL_MODELS, block, CLUSTER_MAIN_METRICS, CLUSTER_MAIN_HEAD,
        OUT / f"cluster_block{bi}.tex",
        f"Document clustering (ACC/NMI), datasets {bi} of {len(cluster_blocks)}. Core baselines and at most two "
        r"embedding-clustering baselines (SBERT-MiniLM, SBERT-GTE) above a horizontal rule; our proposed family below "
        r"it, with \textbf{VAE-BM-PoE} bolded for identification. "
        r"$^{\mathrm{LU}}$ marks a value from the labuai environment (only shown when no FutureLab value exists for "
        r"that model/dataset - the two are never averaged). \textbf{Bold}/\underline{underline} = best/second-best "
        "per dataset+metric among the models shown. -- = not yet available (the 26-dataset VAE-BM-PoE/VAE-BM-DEC "
        "sweep was still running at submission time). Full 11-metric breakdowns are in Appendix~"
        r"\ref{sec:appendix-cluster-full}.",
        f"tab:cluster-block{bi}")

# ---- Appendix: full 11-metric table, one per dataset, all models ----
CLUSTER_FULL_METRICS = ["acc", "nmi", "ari", "ami", "homogeneity", "completeness", "v_measure",
                          "purity", "silhouette", "davies_bouldin", "calinski_harabasz"]
FULL_HEAD = {"acc": "ACC", "nmi": "NMI", "ari": "ARI", "ami": "AMI", "homogeneity": "Hom.",
              "completeness": "Comp.", "v_measure": "V-m.", "purity": "Pur.",
              "silhouette": "Sil.", "davies_bouldin": "DB", "calinski_harabasz": "CH"}
appendix_lines = []
for ds in CLUSTER_DATASETS:
    split_at = first_proposed_index(CLUSTER_ALL_MODELS)
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
              r"\caption{Full clustering metrics, " + lbl(ds, DATASET_LABEL) + r". Lower is better for DB only; "
              r"higher is better for every other metric. \textbf{Bold}/\underline{underline} = best/second-best.}",
              r"\label{tab:cluster-full-" + ds + "}",
              r"\adjustbox{max width=\columnwidth}{",
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
    any_row = False
    emitted = 0
    for mi, model in enumerate(CLUSTER_ALL_MODELS):
        cells = model_cells[model]
        if all(c[0] is None for c in cells):
            continue
        if split_at is not None and mi == split_at and emitted:
            lines.append(r"\midrule")
        any_row = True
        emitted += 1
        rendered = []
        for m, (s, tag) in zip(CLUSTER_FULL_METRICS, cells):
            mark = marks.get((m, model))
            rendered.append(mark_cell(s, mark) + (tag if s else ""))
        lines.append(f"{row_label(model)} & " + " & ".join(rendered) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table}"]
    if any_row:
        fname = f"cluster_full_{ds}.tex"
        (OUT / fname).write_text("\n".join(lines) + "\n", encoding="utf-8")
        appendix_lines.append(fname)
print("Appendix full-metric cluster tables:", len(appendix_lines))

# ======================================================= CLASSIFICATION ====
CLASSIF_ALL_MODELS = ["lda", "glocom", "ecrtm", "s2wtm", "fastopic", "hicot", "bertopic", "sbert_minilm", "sbert_gte"] + \
    ["vaebm", "vaebm_poe", "vaebm_dec"]
CLASSIF_ALL_MODELS = [m for m in CLASSIF_ALL_MODELS if get("classification", model=m)]
CLASSIF_DATASET_ORDER = ["bbc_news", "20ng", "imdb", "agnews_short", "search_snippets"]
ALL_CLASSIF_DATASETS = sorted(set(r["dataset"] for r in manifest if r["experiment"] == "classification" and r["status"] == "ok"))
CLASSIF_DATASETS = [d for d in CLASSIF_DATASET_ORDER if d in ALL_CLASSIF_DATASETS] + \
                    sorted(d for d in ALL_CLASSIF_DATASETS if d not in CLASSIF_DATASET_ORDER)
CLASSIF_METRICS = ["accuracy", "f1"]
CLASSIF_HEAD = {"accuracy": "Acc.", "f1": "F1"}

classif_blocks = [CLASSIF_DATASETS[i:i + DS_PER_BLOCK_CLUSTER] for i in range(0, len(CLASSIF_DATASETS), DS_PER_BLOCK_CLUSTER)]
for bi, block in enumerate(classif_blocks, start=1):
    build_grouped_table(
        "classification", CLASSIF_ALL_MODELS, block, CLASSIF_METRICS, CLASSIF_HEAD,
        OUT / f"classification_block{bi}.tex",
        f"Classification (Accuracy/F1, --split random), datasets {bi} of {len(classif_blocks)}. Core baselines and "
        r"at most two embedding-clustering baselines (SBERT-MiniLM, SBERT-GTE) above a horizontal rule; our proposed "
        r"family below it, with \textbf{VAE-BM-PoE} bolded for identification. "
        r"\textbf{VAE-BM}'s own cells are the mean over 5 seeds (std/CI in Appendix~\ref{sec:appendix}'s manifest); "
        "every baseline cell (LDA/GloCOM/ECRTM/S2WTM/FASTopic/HiCOT/BERTopic/SBERT-MiniLM/SBERT-GTE) is a single run "
        "(seed 42), matching this task's own established single-seed baseline convention. "
        "-- = not yet available (the 26-dataset VAE-BM-PoE/VAE-BM-DEC sweep was still queued at submission time; "
        "only bbc\\_news has a single validation datapoint for those two models).",
        f"tab:classif-block{bi}")

print(f"\nWrote tables to {OUT}")
for f in sorted(OUT.glob("*.tex")):
    print(" ", f.name)
