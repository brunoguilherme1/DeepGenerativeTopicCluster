#!/usr/bin/env python
"""Builds the final RQ3 (cross-domain generalization) LaTeX table:
LDA, BERTopic, FASTopic, HiCOT, GloCOM, VAE-BM x 7 datasets, from
paper_data/results_manifest.json's "cluster" experiment rows.

Reports NMI (primary cross-domain metric, matches HiCOT's own convention)
and C_V where available. HiCOT's m10/tweet rows reflect
scripts/rq3_hicot_dt_fix.py's outcome: m10 REPLACED (weight_loss_DT=5.0,
NMI 0.078->0.375); tweet NOT replaced (DT was not the cause - remains
severely undertrained, actual_k=12/89, NMI=0.067) - this is a genuine,
disclosed negative result, not a bug.

Usage:
    python scripts/build_rq3_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "paper_data" / "results_manifest.json"
OUT_DIR = REPO_ROOT / "NAACL_HLT_2021_Latex_Template__1_" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["20ng", "banking77", "bbc_news", "biomedical", "m10", "stack_overflow", "tweet"]
DATASET_LABELS = {"20ng": "20NG", "banking77": "Banking77", "bbc_news": "BBC News",
                   "biomedical": "Biomedical", "m10": "M10", "stack_overflow": "StackOverflow", "tweet": "Tweet"}
MODELS = ["lda", "bertopic", "fastopic", "hicot", "glocom", "vaebm"]
MODEL_LABELS = {"lda": "LDA", "bertopic": "BERTopic", "fastopic": "FASTopic",
                "hicot": "HiCOT", "glocom": "GloCOM", "vaebm": "VAE-BM (ours)"}
RQ3_SWEEP_PREFIXES = ("rq3_cross_domain_generalization", "rq3_glocom_addition")


def fmt(x, nd=3):
    return "--" if x is None else f"{x:.{nd}f}"


def main():
    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_key = {}
    for r in rows:
        if r["experiment"] != "cluster" or r["status"] != "ok":
            continue
        if not any(r["sweep"].startswith(p) for p in RQ3_SWEEP_PREFIXES):
            continue
        if r["dataset"] not in DATASETS or r["model"] not in MODELS:
            continue
        by_key[(r["model"], r["dataset"])] = r  # only one sweep per model here, safe

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l" + "c" * len(DATASETS) + "}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & " + " & ".join(DATASET_LABELS[d] for d in DATASETS) + r" \\")
    lines.append(r"\midrule")
    for model in MODELS:
        cells = []
        for d in DATASETS:
            row = by_key.get((model, d))
            cells.append(fmt(row["metrics"].get("nmi")) if row else "--")
        lines.append(MODEL_LABELS[model] + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{RQ3: cross-domain generalization (NMI) across 7 out-of-distribution "
                 r"datasets. HiCOT's tweet result reflects a genuine negative finding: after both "
                 r"a training-budget fix and a dedicated weight\_loss\_DT sensitivity rerun "
                 r"(Appendix~\ref{sec:rq3_dt_ablation}), HiCOT still recovers only "
                 r"12/89 requested clusters on this dataset, so the undertraining here is not "
                 r"attributable to weight\_loss\_DT alone.}")
    lines.append(r"\label{tab:rq3_cross_domain}")
    lines.append(r"\end{table*}")

    out_path = OUT_DIR / "rq3_cross_domain.tex"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")

    missing = [(model, d) for model in MODELS for d in DATASETS if (model, d) not in by_key]
    if missing:
        print("Missing cells:", missing)


if __name__ == "__main__":
    main()
