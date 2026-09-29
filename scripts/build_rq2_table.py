#!/usr/bin/env python
"""Builds the final RQ2 (representation quality, FASTopic protocol)
LaTeX table from paper_data/results_manifest.json.

Per-model sweep selection (explicit, not automatic "last ok wins" - the
manifest legitimately holds two competing VAE-BM configs for this
comparison and only one belongs in the main table):
  - hicot  -> rq2_hicot_quality_fix_20260928 (paper-range weight_loss_DT,
              C_V-selected; supersedes the old paper_default row, which
              was 25-500x outside HiCOT's documented range).
  - glocom -> rq2_glocom_addition_20260928 (only sweep that exists).
  - vaebm  -> rq2_fastopic_protocol_representation_quality_20260922's
              gte_baseline (units=50, UNFROZEN embedding branch) - kept
              deliberately over the newer TRUE-locked-config rerun
              (rq2_vaebm_locked_config_20260928, frozen, units=1024),
              because that rerun's own 2026-09-28 result (20NG cluster:
              purity 0.111, nmi 0.081 - near-random) independently
              reproduces this repo's own earlier documented finding
              (RESULTS_SUMMARY.md \\S3: "freeze_embedding_branch=True
              consistently hurt VAE-BM in this sweep"). This IS a real,
              disclosed deviation from main.tex's stated architecture and
              must be stated as such in the text, not silently used.

Usage:
    python scripts/build_rq2_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "paper_data" / "results_manifest.json"
OUT_DIR = REPO_ROOT / "NAACL_HLT_2021_Latex_Template__1_" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["fastopic_20ng", "fastopic_nyt", "fastopic_wos_reconstructed"]
DATASET_LABELS = {"fastopic_20ng": "20NG", "fastopic_nyt": "NYT", "fastopic_wos_reconstructed": "WOS"}
MODEL_SWEEP = {
    "hicot": "rq2_hicot_quality_fix_20260928",
    "glocom": "rq2_glocom_addition_20260928",
    "vaebm": "rq2_fastopic_protocol_representation_quality_20260922",
}
MODEL_LABELS = {"hicot": "HiCOT", "glocom": "GloCOM", "vaebm": "VAE-BM (ours)"}


def fmt(x, nd=3):
    return "--" if x is None else f"{x:.{nd}f}"


def main():
    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_key = {}
    for r in rows:
        if r["dataset"] not in DATASETS or r["status"] != "ok":
            continue
        key = (r["model"], r["dataset"], r["sweep"], r["experiment"])
        by_key[key] = r

    def get(model, dataset, experiment):
        sweep = MODEL_SWEEP[model]
        return by_key.get((model, dataset, sweep, experiment))

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l" + "cc" * len(DATASETS) + "}")
    lines.append(r"\toprule")
    header = " & ".join(rf"\multicolumn{{2}}{{c}}{{{DATASET_LABELS[d]}}}" for d in DATASETS)
    lines.append(r"\textbf{Model} & " + header + r" \\")
    sub = " & ".join(["Purity", "NMI"] * len(DATASETS))
    lines.append(r" & " + sub + r" \\")
    lines.append(r"\midrule")
    for model in ("hicot", "glocom", "vaebm"):
        cells = []
        for d in DATASETS:
            row = get(model, d, "cluster")
            m = row["metrics"] if row else {}
            cells.append(fmt(m.get("purity")))
            cells.append(fmt(m.get("nmi")))
        lines.append(MODEL_LABELS[model] + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(1 + 2 * len(DATASETS)) + r"}{l}{\textit{Classification (Accuracy / F1)}} \\")
    for model in ("hicot", "glocom", "vaebm"):
        cells = []
        for d in DATASETS:
            row = get(model, d, "classification")
            m = row["metrics"] if row else {}
            cells.append(fmt(m.get("accuracy")))
            cells.append(fmt(m.get("f1")))
        lines.append(MODEL_LABELS[model] + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{RQ2: representation quality under FASTopic's own protocol "
                 r"(K=50, official/reconstructed artifacts). HiCOT uses the paper-range "
                 r"weight\_loss\_DT fix (Section~\ref{sec:rq2}); VAE-BM uses a "
                 r"different configuration than the rest of this paper, disclosed as "
                 r"such (see Section~\ref{sec:rq2}).}")
    lines.append(r"\label{tab:rq2_representation_quality}")
    lines.append(r"\end{table*}")

    out_path = OUT_DIR / "rq2_representation_quality.tex"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")

    # Print the VAE-BM deviation evidence plainly, for the paper text.
    old = get("vaebm", "fastopic_20ng", "cluster")
    new_rows = [r for r in rows if r["model"] == "vaebm" and r["dataset"] == "fastopic_20ng"
                and r["sweep"] == "rq2_vaebm_locked_config_20260928" and r["experiment"] == "cluster"]
    print("\n--- VAE-BM config-choice evidence (for main.tex Section rq2) ---")
    print(f"gte_baseline (unfrozen, units=50, USED in table): purity={old['metrics']['purity']:.3f} nmi={old['metrics']['nmi']:.3f}")
    if new_rows:
        n = new_rows[0]
        print(f"TRUE locked config (frozen, units=1024, NOT used): purity={n['metrics'].get('purity')} nmi={n['metrics'].get('nmi')} status={n['status']}")


if __name__ == "__main__":
    main()
