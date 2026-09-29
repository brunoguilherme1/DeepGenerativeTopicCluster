#!/usr/bin/env python
"""Builds the FINAL RQ1 topic-modeling table (tables/topic_vaebm_all5.tex)
from the real multi-seed rerun (docs/rq1_final_protocol.md): global
lambda_relevance=0.1, disclosed epochs=1, Palmetto-only C_V, 5 seeds
(1-5) for 20NG/AGNews/SearchSnippets/GoogleNews, 3 seeds (1-3) for IMDB.

Replaces the prior single-seed, per-dataset-tuned-lambda table, which had
one confirmed transcription error (SearchSnippets K50 printed 0.482,
actually GoogleNews's value copied in by mistake) and a lambda-selection
circularity concern (lambda picked via the same C_V being reported).

A genuine finding from the multi-seed data itself: std across the 5 (or
3) seeds was exactly (or near-exactly) zero for every dataset except
IMDB, which had a small non-zero std (~1e-3 to 1e-5). This is disclosed
explicitly in the table caption rather than silently implying real
stochastic averaging occurred - the locked, frozen configuration makes
the embedding branch and (independently) the clustering step's own
default seeding largely insensitive to the outer --seed flag.

Usage:
    python scripts/build_rq1_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "paper_data" / "results_manifest.json"
OUT_DIR = REPO_ROOT / "NAACL_HLT_2021_Latex_Template__1_" / "tables"

DATASET_ORDER = ["hicot_20ng", "hicot_agnews", "hicot_imdb", "hicot_search_snippets", "hicot_google_news"]

# HiCOT's own published numbers (Findings ACL 2025), reproduced verbatim
# from the table this script replaces - not re-derived here.
HICOT_PUBLISHED = {
    50: {
        "hicot_20ng": (0.451, 0.626, 0.583, 0.852),
        "hicot_agnews": (0.446, 0.857, 0.412, 0.992),
        "hicot_imdb": (0.404, 0.737, 0.082, 0.837),
        "hicot_search_snippets": (0.460, 0.818, 0.478, 1.000),
        "hicot_google_news": (0.454, 0.465, 0.657, 0.920),
    },
    100: {
        "hicot_20ng": (0.424, 0.652, 0.568, 0.741),
        "hicot_agnews": (0.435, 0.862, 0.388, 0.960),
        "hicot_imdb": (0.388, 0.739, 0.071, 0.733),
        "hicot_search_snippets": (0.449, 0.857, 0.480, 0.940),
        "hicot_google_news": (0.470, 0.763, 0.864, 0.802),
    },
}

BASELINE_ROWS_K50 = r"""ETM\textdagger & 0.375 & 0.347 & 0.319 & 0.704 & 0.364 & 0.679 & 0.224 & 0.819 & 0.346 & 0.660 & 0.038 & 0.557 & 0.397 & 0.688 & 0.389 & 0.594 & 0.402 & 0.366 & 0.560 & 0.916 \\
NTM+CL & 0.437 & 0.582 & 0.491 & 0.802 & 0.440 & 0.322 & 0.100 & 0.441 & 0.396 & 0.657 & 0.044 & 0.617 & 0.403 & 0.215 & 0.030 & 0.532 & 0.433 & 0.041 & 0.005 & 0.301 \\
ECRTM\textdagger & 0.431 & 0.560 & 0.524 & 0.964 & 0.466 & 0.802 & 0.367 & 0.961 & 0.393 & 0.694 & 0.058 & 0.974 & 0.450 & 0.711 & 0.419 & 0.998 & 0.441 & 0.396 & 0.615 & 0.987 \\
FASTopic & 0.427 & 0.583 & 0.528 & 0.980 & 0.379 & 0.831 & 0.352 & 0.960 & 0.371 & 0.683 & 0.055 & 0.969 & 0.356 & 0.793 & 0.497 & 0.519 & 0.401 & 0.252 & 0.570 & 0.235 \\
NeuroMax\textdagger & 0.435 & 0.623 & 0.570 & 0.912 & 0.385 & 0.804 & 0.410 & 0.952 & 0.402 & 0.709 & 0.061 & 0.936 & 0.427 & 0.743 & 0.427 & 0.920 & 0.409 & 0.359 & 0.590 & 1.000 \\"""

BASELINE_ROWS_K100 = r"""ETM\textdagger & 0.369 & 0.394 & 0.339 & 0.573 & 0.371 & 0.674 & 0.204 & 0.773 & 0.341 & 0.648 & 0.037 & 0.371 & 0.389 & 0.691 & 0.365 & 0.448 & 0.398 & 0.554 & 0.713 & 0.677 \\
NTM+CL & 0.420 & 0.626 & 0.490 & 0.624 & 0.415 & 0.280 & 0.050 & 0.277 & 0.382 & 0.705 & 0.044 & 0.492 & 0.406 & 0.217 & 0.020 & 0.394 & 0.432 & 0.039 & 0.005 & 0.367 \\
ECRTM\textdagger & 0.405 & 0.555 & 0.494 & 0.904 & 0.416 & 0.812 & 0.428 & 0.981 & 0.373 & 0.694 & 0.049 & 0.887 & 0.432 & 0.789 & 0.443 & 0.966 & 0.418 & 0.342 & 0.491 & 0.991 \\
FASTopic & 0.400 & 0.622 & 0.522 & 0.861 & 0.385 & 0.833 & 0.330 & 0.912 & 0.369 & 0.680 & 0.048 & 0.886 & 0.350 & 0.801 & 0.466 & 0.463 & 0.366 & 0.237 & 0.459 & 0.100 \\
NeuroMax\textdagger & 0.412 & 0.602 & 0.516 & 0.913 & 0.406 & 0.828 & 0.389 & 0.957 & 0.381 & 0.706 & 0.059 & 0.870 & 0.439 & 0.854 & 0.472 & 0.960 & 0.427 & 0.664 & 0.834 & 0.915 \\"""


def load_rq1_final():
    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out = {}
    for r in rows:
        if not r["sweep"].startswith("rq1_final") or r["status"] != "ok":
            continue
        out[(r["dataset"], r["k"])] = r["metrics"]
    return out


def fmt(x, nd=3):
    return "--" if x is None else f"{x:.{nd}f}"


def hicot_row(vals):
    cv, pur, nmi, td = vals
    return f"HiCOT & {cv:.3f} & {pur:.3f} & {nmi:.3f} & {td:.3f}"


def vaebm_row(k, ours):
    cells = []
    for ds in DATASET_ORDER:
        m = ours[(ds, k)]
        hcv, hpur, hnmi, htd = HICOT_PUBLISHED[k][ds]
        cv, pur, nmi, td = m["cv"], m["purity"], m["nmi"], m["td"]
        cell_cv = rf"\textbf{{{cv:.3f}}}" if cv > hcv else f"{cv:.3f}"
        cell_pur = rf"\textbf{{{pur:.3f}}}" if pur > hpur else f"{pur:.3f}"
        cell_nmi = rf"\textbf{{{nmi:.3f}}}" if nmi > hnmi else f"{nmi:.3f}"
        cell_td = f"{td:.3f}"
        cells.extend([cell_cv, cell_pur, cell_nmi, cell_td])
    return " & ".join(cells)


def main():
    ours = load_rq1_final()
    missing = [(ds, k) for ds in DATASET_ORDER for k in (50, 100) if (ds, k) not in ours]
    if missing:
        print("WARNING missing cells (not overwriting table):", missing)
        return

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\adjustbox{max width=\textwidth}{")
    lines.append(r"\begin{tabular}{lrrrr rrrr rrrr rrrr rrrr}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{4}{c}{\textbf{20NG}} & \multicolumn{4}{c}{\textbf{AGNews}} & \multicolumn{4}{c}{\textbf{IMDB}} & \multicolumn{4}{c}{\textbf{SearchSnip.}} & \multicolumn{4}{c}{\textbf{GoogleNews}} \\")
    lines.append(r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}\cmidrule(lr){14-17}\cmidrule(lr){18-21}")
    lines.append(r" & $C_V$ & Pur. & NMI & TD & $C_V$ & Pur. & NMI & TD & $C_V$ & Pur. & NMI & TD & $C_V$ & Pur. & NMI & TD & $C_V$ & Pur. & NMI & TD \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{21}{l}{\textit{50 Topics}} \\")
    lines.append(BASELINE_ROWS_K50)
    cv, pur, nmi, td = HICOT_PUBLISHED[50]["hicot_20ng"]
    # HiCOT row is dataset-specific; build it inline using the same published numbers as the baseline table.
    hicot_cells = []
    for ds in DATASET_ORDER:
        hicot_cells.extend([f"{v:.3f}" for v in HICOT_PUBLISHED[50][ds]])
    lines.append("HiCOT & " + " & ".join(hicot_cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\textbf{VAE-BM (ours)} & " + vaebm_row(50, ours) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{21}{l}{\textit{100 Topics}} \\")
    lines.append(BASELINE_ROWS_K100)
    hicot_cells = []
    for ds in DATASET_ORDER:
        hicot_cells.extend([f"{v:.3f}" for v in HICOT_PUBLISHED[100][ds]])
    lines.append("HiCOT & " + " & ".join(hicot_cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\textbf{VAE-BM (ours)} & " + vaebm_row(100, ours) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\vspace{2pt}")
    lines.append(
        r"\caption{Topic results at $K=50$ and $K=100$, all five of HiCOT's official benchmark datasets. "
        r"Baseline rows (ETM through HiCOT) reproduced from the HiCOT paper's own published table; "
        r"\textdagger\ marks a baseline HiCOT itself reports as its own reproduction, not the original "
        r"paper's number. \textbf{Bold} in our row marks a metric that beats HiCOT's published value. "
        r"VAE-BM (ours) uses a single GLOBAL $\lambda_{\text{relevance}}=0.1$ (not tuned per dataset/$K$ "
        r"against this same $C_V$ - see \S\ref{sec:hp-search} and the $\lambda$-sensitivity appendix "
        r"table, Appendix~\ref{sec:appendix-config}), epochs$=$1 (disclosed), fully unsupervised, "
        r"5 seeds (1--5) for 20NG/AGNews/SearchSnippets/GoogleNews and 3 seeds (1--3) for IMDB; every "
        r"cell above is the mean over these seeds. Seed-to-seed std was exactly or near-exactly zero for "
        r"every dataset except IMDB (std $\le 0.003$) - the locked, frozen configuration makes both the "
        r"embedding branch and the clustering step largely insensitive to the outer seed, disclosed here "
        r"rather than silently implied as genuine stochastic averaging.}"
    )
    lines.append(r"\label{tab:topic-k50}")
    lines.append(r"\end{table*}")

    out_path = OUT_DIR / "topic_vaebm_all5.tex"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
