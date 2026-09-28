#!/usr/bin/env python
"""Builds the RQ4 completion-status table (done/missing per backbone x
dataset x experiment) plus a numeric results summary, straight from
results/rq4_llm/partial_results.csv (Experiment A: topic-word refinement)
and results/rq4_llm/expB_partial_results.csv (Experiment B: cluster/
document refinement). Same LLM (Qwen2.5-7B-Instruct, 4-bit) throughout.

Usage:
    python scripts/build_rq4_status.py
"""
from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RQ4_DIR = REPO_ROOT / "results" / "rq4_llm"
OUT_DIR = REPO_ROOT / "NAACL_HLT_2021_Latex_Template__1_" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BACKBONES = ["fastopic", "hicot", "vaebm"]
DATASETS_A = ["20ng", "agnews_short", "search_snippets"]
DATASETS_B = ["20ng", "agnews_short", "search_snippets"]


def load_last(fname):
    """Dedup by (backbone, dataset), keeping the last row with status=ok
    if any exists (matches this project's own "last ok wins" convention),
    else the last row seen."""
    path = RQ4_DIR / fname
    if not path.exists():
        return {}
    rows = list(csv.DictReader(open(path)))
    out = {}
    for r in rows:
        if not r.get("backbone") or not r.get("dataset"):
            continue
        key = (r["backbone"], r["dataset"])
        if key not in out or r.get("status") == "ok" or (r.get("base_cv") not in (None, "")):
            out[key] = r
    return out


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    a = load_last("partial_results.csv")
    b = load_last("expB_partial_results.csv")

    lines = ["=== RQ4 Completion Status (Qwen2.5-7B-Instruct, 4-bit throughout) ===\n"]
    lines.append("Experiment A (topic-word refinement, C_V):")
    for backbone in BACKBONES:
        for d in DATASETS_A:
            row = a.get((backbone, d))
            if row and f(row.get("base_cv")) is not None:
                lines.append(f"  [DONE]    {backbone:10s} x {d:16s}  cv {f(row['base_cv']):.4f} -> {f(row['refined_cv']):.4f}  "
                              f"(delta={f(row['delta_cv']):+.4f}, llm_calls={row.get('llm_calls', '?')})")
            else:
                lines.append(f"  [MISSING] {backbone:10s} x {d}")

    lines.append("\nExperiment B (cluster/document refinement, NMI/ACC):")
    for backbone in BACKBONES:
        for d in DATASETS_B:
            row = b.get((backbone, d))
            if row and f(row.get("base_nmi")) is not None:
                lines.append(f"  [DONE]    {backbone:10s} x {d:16s}  nmi {f(row['base_nmi']):.4f} -> {f(row['refined_nmi']):.4f}  "
                              f"acc {f(row.get('base_acc')) or float('nan'):.4f} -> {f(row.get('refined_acc')) or float('nan'):.4f}  "
                              f"(llm_calls={row.get('llm_calls', '?')})")
            else:
                lines.append(f"  [MISSING] {backbone:10s} x {d}")

    text = "\n".join(lines)
    print(text)
    (OUT_DIR.parent / "rq4_completion_status.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_DIR.parent / 'rq4_completion_status.txt'}")


if __name__ == "__main__":
    main()
