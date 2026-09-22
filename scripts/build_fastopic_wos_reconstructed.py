#!/usr/bin/env python
"""Builds data/raw/fastopic_wos_reconstructed/{texts,labels,vocab}.txt +
provenance.json from the raw WOS-11967 (HDLTex, Kowsari et al. 2017)
source, downloading it directly from Mendeley Data (DOI
10.17632/9rw3vkcfy4.2) if not already present.

See src/vaebm_benchmark/datasets/definitions/fastopic_wos.py's module
docstring for full provenance detail and exactly what is/isn't an exact
FASTopic-paper reproduction here. In short: the raw source and the 5-step
TopMost preprocessing pipeline are exact/documented (FASTopic paper
Appendix B); the specific 11,967->10,000 document subsampling FASTopic/
TopMost used is not published anywhere, so this script's own MIN_TERM
document-length filter + seed=42 subsample is a documented substitute,
calibrated against the paper's own reported avg length (110.0) rather
than guessed - achieves 110.36.

Usage:
    python scripts/build_fastopic_wos_reconstructed.py
"""
from __future__ import annotations

import json
import re
import shutil
import string
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import requests
from gensim.parsing.preprocessing import STOPWORDS

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "raw" / "fastopic_wos_reconstructed"
CACHE_ZIP = OUT_DIR / "_source" / "WebOfScience.zip"

MENDELEY_DOWNLOAD_URL = (
    "https://data.mendeley.com/public-files/datasets/9rw3vkcfy4/files/"
    "4ed4e75a-a656-4e55-9cb2-f5638b4de00a/file_downloaded"
)
MENDELEY_DOI = "10.17632/9rw3vkcfy4.2"
EXPECTED_SHA256_RAW_ZIP = "b787d484bff88b0dcdb3fa291d06ec9d2f025dc2a67ce1045d0c688cd96ccf8a"

SEED = 42
TARGET_DOCS = 10000
VOCAB_SIZE = 10000
MIN_LENGTH = 3
MIN_TERM = 65  # calibrated against achieved avg length matching the 110.0 target (achieves 110.36)

punct_chars = list(set(string.punctuation) - set("'"))
punct_chars.sort()
punctuation = ''.join(punct_chars)
replace = re.compile('[%s]' % re.escape(punctuation))
alpha = re.compile('^[a-zA-Z_]+$')


def clean_text(text: str) -> str:
    text = re.sub(r'<', '(', text)
    text = re.sub(r'>', ')', text)
    text = text.lower()
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\s@\S+', ' ', text)
    text = re.sub(r'_', ' ', text)
    text = re.sub(r"\s'", ' ', text)
    text = re.sub(r"'\s", ' ', text)
    text = re.sub(r'\.', '', text)
    text = replace.sub(' ', text)
    text = re.sub(r"'", '', text)
    text = re.sub(r'\s', ' ', text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    text = clean_text(text)
    tokens = text.split()
    tokens = ['_' if t in STOPWORDS else t for t in tokens]
    tokens = [t if alpha.match(t) else '_' for t in tokens]
    tokens = [t if len(t) >= MIN_LENGTH else '_' for t in tokens]
    return [t for t in tokens if t != '_']


def fetch_raw() -> Path:
    CACHE_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE_ZIP.exists():
        print(f"Downloading {MENDELEY_DOWNLOAD_URL} ...")
        resp = requests.get(MENDELEY_DOWNLOAD_URL, timeout=300)
        resp.raise_for_status()
        CACHE_ZIP.write_bytes(resp.content)
    import hashlib
    actual = hashlib.sha256(CACHE_ZIP.read_bytes()).hexdigest()
    if actual != EXPECTED_SHA256_RAW_ZIP:
        raise ValueError(
            f"WebOfScience.zip SHA256 {actual} != expected {EXPECTED_SHA256_RAW_ZIP} "
            f"(Mendeley content_details.sha256_hash) - refusing to proceed silently."
        )
    extract_dir = CACHE_ZIP.parent / "extracted"
    if not (extract_dir / "WOS11967" / "X.txt").exists():
        with zipfile.ZipFile(CACHE_ZIP) as zf:
            zf.extractall(extract_dir)
    return extract_dir / "WOS11967"


def main() -> None:
    raw_dir = fetch_raw()
    raw_texts = [l.strip() for l in (raw_dir / "X.txt").read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    raw_labels = [int(x) for x in (raw_dir / "YL1.txt").read_text().split()]
    assert len(raw_texts) == len(raw_labels), (len(raw_texts), len(raw_labels))
    print(f"raw docs: {len(raw_texts)}, labels: {sorted(set(raw_labels))}")

    tokenized = [tokenize(t) for t in raw_texts]
    doc_counts = Counter()
    for toks in tokenized:
        doc_counts.update(set(toks))

    vocab = sorted([w for w, _ in doc_counts.most_common(VOCAB_SIZE)])
    vocab_set = set(vocab)
    parsed = [[t for t in toks if t in vocab_set] for toks in tokenized]
    lengths = np.array([len(p) for p in parsed])

    keep_idx = np.where(lengths >= MIN_TERM)[0]
    rng = np.random.RandomState(SEED)
    if len(keep_idx) < TARGET_DOCS:
        raise RuntimeError(f"only {len(keep_idx)} docs survive MIN_TERM={MIN_TERM}, need {TARGET_DOCS}")
    sample_idx = np.sort(rng.choice(keep_idx, size=TARGET_DOCS, replace=False))

    final_lengths = lengths[sample_idx]
    final_labels = [raw_labels[i] for i in sample_idx]
    print(f"FINAL: {len(sample_idx)} docs, avg length={final_lengths.mean():.2f} (target 110.0), "
          f"labels={sorted(set(final_labels))} (target 7), vocab={len(vocab)} (target {VOCAB_SIZE})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "texts.txt").write_text("\n".join(" ".join(parsed[i]) for i in sample_idx) + "\n")
    (OUT_DIR / "labels.txt").write_text("\n".join(str(raw_labels[i]) for i in sample_idx) + "\n")
    (OUT_DIR / "vocab.txt").write_text("\n".join(vocab) + "\n")

    provenance = {
        "dataset_id": "fastopic_wos_reconstructed",
        "exact_fastopic_artifact": False,
        "source": f"Mendeley Data DOI {MENDELEY_DOI} (Kowsari et al. 2017, HDLTex), WOS11967 subset (7 parent categories = YL1.txt)",
        "source_citation": "Kowsari, Brown, Heidarysafa, Jafari Meimandi, Gerber, Barnes. HDLTex: Hierarchical Deep Learning for Text Classification. ICMLA 2017.",
        "cited_by_fastopic_as": "reference [32] for the WoS dataset",
        "preprocessing": "TopMost's exact 5-step pipeline (tokenize+lowercase, remove punctuation, remove tokens with numbers, remove tokens <3 chars, remove English/gensim stopwords) per FASTopic paper Appendix B, reimplemented from topmost/preprocess/preprocess.py's Tokenizer class read directly from the official repo",
        "unrecoverable_detail": f"The exact 11,967 -> 10,000 document subsampling/filtering procedure FASTopic/TopMost used is not published or present in TopMost's repo (verified via full git history of its data/ directory - no WoS file ever committed). This script filters docs with fewer than MIN_TERM={MIN_TERM} post-vocab tokens, then takes a seed=42 random sample of exactly 10,000 of the survivors.",
        "min_term_threshold": MIN_TERM,
        "seed": SEED,
        "target_stats": {"docs": 10000, "labels": 7, "vocab": 10000, "avg_length": 110.0},
        "achieved_stats": {
            "docs": int(len(sample_idx)),
            "labels": len(set(final_labels)),
            "vocab": len(vocab),
            "avg_length": float(final_lengths.mean()),
        },
    }
    (OUT_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
