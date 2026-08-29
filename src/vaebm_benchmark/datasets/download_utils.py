"""Shared HTTP + parsing helpers. Dataset definitions call these; they
never issue requests directly, so every dataset's provenance capture
(checksums, MANIFEST.yaml) goes through one code path."""

from __future__ import annotations

from pathlib import Path

import requests

TIMEOUT_SECONDS = 30


def fetch_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.content


def save_url(url: str, dest: Path) -> None:
    dest.write_bytes(fetch_bytes(url))


def parse_label_tab_text(path: Path) -> tuple[list[str], list[str]]:
    """Parses the "<label>\\t<text>" single-file layout used by the STC2
    short-text-clustering benchmark family (SearchSnippets, Biomedical,
    StackOverflow, ...) - the exact artifact GloCOM's own paper evaluates
    on for these corpora."""
    texts: list[str] = []
    raw_labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip("\n")
        if not line.strip():
            continue
        label, _, text = line.partition("\t")
        texts.append(text.strip())
        raw_labels.append(label.strip())
    return texts, raw_labels


def encode_labels(raw_labels: list[str]) -> tuple[list[int], dict[int, str]]:
    unique = sorted(set(raw_labels))
    label_to_id = {label: idx for idx, label in enumerate(unique)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    encoded = [label_to_id[label] for label in raw_labels]
    return encoded, id_to_label
