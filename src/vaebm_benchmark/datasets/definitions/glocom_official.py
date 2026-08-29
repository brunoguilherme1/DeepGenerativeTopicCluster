"""Fetches the official, precomputed GloCOM artifacts for SearchSnippets
directly from github.com/qducnguyen/GloCOM's `data/SearchSnippets/`
folder - the ONLY dataset the official repo ships data for (confirmed by
listing its full repo tree; GoogleNews/StackOverflow/Biomedical require
external STTM+TopMost preprocessing the repo does not itself perform or
document precisely - see docs/methodological_notes.md).

Using these files verbatim (rather than reconstructing the "global
clustering context" ourselves from raw text, which glocom_adapter.py's
`fit()` still supports as a fallback for datasets without shipped
artifacts) is the most faithful possible reproduction: the bow.npz /
global_bow.npz / global_maps.txt / vocab.txt are exactly what the paper's
own `run.py` trains on, not our own re-derivation of them.

Commit pinned (not a branch) so re-running this next year reproduces
byte-identical raw data even if the upstream repo's default branch moves:
`4094055b9e2d0169b0aa75d5aed7220e9509f0de` ("Initial commit",
2024-10-13T16:42:40Z - the latest commit on `main` as of this project's
own research pass, 2026-08; `git log --all` on the official repo shows
only this commit and one earlier "first commit").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from vaebm_benchmark.datasets.download_utils import save_url
from vaebm_benchmark.utils.paths import RAW_DIR
from vaebm_benchmark.utils.provenance import verify_manifest, write_manifest

GLOCOM_COMMIT = "4094055b9e2d0169b0aa75d5aed7220e9509f0de"
GLOCOM_BASE_URL = f"https://raw.githubusercontent.com/qducnguyen/GloCOM/{GLOCOM_COMMIT}/data/SearchSnippets"

# relative path in the official repo -> flattened local filename
_FILES = {
    "bow.npz": "bow.npz",
    "vocab.txt": "vocab.txt",
    "texts.txt": "texts.txt",
    "labels.txt": "labels.txt",
    "word_embeddings.npz": "word_embeddings.npz",
    "global/global_bow.npz": "global_bow.npz",
    "global/global_maps.txt": "global_maps.txt",
}


@dataclass
class GloCOMArtifactBundle:
    """Everything GloCOM's own `run.py` trains on for SearchSnippets,
    loaded exactly as shipped - no re-tokenization, no re-clustering."""

    dataset_id: str
    documents: list[str]  # texts.txt: whitespace-tokenized, already vocab-filtered text
    labels: list[int]
    vocab: list[str]
    bow: np.ndarray  # dense [N, V], from bow.npz
    global_bow: np.ndarray  # dense [G, V], from global_bow.npz
    global_maps: list[int]  # len N, cluster id in [0, G) per document, from global_maps.txt
    word_embeddings: Optional[np.ndarray] = None  # dense [V, 200], from word_embeddings.npz


class GloCOMOfficialSearchSnippets:
    dataset_id = "glocom_search_snippets_official"

    def raw_dir(self):
        d = RAW_DIR / self.dataset_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def download(self, force: bool = False) -> None:
        manifest_path = self.raw_dir() / "MANIFEST.yaml"
        if manifest_path.exists() and not force:
            return
        for repo_relpath, local_name in _FILES.items():
            save_url(f"{GLOCOM_BASE_URL}/{repo_relpath}", self.raw_dir() / local_name)
        write_manifest(self.raw_dir(), extra={"source_commit": GLOCOM_COMMIT, "source_base_url": GLOCOM_BASE_URL})

    def verify(self) -> tuple[bool, list[str]]:
        return verify_manifest(self.raw_dir())

    def load(self) -> GloCOMArtifactBundle:
        import scipy.sparse as sp

        self.download()
        d = self.raw_dir()

        vocab = d.joinpath("vocab.txt").read_text(encoding="utf-8").splitlines()
        texts = d.joinpath("texts.txt").read_text(encoding="utf-8").splitlines()
        labels = [int(x) for x in d.joinpath("labels.txt").read_text(encoding="utf-8").splitlines()]
        global_maps = [int(x) for x in d.joinpath("global_maps.txt").read_text(encoding="utf-8").splitlines()]

        bow = sp.load_npz(d / "bow.npz")
        bow = bow.toarray() if hasattr(bow, "toarray") else np.asarray(bow)
        global_bow = sp.load_npz(d / "global_bow.npz")
        global_bow = global_bow.toarray() if hasattr(global_bow, "toarray") else np.asarray(global_bow)

        word_embeddings = None
        we_path = d / "word_embeddings.npz"
        if we_path.exists():
            we = sp.load_npz(we_path)
            word_embeddings = we.toarray() if hasattr(we, "toarray") else np.asarray(we)

        return GloCOMArtifactBundle(
            dataset_id="search_snippets",
            documents=texts,
            labels=labels,
            vocab=vocab,
            bow=bow.astype("float32"),
            global_bow=global_bow.astype("float32"),
            global_maps=global_maps,
            word_embeddings=word_embeddings,
        )
