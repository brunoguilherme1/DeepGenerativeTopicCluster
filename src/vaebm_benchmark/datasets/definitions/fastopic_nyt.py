"""NYT (New York Times), fetched via `topmost.download_dataset('NYT', ...)`
- the exact artifact FASTopic's own paper protocol uses (FASTopic's
companion toolkit, github.com/bobxwu/topmost, mirrors it at
raw.githubusercontent.com/BobXWu/TopMost/master/data/NYT.zip). Chosen as
this project's FASTopic smoke-test dataset because it is the smallest of
the paper's downloadable, labeled datasets (9,172 docs total: 8,254 train
/ 918 test - see protocols/fastopic_protocol.py's docstring for how this
was verified against the paper's own Table 7).

Ships PRE-SPLIT train/test files (`train_bow.npz`/`test_bow.npz`/
`train_texts.txt`/`test_texts.txt`/`vocab.txt`/`train_labels.txt`/
`test_labels.txt`) - this project does not generate its own split for
this dataset, per its own "don't invent a split the paper didn't use"
rule; the split is whatever ships in the official artifact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vaebm_benchmark.utils.paths import RAW_DIR
from vaebm_benchmark.utils.provenance import sha256_of, verify_manifest, write_manifest

# Expected SHA256 per extracted file, pinned by THIS PROJECT'S OWN first
# download (2026-08) of topmost.download_dataset('NYT', ...) - unlike
# GloCOM's EXPECTED_SHA256 (glocom_official.py), no independent third
# party has published a reference hash for NYT.zip specifically (only for
# 20NG), so this is trust-on-first-download, not independent
# corroboration. Still strictly better than recording no expected hash at
# all: it now catches "the upstream TopMost mirror changed since this was
# pinned" on every subsequent run, not only local corruption.
EXPECTED_SHA256 = {
    "train_bow.npz": "d62e5d4453aaf8b8482b8aa34d14f9113346e641d116c8c2de4a5e64edcd5546",
    "test_bow.npz": "c40537ab632fde34c27d16e1d9cef3f5cd56702588328367a8497107249bb68f",
    "train_texts.txt": "50fea66eb4eee8b9e6b47ec5e2dd6436d1712f9bcd340c3d0349000ab77e089a",
    "test_texts.txt": "1aaa2719171ada85df330cd4074c32dc402a69e9f9ae14fcf17c2add96e24895",
    "vocab.txt": "74046b12410b5ad711d596ca3cfcd0fb6c39e01fdc3624d270b28b874eb859d5",
    "train_labels.txt": "4a34f158bb706dbac02ab4fd3af27bd8949253afec0ece10b57e50e39d0d78d6",
    "test_labels.txt": "8ed29679d83934ce084b397f3460bd47d8b377124706a192a3263fad4dfe8e1a",
    "word_embeddings.npz": "b46363a49c0e6d99427ff0e01bdc7491f17614c02939847861d9b0502a3c501d",
}


@dataclass
class NYTBundle:
    train_texts: list[str]
    test_texts: list[str]
    train_labels: list[int]
    test_labels: list[int]
    vocab: list[str]
    train_bow: np.ndarray
    test_bow: np.ndarray


class NYTDataset:
    dataset_id = "nyt"

    def raw_dir(self):
        d = RAW_DIR / self.dataset_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def download(self, force: bool = False) -> None:
        manifest_path = self.raw_dir() / "MANIFEST.yaml"
        if manifest_path.exists() and not force:
            return
        import topmost

        topmost.download_dataset("NYT", cache_path=str(self.raw_dir()))
        nyt_dir = self.raw_dir() / "NYT"
        for filename, expected in EXPECTED_SHA256.items():
            path = nyt_dir / filename
            if not path.exists():
                raise ValueError(f"Expected file {filename} missing after topmost.download_dataset('NYT', ...)")
            actual = sha256_of(path)
            if actual != expected:
                raise ValueError(
                    f"{filename}: downloaded SHA256 {actual} does not match the hash this project "
                    f"pinned on first download ({expected}) - the upstream TopMost mirror may have "
                    f"changed, or the download was corrupted. Refusing to proceed silently."
                )
        write_manifest(
            nyt_dir,
            extra={"source": "https://raw.githubusercontent.com/BobXWu/TopMost/master/data/NYT.zip"},
        )

    def verify(self) -> tuple[bool, list[str]]:
        return verify_manifest(self.raw_dir() / "NYT")

    def load(self) -> NYTBundle:
        import topmost

        self.download()
        dataset = topmost.BasicDataset(str(self.raw_dir() / "NYT"), device="cpu", read_labels=True)

        def _dense(x):
            return x.toarray() if hasattr(x, "toarray") else np.asarray(x)

        return NYTBundle(
            train_texts=list(dataset.train_texts),
            test_texts=list(dataset.test_texts),
            train_labels=[int(x) for x in dataset.train_labels],
            test_labels=[int(x) for x in dataset.test_labels],
            vocab=list(dataset.vocab),
            train_bow=_dense(dataset.train_bow).astype("float32"),
            test_bow=_dense(dataset.test_bow).astype("float32"),
        )
