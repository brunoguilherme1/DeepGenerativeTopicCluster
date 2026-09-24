"""Wikitext-103 (Wikipedia articles), fetched directly from TopMost's
official mirror - UNLIKE every other fastopic_*.py dataset here, this one
file is Git-LFS-tracked in the TopMost repo, so raw.githubusercontent.com
serves only a ~130-byte LFS pointer, not the actual zip (verified
2026-09-24: a plain `requests.get` on the raw.githubusercontent.com URL
returns `version https://git-lfs.github.com/spec/v1 ...`). The real bytes
are fetched from GitHub's LFS media endpoint instead
(media.githubusercontent.com/media/.../main/...) - same content, just a
different URL, still no `topmost` package dependency. Also note the
branch is `main` here, not `master` (`master` 404s against this endpoint).

28,532 docs total: 28,472 train / 60 test, vocab_size=10,000 - matches
FASTopic paper Table 7 exactly. NO ground-truth labels (Table 7 lists "-"
for #labels) - used for topic-quality (C_V/TD, Table 1) only.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass

import numpy as np
import scipy.sparse

from vaebm_benchmark.utils.paths import RAW_DIR
from vaebm_benchmark.utils.provenance import sha256_of, verify_manifest, write_manifest

SOURCE_URL = "https://media.githubusercontent.com/media/BobXWu/TopMost/main/data/Wikitext-103.zip"

EXPECTED_SHA256 = {
    "test_bow.npz": "18df7f6945fbd0ca86e9d0cc29f0b927b179afd560b1f2a9188b8aa50fb053fb",
    "test_texts.txt": "df68c52c9d26097893032bf25934d9b39e631a1e6b8f1d94477b2981802e980d",
    "train_bow.npz": "93981bd14ffc51509cb56eedf2f64a7018435694d6cd7763f3e168444753b370",
    "train_texts.txt": "6468a14a72eb82e3a4199fa2cbecd62f0855da6e99d30a93ea578e48969ecc00",
    "vocab.txt": "28b68d061d66cba54efc1f58e04e6b0898b03e28747ec3688b070462e088f04b",
    "word_embeddings.npz": "defca3c07b6f8a33812fbc85bf9524579fb3dec8255be6560b3c78824bdf3cb4",
}


@dataclass
class Wikitext103FASTopicBundle:
    train_texts: list[str]
    test_texts: list[str]
    vocab: list[str]
    train_bow: np.ndarray
    test_bow: np.ndarray


class Wikitext103FASTopicDataset:
    dataset_id = "fastopic_wikitext103"

    def raw_dir(self):
        d = RAW_DIR / self.dataset_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def download(self, force: bool = False) -> None:
        manifest_path = self.raw_dir() / "MANIFEST.yaml"
        if manifest_path.exists() and not force:
            return
        import requests

        ds_dir = self.raw_dir() / "Wikitext-103"
        if not (ds_dir / "train_texts.txt").exists():
            zip_path = self.raw_dir() / "Wikitext-103.zip"
            resp = requests.get(SOURCE_URL, timeout=600)
            resp.raise_for_status()
            zip_path.write_bytes(resp.content)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(self.raw_dir())
        for filename, expected in EXPECTED_SHA256.items():
            path = ds_dir / filename
            if not path.exists():
                raise ValueError(f"Expected file {filename} missing after downloading {SOURCE_URL}")
            actual = sha256_of(path)
            if actual != expected:
                raise ValueError(
                    f"{filename}: downloaded SHA256 {actual} does not match the hash this project "
                    f"pinned on first download ({expected}) - the upstream TopMost mirror may have "
                    f"changed, or the download was corrupted. Refusing to proceed silently."
                )
        write_manifest(ds_dir, extra={"source": SOURCE_URL})

    def verify(self) -> tuple[bool, list[str]]:
        return verify_manifest(self.raw_dir() / "Wikitext-103")

    def load(self) -> Wikitext103FASTopicBundle:
        self.download()
        ds_dir = self.raw_dir() / "Wikitext-103"

        def _read_lines(name):
            return (ds_dir / name).read_text(encoding="utf-8").splitlines()

        return Wikitext103FASTopicBundle(
            train_texts=_read_lines("train_texts.txt"),
            test_texts=_read_lines("test_texts.txt"),
            vocab=_read_lines("vocab.txt"),
            train_bow=scipy.sparse.load_npz(ds_dir / "train_bow.npz").toarray().astype("float32"),
            test_bow=scipy.sparse.load_npz(ds_dir / "test_bow.npz").toarray().astype("float32"),
        )
