"""NeurIPS (conference papers, 1987-2017), fetched directly from
raw.githubusercontent.com/BobXWu/TopMost/master/data/NeurIPS.zip - same
official mirror pattern as fastopic_20ng.py/fastopic_nyt.py, no `topmost`
package dependency.

7,237 docs total: 6,512 train / 725 test, vocab_size=10,000 - matches
FASTopic paper Table 7 exactly. NO ground-truth labels (Table 7 lists
"-" for #labels) - used for topic-quality (C_V/TD, Table 1) only, never
for clustering/classification (Table 2/Figure 4 only cover 20NG/NYT/WoS).
Ships train_times.txt/test_times.txt/time2id.txt too (a temporal-topic-
model variant of this same artifact) - unused here, this protocol has no
temporal component.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass

import numpy as np
import scipy.sparse

from vaebm_benchmark.utils.paths import RAW_DIR
from vaebm_benchmark.utils.provenance import sha256_of, verify_manifest, write_manifest

SOURCE_URL = "https://raw.githubusercontent.com/BobXWu/TopMost/master/data/NeurIPS.zip"

EXPECTED_SHA256 = {
    "test_bow.npz": "8890ed4ce2bc41451dde35dff099a46ab259a5dac8206841d0d155658dd04e3a",
    "test_texts.txt": "16d7fd44f1187aacf69621e1560cab825cfce360e9a575d814387ef76b979145",
    "train_bow.npz": "ac2bb75b892daea2be6c5348ea65074f8afebbc85c8a8c3e3326c24bbd189e8c",
    "train_texts.txt": "55ab7dd81832301ad2b02d8b2c9417192074fd2087d47ecc4edfb63d440882d8",
    "vocab.txt": "1601d6fef9eb6ba748d32254121fef5b5cd9646716c48e00917e15d95fd6a262",
    "word_embeddings.npz": "7d0f51b47d24b65b1d202b6b99faeb3d3f322693d38cd91b4c194d208bb995b9",
}


@dataclass
class NeurIPSFASTopicBundle:
    train_texts: list[str]
    test_texts: list[str]
    vocab: list[str]
    train_bow: np.ndarray
    test_bow: np.ndarray


class NeurIPSFASTopicDataset:
    dataset_id = "fastopic_neurips"

    def raw_dir(self):
        d = RAW_DIR / self.dataset_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def download(self, force: bool = False) -> None:
        manifest_path = self.raw_dir() / "MANIFEST.yaml"
        if manifest_path.exists() and not force:
            return
        import requests

        ds_dir = self.raw_dir() / "NeurIPS"
        if not (ds_dir / "train_texts.txt").exists():
            zip_path = self.raw_dir() / "NeurIPS.zip"
            resp = requests.get(SOURCE_URL, timeout=300)
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
        return verify_manifest(self.raw_dir() / "NeurIPS")

    def load(self) -> NeurIPSFASTopicBundle:
        self.download()
        ds_dir = self.raw_dir() / "NeurIPS"

        def _read_lines(name):
            return (ds_dir / name).read_text(encoding="utf-8").splitlines()

        return NeurIPSFASTopicBundle(
            train_texts=_read_lines("train_texts.txt"),
            test_texts=_read_lines("test_texts.txt"),
            vocab=_read_lines("vocab.txt"),
            train_bow=scipy.sparse.load_npz(ds_dir / "train_bow.npz").toarray().astype("float32"),
            test_bow=scipy.sparse.load_npz(ds_dir / "test_bow.npz").toarray().astype("float32"),
        )
