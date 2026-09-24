"""ACL Anthology papers (1970-2015), fetched directly from
raw.githubusercontent.com/BobXWu/TopMost/master/data/ACL.zip - same
official mirror pattern as fastopic_20ng.py/fastopic_nyt.py, no `topmost`
package dependency.

10,560 docs total: 9,507 train / 1,053 test, vocab_size=10,000 - matches
FASTopic paper Table 7 exactly. NO ground-truth labels (Table 7 lists
"-" for #labels) - used for topic-quality (C_V/TD, Table 1) only, never
for clustering/classification (Table 2/Figure 4 only cover 20NG/NYT/WoS).
Ships train_times.txt/test_times.txt/time2id.txt too (a temporal-topic-
model variant of this same artifact) - unused here.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass

import numpy as np
import scipy.sparse

from vaebm_benchmark.utils.paths import RAW_DIR
from vaebm_benchmark.utils.provenance import sha256_of, verify_manifest, write_manifest

SOURCE_URL = "https://raw.githubusercontent.com/BobXWu/TopMost/master/data/ACL.zip"

EXPECTED_SHA256 = {
    "test_bow.npz": "9a92fe337f622b7c45ebc16dc938877660ec24e73e2c67e0a135b209f85210b4",
    "test_texts.txt": "a631d069663002ecf0d1ca598cd8652a98f4723cc8776a2e206df572a8d13de0",
    "train_bow.npz": "6346e56aac0d417eed4b0b0b2f574a3a0385c6484b7e3acb1f566cab398b7298",
    "train_texts.txt": "ac9baf66ea81d49209c60b4ac17d49d7687da118f7cd80293dfa43b9c582d115",
    "vocab.txt": "389eff0df6c96f7c84b6a5f99e5c911fceab618e9fe7bece8f35a4d4f476e71f",
    "word_embeddings.npz": "60e74d4c0fdcb3de6b3a0cc07d41bf41817c0753c79f591aae60f0ad5d3c991a",
}


@dataclass
class ACLFASTopicBundle:
    train_texts: list[str]
    test_texts: list[str]
    vocab: list[str]
    train_bow: np.ndarray
    test_bow: np.ndarray


class ACLFASTopicDataset:
    dataset_id = "fastopic_acl"

    def raw_dir(self):
        d = RAW_DIR / self.dataset_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def download(self, force: bool = False) -> None:
        manifest_path = self.raw_dir() / "MANIFEST.yaml"
        if manifest_path.exists() and not force:
            return
        import requests

        ds_dir = self.raw_dir() / "ACL"
        if not (ds_dir / "train_texts.txt").exists():
            zip_path = self.raw_dir() / "ACL.zip"
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
        return verify_manifest(self.raw_dir() / "ACL")

    def load(self) -> ACLFASTopicBundle:
        self.download()
        ds_dir = self.raw_dir() / "ACL"

        def _read_lines(name):
            return (ds_dir / name).read_text(encoding="utf-8").splitlines()

        return ACLFASTopicBundle(
            train_texts=_read_lines("train_texts.txt"),
            test_texts=_read_lines("test_texts.txt"),
            vocab=_read_lines("vocab.txt"),
            train_bow=scipy.sparse.load_npz(ds_dir / "train_bow.npz").toarray().astype("float32"),
            test_bow=scipy.sparse.load_npz(ds_dir / "test_bow.npz").toarray().astype("float32"),
        )
