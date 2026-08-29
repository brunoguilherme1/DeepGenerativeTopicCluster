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
from vaebm_benchmark.utils.provenance import verify_manifest, write_manifest


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
        write_manifest(
            self.raw_dir() / "NYT",
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
