"""WoS (Web of Science), reconstructed -- NOT the exact FASTopic/TopMost
artifact, because no such artifact exists to obtain: TopMost's own `data/`
directory (github.com/bobxwu/topmost) has never distributed a WoS file at
any point in its git history (verified via the GitHub API's full commit
history for that path), and the FASTopic PyPI package/repo ship no dataset
files of their own either.

Provenance that IS exact: FASTopic's paper cites reference [32] for WoS,
which is Kowsari et al. 2017 ("HDLTex: Hierarchical Deep Learning for Text
Classification"). That paper's own repo (github.com/kk7nc/HDLTex) links
to the source dataset at Mendeley Data, DOI 10.17632/9rw3vkcfy4.2 ("Web of
Science"), which ships three subsets: WOS-5736, WOS-11967, WOS-46985.
WOS-11967 (11,967 docs, 7 parent categories via YL1.txt) is the only one
matching FASTopic Table 7's 7-label count for WoS, and its raw abstracts
were downloaded directly from that Mendeley record (file `WebOfScience.zip`,
sha256 pinned below).

What is NOT exact: the specific 11,967 -> 10,000 document subsampling/
filtering procedure FASTopic/TopMost used to reach their reported 10,000-
doc/110.0-avg-length subset is not published or present anywhere in
TopMost's repo. This dataset instead applies TopMost's own documented
5-step preprocessing (FASTopic paper Appendix B: tokenize+lowercase,
remove punctuation, remove tokens containing numbers, remove tokens <3
chars, remove English/gensim stopwords -- reimplemented directly from
topmost/preprocess/preprocess.py's Tokenizer class, not approximated),
then filters documents with fewer than MIN_TERM=65 surviving tokens and
takes a seed=42 random sample of exactly 10,000 of the survivors. MIN_TERM
was calibrated (not guessed) against the achieved avg length matching the
paper's reported 110.0 (achieved: 110.36).

Because of this gap, `exact_fastopic_artifact` is False for this dataset
(see provenance.json written alongside the reconstructed files) -- every
result produced against this dataset must be reported as
"fastopic_wos_reconstructed", never silently presented as the paper's own
WoS number.

No official train/test split exists for this dataset (there is no
official artifact to ship one). This project generates its own
stratified 80/20 split at seed=42 for the classification experiment,
per its own established "don't invent a split when one exists, but you
must still evaluate this experiment when none does" convention -- unlike
20NG/NYT, this is NOT a case of overriding an available official split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from vaebm_benchmark.utils.paths import RAW_DIR
from vaebm_benchmark.utils.provenance import sha256_of, verify_manifest, write_manifest

EXPECTED_SHA256_RAW_ZIP = "b787d484bff88b0dcdb3fa291d06ec9d2f025dc2a67ce1045d0c688cd96ccf8a"  # WebOfScience.zip, per Mendeley's own content_details.sha256_hash
MENDELEY_DOI = "10.17632/9rw3vkcfy4.2"
MENDELEY_DOWNLOAD_URL = (
    "https://data.mendeley.com/public-files/datasets/9rw3vkcfy4/files/"
    "4ed4e75a-a656-4e55-9cb2-f5638b4de00a/file_downloaded"
)


@dataclass
class WoSReconstructedBundle:
    train_texts: list[str]
    test_texts: list[str]
    train_labels: list[int]
    test_labels: list[int]
    vocab: list[str]
    train_bow: np.ndarray
    test_bow: np.ndarray
    exact_fastopic_artifact: bool
    provenance: dict


class WoSReconstructedDataset:
    dataset_id = "fastopic_wos_reconstructed"

    def raw_dir(self):
        d = RAW_DIR / self.dataset_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _require_files(self) -> None:
        d = self.raw_dir()
        for name in ("texts.txt", "labels.txt", "vocab.txt", "provenance.json"):
            if not (d / name).exists():
                raise FileNotFoundError(
                    f"{d / name} missing - run scripts/build_fastopic_wos_reconstructed.py first "
                    f"(reconstructs this dataset from the raw Mendeley WOS-11967 source; "
                    f"see this module's own docstring for full provenance)."
                )

    def write_manifest_if_needed(self) -> None:
        manifest_path = self.raw_dir() / "MANIFEST.yaml"
        if manifest_path.exists():
            return
        self._require_files()
        write_manifest(
            self.raw_dir(),
            extra={
                "source": MENDELEY_DOWNLOAD_URL,
                "source_doi": MENDELEY_DOI,
                "exact_fastopic_artifact": False,
            },
        )

    def verify(self) -> tuple[bool, list[str]]:
        return verify_manifest(self.raw_dir())

    def load(self, test_p: float = 0.2, seed: int = 42) -> WoSReconstructedBundle:
        self._require_files()
        self.write_manifest_if_needed()
        d = self.raw_dir()

        texts = [line.strip() for line in (d / "texts.txt").read_text().splitlines()]
        labels = [int(x) for x in (d / "labels.txt").read_text().split()]
        vocab = [line.strip() for line in (d / "vocab.txt").read_text().splitlines() if line.strip()]
        provenance = json.loads((d / "provenance.json").read_text())
        assert len(texts) == len(labels), (len(texts), len(labels))

        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.model_selection import train_test_split

        idx = np.arange(len(texts))
        train_idx, test_idx = train_test_split(
            idx, test_size=test_p, random_state=seed, stratify=labels,
        )
        train_idx, test_idx = np.sort(train_idx), np.sort(test_idx)

        vectorizer = CountVectorizer(vocabulary=vocab, tokenizer=lambda x: x.split())
        train_bow = vectorizer.fit_transform([texts[i] for i in train_idx]).toarray().astype("float32")
        test_bow = vectorizer.transform([texts[i] for i in test_idx]).toarray().astype("float32")

        return WoSReconstructedBundle(
            train_texts=[texts[i] for i in train_idx],
            test_texts=[texts[i] for i in test_idx],
            train_labels=[labels[i] for i in train_idx],
            test_labels=[labels[i] for i in test_idx],
            vocab=vocab,
            train_bow=train_bow,
            test_bow=test_bow,
            exact_fastopic_artifact=provenance["exact_fastopic_artifact"],
            provenance=provenance,
        )
