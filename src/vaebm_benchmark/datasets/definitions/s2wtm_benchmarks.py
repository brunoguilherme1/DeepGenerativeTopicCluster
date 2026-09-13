"""Datasets sourced from the AdhyaSuman/S2WTM preprocessed_datasets
mirror (corpus.txt + labels.txt two-file layout, one document/label per
line). Independently implemented against the same public mirror DTEA
uses (document-topic-evaluatio-arena/src/dtea/datasets/definitions/
s2wtm_benchmarks.py) - this project never imports DTEA's code (see
models/base.py's own docstring on why the two projects stay independent).

Caveat carried over from DTEA's own provenance records: this mirror
applies its own vocabulary filtering before release, which is not
independently verifiable against each dataset's original source corpus.
"""

from __future__ import annotations

from vaebm_benchmark.datasets.base import BenchmarkDataset, LoadedDataset
from vaebm_benchmark.datasets.download_utils import encode_labels, parse_two_file, save_url

S2WTM_BASE_URL = "https://raw.githubusercontent.com/AdhyaSuman/S2WTM/main/preprocessed_datasets"


class S2WTMTwoFileDataset(BenchmarkDataset):
    _corpus_path: str = ""
    _labels_path: str = ""

    def _download_raw(self) -> None:
        save_url(f"{S2WTM_BASE_URL}/{self._corpus_path}", self.raw_dir() / "corpus.txt")
        save_url(f"{S2WTM_BASE_URL}/{self._labels_path}", self.raw_dir() / "labels.txt")

    def _load_raw(self) -> LoadedDataset:
        texts, raw_labels = parse_two_file(self.raw_dir() / "corpus.txt", self.raw_dir() / "labels.txt")
        encoded, id_to_label = encode_labels(raw_labels)
        return LoadedDataset(self.dataset_id, texts, encoded, id_to_label)


class BBCNewsDataset(S2WTMTwoFileDataset):
    dataset_id = "bbc_news"
    _corpus_path = "BBC_News/corpus.txt"
    _labels_path = "BBC_News/labels.txt"


class DBLPDataset(S2WTMTwoFileDataset):
    dataset_id = "dblp"
    _corpus_path = "DBLP/corpus.txt"
    _labels_path = "DBLP/labels.txt"


class M10Dataset(S2WTMTwoFileDataset):
    dataset_id = "m10"
    _corpus_path = "M10/corpus.txt"
    _labels_path = "M10/labels.txt"


class PascalFlickrDataset(S2WTMTwoFileDataset):
    dataset_id = "pascal_flickr"
    _corpus_path = "Pascal_Flickr/raw/Pascal_Flickr.txt"
    _labels_path = "Pascal_Flickr/raw/Pascal_Flickr_LABEL.txt"


class TwentyNewsgroupsS2WTMDataset(S2WTMTwoFileDataset):
    """The S2WTM-preprocessed 20 Newsgroups variant (~16,309 docs) -
    distinct from `20ng` (the raw scikit-learn variant, ~18,846 docs,
    see datasets/definitions/twenty_newsgroups.py) - different
    preprocessing, never merged with it."""

    dataset_id = "20ng_s2wtm"
    _corpus_path = "20NewsGroup/corpus.txt"
    _labels_path = "20NewsGroup/labels.txt"


DATASETS = {
    "bbc_news": BBCNewsDataset,
    "dblp": DBLPDataset,
    "m10": M10Dataset,
    "pascal_flickr": PascalFlickrDataset,
    "20ng_s2wtm": TwentyNewsgroupsS2WTMDataset,
}
