"""Datasets sourced from the Hugging Face Hub, for the simplified VAE-BM
vs. BERTopic experiment runner (scripts/run_experiment.py). Mirrors DTEA's
own `HFDatasetSource` pattern (document-topic-evaluatio-arena/src/dtea/
datasets/definitions/hf_benchmarks.py): every `_revision` is a commit SHA
pinned at definition time, not a branch name, so re-running next year
reproduces byte-identical raw data even if the upstream repo's default
branch moves.
"""

from __future__ import annotations

from vaebm_benchmark.datasets.base import BenchmarkDataset, LoadedDataset
from vaebm_benchmark.datasets.download_utils import encode_labels


class HFDatasetSource(BenchmarkDataset):
    _repo_id: str = ""
    _revision: str = ""
    _splits: tuple[str, ...] = ("train", "test")
    _text_fields: tuple[str, ...] = ("text",)
    _label_field: str = "label"

    def _download_raw(self) -> None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                f"{self.dataset_id} requires the `datasets` package: pip install datasets"
            ) from exc

        if not self._revision:
            raise ValueError(f"{self.dataset_id}: _revision must be pinned to a commit SHA")

        out_path = self.raw_dir() / "data.tsv"
        with open(out_path, "w", encoding="utf-8") as f:
            for split in self._splits:
                hf_split = load_dataset(self._repo_id, revision=self._revision, split=split)
                label_names = getattr(hf_split.features[self._label_field], "names", None)
                for row in hf_split:
                    text = " ".join(str(row[field]).strip() for field in self._text_fields if row.get(field))
                    flat_text = " ".join(text.split())
                    if not flat_text:
                        continue
                    label_value = row[self._label_field]
                    label = label_names[label_value] if label_names else str(label_value)
                    f.write(f"{label}\t{flat_text}\n")

    def _load_raw(self) -> LoadedDataset:
        from vaebm_benchmark.datasets.download_utils import parse_label_tab_text

        texts, raw_labels = parse_label_tab_text(self.raw_dir() / "data.tsv")
        encoded, id_to_label = encode_labels(raw_labels)
        return LoadedDataset(self.dataset_id, texts, encoded, id_to_label)


class IMDBDataset(HFDatasetSource):
    """25k train + 25k test labeled reviews (pos/neg) - the additional
    50k "unsupervised" split has no label and is intentionally excluded
    (it cannot support Purity/NMI)."""

    dataset_id = "imdb"
    _repo_id = "stanfordnlp/imdb"
    _revision = "e6281661ce1c48d982bc483cf8a173c1bbeb5d31"
    _splits = ("train", "test")
    _text_fields = ("text",)
    _label_field = "label"


DATASETS = {
    "imdb": IMDBDataset,
}
