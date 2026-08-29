"""20 Newsgroups via scikit-learn's own built-in loader - the artifact
FASTopic's paper and most neural-topic-model papers use, headers/footers/
quotes stripped (the standard "raw" preprocessing convention in this
literature, e.g. topmost/OCTIS's own 20NG recipes)."""

from __future__ import annotations

from vaebm_benchmark.datasets.base import BenchmarkDataset, LoadedDataset
from vaebm_benchmark.datasets.download_utils import encode_labels


class TwentyNewsgroupsDataset(BenchmarkDataset):
    dataset_id = "20ng"

    def _download_raw(self) -> None:
        import pickle

        from sklearn.datasets import fetch_20newsgroups

        bunch = fetch_20newsgroups(
            subset="all", remove=("headers", "footers", "quotes"), data_home=str(self.raw_dir() / "sklearn_cache")
        )
        with open(self.raw_dir() / "bunch.pkl", "wb") as f:
            pickle.dump({"data": bunch.data, "target": bunch.target, "target_names": list(bunch.target_names)}, f)

    def _load_raw(self) -> LoadedDataset:
        import pickle

        with open(self.raw_dir() / "bunch.pkl", "rb") as f:
            raw = pickle.load(f)
        texts = [" ".join(doc.split()) for doc in raw["data"]]
        labels = list(raw["target"])
        id_to_label = dict(enumerate(raw["target_names"]))
        _ = encode_labels  # labels already integer-encoded by sklearn
        return LoadedDataset(self.dataset_id, texts, labels, id_to_label)
