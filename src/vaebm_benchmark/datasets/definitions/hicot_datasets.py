"""Official HiCOT (Tran et al., 2025, github.com/HoangTran223/HiCOT)
pre-processed dataset artifacts, downloaded and used VERBATIM - no
re-tokenizing, re-splitting, or resampling of the texts. Each dataset
downloads the exact `datasets/<hicot_folder>/{train,test}_{texts,labels}
.txt` files, plus `vocab.txt`/`word_embeddings.npz`/`{train,test}_bow
.npz` (kept on disk for provenance and for a future caller that wants
the paper's own BoW/vocabulary/embeddings - see load_hicot_vocab()/
load_hicot_word_embeddings()/load_hicot_bow() below - though this
registry's own `_load_raw()` only exposes texts+labels, matching every
other dataset's `LoadedDataset` contract).

Two things about these official files that matter for how `_load_raw()`
combines train+test into the single corpus this project's topic
experiment expects:

1. For 20NG/IMDB/AGNews, train and test are genuinely disjoint splits -
   verified directly: 11,314+7,532=18,846 / 25,000+25,000=50,000 /
   10,000+2,500=12,500 documents respectively, matching ECRTM (Wu et
   al., ICML 2023) Table 9's dataset-size column EXACTLY for all three
   (this is, in particular, the first artifact in this project matching
   the paper's own 12,500-doc AG News subsample - `agnews_short`/
   `agnews_full` elsewhere in this registry are both different cuts of
   the same underlying corpus, see docs/methodological_notes.md #10/#11).
   `_load_raw()` concatenates them (train first, then test, in file
   order, texts/labels unmodified) into one corpus: this project's topic
   experiment - like ECRTM's own Table 2/3 - evaluates topic quality
   transductively over the full corpus, not via a held-out split (the
   paper's train/test split is for its OWN downstream text-
   classification experiment, Sec 4.4, not topic-quality evaluation).
2. For SearchSnippets/GoogleNews, HiCOT's own `train_texts.txt` and
   `test_texts.txt` are BYTE-IDENTICAL (verified directly - same for the
   label files): there is no genuine held-out split for these two at
   all, only one corpus. Concatenating them would silently duplicate
   every document, corrupting corpus size and every topic-quality metric
   computed over it rather than "preserving the official split" -
   `_load_raw()` detects this (byte-for-byte equality of both the texts
   and the labels) and uses train alone instead.
"""

from __future__ import annotations

from vaebm_benchmark.datasets.base import BenchmarkDataset, LoadedDataset

_HICOT_BASE_URL = "https://raw.githubusercontent.com/HoangTran223/HiCOT/main/datasets"

# The exact 8 official artifacts this project downloads per dataset -
# see this module's own docstring for what each is used for.
_HICOT_FILES = (
    "train_texts.txt", "train_labels.txt", "train_bow.npz",
    "test_texts.txt", "test_labels.txt", "test_bow.npz",
    "vocab.txt", "word_embeddings.npz",
)


def _read_lines(path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _read_int_lines(path) -> list[int]:
    return [int(line) for line in path.read_text(encoding="utf-8").splitlines()]


class HiCOTDataset(BenchmarkDataset):
    """One dataset from HiCOT's official `datasets/<hicot_folder>/`
    artifacts. Subclasses set `dataset_id` and `hicot_folder` only -
    `hicot_folder` is the exact folder name under HiCOT's own repo (e.g.
    "20NG", "AGNews" - case-sensitive, matches the GitHub path)."""

    hicot_folder: str = ""

    def __init__(self) -> None:
        if not self.hicot_folder:
            raise ValueError(f"{type(self).__name__} must set hicot_folder")
        super().__init__()

    def _download_raw(self) -> None:
        from vaebm_benchmark.datasets.download_utils import save_url

        for filename in _HICOT_FILES:
            save_url(f"{_HICOT_BASE_URL}/{self.hicot_folder}/{filename}", self.raw_dir() / filename)

    def _load_raw(self) -> LoadedDataset:
        train_texts = _read_lines(self.raw_dir() / "train_texts.txt")
        test_texts = _read_lines(self.raw_dir() / "test_texts.txt")
        train_labels = _read_int_lines(self.raw_dir() / "train_labels.txt")
        test_labels = _read_int_lines(self.raw_dir() / "test_labels.txt")

        if len(train_texts) != len(train_labels):
            raise ValueError(
                f"{self.dataset_id}: train_texts.txt has {len(train_texts)} lines "
                f"but train_labels.txt has {len(train_labels)}"
            )
        if len(test_texts) != len(test_labels):
            raise ValueError(
                f"{self.dataset_id}: test_texts.txt has {len(test_texts)} lines "
                f"but test_labels.txt has {len(test_labels)}"
            )

        if train_texts == test_texts and train_labels == test_labels:
            # No genuine held-out split for this dataset (verified for
            # SearchSnippets/GoogleNews - see module docstring) -
            # concatenating would duplicate every document.
            documents, labels = train_texts, train_labels
        else:
            documents, labels = train_texts + test_texts, train_labels + test_labels

        num_classes = max(labels) + 1
        label_names = {i: str(i) for i in range(num_classes)}
        return LoadedDataset(self.dataset_id, documents, labels, label_names)


class HiCOT20NGDataset(HiCOTDataset):
    dataset_id = "hicot_20ng"
    hicot_folder = "20NG"


class HiCOTIMDBDataset(HiCOTDataset):
    dataset_id = "hicot_imdb"
    hicot_folder = "IMDB"


class HiCOTAGNewsDataset(HiCOTDataset):
    dataset_id = "hicot_agnews"
    hicot_folder = "AGNews"


class HiCOTSearchSnippetsDataset(HiCOTDataset):
    dataset_id = "hicot_search_snippets"
    hicot_folder = "SearchSnippets"


class HiCOTGoogleNewsDataset(HiCOTDataset):
    dataset_id = "hicot_google_news"
    hicot_folder = "GoogleNews"


DATASETS = {
    "hicot_20ng": HiCOT20NGDataset,
    "hicot_imdb": HiCOTIMDBDataset,
    "hicot_agnews": HiCOTAGNewsDataset,
    "hicot_search_snippets": HiCOTSearchSnippetsDataset,
    "hicot_google_news": HiCOTGoogleNewsDataset,
}


def load_hicot_vocab(dataset_id: str) -> list[str]:
    """The exact vocabulary (5,000 words for 20NG/IMDB/AGNews, matching
    ECRTM's own Table 9 vocab-size column; 4,618 for SearchSnippets;
    3,473 for GoogleNews) HiCOT's own `{train,test}_bow.npz`/
    `word_embeddings.npz` are indexed against - for a future caller that
    wants to pass this as VAEBMAdapter's own `vocabulary=` parameter (a
    protocol-fidelity hook that already exists, see
    models/vaebm.py::fit_predict) instead of letting VAE-BM fit its own
    vocabulary from these texts. NOT wired into experiment/runner.py
    automatically - forcing a fixed external vocabulary would change
    VAE-BM's own vocabulary-selection behavior for these datasets, a
    model-behavior decision left for a future, explicit change."""
    dataset = DATASETS[dataset_id]()
    dataset.download()
    return dataset.raw_dir().joinpath("vocab.txt").read_text(encoding="utf-8").splitlines()


def load_hicot_word_embeddings(dataset_id: str):
    """The exact word-embedding matrix HiCOT ships for this dataset (200-
    dim GloVe, matching ECRTM's own Appendix B spec; rows aligned to
    load_hicot_vocab()'s own word order) - scipy.sparse CSR (~99.5%
    dense; HiCOT saves it via scipy.sparse.save_npz regardless of
    density). Not wired into any model automatically - see
    load_hicot_vocab()'s own docstring."""
    import scipy.sparse as sp

    dataset = DATASETS[dataset_id]()
    dataset.download()
    return sp.load_npz(dataset.raw_dir() / "word_embeddings.npz")


def load_hicot_bow(dataset_id: str, split: str = "train"):
    """The exact BoW matrix (scipy.sparse CSR, over load_hicot_vocab()'s
    own vocabulary) HiCOT ships for `split` ("train" or "test") - not
    wired into any model automatically, see load_hicot_vocab()'s own
    docstring. Note: for SearchSnippets/GoogleNews, `train_bow.npz` and
    `test_bow.npz` are the SAME matrix twice (train_texts.txt ==
    test_texts.txt for those two - see this module's own docstring), not
    two different splits."""
    import scipy.sparse as sp

    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    dataset = DATASETS[dataset_id]()
    dataset.download()
    return sp.load_npz(dataset.raw_dir() / f"{split}_bow.npz")


# Datasets where HiCOT's own train_texts.txt/test_texts.txt (and labels)
# are byte-identical - see this module's own docstring. load_hicot_split()
# raises for these rather than silently handing back a "held-out" test
# set that is actually the training set verbatim (a data leak that would
# make classification accuracy meaningless, not merely optimistic).
NO_GENUINE_SPLIT_DATASET_IDS = {"hicot_search_snippets", "hicot_google_news"}


def load_hicot_split(dataset_id: str) -> tuple[list[str], list[int], list[str], list[int], int]:
    """The genuine, UN-concatenated official train/test split - for
    experiment/classification_runner.py, which (per ECRTM's own Sec 4.4
    text-classification protocol) needs a real held-out test set, unlike
    the `topic`/`cluster` experiments (load_dataset() above, which
    combines train+test transductively, matching how this project's
    other experiments and ECRTM's own Table 2/3 evaluate topic quality).

    Raises ValueError for `dataset_id` in NO_GENUINE_SPLIT_DATASET_IDS
    (SearchSnippets/GoogleNews) - HiCOT ships the same corpus under both
    filenames for these two, so there is no real held-out set to use
    here; silently returning train==test would report a data-leaked
    accuracy that isn't a real generalization measurement."""
    if dataset_id in NO_GENUINE_SPLIT_DATASET_IDS:
        raise ValueError(
            f"'{dataset_id}' has no genuine held-out test split - HiCOT ships the identical corpus "
            "under both train_texts.txt and test_texts.txt for this dataset (verified directly - see "
            "this module's own docstring). Using it for classification would silently data-leak "
            "(train == test). Choose a dataset with a real split instead: "
            f"{sorted(set(DATASETS) - NO_GENUINE_SPLIT_DATASET_IDS)}."
        )
    dataset = DATASETS[dataset_id]()
    dataset.download()
    train_texts = _read_lines(dataset.raw_dir() / "train_texts.txt")
    test_texts = _read_lines(dataset.raw_dir() / "test_texts.txt")
    train_labels = _read_int_lines(dataset.raw_dir() / "train_labels.txt")
    test_labels = _read_int_lines(dataset.raw_dir() / "test_labels.txt")
    num_classes = max(train_labels + test_labels) + 1
    return train_texts, train_labels, test_texts, test_labels, num_classes
