"""Plain labeled-text dataset registry for the simplified experiment
runners (scripts/run_experiment.py, both `topic` and `cluster` modes) -
deliberately independent of the FASTopic/GloCOM protocol machinery in
protocols/*.py: these runners are direct, symmetric model-vs-model
comparisons on shared corpora, not per-paper faithful reproductions, so
they do not need per-protocol vocabulary/checksum/split pinning.

Each entry maps a dataset_id to a zero-arg loader returning
(documents, labels, num_classes). Adding a new dataset here is exactly
one registry entry pointing at an existing (or new) BenchmarkDataset
definition under datasets/definitions/ - no other file needs to change.
"""

from __future__ import annotations


def _short_text_loader(dataset_id: str):
    def _load():
        from vaebm_benchmark.datasets.definitions.short_text_benchmarks import DATASETS

        loaded = DATASETS[dataset_id]().load()
        return loaded.documents, loaded.labels, len(loaded.label_names)

    return _load


def _twenty_newsgroups():
    from vaebm_benchmark.datasets.definitions.twenty_newsgroups import TwentyNewsgroupsDataset

    loaded = TwentyNewsgroupsDataset().load()
    return loaded.documents, loaded.labels, len(loaded.label_names)


def _imdb():
    from vaebm_benchmark.datasets.definitions.hf_benchmarks import IMDBDataset

    loaded = IMDBDataset().load()
    return loaded.documents, loaded.labels, len(loaded.label_names)


def _hicot_loader(dataset_id: str):
    def _load():
        from vaebm_benchmark.datasets.definitions.hicot_datasets import DATASETS

        loaded = DATASETS[dataset_id]().load()
        return loaded.documents, loaded.labels, len(loaded.label_names)

    return _load


# Short-text-clustering family (STC2 + GoogleNews + Tweet), all from the
# same brunoguilherme1/TopicClusterDocument mirror via LabelTabTextDataset
# - see datasets/definitions/short_text_benchmarks.py.
SHORT_TEXT_DATASET_IDS = [
    "agnews_short",
    "search_snippets",
    "stack_overflow",
    "biomedical",
    "google_news_ts",
    "google_news_t",
    "google_news_s",
    "tweet",
]

# The official HiCOT (Tran et al., 2025) pre-processed artifacts - used
# verbatim, see datasets/definitions/hicot_datasets.py's own module
# docstring for exact provenance/train-test-combination rules per
# dataset. Kept as separate ids from the pre-existing 20ng/imdb/
# agnews_short/search_snippets/google_news_* above (different artifacts,
# different preprocessing/vocab - never silently merged with them).
HICOT_DATASET_IDS = [
    "hicot_20ng",
    "hicot_imdb",
    "hicot_agnews",
    "hicot_search_snippets",
    "hicot_google_news",
]

LOADERS = {dataset_id: _short_text_loader(dataset_id) for dataset_id in SHORT_TEXT_DATASET_IDS}
LOADERS["20ng"] = _twenty_newsgroups
LOADERS["imdb"] = _imdb
LOADERS.update({dataset_id: _hicot_loader(dataset_id) for dataset_id in HICOT_DATASET_IDS})

# Aliases: alternate spellings that map onto an already-registered id,
# rather than duplicating a dataset definition under two names.
ALIASES = {
    "agnews": "agnews_short",
    "ag_news": "agnews_short",
    "googlenews_ts": "google_news_ts",
    "googlenews_t": "google_news_t",
    "googlenews_s": "google_news_s",
}


def resolve_dataset_id(dataset_id: str) -> str:
    return ALIASES.get(dataset_id, dataset_id)


def load_dataset(dataset_id: str):
    """Returns (documents, labels, num_classes). Downloads on first call
    (cached under data/raw/ afterward, same as every other dataset in
    this project)."""
    resolved = resolve_dataset_id(dataset_id)
    if resolved not in LOADERS:
        raise KeyError(f"Unknown dataset '{dataset_id}'. Available: {sorted(LOADERS)} (aliases: {sorted(ALIASES)})")
    return LOADERS[resolved]()


def list_datasets() -> list[str]:
    return sorted(LOADERS)


def list_short_text_datasets() -> list[str]:
    """The `--datasets all-short` expansion for scripts/run_experiment.py."""
    return list(SHORT_TEXT_DATASET_IDS)
