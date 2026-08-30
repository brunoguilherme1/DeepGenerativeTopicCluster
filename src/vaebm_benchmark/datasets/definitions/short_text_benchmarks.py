"""Short-text-clustering benchmark corpora (STC2 family: SearchSnippets,
Biomedical, StackOverflow), from the brunoguilherme1/TopicClusterDocument
mirror (label\\tTAB\\ttext layout, one document per line) - the same
canonical artifact GloCOM's own paper (Nguyen et al., NAACL 2025)
evaluates on for these three corpora (see configs/datasets/*.yaml for
citation/provenance detail and the official-vs-mirror checksum note).
"""

from __future__ import annotations

from vaebm_benchmark.datasets.base import BenchmarkDataset, LoadedDataset
from vaebm_benchmark.datasets.download_utils import encode_labels, parse_label_tab_text, save_url

TCD_BASE_URL = "https://raw.githubusercontent.com/brunoguilherme1/TopicClusterDocument/main"


class LabelTabTextDataset(BenchmarkDataset):
    _source_path: str = ""

    def _download_raw(self) -> None:
        save_url(f"{TCD_BASE_URL}/{self._source_path}", self.raw_dir() / "data.tsv")

    def _load_raw(self) -> LoadedDataset:
        texts, raw_labels = parse_label_tab_text(self.raw_dir() / "data.tsv")
        encoded, id_to_label = encode_labels(raw_labels)
        return LoadedDataset(self.dataset_id, texts, encoded, id_to_label)


class StackOverflowDataset(LabelTabTextDataset):
    dataset_id = "stack_overflow"
    _source_path = "stack_text.txt"


class BiomedicalDataset(LabelTabTextDataset):
    dataset_id = "biomedical"
    _source_path = "bio_text.txt"


class SearchSnippetsDataset(LabelTabTextDataset):
    dataset_id = "search_snippets"
    _source_path = "search_text.txt"


class AGNewsShortDataset(LabelTabTextDataset):
    """8,000-doc short-text-clustering subsample (4 classes:
    World/Sports/Business/Sci-Tech) - distinct from the full 127,600-doc
    AG News classification benchmark (`agnews_full`, HuggingFace
    fancyzhx/ag_news), same underlying corpus, different standard cut."""

    dataset_id = "agnews_short"
    _source_path = "agnewsdataraw-8000"


class GoogleNewsTitleSnippetDataset(LabelTabTextDataset):
    """Title+Snippet view (152 classes) - the default "GoogleNews" view."""

    dataset_id = "google_news_ts"
    _source_path = "TS"


class GoogleNewsTitleDataset(LabelTabTextDataset):
    """Title-only view of the same 152-class GoogleNews corpus."""

    dataset_id = "google_news_t"
    _source_path = "T"


class GoogleNewsSnippetDataset(LabelTabTextDataset):
    """Snippet-only view of the same 152-class GoogleNews corpus."""

    dataset_id = "google_news_s"
    _source_path = "S"


class TweetDataset(LabelTabTextDataset):
    dataset_id = "tweet"
    _source_path = "tweet-original-order.txt"


DATASETS = {
    "stack_overflow": StackOverflowDataset,
    "biomedical": BiomedicalDataset,
    "search_snippets": SearchSnippetsDataset,
    "agnews_short": AGNewsShortDataset,
    "google_news_ts": GoogleNewsTitleSnippetDataset,
    "google_news_t": GoogleNewsTitleDataset,
    "google_news_s": GoogleNewsSnippetDataset,
    "tweet": TweetDataset,
}
