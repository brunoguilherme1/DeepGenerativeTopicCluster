"""FASTopic protocol: reproduces github.com/bobxwu/FASTopic's own
evaluation on **NYT**, verified directly against the cloned official repo,
its companion toolkit `topmost` (github.com/BobXWu/TopMost, where FASTopic's
own preprocessing/dataset-loading code actually lives), and the NeurIPS
2024 paper PDF (arXiv:2405.17978).

Protocol facts (all verified against source, not inferred):
  - Dataset artifact: `topmost.download_dataset('NYT', ...)` fetches
    `raw.githubusercontent.com/BobXWu/TopMost/master/data/NYT.zip` - the
    exact mirror FASTopic's own paper protocol uses. 9,172 docs total
    (8,254 train / 918 test), avg length 175.4, vocab 10,000, 12 classes
    (paper Table 7) - the smallest of the paper's six datasets that is
    both downloadable and labeled (WoS is undownloadable - see
    docs/methodological_notes.md; 20NG/ACL/NeurIPS/Wikitext-103 are all
    larger or, for 20NG, unlabeled-for-clustering-relevant-K... no, 20NG
    IS labeled but is nearly twice NYT's size and this project's own
    instructions ask for "one small official dataset").
  - Split: topmost.BasicDataset ships PRE-SPLIT train/test files - this
    project does not generate its own split. FASTopic's own training
    (`fit_transform`) only ever sees `train_texts`; `test_texts` is used
    solely at inference time (`model.transform`) to get held-out theta
    for the paper's clustering metrics. Topic-quality metrics (CV, TD)
    describe the topics discovered FROM TRAINING - there is no train/test
    distinction for those (see protocols/base.py's `evaluate()`
    docstring for how this project's own evaluate() reflects that split).
  - K: paper's Table 1/2 primary results use K=50 (a sensitivity sweep
    over K uses WoS only, which this protocol doesn't touch).
  - Hyperparameters: fastopic's OWN constructor/fit_transform defaults -
    DT_alpha=3.0, TW_alpha=2.0, theta_temp=1.0, epochs=200,
    learning_rate=0.002 - verified to be EXACTLY what Appendix D states
    (epsilon_1=1/3, epsilon_2=1/2, tau=1.0, Adam/200/0.002, "same
    hyperparameters for all reported experiments"). `vocab_size=10000,
    stopwords="English"` explicitly passed to `topmost.Preprocess` to
    match Table 7's vocab size (the package's own bare default is
    uncapped). `low_memory=False` (full-batch training, matching the
    paper's actual training regime - see fastopic_adapter.py's
    docstring for why this is NOT silently auto-enabled above some doc
    count the way a "helpful" wrapper might).
  - Metrics: CV coherence (gensim CoherenceModel, c_v, **topn=20** - note
    this differs from GloCOM's protocol, which uses topn=15; each
    protocol keeps its own paper's own convention, never a shared
    default), Topic Diversity (proportion-of-unique-words - the SAME
    formula as this project's generic `topic_diversity`, unlike GloCOM's
    distinct `glocom_td`), Purity and NMI (sklearn, argmax(theta) on the
    TEST split vs. test labels).
  - Seeds: fastopic's own code sets no random seed anywhere; the paper
    reports significance markers across tables but never states run
    count/seeds/test method. This protocol's `seeds` is this project's
    OWN choice for its own runs.

KNOWN DEVIATION (documented, not silently absorbed - see
docs/methodological_notes.md #5): CV coherence is NOT computed against
"a widely-used large Wikipedia article collection" (the paper's own
words, §4.1) - neither the FASTopic nor topmost repos ship or name that
corpus. This protocol instead uses the NYT training corpus itself as the
reference corpus for gensim's CoherenceModel, the same convention
topmost's own shipped demos/tests use as an illustration of the metric
API (see topmost/eva/topic_coherence.py). Treat the `cv` published-vs-
reproduced comparison as indicative only, not apples-to-apples, until an
equivalent Wikipedia reference corpus is sourced and wired in.
"""

from __future__ import annotations

from vaebm_benchmark.protocols.base import (
    BaselineProtocol,
    DatasetSpec,
    MetricSpec,
    PublishedResult,
    SplitSpec,
)


class FASTopicProtocol(BaselineProtocol):
    name = "fastopic"
    paper = (
        "Wu, X., Nguyen, T., Zhang, D. C., Wang, W. Y., & Luu, A. T. (2024). "
        "\"FASTopic: Pretrained Transformer is a Fast, Adaptive, Stable, and Transferable Topic Model.\" "
        "NeurIPS 2024. arXiv:2405.17978"
    )
    official_repository = "https://github.com/bobxwu/FASTopic"

    datasets = [
        DatasetSpec(
            id="nyt",
            name="NYT (New York Times), via topmost's official mirror",
            source_url="https://raw.githubusercontent.com/BobXWu/TopMost/master/data/NYT.zip",
            source_repository="https://github.com/BobXWu/TopMost",
            num_docs_expected=9172,
            num_classes=12,
            notes="Pre-split train (8,254) / test (918) files, vocab_size=10000, shipped by topmost's own download_dataset('NYT').",
        ),
    ]
    split_strategy = SplitSpec(
        strategy="predefined_train_test",
        description="Ships pre-split train/test files; FASTopic trains on train_texts only, "
        "infers held-out theta on test_texts via model.transform() for clustering metrics.",
    )
    topic_count = {"nyt": 50}
    metric_specs = [
        MetricSpec(name="cv", kind="topic", top_n=20),
        MetricSpec(name="topic_diversity", kind="topic", top_n=20),
        MetricSpec(name="purity", kind="clustering"),
        MetricSpec(name="nmi", kind="clustering"),
    ]
    seeds = [42]

    published_results = [
        PublishedResult(dataset_id="nyt", metric="cv", value=0.437,
                         source="Table 1, FASTopic row, NYT K=50 (Wu et al., NeurIPS 2024) - "
                                "PUBLISHED VALUE USES AN UNRELEASED WIKIPEDIA REFERENCE CORPUS; see module docstring"),
        PublishedResult(dataset_id="nyt", metric="topic_diversity", value=0.999,
                         source="Table 1, FASTopic row, NYT K=50 (Wu et al., NeurIPS 2024)"),
        PublishedResult(dataset_id="nyt", metric="purity", value=0.662,
                         source="Table 2, FASTopic row, NYT K=50 (Wu et al., NeurIPS 2024)"),
        PublishedResult(dataset_id="nyt", metric="nmi", value=0.369,
                         source="Table 2, FASTopic row, NYT K=50 (Wu et al., NeurIPS 2024)"),
    ]

    def __init__(self, smoke_test: bool = True) -> None:
        self.smoke_test = smoke_test
        # Paper default is 200 epochs, full-batch on an A6000 GPU.
        # Reduced for a CPU smoke test per this project's own instructions
        # ("test only small experiments") - documented, not silent;
        # verify() surfaces this as a DIFFERENCE.
        self.epochs = 20 if smoke_test else 200

    def prepare_dataset(self, dataset_id: str) -> list[str]:
        if dataset_id != "nyt":
            raise KeyError(f"FASTopicProtocol only supports 'nyt' currently, got '{dataset_id}'")
        from vaebm_benchmark.datasets.definitions.fastopic_nyt import NYTDataset

        return NYTDataset().load().train_texts

    def prepare_eval_documents(self, dataset_id: str) -> list[str]:
        if dataset_id != "nyt":
            raise KeyError(f"FASTopicProtocol only supports 'nyt' currently, got '{dataset_id}'")
        from vaebm_benchmark.datasets.definitions.fastopic_nyt import NYTDataset

        return NYTDataset().load().test_texts

    def prepare_labels(self, dataset_id: str):
        if dataset_id != "nyt":
            raise KeyError(f"FASTopicProtocol only supports 'nyt' currently, got '{dataset_id}'")
        from vaebm_benchmark.datasets.definitions.fastopic_nyt import NYTDataset

        return NYTDataset().load().test_labels

    def build_baseline(self, dataset_id: str, seed: int):
        from vaebm_benchmark.models.fastopic_adapter import FASTopicAdapter

        return FASTopicAdapter(
            num_topics=self.topic_count[dataset_id],
            num_top_words=20,  # matches this protocol's metric top_n (see metric_specs)
            doc_embed_model="all-MiniLM-L6-v2",
            epochs=self.epochs,
            learning_rate=0.002,
            vocab_size_cap=10_000,
            stopwords="English",
            DT_alpha=3.0,
            TW_alpha=2.0,
            theta_temp=1.0,
            low_memory=False,
        )

    def build_vaebm(self, dataset_id: str, seed: int):
        from vaebm_benchmark.datasets.definitions.fastopic_nyt import NYTDataset
        from vaebm_benchmark.models.vaebm_adapter import VAEBMAdapter

        bundle = NYTDataset().load()
        return VAEBMAdapter(
            n_clusters=self.topic_count[dataset_id],
            voc_size=len(bundle.vocab),
            units=50,
            epochs=self.epochs,
            batch_size=128,
            lr=1e-2,
            random_state=seed,
            vectorizer_type="tfidf",
            embedder="all-MiniLM-L6-v2",  # matches FASTopic's own doc_embed_model
            dim=(1500, 1000, 500),
            dim_emb=(368,),
            alpha=0.99,
            top_words_mode="energy",
            vocabulary=bundle.vocab,  # exact same vocab.txt as the FASTopic baseline
        )

    def verify(self) -> dict:
        return {
            "paper": self.paper,
            "official_repository": self.official_repository,
            "dataset": {
                "id": "nyt",
                "artifact": self.datasets[0].source_url,
                "num_docs_expected": self.datasets[0].num_docs_expected,
                "verdict": "MATCH - fetched via topmost's own download_dataset('NYT'), the same mirror the FASTopic ecosystem uses",
            },
            "preprocessing": {
                "description": "topmost.Preprocess(vocab_size=10000, stopwords='English') - tokenize/lowercase, "
                "strip punctuation/numbers, drop tokens <3 chars, remove gensim's English stopword list",
                "verdict": "MATCH",
            },
            "vocabulary": {"source": "vocab.txt from the official NYT.zip artifact, size 10000", "verdict": "MATCH"},
            "K": {"value": self.topic_count["nyt"], "paper_reports": "K=50 (primary results)", "verdict": "MATCH"},
            "split_strategy": {"value": self.split_strategy.strategy, "verdict": "MATCH - pre-split train/test files, as shipped"},
            "seeds": {"value": self.seeds, "verdict": "UNKNOWN - official repo sets no seed; paper reports significance markers with undisclosed run count/seed/test method"},
            "metrics": {
                "cv": "DIFFERENCE - paper uses an unreleased 'large Wikipedia article collection' as reference corpus; "
                      "this project uses the NYT train corpus itself (same fallback topmost's own demos use)",
                "topic_diversity": "MATCH - same proportion-of-unique-words formula as topmost/eva/topic_diversity.py",
                "purity": "MATCH - same as topmost/eva/clustering.py::purity_score",
                "nmi": "MATCH - same as sklearn.metrics.normalized_mutual_info_score, same as official repo",
            },
            "baseline_implementation": "MATCH - official `fastopic` PyPI package, default hyperparameters (verified against paper Appendix D)",
            "vaebm_implementation": "N/A - VAE-BM is not a baseline being reproduced; see docs/methodological_notes.md",
            "epochs": {
                "value": self.epochs,
                "paper_value": 200,
                "verdict": "MATCH" if not self.smoke_test else "DIFFERENCE - reduced for smoke test, see BaselineProtocol.__init__",
            },
        }
