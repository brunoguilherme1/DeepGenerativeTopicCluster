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
    larger).
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
  - Preprocessing / vocabulary: the baseline is trained on the RELEASED
    `train_bow.npz`/`vocab.txt` directly (see
    models/fastopic_adapter.py::_ReleasedArtifactPreprocess) - NOT a
    re-tokenization via `topmost.Preprocess` that merely aims to be
    equivalent. This makes the `vocabulary` check below a provable MATCH:
    VAE-BM is pinned to this exact same vocab list AND (see
    models/vaebm.py's `tokenizer=str.split` fix) the exact same
    whitespace-tokenization rule, so both models' BoW inputs are built
    from literally the same counts.
  - Hyperparameters: fastopic's OWN constructor/fit_transform defaults -
    DT_alpha=3.0, TW_alpha=2.0, theta_temp=1.0, epochs=200,
    learning_rate=0.002 - verified to be EXACTLY what Appendix D states
    (epsilon_1=1/3, epsilon_2=1/2, tau=1.0, Adam/200/0.002, "same
    hyperparameters for all reported experiments"). `low_memory=False`
    (full-batch training, matching the paper's actual training regime -
    see fastopic_adapter.py's docstring for why this is NOT silently
    auto-enabled above some doc count the way a "helpful" wrapper might).
  - Metrics: CV coherence - `cv_palmetto_wikipedia` is the paper's actual
    metric (Palmetto + an unreleased Wikipedia reference corpus - see
    metrics/palmetto.py); this project also computes `cv_local_corpus`
    (gensim CoherenceModel against the NYT train corpus) as a clearly
    DIFFERENTLY-NAMED, non-paper-equivalent number - the two are NEVER
    merged into one comparison column (see docs/methodological_notes.md
    #5 and metrics/topic_quality.py). Topic Diversity (proportion-of-
    unique-words - the SAME formula as this project's generic
    `topic_diversity`, unlike GloCOM's distinct `glocom_td`), Purity and
    NMI (sklearn, argmax(theta) on the TEST split vs. test labels).
  - Seeds: fastopic's own code sets no random seed anywhere; the paper
    reports significance markers across tables but never states run
    count/seeds/test method. This protocol's `seeds` is this project's
    OWN choice for its own runs.
"""

from __future__ import annotations

from vaebm_benchmark.protocols.base import (
    BaselineProtocol,
    DatasetSpec,
    MatchStatus,
    MetricSpec,
    ProtocolCheck,
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
    upstream_commit = "51150f1ac22c4599ab0e390b14c031a98cffed68"  # fastopic==1.0.1 package commit

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
        MetricSpec(name="cv_palmetto_wikipedia", kind="topic", top_n=20),
        MetricSpec(name="cv_local_corpus", kind="topic", top_n=20),
        MetricSpec(name="topic_diversity", kind="topic", top_n=20),
        MetricSpec(name="purity", kind="clustering"),
        MetricSpec(name="nmi", kind="clustering"),
    ]
    seeds = [42]

    published_results = [
        PublishedResult(dataset_id="nyt", metric="cv_palmetto_wikipedia", value=0.437,
                         source="Table 1, FASTopic row, NYT K=50 (Wu et al., NeurIPS 2024) - "
                                "uses an unreleased Wikipedia reference corpus; see metrics/palmetto.py"),
        PublishedResult(dataset_id="nyt", metric="topic_diversity", value=0.999,
                         source="Table 1, FASTopic row, NYT K=50 (Wu et al., NeurIPS 2024)"),
        PublishedResult(dataset_id="nyt", metric="purity", value=0.662,
                         source="Table 2, FASTopic row, NYT K=50 (Wu et al., NeurIPS 2024)"),
        PublishedResult(dataset_id="nyt", metric="nmi", value=0.369,
                         source="Table 2, FASTopic row, NYT K=50 (Wu et al., NeurIPS 2024)"),
        # cv_local_corpus has no published counterpart - it is not the paper's metric (see module docstring).
    ]

    def __init__(self, smoke_test: bool = True) -> None:
        self.smoke_test = smoke_test
        self.mode = "smoke" if smoke_test else "full"
        # Paper default is 200 epochs, full-batch on an A6000 GPU.
        # Reduced for a CPU smoke test per this project's own instructions
        # ("test only small experiments") - documented, not silent;
        # verify() surfaces this as a DIFFERENCE.
        self.epochs = 20 if smoke_test else 200

    def _require_nyt(self, dataset_id: str) -> None:
        if dataset_id != "nyt":
            raise KeyError(f"FASTopicProtocol only supports 'nyt' currently, got '{dataset_id}'")

    def prepare_dataset(self, dataset_id: str) -> list[str]:
        self._require_nyt(dataset_id)
        from vaebm_benchmark.datasets.definitions.fastopic_nyt import NYTDataset

        return NYTDataset().load().train_texts

    def prepare_eval_documents(self, dataset_id: str) -> list[str]:
        self._require_nyt(dataset_id)
        from vaebm_benchmark.datasets.definitions.fastopic_nyt import NYTDataset

        return NYTDataset().load().test_texts

    def prepare_labels(self, dataset_id: str):
        self._require_nyt(dataset_id)
        from vaebm_benchmark.datasets.definitions.fastopic_nyt import NYTDataset

        return NYTDataset().load().test_labels

    def artifact_checksum(self, dataset_id: str) -> str:
        self._require_nyt(dataset_id)
        from vaebm_benchmark.datasets.definitions.fastopic_nyt import EXPECTED_SHA256

        return "|".join(f"{name}:{digest}" for name, digest in sorted(EXPECTED_SHA256.items()))

    def preprocessing_version(self, dataset_id: str) -> str:
        self._require_nyt(dataset_id)
        return "released_artifact_train_bow_v1"  # bypasses topmost.Preprocess entirely - see build_baseline()

    def vocabulary_for(self, dataset_id: str) -> list[str]:
        self._require_nyt(dataset_id)
        from vaebm_benchmark.datasets.definitions.fastopic_nyt import NYTDataset

        return NYTDataset().load().vocab

    def build_baseline(self, dataset_id: str, seed: int):
        import scipy.sparse as sp

        from vaebm_benchmark.datasets.definitions.fastopic_nyt import NYTDataset
        from vaebm_benchmark.models.fastopic_adapter import FASTopicAdapter

        bundle = NYTDataset().load()
        return FASTopicAdapter(
            num_topics=self.topic_count[dataset_id],
            num_top_words=20,  # matches this protocol's metric top_n (see metric_specs)
            doc_embed_model="all-MiniLM-L6-v2",
            epochs=self.epochs,
            learning_rate=0.002,
            DT_alpha=3.0,
            TW_alpha=2.0,
            theta_temp=1.0,
            low_memory=False,
            # Reuse the released train_bow.npz/vocab.txt directly instead
            # of re-tokenizing train_texts via topmost.Preprocess - see
            # models/fastopic_adapter.py::_ReleasedArtifactPreprocess and
            # this module's own docstring.
            released_train_bow=sp.csr_matrix(bundle.train_bow),
            released_vocab=bundle.vocab,
        )

    def vaebm_variants(self) -> list[str]:
        return ["protocol_faithful", "stability_adjusted"]

    def build_vaebm(self, dataset_id: str, seed: int, variant: str = "stability_adjusted"):
        from vaebm_benchmark.datasets.definitions.fastopic_nyt import NYTDataset
        from vaebm_benchmark.models.vaebm_adapter import VAEBMAdapter

        if variant not in self.vaebm_variants():
            raise ValueError(f"Unknown VAE-BM variant '{variant}'; available: {self.vaebm_variants()}")

        # protocol_faithful: the AS-SUPPLIED notebook default (lr=1e-2).
        # Confirmed by direct diagnostic (docs/methodological_notes.md #8)
        # to diverge to inf/NaN within a couple of epochs at this vocab
        # scale (10,000 words) - this is EXPECTED and recorded as such,
        # never hidden. stability_adjusted: lr=1e-3, which trains stably;
        # this is a documented hyperparameter substitution, not a change
        # to VAE-BM's architecture/loss, and is never presented as if it
        # were the original formulation - see evaluation/runner.py's
        # `variant` field in every persisted run.
        lr = 1e-2 if variant == "protocol_faithful" else 1e-3

        bundle = NYTDataset().load()
        return VAEBMAdapter(
            n_clusters=self.topic_count[dataset_id],
            voc_size=len(bundle.vocab),
            units=50,
            epochs=self.epochs,
            batch_size=128,
            lr=lr,
            random_state=seed,
            vectorizer_type="tfidf",
            embedder="all-MiniLM-L6-v2",  # matches FASTopic's own doc_embed_model
            dim=(1500, 1000, 500),
            dim_emb=(368,),
            alpha=0.99,
            top_words_mode="energy",
            vocabulary=bundle.vocab,  # exact same vocab.txt as the FASTopic baseline, exact-count tokenizer (see vaebm.py)
        )

    def checks(self) -> list[ProtocolCheck]:
        from vaebm_benchmark.datasets.definitions.fastopic_nyt import NYTDataset
        from vaebm_benchmark.metrics.palmetto import palmetto_available

        checksum_ok, checksum_problems = NYTDataset().verify()
        checksum_status = MatchStatus.MATCH if checksum_ok else MatchStatus.DIFFERENCE
        checksum_note = (
            "every file's SHA256 matches EXPECTED_SHA256, pinned by this project's own first download "
            "(no independent third-party hash exists for NYT.zip specifically - see fastopic_nyt.py)"
            if checksum_ok
            else "; ".join(checksum_problems) or "not yet downloaded - run prepare_dataset() first"
        )
        palmetto_status = MatchStatus.MATCH if palmetto_available() else MatchStatus.UNKNOWN
        palmetto_note = (
            "tools/palmetto/palmetto.jar + wiki_data/wikipedia_bd found - real Palmetto C_V will be computed"
            if palmetto_available()
            else "tools/palmetto/{palmetto.jar,wiki_data/wikipedia_bd} not present - cv_palmetto_wikipedia will be "
                 "recorded as unavailable, never silently replaced by cv_local_corpus"
        )

        return [
            ProtocolCheck("dataset", self.datasets[0].source_url, MatchStatus.MATCH,
                          "fetched via topmost's own download_dataset('NYT'), the same mirror the FASTopic ecosystem uses"),
            ProtocolCheck("checksum", "per-file SHA256 vs. EXPECTED_SHA256 (fastopic_nyt.py)", checksum_status,
                          checksum_note),
            ProtocolCheck("upstream_commit", self.upstream_commit, MatchStatus.MATCH,
                          "pinned fastopic==1.0.1 package commit, verified against PyPI/GitHub"),
            ProtocolCheck("preprocessing", "released train_bow.npz/vocab.txt injected directly (no re-tokenization)",
                          MatchStatus.MATCH,
                          "bypasses topmost.Preprocess entirely via _ReleasedArtifactPreprocess - byte-identical to the official artifact by construction"),
            ProtocolCheck("vocabulary", f"{len(self.vocabulary_for('nyt'))}-word vocab.txt from the official NYT.zip artifact",
                          MatchStatus.MATCH,
                          "VAE-BM's vectorizer is pinned to this exact vocab list AND the same whitespace-split "
                          "tokenization (models/vaebm.py) - counts are provably identical, not assumed equivalent"),
            ProtocolCheck("K", f"{self.topic_count['nyt']}", MatchStatus.MATCH, "paper's Table 1/2 primary results use K=50"),
            ProtocolCheck("split_strategy", self.split_strategy.strategy, MatchStatus.MATCH,
                          "pre-split train/test files, as shipped; confirmed against topmost's own reference "
                          "FASTopicTrainer (fits train_texts, evaluates held-out test_texts for clustering)"),
            ProtocolCheck("seeds", f"{self.seeds}", MatchStatus.UNKNOWN,
                          "official repo sets no seed; paper reports significance markers with undisclosed run count/seed/test method"),
            ProtocolCheck("metric:cv_palmetto_wikipedia", "metrics/palmetto.py::palmetto_cv()", palmetto_status, palmetto_note),
            ProtocolCheck("metric:cv_local_corpus", "gensim CoherenceModel c_v vs. training corpus", MatchStatus.DIFFERENCE,
                          "NOT the paper's metric - a separately-named, documented approximation (see module docstring); "
                          "never compared against the paper's published cv_palmetto_wikipedia number"),
            ProtocolCheck("metric:topic_diversity", "topic_quality.topic_diversity()", MatchStatus.MATCH,
                          "same proportion-of-unique-words formula as topmost/eva/topic_diversity.py"),
            ProtocolCheck("metric:purity", "clustering_quality.purity()", MatchStatus.MATCH,
                          "same as topmost/eva/clustering.py::purity_score"),
            ProtocolCheck("metric:nmi", "clustering_quality.nmi()", MatchStatus.MATCH,
                          "same as sklearn.metrics.normalized_mutual_info_score, same as official repo"),
            ProtocolCheck("baseline_implementation", "official `fastopic` PyPI package, default hyperparameters",
                          MatchStatus.MATCH, "verified against paper Appendix D (DT_alpha/TW_alpha/theta_temp/epochs/lr)"),
            ProtocolCheck("vaebm_implementation", "models/vaebm.py, unmodified architecture/initializers/callbacks",
                          MatchStatus.MATCH,
                          "protocol_faithful variant uses the AS-SUPPLIED lr=1e-2; stability_adjusted uses lr=1e-3 - "
                          "both persisted and labeled, never conflated (see docs/methodological_notes.md #8/#9)"),
            ProtocolCheck("mode", self.mode, MatchStatus.MATCH if not self.smoke_test else MatchStatus.DIFFERENCE,
                          "smoke mode uses 20 epochs, not the paper's 200 - never described as paper reproduction"
                          if self.smoke_test else "matches paper default (200 epochs)"),
        ]
