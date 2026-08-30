"""GloCOM protocol: reproduces github.com/qducnguyen/GloCOM's own
evaluation exactly as `run.py` performs it for SearchSnippets - the ONLY
dataset the official repo ships precomputed artifacts for (confirmed by
listing its full repo tree; GoogleNews/StackOverflow/Biomedical require
external STTM+TopMost preprocessing this project has not reproduced - see
docs/methodological_notes.md and configs/datasets/*.yaml for those three).

This project does NOT clone or subprocess-execute the official repo (a
deliberate choice - see docs/methodological_notes.md #10): the GloCOM
model (`_glocom_source.py`) is vendored verbatim, trained via this
project's own data pipeline
(`models/glocom_adapter.py::fit_precomputed()`), built directly from the
official repo's own precomputed artifacts (bow.npz/global_bow.npz/
global_maps.txt/vocab.txt/word_embeddings.npz - no re-derivation). Every
hyperparameter below is verified against the official repo's own
`run.py`/`GloCOM.py` source, not assumed.

Protocol facts, each verified directly against the cloned official repo
(read directly during this project's own research pass, not executed)
and the NAACL 2025 paper PDF:
  - Dataset artifact: the exact bow.npz / global_bow.npz / global_maps.txt
    / vocab.txt / word_embeddings.npz files shipped in
    data/SearchSnippets/, fetched verbatim (see
    datasets/definitions/glocom_official.py) - no re-tokenization, no
    re-derivation of the global clustering context.
  - Split: full-corpus, transductive. The official repo's own
    dataloader.py reloads the training bow.npz as "test" too
    (`self.test_bow = scipy.sparse.load_npz(f'{path}/bow.npz')`) and
    `run.py` itself comments "train_theta == test_theta for the short
    text problem." This protocol reproduces exactly that.
  - K: paper's Table 2/11 report K=50 and K=100 for SearchSnippets; K=50
    is used here (the paper's primary reported setting).
  - num_global_clusters: 40 (Table 4), the paper's own choice for
    SearchSnippets specifically (GoogleNews instead uses 200 - not
    relevant here since GoogleNews isn't part of this protocol).
  - Hyperparameters: run.py's own CLI DEFAULTS (verified by reading
    run.py::parse_args directly) - aug_coef=0.5, prior_var=0.1 (NOT the
    GloCOM class's own default of 0.01 - run.py always overrides it),
    weight_loss_ECR=60.0 (NOT the class default of 30.0, same reason),
    en_units=200, embed_size=200, sinkhorn_alpha=20.0,
    sinkhorn_max_iter=100, beta_temp=0.2 (fixed, not CLI-exposed),
    epochs=200, lr=0.002, batch_size=200, num_top_words=15.
  - Training loop equivalence: this project trains the vendored GloCOM
    module via `topmost.BasicTrainer`, not the official
    `trainer/Trainer.py` class. VERIFIED by fetching
    `trainer/Trainer.py` directly from the pinned commit and diffing its
    `train()`/`test()`/`make_optimizer()` against `topmost.BasicTrainer`'s
    (installed, inspected via `inspect.getsource`): both construct
    `torch.optim.Adam(model.parameters(), lr=learning_rate)` identically,
    both loops are `for epoch: for batch_data in train_dataloader:
    rst_dict = model(batch_data); loss = rst_dict['loss'];
    optimizer.zero_grad(); loss.backward(); optimizer.step()` verbatim,
    and both `test()` methods batch `model.get_theta(...)` identically.
    The only difference is logging cosmetics (`print()` vs. a logger) -
    nothing that affects trained weights, theta, or beta. Recorded as
    MATCH below on this verified basis, not assumed - see
    docs/methodological_notes.md #10 for the full comparison.
  - Metrics: the paper reports TC (Topic Coherence, C_V via the Palmetto
    Java library against a bundled Wikipedia reference corpus -
    `cv_palmetto_wikipedia`, computed for real when
    `tools/palmetto/{palmetto.jar,wiki_data/wikipedia_bd}` are present,
    else recorded as UNAVAILABLE and never silently replaced), TD (this
    project's `glocom_td` - see metrics/topic_quality.py, NOT the same
    formula as standard `topic_diversity` - this distinction is preserved
    exactly and unit-tested, see tests/test_metrics.py), Purity, and NMI
    (both via argmax(theta) vs. ground-truth labels).
  - Seeds: the official codebase sets NO random seed anywhere (`grep -rni
    seed` across its own .py files returns nothing) - the paper instead
    reports mean+-std over 3 runs. This protocol's own `seeds` list is
    this project's OWN choice (for reproducibility of THIS project's
    runs), not something recovered from the paper.

Separately-named, non-paper coherence: `cv_local_corpus` (gensim
CoherenceModel against the SearchSnippets corpus itself) is ALSO computed
and reported, but under a name that can never be confused with the
paper's own `cv_palmetto_wikipedia` metric - see
docs/methodological_notes.md #5/#11.
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


class _GloCOMOfficialArtifactAdapter:
    """Wraps GloCOMAdapter so protocols.base's uniform `model.fit(documents)`
    call resolves to `fit_precomputed()` under the hood - the runner never
    needs to know a given protocol prefers the precomputed-artifact path."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def fit(self, documents: list[str]):
        from vaebm_benchmark.datasets.definitions.glocom_official import (
            GloCOMOfficialSearchSnippets,
        )

        bundle = GloCOMOfficialSearchSnippets().load()
        if list(documents) != bundle.documents:
            raise ValueError(
                "GloCOM protocol expects `documents` to be exactly "
                "GloCOMOfficialSearchSnippets().load().documents (texts.txt) - "
                "got a different document list. Call protocol.prepare_dataset() "
                "to obtain the correct documents."
            )
        self._inner.fit_precomputed(bundle)
        return self

    def get_topics(self, top_n: int = 10):
        return self._inner.get_topics(top_n)

    def get_topics_both_views(self, top_n: int = 10):
        return self._inner.get_topics_both_views(top_n) if hasattr(self._inner, "get_topics_both_views") else None

    def get_document_topics(self, documents: list[str]):
        return self._inner.get_document_topics(documents)

    def get_document_clusters(self, documents: list[str]):
        return self._inner.get_document_clusters(documents)


class GloCOMProtocol(BaselineProtocol):
    name = "glocom"
    paper = (
        "Nguyen, Q. D., Nguyen, T., Nguyen, D. A., Ngo Van, L., Dinh, S., & Nguyen, T. H. (2025). "
        "\"GloCOM: A Short Text Neural Topic Model via Global Clustering Context.\" NAACL 2025. "
        "https://aclanthology.org/2025.naacl-long.51/"
    )
    official_repository = "https://github.com/qducnguyen/GloCOM"
    upstream_commit = "4094055b9e2d0169b0aa75d5aed7220e9509f0de"

    datasets = [
        DatasetSpec(
            id="search_snippets",
            name="SearchSnippets (official GloCOM precomputed artifact)",
            source_url="https://raw.githubusercontent.com/qducnguyen/GloCOM/4094055b9e2d0169b0aa75d5aed7220e9509f0de/data/SearchSnippets",
            source_repository="https://github.com/qducnguyen/GloCOM",
            num_docs_expected=12294,
            num_classes=8,
            notes="Precomputed bow.npz/global_bow.npz/global_maps.txt/vocab.txt/word_embeddings.npz, fetched verbatim.",
        ),
    ]
    split_strategy = SplitSpec(
        strategy="full_corpus",
        description="Transductive: fit and evaluate on the same full corpus, matching the official repo exactly.",
    )
    topic_count = {"search_snippets": 50}
    metric_specs = [
        MetricSpec(name="cv_palmetto_wikipedia", kind="topic", top_n=15),
        MetricSpec(name="cv_local_corpus", kind="topic", top_n=15),
        MetricSpec(name="glocom_td", kind="topic", top_n=15),
        MetricSpec(name="purity", kind="clustering"),
        MetricSpec(name="nmi", kind="clustering"),
    ]
    seeds = [42]

    published_results = [
        PublishedResult(dataset_id="search_snippets", metric="cv_palmetto_wikipedia", value=0.453,
                         source="Table 2, GloCOM row, SearchSnippets K=50 (Nguyen et al., NAACL 2025)"),
        PublishedResult(dataset_id="search_snippets", metric="glocom_td", value=0.956,
                         source="Table 2, GloCOM row, SearchSnippets K=50 (Nguyen et al., NAACL 2025)"),
        PublishedResult(dataset_id="search_snippets", metric="purity", value=0.806,
                         source="Table 2, GloCOM row, SearchSnippets K=50 (Nguyen et al., NAACL 2025)"),
        PublishedResult(dataset_id="search_snippets", metric="nmi", value=0.502,
                         source="Table 2, GloCOM row, SearchSnippets K=50 (Nguyen et al., NAACL 2025)"),
        # cv_local_corpus has no published counterpart - it is not the paper's metric (see module docstring).
    ]

    def __init__(self, smoke_test: bool = True) -> None:
        self.smoke_test = smoke_test
        self.mode = "smoke" if smoke_test else "full"
        # Paper default is 200 epochs (~<10 min on an RTX 3090; slower on
        # CPU). Reduced for a smoke test per this project's own
        # instructions ("test only small experiments") - documented here,
        # not silently substituted; verify() surfaces this as a DIFFERENCE.
        self.epochs = 20 if smoke_test else 200

    def _require_search_snippets(self, dataset_id: str) -> None:
        if dataset_id != "search_snippets":
            raise KeyError(f"GloCOMProtocol only supports 'search_snippets' currently, got '{dataset_id}'")

    def prepare_dataset(self, dataset_id: str) -> list[str]:
        self._require_search_snippets(dataset_id)
        from vaebm_benchmark.datasets.definitions.glocom_official import GloCOMOfficialSearchSnippets

        return GloCOMOfficialSearchSnippets().load().documents

    def prepare_labels(self, dataset_id: str):
        self._require_search_snippets(dataset_id)
        from vaebm_benchmark.datasets.definitions.glocom_official import GloCOMOfficialSearchSnippets

        return GloCOMOfficialSearchSnippets().load().labels

    def artifact_checksum(self, dataset_id: str) -> str:
        self._require_search_snippets(dataset_id)
        from vaebm_benchmark.datasets.definitions.glocom_official import EXPECTED_SHA256

        return "|".join(f"{name}:{digest}" for name, digest in sorted(EXPECTED_SHA256.items()))

    def preprocessing_version(self, dataset_id: str) -> str:
        self._require_search_snippets(dataset_id)
        return "official_precomputed_artifact_no_preprocessing_v1"

    def vocabulary_for(self, dataset_id: str) -> list[str]:
        self._require_search_snippets(dataset_id)
        from vaebm_benchmark.datasets.definitions.glocom_official import GloCOMOfficialSearchSnippets

        return GloCOMOfficialSearchSnippets().load().vocab

    def build_baseline(self, dataset_id: str, seed: int):
        from vaebm_benchmark.models.glocom_adapter import GloCOMAdapter

        inner = GloCOMAdapter(
            num_topics=self.topic_count[dataset_id],
            num_global_clusters=40,
            epochs=self.epochs,
            learning_rate=0.002,
            batch_size=200,
            aug_coef=0.5,
            prior_var=0.1,
            weight_loss_ECR=60.0,
            en_units=200,
            embed_size=200,
            sinkhorn_alpha=20.0,
            sinkhorn_max_iter=100,
            beta_temp=0.2,
            num_top_words=15,
            seed=seed,
        )
        return _GloCOMOfficialArtifactAdapter(inner)

    def vaebm_variants(self) -> list[str]:
        return ["protocol_faithful", "stability_adjusted"]

    def build_vaebm(self, dataset_id: str, seed: int, variant: str = "stability_adjusted"):
        from vaebm_benchmark.datasets.definitions.glocom_official import GloCOMOfficialSearchSnippets
        from vaebm_benchmark.models.vaebm_adapter import VAEBMAdapter

        if variant not in self.vaebm_variants():
            raise ValueError(f"Unknown VAE-BM variant '{variant}'; available: {self.vaebm_variants()}")

        # protocol_faithful: the AS-SUPPLIED notebook default (lr=1e-2).
        # Confirmed by direct diagnostic (docs/methodological_notes.md #8)
        # to diverge to inf/NaN within 2 epochs on this vocab size (4618
        # words), producing degenerate KMeans clusters (Purity=0.224,
        # NMI=0.014 - near-random). This is EXPECTED and recorded as such,
        # never hidden. stability_adjusted: lr=1e-3, trains stably over
        # the full run and produces non-degenerate results (Purity=0.742,
        # NMI=0.393) - a documented hyperparameter substitution, not a
        # change to VAE-BM's architecture/loss, never presented as if it
        # were the original formulation.
        lr = 1e-2 if variant == "protocol_faithful" else 1e-3

        bundle = GloCOMOfficialSearchSnippets().load()
        return VAEBMAdapter(
            n_clusters=self.topic_count[dataset_id],
            voc_size=len(bundle.vocab),
            units=50,
            epochs=self.epochs,
            batch_size=128,
            lr=lr,
            random_state=seed,
            vectorizer_type="tfidf",
            embedder="all-MiniLM-L6-v2",  # matches GloCOM's own embedding_model
            dim=(1500, 1000, 500),
            dim_emb=(368,),
            alpha=0.99,
            top_words_mode="energy",
            vocabulary=bundle.vocab,  # exact same vocab.txt as the GloCOM baseline, exact-count tokenizer (see vaebm.py)
        )

    def checks(self) -> list[ProtocolCheck]:
        from vaebm_benchmark.datasets.definitions.glocom_official import (
            GLOCOM_COMMIT,
            GloCOMOfficialSearchSnippets,
        )
        from vaebm_benchmark.metrics.palmetto import palmetto_available

        checksum_ok, checksum_problems = GloCOMOfficialSearchSnippets().verify()
        checksum_status = MatchStatus.MATCH if checksum_ok else MatchStatus.DIFFERENCE
        checksum_note = (
            f"every file's SHA256 matches EXPECTED_SHA256, pinned to commit {GLOCOM_COMMIT}"
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
                          "fetched verbatim from the official repo's own data/SearchSnippets/ at the pinned commit"),
            ProtocolCheck("checksum", "per-file SHA256 vs. EXPECTED_SHA256 (glocom_official.py)", checksum_status,
                          checksum_note),
            ProtocolCheck("upstream_commit", self.upstream_commit, MatchStatus.MATCH,
                          "pinned commit, verified directly by reading run.py/GloCOM.py/dataloader.py at this commit"),
            ProtocolCheck("preprocessing", "none applied by this project", MatchStatus.MATCH,
                          "documents/vocab/bow are the official repo's own precomputed artifact, unmodified"),
            ProtocolCheck("vocabulary", f"{len(self.vocabulary_for('search_snippets'))}-word vocab.txt (official artifact)",
                          MatchStatus.MATCH,
                          "VAE-BM's vectorizer is pinned to this exact vocab list AND the same whitespace-split "
                          "tokenization (models/vaebm.py) - counts are provably identical, not assumed equivalent"),
            ProtocolCheck("K", f"{self.topic_count['search_snippets']}", MatchStatus.MATCH,
                          "paper reports K in {50, 100} for SearchSnippets; K=50 used here"),
            ProtocolCheck("split_strategy", self.split_strategy.strategy, MatchStatus.MATCH,
                          "official repo is transductive - dataloader.py reloads train bow.npz as test too"),
            ProtocolCheck("seeds", f"{self.seeds}", MatchStatus.UNKNOWN,
                          "official repo sets no seed anywhere (grep -rni seed on the cloned repo is empty); "
                          "paper reports mean/std over 3 runs, not a seed-reproducible single number"),
            ProtocolCheck("metric:cv_palmetto_wikipedia", "metrics/palmetto.py::palmetto_cv()", palmetto_status, palmetto_note),
            ProtocolCheck("metric:cv_local_corpus", "gensim CoherenceModel c_v vs. training corpus", MatchStatus.DIFFERENCE,
                          "NOT the paper's metric - a separately-named, documented approximation; never compared "
                          "against the paper's published cv_palmetto_wikipedia number"),
            ProtocolCheck("metric:glocom_td", "topic_diversity_glocom()", MatchStatus.MATCH,
                          "same TF==1 formula as the official evaluations/topic_diversity.py::compute_TD"),
            ProtocolCheck("metric:purity", "clustering_quality.purity()", MatchStatus.MATCH,
                          "same as evaluations/clustering.py::purity_score"),
            ProtocolCheck("metric:nmi", "clustering_quality.nmi()", MatchStatus.MATCH,
                          "same as sklearn.metrics.normalized_mutual_info_score, same as official repo"),
            ProtocolCheck("baseline_implementation", "GloCOM/ECR vendored verbatim + official precomputed artifacts",
                          MatchStatus.MATCH,
                          "hyperparameters match run.py's own CLI defaults (prior_var=0.1, weight_loss_ECR=60.0, "
                          "not the class defaults); training loop is topmost.BasicTrainer, compared directly "
                          "against the official trainer/Trainer.py - see module docstring"),
            ProtocolCheck("vaebm_implementation", "models/vaebm.py, unmodified architecture/initializers/callbacks",
                          MatchStatus.MATCH,
                          "protocol_faithful variant uses the AS-SUPPLIED lr=1e-2; stability_adjusted uses lr=1e-3 - "
                          "both persisted and labeled, never conflated (see docs/methodological_notes.md #8/#9)"),
            ProtocolCheck("mode", self.mode, MatchStatus.MATCH if not self.smoke_test else MatchStatus.DIFFERENCE,
                          "smoke mode uses 20 epochs, not the paper's 200 - never described as paper reproduction"
                          if self.smoke_test else "matches paper default (200 epochs)"),
        ]
