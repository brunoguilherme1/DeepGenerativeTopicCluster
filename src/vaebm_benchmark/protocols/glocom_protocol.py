"""GloCOM protocol: reproduces github.com/qducnguyen/GloCOM's own
evaluation exactly as `run.py` performs it for SearchSnippets - the ONLY
dataset the official repo ships precomputed artifacts for (confirmed by
listing its full repo tree; GoogleNews/StackOverflow/Biomedical require
external STTM+TopMost preprocessing this project has not reproduced - see
docs/methodological_notes.md and configs/datasets/*.yaml for those three).

Protocol facts, each verified directly against the cloned official repo
and the NAACL 2025 paper PDF (not inferred):
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
  - Metrics: the paper reports TC (Topic Coherence, C_V via the Palmetto
    Java library against a bundled Wikipedia reference corpus), TD (this
    project's `glocom_td` - see metrics/topic_quality.py, NOT the same
    formula as standard `topic_diversity`), Purity, and NMI (both via
    argmax(theta) vs. ground-truth labels).
  - Seeds: the official codebase sets NO random seed anywhere (`grep -rni
    seed` across its own .py files returns nothing) - the paper instead
    reports mean+-std over 3 runs. This protocol's own `seeds` list is
    this project's OWN choice (for reproducibility of THIS project's
    runs), not something recovered from the paper.

KNOWN DEVIATION (documented, not silently absorbed - see
docs/methodological_notes.md #5 and comparison.csv's `published_source`
column): TC is NOT computed via Palmetto+Wikipedia here (that requires a
Java runtime and a ~multi-GB bundled Wikipedia reference corpus this
project does not set up for a smoke test) - `cv` below uses gensim's
CoherenceModel against the SearchSnippets corpus itself as the reference,
the SAME fallback method the official repo's own
`evaluations/topic_coherence.py` provides as a non-default alternative to
Palmetto. This makes the `cv` published-vs-reproduced comparison NOT
apples-to-apples; treat it as indicative only until Palmetto is wired up.
"""

from __future__ import annotations

from vaebm_benchmark.protocols.base import (
    BaselineProtocol,
    DatasetSpec,
    MetricSpec,
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
        MetricSpec(name="cv", kind="topic", top_n=15),
        MetricSpec(name="glocom_td", kind="topic", top_n=15),
        MetricSpec(name="purity", kind="clustering"),
        MetricSpec(name="nmi", kind="clustering"),
    ]
    seeds = [42]

    published_results = [
        PublishedResult(dataset_id="search_snippets", metric="cv", value=0.453,
                         source="Table 2, GloCOM row, SearchSnippets K=50 (Nguyen et al., NAACL 2025) - "
                                "PUBLISHED VALUE USES PALMETTO+WIKIPEDIA, NOT gensim CoherenceModel; see module docstring"),
        PublishedResult(dataset_id="search_snippets", metric="glocom_td", value=0.956,
                         source="Table 2, GloCOM row, SearchSnippets K=50 (Nguyen et al., NAACL 2025)"),
        PublishedResult(dataset_id="search_snippets", metric="purity", value=0.806,
                         source="Table 2, GloCOM row, SearchSnippets K=50 (Nguyen et al., NAACL 2025)"),
        PublishedResult(dataset_id="search_snippets", metric="nmi", value=0.502,
                         source="Table 2, GloCOM row, SearchSnippets K=50 (Nguyen et al., NAACL 2025)"),
    ]

    def __init__(self, smoke_test: bool = True) -> None:
        self.smoke_test = smoke_test
        # Paper default is 200 epochs (~<10 min on an RTX 3090; slower on
        # CPU). Reduced for a smoke test per this project's own
        # instructions ("test only small experiments") - documented here,
        # not silently substituted; verify() surfaces this as a DIFFERENCE.
        self.epochs = 20 if smoke_test else 200

    def prepare_dataset(self, dataset_id: str) -> list[str]:
        if dataset_id != "search_snippets":
            raise KeyError(f"GloCOMProtocol only supports 'search_snippets' currently, got '{dataset_id}'")
        from vaebm_benchmark.datasets.definitions.glocom_official import GloCOMOfficialSearchSnippets

        return GloCOMOfficialSearchSnippets().load().documents

    def prepare_labels(self, dataset_id: str):
        if dataset_id != "search_snippets":
            raise KeyError(f"GloCOMProtocol only supports 'search_snippets' currently, got '{dataset_id}'")
        from vaebm_benchmark.datasets.definitions.glocom_official import GloCOMOfficialSearchSnippets

        return GloCOMOfficialSearchSnippets().load().labels

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

    def build_vaebm(self, dataset_id: str, seed: int):
        from vaebm_benchmark.datasets.definitions.glocom_official import GloCOMOfficialSearchSnippets
        from vaebm_benchmark.models.vaebm_adapter import VAEBMAdapter

        bundle = GloCOMOfficialSearchSnippets().load()
        return VAEBMAdapter(
            n_clusters=self.topic_count[dataset_id],
            voc_size=len(bundle.vocab),
            units=50,
            epochs=self.epochs,
            batch_size=128,
            # NOT the supplied notebook's own default (1e-2): confirmed by
            # direct diagnostic (see docs/methodological_notes.md) that
            # 1e-2 diverges to inf/NaN within 2 epochs on this vocab size
            # (4618 words), producing degenerate KMeans clusters
            # (Purity=0.224, NMI=0.014 - near-random). 1e-3 trains stably
            # over the full smoke-test run and produces non-degenerate
            # results (Purity=0.742, NMI=0.393). This is a hyperparameter
            # substitution, not a change to VAE-BM's architecture/loss -
            # documented per this project's own "don't silently change
            # the model" rule, not hidden.
            lr=1e-3,
            random_state=seed,
            vectorizer_type="tfidf",
            embedder="all-MiniLM-L6-v2",  # matches GloCOM's own embedding_model
            dim=(1500, 1000, 500),
            dim_emb=(368,),
            alpha=0.99,
            top_words_mode="energy",
            vocabulary=bundle.vocab,  # exact same vocab.txt as the GloCOM baseline, not just matching size
        )

    def verify(self) -> dict:
        return {
            "paper": self.paper,
            "official_repository": self.official_repository,
            "dataset": {
                "id": "search_snippets",
                "artifact": self.datasets[0].source_url,
                "num_docs_expected": self.datasets[0].num_docs_expected,
                "verdict": "MATCH - fetched verbatim from the official repo's own data/SearchSnippets/",
            },
            "preprocessing": {
                "description": "None applied by this project - documents/vocab/bow are the official repo's own precomputed artifact.",
                "verdict": "MATCH",
            },
            "vocabulary": {"source": "vocab.txt (official artifact)", "verdict": "MATCH"},
            "K": {"value": self.topic_count["search_snippets"], "paper_reports": "K in {50, 100}", "verdict": "MATCH (K=50)"},
            "split_strategy": {"value": self.split_strategy.strategy, "verdict": "MATCH - official repo is transductive"},
            "seeds": {"value": self.seeds, "verdict": "UNKNOWN - official repo sets no seed; paper reports mean/std over 3 runs, not seed-reproducible"},
            "metrics": {
                "cv": "DIFFERENCE - paper uses Palmetto+Wikipedia C_V, this project uses gensim CoherenceModel c_v (see module docstring)",
                "glocom_td": "MATCH - same formula as evaluations/topic_diversity.py::compute_TD",
                "purity": "MATCH - same as evaluations/clustering.py::purity_score",
                "nmi": "MATCH - same as sklearn.metrics.normalized_mutual_info_score, same as official repo",
            },
            "baseline_implementation": "MATCH - GloCOM model vendored verbatim; trained on official precomputed artifacts, hyperparameters match run.py's own CLI defaults",
            "vaebm_implementation": "N/A - VAE-BM is not a baseline being reproduced; see docs/methodological_notes.md",
            "epochs": {
                "value": self.epochs,
                "paper_value": 200,
                "verdict": "MATCH" if not self.smoke_test else "DIFFERENCE - reduced for smoke test, see BaselineProtocol.__init__",
            },
        }
