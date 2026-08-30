"""BaselineProtocol: the one abstraction this whole repo is organized
around. Each baseline paper (FASTopic, GloCOM, ...) gets exactly one
Protocol subclass that pins down everything the paper's own experiments
actually did - dataset artifact, preprocessing, vocabulary, K, split
strategy, metrics, seeds - and builds BOTH the baseline and VAE-BM through
that same pinned-down configuration. Nothing here is shared across
protocols by default; a protocol only shares what its own docstring/config
says the paper itself used identically to another (e.g. two datasets from
the same paper, not two different papers).

The one rule every protocol must satisfy, and every script in scripts/
checks: VAE-BM must be evaluated under the EXACT protocol of the baseline
being compared. If FASTopic's paper used K=20 on 20ng, GloCOM's used K=20
on StackOverflow, then vaebm gets KMeans(n_clusters=20) in both cases
respectively - never a K independently tuned for VAE-BM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from vaebm_benchmark.models.base import ProtocolModelAdapter


class MatchStatus(str, Enum):
    """Adopted from an independent second implementation of this same
    project (see docs/repository_comparison_report.md) - a formal enum
    plus an aggregate boolean is a genuinely stronger fair-comparison
    guardrail than free-text verdict strings alone. Unlike that other
    implementation's checks (which mostly compared a config value to
    itself - see the report's Finding C.3), every ProtocolCheck emitted
    by this project's own `checks()` methods below is backed by cited
    evidence from this project's own upstream research (paper text,
    official repo source, or a runtime-computed checksum) - never a
    tautological self-comparison."""

    MATCH = "MATCH"
    DIFFERENCE = "DIFFERENCE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ProtocolCheck:
    """One verifiable fair-comparison field. `note` should cite WHY the
    status is what it is (a paper section, a repo file, a checksum
    comparison) - never left to speak for itself."""

    field: str
    detail: str
    status: MatchStatus
    note: str = ""


def fair_comparison(checks: list[ProtocolCheck]) -> bool:
    """True iff EVERY check is MATCH. A single DIFFERENCE or UNKNOWN in
    any field (dataset artifact, checksum, preprocessing, vocabulary, K,
    split, metrics, baseline source, environment) means the comparison is
    not yet defensible as fair - this function is never overridden to
    special-case a field away."""
    return all(check.status is MatchStatus.MATCH for check in checks)


@dataclass(frozen=True)
class DatasetSpec:
    """What "the dataset" means for this protocol - the artifact, not just
    a name. Two protocols using "the same named dataset" but a different
    artifact/preprocessing are NOT interchangeable; keep them as separate
    DatasetSpecs even if the human-readable name repeats."""

    id: str
    name: str
    source_url: str
    source_repository: str
    num_docs_expected: Optional[int]
    num_classes: Optional[int]
    checksum_sha256: Optional[str] = None  # of the raw artifact file, once downloaded
    notes: str = ""


@dataclass(frozen=True)
class SplitSpec:
    """How the paper evaluates - not a DTEA-style generated split. Most
    topic-model papers (including both FASTopic and GloCOM, per this
    repo's protocol notes) fit and evaluate on the SAME full corpus
    (transductive) rather than holding out a test set - `strategy` should
    say so explicitly rather than defaulting to an assumed train/test."""

    strategy: str  # "full_corpus" | "predefined_train_test" | "random_split" | ...
    description: str


@dataclass(frozen=True)
class MetricSpec:
    name: str  # matches a key in metrics/topic_quality.py or metrics/clustering_quality.py
    kind: str  # "topic" | "clustering"
    top_n: int = 10


@dataclass(frozen=True)
class PublishedResult:
    """A number FROM THE PAPER ITSELF (a table, a footnote), kept
    separate from anything this repo computes - see
    docs/methodological_notes.md and results/<baseline>/comparison.csv for
    why published != reproduced is always shown, never collapsed into one
    number."""

    dataset_id: str
    metric: str
    value: float
    source: str  # e.g. "Table 2, FASTopic (Wu et al., 2024), 20NG row"


@dataclass
class ProtocolResult:
    protocol_name: str
    dataset_id: str
    seed: int
    baseline_topics: Optional[list[list[str]]] = None
    baseline_document_clusters: Optional[list[int]] = None
    vaebm_topics_energy: Optional[list[list[str]]] = None
    vaebm_topics_freq: Optional[list[list[str]]] = None
    vaebm_document_clusters: Optional[list[int]] = None
    baseline_metrics: dict = field(default_factory=dict)
    vaebm_metrics: dict = field(default_factory=dict)


class BaselineProtocol:
    """One reproducible protocol per baseline paper. Subclasses fill in the
    class-level provenance fields and implement the four methods below.
    """

    name: str = ""
    paper: str = ""
    official_repository: str = ""
    upstream_commit: str = ""  # pinned commit/version this protocol was verified against - never a moving branch
    mode: str = "smoke"  # "smoke" | "full" - every persisted run stamps this; smoke results are never paper reproduction

    datasets: list[DatasetSpec] = []
    split_strategy: SplitSpec = SplitSpec(strategy="full_corpus", description="")
    topic_count: dict[str, int] = {}  # dataset_id -> K, exactly as the paper used
    metric_specs: list[MetricSpec] = []
    seeds: list[int] = [42]
    published_results: list[PublishedResult] = []

    baseline_model_config: dict[str, Any] = {}
    vaebm_model_config: dict[str, Any] = {}

    def prepare_dataset(self, dataset_id: str) -> list[str]:
        """Returns the exact documents this protocol evaluates on, after
        this protocol's own preprocessing - no protocol-independent
        preprocessing is applied elsewhere."""
        raise NotImplementedError

    def prepare_eval_documents(self, dataset_id: str) -> list[str]:
        """Documents clustering metrics are computed on - may differ from
        prepare_dataset()'s (what gets fit). Default: identical to
        prepare_dataset() (transductive protocols, e.g. GloCOM's, where
        the official repo evaluates on the same full corpus it trained
        on). A protocol with a real held-out split (e.g. FASTopic's, whose
        topmost.BasicDataset ships pre-split train/test files) overrides
        this to return the TEST documents, since that paper's own Purity/
        NMI numbers come from inferring theta on held-out documents via
        `model.transform(test_texts)`, never from the training documents."""
        return self.prepare_dataset(dataset_id)

    def prepare_labels(self, dataset_id: str) -> Optional[list[int]]:
        """Ground-truth labels ALIGNED to prepare_eval_documents()'s
        documents (NOT necessarily prepare_dataset()'s - see that method's
        docstring for why), for clustering metrics - never used during
        fit(). Each protocol owns this explicitly (rather than a generic
        dataset-id -> labels lookup) because two protocols using a dataset
        with "the same name" may use different artifacts/preprocessing
        with different document counts/ordering (e.g. this project's
        GloCOM protocol uses the official repo's own 12,294-doc
        SearchSnippets artifact, not a 12,340-doc mirror another protocol
        might use for the same-named dataset) - a shared lookup would
        silently misalign labels."""
        raise NotImplementedError

    def build_baseline(self, dataset_id: str, seed: int) -> ProtocolModelAdapter:
        raise NotImplementedError

    def build_vaebm(self, dataset_id: str, seed: int, variant: str = "stability_adjusted") -> ProtocolModelAdapter:
        """`variant` distinguishes the AS-SUPPLIED hyperparameters
        ("protocol_faithful") from any documented substitution this
        project made for training stability ("stability_adjusted" - see
        docs/methodological_notes.md #8). Both are real, runnable
        configurations - never silently pick one and call it "the"
        VAE-BM result; every persisted run states which variant produced
        it (see evaluation/runner.py)."""
        raise NotImplementedError

    def vaebm_variants(self) -> list[str]:
        """Which variants build_vaebm() actually supports for this
        protocol - default both. A protocol can override this to add
        more (e.g. a third "tuned" variant) but must never silently drop
        "protocol_faithful", even where it is known to diverge - the
        divergence itself is the result worth recording."""
        return ["protocol_faithful", "stability_adjusted"]

    def artifact_checksum(self, dataset_id: str) -> str:
        """A single stable hash identifying the exact dataset artifact
        this protocol trains on for `dataset_id` - used both for
        provenance (checks()) and as a run-key component (utils/
        run_identity.py) so a baseline and VAE-BM run can only ever be
        paired if they trained on the byte-identical artifact."""
        raise NotImplementedError

    def preprocessing_version(self, dataset_id: str) -> str:
        """A short, stable tag identifying the exact preprocessing recipe
        applied for `dataset_id` (e.g. "topmost_preprocess_v1_vocab10000"
        or "official_precomputed_artifact_no_preprocessing"). Two runs
        with a different tag are NOT comparable even if everything else
        matches, and run_identity.py's run key reflects that."""
        raise NotImplementedError

    def vocabulary_for(self, dataset_id: str) -> list[str]:
        """The exact vocabulary (ordered) this protocol's baseline AND
        VAE-BM both train against for `dataset_id` - both models must use
        this SAME list, not independently-derived lists that merely have
        the same size. Used to compute the vocabulary_checksum run-key
        component and the `vocabulary` ProtocolCheck."""
        raise NotImplementedError

    def evaluate(
        self,
        model: ProtocolModelAdapter,
        train_documents: list[str],
        eval_documents: Optional[list[str]] = None,
        true_labels: Optional[list[int]] = None,
        reference_corpus: Optional[list[list[str]]] = None,
    ) -> tuple[dict, dict]:
        """Computes every metric in self.metric_specs against `model`,
        already fit on `train_documents`. Topic-quality metrics (NPMI/CV/
        Diversity/...) always describe the topics the model learned from
        fitting - there is no train/test distinction for them. Clustering
        metrics are computed on `eval_documents` (defaults to
        `train_documents` for a transductive protocol like GloCOM's;
        FASTopic's protocol passes the held-out test split instead - see
        prepare_eval_documents()). Identical code path for the baseline
        and for VAE-BM - the only difference between the two runs is which
        model was fit, never how it's scored.

        Returns `(metrics, metric_errors)`. `metrics[name]` is `None` (not
        omitted) for any OPTIONAL metric that could not be computed (e.g.
        `cv_palmetto_wikipedia` with no Java/Palmetto installed);
        `metric_errors[name]` then explains why - see metrics/
        topic_quality.py's `OPTIONAL_METRICS`. A non-optional metric that
        fails raises instead of silently degrading the result."""
        from vaebm_benchmark.metrics.clustering_quality import compute_clustering_metrics
        from vaebm_benchmark.metrics.topic_quality import compute_topic_metrics

        if eval_documents is None:
            eval_documents = train_documents

        topic_metric_names = [m.name for m in self.metric_specs if m.kind == "topic"]
        clustering_metric_names = [m.name for m in self.metric_specs if m.kind == "clustering"]
        top_n = self.metric_specs[0].top_n if self.metric_specs else 10

        results: dict = {}
        metric_errors: dict = {}
        if topic_metric_names:
            if reference_corpus is None:
                reference_corpus = [doc.split() for doc in train_documents]
            topics = model.get_topics(top_n=top_n)
            topic_values, topic_errors = compute_topic_metrics(topics, reference_corpus, topic_metric_names, top_n)
            results.update(topic_values)
            metric_errors.update(topic_errors)
        if clustering_metric_names and true_labels is not None:
            predicted = model.get_document_clusters(eval_documents)
            results.update(compute_clustering_metrics(predicted, true_labels, clustering_metric_names))
        return results, metric_errors

    def checks(self) -> list[ProtocolCheck]:
        """The one place a protocol declares its own MATCH/DIFFERENCE/
        UNKNOWN verdicts, as a flat, machine-readable list - subclasses
        implement this; `verify()` below renders it for humans and
        computes the aggregate `fair_comparison` boolean automatically, so
        the two can never drift out of sync with each other."""
        raise NotImplementedError

    def verify(self) -> dict:
        """Structured self-description for scripts/verify_protocol.py:
        paper, repo, dataset, checksum, preprocessing, vocab, K, seeds,
        metrics, baseline/vaebm implementation - each derived from
        checks(), plus the aggregate `fair_comparison` boolean (True only
        if every single check is MatchStatus.MATCH - see fair_comparison()
        above). A caller that only reads `fair_comparison` still gets the
        correct, conservative answer without having to parse prose."""
        checks = self.checks()
        return {
            "paper": self.paper,
            "official_repository": self.official_repository,
            "checks": {
                check.field: {
                    "detail": check.detail,
                    "status": check.status.value,
                    "note": check.note,
                }
                for check in checks
            },
            "fair_comparison": fair_comparison(checks),
            "summary": {
                status.value: sum(1 for c in checks if c.status is status)
                for status in MatchStatus
            },
        }
