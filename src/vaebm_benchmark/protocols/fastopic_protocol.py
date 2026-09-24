"""FASTopic protocol: reproduces github.com/bobxwu/FASTopic's own
evaluation on **20NG, NYT, and WoS**, verified directly against the
FASTopic NeurIPS 2024 paper PDF (arXiv:2405.17978) and its companion
toolkit `topmost` (github.com/BobXWu/TopMost, where FASTopic's own
preprocessing/dataset-loading code actually lives).

Protocol facts (all verified against source, not inferred):
  - K: paper Table 1/2/Figure 4/Figure 7 all use K=50 as the main setting
    (Figure 7's own caption: "under 50 topics (K=50)"). Tables 4/5 sweep
    K in {75,...,200} but ONLY on WoS, as a separate adaptivity study -
    not this protocol's setting.
  - Datasets: 20NG (18,846 docs, vocab 5,000), NYT (9,172 docs, vocab
    10,000) via TopMost's official pre-split artifacts (fastopic_20ng.py/
    fastopic_nyt.py) - byte-verified against the paper's own Table 7. WoS
    (10,000 docs, 7 labels, vocab 10,000) has NO official FASTopic/
    TopMost artifact (verified via TopMost's full git history - never
    distributed one); fastopic_wos.py reconstructs it from the dataset
    the paper actually cites (Kowsari et al. 2017, HDLTex, WOS-11967),
    marked exact_fastopic_artifact=false throughout - see that module's
    own docstring.
  - Split: 20NG/NYT ship pre-split train/test files (topmost.BasicDataset)
    - this protocol does not generate its own split for them. WoS has no
      official split to preserve (no official artifact exists at all);
      fastopic_wos.py generates its own stratified 80/20 split at seed=42.
    Classification fits each model on the TRAIN split only, then infers
    train/test representations separately (theta for HiCOT, mu for
    VAE-BM) - never re-fit on test. Clustering is transductive: fit+eval
    on the full corpus (train+test concatenated), matching FASTopic's own
    clustering protocol (topmost.BasicTrainer fits/evaluates the same
    way).
  - Classification protocol: "we train SVM classifiers with the inferred
    doc-topic distributions as document features" (Sec 4.3), following
    Wu et al. [73] = ECRTM. The EXACT rule (kernel/gamma/C, F1 averaging)
    is not stated in FASTopic's own paper text - recovered instead from
    TopMost's own shared evaluation code (topmost/eva/classification.py:
    `SVC(gamma='scale')` - i.e. default kernel='rbf', default C=1.0 - and
    `f1_score(..., average='macro')`). This protocol uses that exact
    recovered rule, NOT this repo's other/different SVC(kernel='linear',
    C=1.0) convention used elsewhere (see docs/methodological_notes.md's
    own SVM-protocol note, which is a documented default for a DIFFERENT
    experiment, not a claim about this one).
  - Clustering assignment: HiCOT uses argmax(theta) (topmost/eva/
    clustering.py's own `_clustering`); VAE-BM has no genuine theta, so
    per this project's own established VAE-BM semantics it uses
    KMeans(mu, n_clusters=K) instead - documented explicitly as a
    necessary substitution, never called "theta".
  - Baselines: this protocol's own published_results below are COPIED
    from the paper's Table 2 (clustering) and Figure 4 (classification) -
    FASTopic itself, and every other baseline in those tables, is NEVER
    rerun here. Only VAE-BM and HiCOT are executed.
  - Hyperparameters: VAE-BM's alpha is locked at 0 (this project's own
    established finding - see main.tex/docs - that any lexical-branch
    contribution hurts clustering quality relative to pure-embedding).
    HiCOT uses its own paper-documented defaults. Both use the same
    embedder FASTopic itself uses (all-MiniLM-L6-v2, Appendix D) unless
    a search later finds a better one - see run_fastopic_protocol_search.py.
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

K = 50


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
            id="fastopic_20ng",
            name="20NG, via topmost's official mirror",
            source_url="https://raw.githubusercontent.com/BobXWu/TopMost/master/data/20NG.zip",
            source_repository="https://github.com/BobXWu/TopMost",
            num_docs_expected=18846,
            num_classes=20,
            notes="Pre-split train (11,314) / test (7,532) files, vocab_size=5000 - verified byte-identical to paper Table 7.",
        ),
        DatasetSpec(
            id="fastopic_nyt",
            name="NYT (New York Times), via topmost's official mirror",
            source_url="https://raw.githubusercontent.com/BobXWu/TopMost/master/data/NYT.zip",
            source_repository="https://github.com/BobXWu/TopMost",
            num_docs_expected=9172,
            num_classes=12,
            notes="Pre-split train (8,254) / test (918) files, vocab_size=10000.",
        ),
        DatasetSpec(
            id="fastopic_wos_reconstructed",
            name="WoS (Web of Science), RECONSTRUCTED - no official artifact exists",
            source_url="https://data.mendeley.com/datasets/9rw3vkcfy4/2",
            source_repository="https://github.com/kk7nc/HDLTex",
            num_docs_expected=10000,
            num_classes=7,
            notes="exact_fastopic_artifact=False - see fastopic_wos.py's own module docstring for full provenance.",
        ),
    ]
    split_strategy = SplitSpec(
        strategy="predefined_train_test (20NG/NYT) / generated_stratified_80_20_seed42 (WoS)",
        description="20NG/NYT ship pre-split train/test files; FASTopic trains on train_texts only, "
        "infers held-out theta on test_texts via model.transform() for clustering metrics. WoS has no "
        "official artifact to ship a split, so this protocol generates its own stratified 80/20 split "
        "(seed=42) - documented explicitly as this project's own choice, not FASTopic's.",
    )
    topic_count = {"fastopic_20ng": K, "fastopic_nyt": K, "fastopic_wos_reconstructed": K}
    metric_specs = [
        MetricSpec(name="purity", kind="clustering"),
        MetricSpec(name="nmi", kind="clustering"),
    ]
    seeds = [42]

    # Published reference rows - COPIED from the paper, never rerun here.
    published_results = [
        # Table 2 (clustering, Purity/NMI), K=50
        PublishedResult(dataset_id="fastopic_20ng", metric="purity", value=0.577, source="Table 2, FASTopic row, 20NG"),
        PublishedResult(dataset_id="fastopic_20ng", metric="nmi", value=0.525, source="Table 2, FASTopic row, 20NG"),
        PublishedResult(dataset_id="fastopic_nyt", metric="purity", value=0.662, source="Table 2, FASTopic row, NYT"),
        PublishedResult(dataset_id="fastopic_nyt", metric="nmi", value=0.369, source="Table 2, FASTopic row, NYT"),
        PublishedResult(dataset_id="fastopic_wos_reconstructed", metric="purity", value=0.672, source="Table 2, FASTopic row, WoS"),
        PublishedResult(dataset_id="fastopic_wos_reconstructed", metric="nmi", value=0.365, source="Table 2, FASTopic row, WoS"),
        # Figure 4 (classification, Accuracy/F1)
        PublishedResult(dataset_id="fastopic_20ng", metric="accuracy", value=0.604, source="Figure 4, FASTopic row, 20NG"),
        PublishedResult(dataset_id="fastopic_20ng", metric="f1", value=0.593, source="Figure 4, FASTopic row, 20NG"),
        PublishedResult(dataset_id="fastopic_nyt", metric="accuracy", value=0.754, source="Figure 4, FASTopic row, NYT"),
        PublishedResult(dataset_id="fastopic_nyt", metric="f1", value=0.596, source="Figure 4, FASTopic row, NYT"),
        PublishedResult(dataset_id="fastopic_wos_reconstructed", metric="accuracy", value=0.739, source="Figure 4, FASTopic row, WoS"),
        PublishedResult(dataset_id="fastopic_wos_reconstructed", metric="f1", value=0.703, source="Figure 4, FASTopic row, WoS"),
    ]

    def __init__(self, smoke_test: bool = True) -> None:
        self.smoke_test = smoke_test
        self.mode = "smoke" if smoke_test else "full"
        # Paper default is 200 epochs, full-batch. Reduced for a smoke
        # test - documented, not silent (verify() surfaces this).
        self.epochs_vaebm = 5 if smoke_test else 50
        self.epochs_hicot = 5 if smoke_test else 200

    def _dataset_loader(self, dataset_id: str):
        if dataset_id == "fastopic_20ng":
            from vaebm_benchmark.datasets.definitions.fastopic_20ng import TwentyNGFASTopicDataset
            return TwentyNGFASTopicDataset()
        elif dataset_id == "fastopic_nyt":
            from vaebm_benchmark.datasets.definitions.fastopic_nyt import NYTDataset
            return NYTDataset()
        elif dataset_id == "fastopic_wos_reconstructed":
            from vaebm_benchmark.datasets.definitions.fastopic_wos import WoSReconstructedDataset
            return WoSReconstructedDataset()
        raise KeyError(f"FASTopicProtocol supports fastopic_20ng/fastopic_nyt/fastopic_wos_reconstructed, got '{dataset_id}'")

    def _load(self, dataset_id: str):
        loader = self._dataset_loader(dataset_id)
        if dataset_id == "fastopic_wos_reconstructed":
            return loader.load(test_p=0.2, seed=42)
        return loader.load()

    def prepare_dataset(self, dataset_id: str) -> list[str]:
        return self._load(dataset_id).train_texts

    def prepare_eval_documents(self, dataset_id: str) -> list[str]:
        return self._load(dataset_id).test_texts

    def prepare_labels(self, dataset_id: str):
        return self._load(dataset_id).test_labels

    def prepare_all_documents_and_labels(self, dataset_id: str):
        """Transductive clustering view: full corpus (train+test) and its
        aligned labels, in the same concatenation order both times."""
        bundle = self._load(dataset_id)
        return (
            list(bundle.train_texts) + list(bundle.test_texts),
            list(bundle.train_labels) + list(bundle.test_labels),
        )

    def vocabulary_for(self, dataset_id: str) -> list[str]:
        return self._load(dataset_id).vocab

    def artifact_checksum(self, dataset_id: str) -> str:
        loader = self._dataset_loader(dataset_id)
        if hasattr(loader, "verify"):
            ok, problems = loader.verify()
            return "verified" if ok else f"unverified: {problems}"
        return "n/a"

    def preprocessing_version(self, dataset_id: str) -> str:
        if dataset_id == "fastopic_wos_reconstructed":
            return "wos_reconstructed_topmost_5step_v1_mintrm65"
        return "topmost_official_artifact_v1"

    def build_vaebm(self, dataset_id: str, seed: int, config: dict | None = None):
        from vaebm_benchmark.models.vaebm_adapter import VAEBMAdapter

        vocab = self.vocabulary_for(dataset_id)
        cfg = {
            "alpha": 0.0,
            "lr": 1e-3,
            "units": 50,
            "normalize_mu": False,
            "embedder": "all-MiniLM-L6-v2",  # matches FASTopic's own doc_embed_model, Appendix D
            "freeze_embedding_branch": False,
        }
        if config:
            cfg.update(config)
        return VAEBMAdapter(
            n_clusters=self.topic_count[dataset_id],
            voc_size=len(vocab),
            vocabulary=vocab,
            random_state=seed,
            alpha=cfg["alpha"],
            lr=cfg["lr"],
            units=cfg["units"],
            epochs=self.epochs_vaebm,
            normalize_mu=cfg["normalize_mu"],
            embedder=cfg["embedder"],
            freeze_embedding_branch=cfg["freeze_embedding_branch"],
            batch_size=128,
            verbose=0,
        )

    def build_hicot(self, dataset_id: str, seed: int, config: dict | None = None):
        from vaebm_benchmark.models.hicot_adapter import HiCOTAdapter

        vocab = self.vocabulary_for(dataset_id)
        cfg = {"lr": 0.002, "weight_loss_ECR": 40.0, "weight_loss_DT": 250.0, "en_units": 200, "max_fit_seconds": None}
        if config:
            cfg.update(config)
        return HiCOTAdapter(
            n_clusters=self.topic_count[dataset_id],
            voc_size=len(vocab),
            vocabulary=vocab,
            random_state=seed,
            lr=cfg["lr"],
            epochs=self.epochs_hicot,
            weight_loss_ECR=cfg["weight_loss_ECR"],
            weight_loss_DT=cfg["weight_loss_DT"],
            en_units=cfg["en_units"],
            max_clusters=self.topic_count[dataset_id],
            max_fit_seconds=cfg["max_fit_seconds"],
        )

    def build_model(self, model_name: str, dataset_id: str, seed: int, config: dict | None = None):
        if model_name == "vaebm":
            return self.build_vaebm(dataset_id, seed, config)
        elif model_name == "hicot":
            return self.build_hicot(dataset_id, seed, config)
        raise KeyError(f"FASTopicProtocol runs vaebm/hicot only, got '{model_name}'")

    def checks(self) -> list[ProtocolCheck]:
        return [
            ProtocolCheck("K", str(K), MatchStatus.MATCH,
                          "paper Table 1/2/Figure 4/Figure 7 all use K=50 as the main setting"),
            ProtocolCheck("dataset:fastopic_20ng", self.datasets[0].source_url, MatchStatus.MATCH,
                          "official topmost mirror, doc/vocab counts verified against paper Table 7"),
            ProtocolCheck("dataset:fastopic_nyt", self.datasets[1].source_url, MatchStatus.MATCH,
                          "official topmost mirror, doc/vocab counts verified against paper Table 7"),
            ProtocolCheck("dataset:fastopic_wos_reconstructed", self.datasets[2].source_url, MatchStatus.DIFFERENCE,
                          "no official FASTopic/TopMost WoS artifact exists (verified via TopMost's full git "
                          "history) - reconstructed from the paper's own cited source (Kowsari et al. 2017); "
                          "exact 11,967->10,000 subsampling procedure is not published anywhere"),
            ProtocolCheck("classification_svm", "SVC(gamma='scale'), macro-F1", MatchStatus.MATCH,
                          "recovered from topmost/eva/classification.py::_cls - not stated in the paper's own text"),
            ProtocolCheck("clustering_assignment", "argmax(theta) [HiCOT] / KMeans(mu) [VAE-BM]", MatchStatus.MATCH,
                          "argmax(theta) matches topmost/eva/clustering.py::_clustering; VAE-BM has no genuine "
                          "theta so uses this project's own established KMeans(mu) substitute instead"),
            ProtocolCheck("baselines_rerun", "none - FASTopic/LDA-Mallet/NMF/BERTopic/etc. are reference rows only",
                          MatchStatus.MATCH, "published_results copied directly from Table 2/Figure 4, never recomputed"),
            ProtocolCheck("mode", self.mode, MatchStatus.MATCH if not self.smoke_test else MatchStatus.DIFFERENCE,
                          f"smoke mode uses {self.epochs_vaebm}/{self.epochs_hicot} epochs (vaebm/hicot), "
                          f"not a paper-scale run" if self.smoke_test else "full-scale run"),
        ]
