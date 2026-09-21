"""Downstream text-classification experiment (--experiment classification):

    document -> model representation (theta, mu, or SBERT embeddings) -> SVM -> labels

Following ECRTM (Wu et al., ICML 2023) Sec 4.4 / HiCOT's own `--tune_SVM`
protocol reference: "we use the doc-topic distributions learned by topic
models as document features and train SVMs to predict the class of each
document." Neither paper's own text/code (HiCOT's `evaluations/` package
has no SVM module) specifies an exact kernel/C - this experiment uses
scikit-learn's `SVC(kernel="linear", C=1.0)` (a documented choice, not a
claim of reproducing either paper's own SVM tuning).

Uses HiCOT's own OFFICIAL train/test split
(datasets/definitions/hicot_datasets.py::load_hicot_split) - required,
since classification accuracy is only meaningful against a genuinely
held-out test set (unlike `topic`/`cluster`, which evaluate
transductively over the combined corpus - see
docs/methodological_notes.md #11/#12). Only `hicot_*` dataset ids with a
real split work here; SearchSnippets/GoogleNews raise (see
load_hicot_split()'s own docstring - HiCOT ships the identical corpus
under both train/test filenames for those two).

`--k` here follows the same convention as `--experiment topic` (a
user-specified topic/cluster count, e.g. 50/100 - the same K used in
ECRTM/HiCOT's own Table 2/3), NOT `--experiment cluster`'s
num_classes-derived K.

Multi-seed: the TOPIC MODEL itself is refit once per seed (not merely
the SVM) - matching this project's own established "seed reseeds the
whole pipeline" convention (experiment/runner.py, cluster_runner.py),
and capturing the model's own training variance, not only the SVM's.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class ClassificationRunResult:
    experiment: str
    model: str
    dataset: str
    k: int
    seed: int
    accuracy: Optional[float]
    f1: Optional[float]
    representation_source: str
    num_train_docs: int
    num_test_docs: int
    runtime_seconds: float
    status: str  # "ok" | "error"
    error: str = ""
    split_stratified: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


def _representation(model, documents: list[str], representation_source: str):
    if representation_source == "theta":
        return model.get_document_topics(documents)
    if representation_source == "mu":
        # VAEBMAdapter's own native-representation alias - the SAME mu
        # get_document_topics() would also return for this model, fetched
        # via the unambiguously-named getter instead - see
        # docs/methodological_notes.md #1 on why mu is not theta.
        return model.get_document_embeddings(documents)
    if representation_source == "embeddings":
        # sbert_kmeans has no theta/mu at all - its own SBERT embedding
        # space (the same space it clustered in) is the document
        # representation here, a completely standard "embed -> SVM"
        # classification baseline in its own right, not a fallback.
        return model.get_document_embeddings(documents)
    raise ValueError(f"Unknown representation_source '{representation_source}'")


def _make_svm(svm_kernel: str, svm_C: float, seed: int):
    """Linear-SVM classifier, swapped from `SVC(kernel="linear")` (libsvm)
    to `LinearSVC` (liblinear) on 2026-09-21 - found via direct
    diagnosis that libsvm's uncapped SMO solver was not converging on
    the locked architecture's raw (high-dimensional, continuous) `mu`
    representation, hanging the classification sweep's 3600s-timeout
    combos indefinitely (confirmed on a synthetic worst-case benchmark
    at the same n/d: SVC took 84s where LinearSVC took 3.8s; the real
    `agnews_short` run sat with zero progress for 9+ minutes past
    model-fit). Both are linear decision boundaries - this is a solver
    swap, not a change of model family - and the module's own docstring
    already flags the exact kernel/C as "a documented choice, not a
    claim of reproducing either paper's own SVM tuning."."""
    from sklearn.svm import SVC, LinearSVC

    if svm_kernel != "linear":
        return SVC(kernel=svm_kernel, C=svm_C, random_state=seed)
    return LinearSVC(C=svm_C, random_state=seed, max_iter=10000, dual="auto")


def _build_model_for_classification(model_name: str, k: int, seed: int, voc_size: int, dataset_id: str):
    """build_model() dispatch, EXCEPT for hicot: wires the same
    VAEBM_HICOT_MAX_FIT_SECONDS-configurable wall-clock early-stop
    experiment/cluster_runner.py's own _build_hicot() wrapper already
    gives the `cluster` experiment (see its own docstring/
    docs/methodological_notes.md #14) - build_model()'s own generic
    dispatch never passes max_fit_seconds at all (hicot's classification
    fit() would otherwise have no internal early-stop, and risk being
    hard-killed by run_single_random_split's own external subprocess
    timeout mid-epoch with nothing usable to show for it, unlike the
    `cluster` experiment's graceful stop). Not applied to run_single()'s
    official-split path - unasked-for scope, and that path already ran
    hicot without this before with no reported issue there."""
    from vaebm_benchmark.experiment.scientific_models import build_model

    if model_name != "hicot":
        return build_model(model_name, k, seed, voc_size, dataset_id=dataset_id)

    import os

    from vaebm_benchmark.experiment.scientific_models import build_hicot

    raw = os.environ.get("VAEBM_HICOT_MAX_FIT_SECONDS", "1200").strip().lower()
    max_fit_seconds = None if raw in ("0", "none", "") else float(raw)
    return build_hicot(k, seed, voc_size, dataset_id=dataset_id, max_fit_seconds=max_fit_seconds)


def run_single(
    model_name: str,
    dataset_id: str,
    k: int,
    seed: int = 42,
    voc_size: int = 5000,
    svm_kernel: str = "linear",
    svm_C: float = 1.0,
) -> ClassificationRunResult:
    from sklearn.metrics import accuracy_score, f1_score

    from vaebm_benchmark.datasets.definitions.hicot_datasets import load_hicot_split
    from vaebm_benchmark.experiment.scientific_models import build_model, representation_source_for_model
    from vaebm_benchmark.utils.seeding import set_all_seeds

    representation_source = representation_source_for_model(model_name)
    start = time.perf_counter()
    try:
        set_all_seeds(seed)
        train_docs, train_labels, test_docs, test_labels, _num_classes = load_hicot_split(dataset_id)

        model = build_model(model_name, k, seed, voc_size, dataset_id=dataset_id)
        model.fit(train_docs)  # labels never passed to fit()

        train_repr = _representation(model, train_docs, representation_source)
        test_repr = _representation(model, test_docs, representation_source)

        clf = _make_svm(svm_kernel, svm_C, seed)
        clf.fit(train_repr, train_labels)
        preds = clf.predict(test_repr)

        accuracy = float(accuracy_score(test_labels, preds))
        f1 = float(f1_score(test_labels, preds, average="macro"))

        runtime = time.perf_counter() - start
        return ClassificationRunResult(
            experiment="classification", model=model_name, dataset=dataset_id, k=k, seed=seed,
            accuracy=accuracy, f1=f1, representation_source=representation_source,
            num_train_docs=len(train_docs), num_test_docs=len(test_docs),
            runtime_seconds=runtime, status="ok",
        )
    except Exception as exc:  # noqa: BLE001 - one failed combination must not abort the whole sweep
        runtime = time.perf_counter() - start
        return ClassificationRunResult(
            experiment="classification", model=model_name, dataset=dataset_id, k=k, seed=seed,
            accuracy=None, f1=None, representation_source=representation_source,
            num_train_docs=0, num_test_docs=0,
            runtime_seconds=runtime, status="error", error=f"{exc}\n{traceback.format_exc(limit=3)}",
        )
    finally:
        # Best-effort GPU/accelerator memory release before the NEXT
        # (model, dataset, k, seed) combination tries to fit - see
        # utils/gpu_memory.py's own module docstring for why.
        from vaebm_benchmark.utils.gpu_memory import release_accelerator_memory

        try:
            del model
        except NameError:
            pass
        release_accelerator_memory()


def run_single_random_split(
    model_name: str,
    dataset_id: str,
    k: Optional[int],
    seed: int = 42,
    voc_size: int = 5000,
    test_size: float = 0.2,
    svm_kernel: str = "linear",
    svm_C: float = 1.0,
) -> ClassificationRunResult:
    """Same document -> representation -> SVM protocol as run_single()
    above, but for datasets/simple_registry.py's own FULL dataset list -
    the same ids experiment/cluster_runner.py's own `cluster` experiment
    sweeps over - rather than only hicot_* ids with an official split.
    The held-out test set is a stratified random 80/20 split we draw
    ourselves (reseeded per `seed`, like the topic model refit below),
    not a paper-provided one - so this is a distinct protocol from
    run_single()'s ECRTM/HiCOT Sec 4.4 reproduction, not a replacement
    for it (see run_experiment.py's own --split flag).

    k=None auto-derives K from the dataset's own num_classes (cluster's
    own convention, sensible here since this path runs over cluster's
    own broad, varied-class-count dataset list rather than the fixed
    50/100 topic counts run_single()'s ECRTM-reproduction protocol
    uses) - pass an explicit k to instead fix one topic count for every
    dataset."""
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    from vaebm_benchmark.datasets.simple_registry import load_dataset, resolve_dataset_id
    from vaebm_benchmark.experiment.scientific_models import representation_source_for_model
    from vaebm_benchmark.utils.seeding import set_all_seeds

    representation_source = representation_source_for_model(model_name)
    start = time.perf_counter()
    model = None
    stratified = True
    try:
        set_all_seeds(seed)
        resolved_id = resolve_dataset_id(dataset_id)
        documents, labels, num_classes = load_dataset(resolved_id)
        effective_k = k if k is not None else num_classes

        try:
            train_docs, test_docs, train_labels, test_labels = train_test_split(
                documents, labels, test_size=test_size, random_state=seed, stratify=labels,
            )
        except ValueError as split_exc:
            # Some datasets (e.g. "tweet") have singleton classes (exactly
            # 1 document) - a stratified split is mathematically
            # impossible for those (a class needs >=2 members to put
            # >=1 in EACH of train/test), not a transient error retrying
            # would fix. Fall back to a plain random split for this combo
            # only - flagged via split_stratified=False below, never
            # silent, rather than leaving every model's result on this
            # dataset as a permanent, avoidable error.
            if "least populated class" not in str(split_exc):
                raise
            stratified = False
            train_docs, test_docs, train_labels, test_labels = train_test_split(
                documents, labels, test_size=test_size, random_state=seed,
            )

        model = _build_model_for_classification(model_name, effective_k, seed, voc_size, resolved_id)
        model.fit(train_docs)  # labels never passed to fit()

        train_repr = _representation(model, train_docs, representation_source)
        test_repr = _representation(model, test_docs, representation_source)

        clf = _make_svm(svm_kernel, svm_C, seed)
        clf.fit(train_repr, train_labels)
        preds = clf.predict(test_repr)

        accuracy = float(accuracy_score(test_labels, preds))
        f1 = float(f1_score(test_labels, preds, average="macro"))

        runtime = time.perf_counter() - start
        return ClassificationRunResult(
            experiment="classification", model=model_name, dataset=dataset_id, k=effective_k, seed=seed,
            accuracy=accuracy, f1=f1, representation_source=representation_source,
            num_train_docs=len(train_docs), num_test_docs=len(test_docs),
            runtime_seconds=runtime, status="ok", split_stratified=stratified,
        )
    except Exception as exc:  # noqa: BLE001 - one failed combination must not abort the whole sweep
        runtime = time.perf_counter() - start
        return ClassificationRunResult(
            experiment="classification", model=model_name, dataset=dataset_id, k=k or 0, seed=seed,
            accuracy=None, f1=None, representation_source=representation_source,
            num_train_docs=0, num_test_docs=0,
            runtime_seconds=runtime, status="error", error=f"{exc}\n{traceback.format_exc(limit=3)}",
            split_stratified=stratified,
        )
    finally:
        from vaebm_benchmark.utils.gpu_memory import release_accelerator_memory

        del model
        release_accelerator_memory()


def run_sweep(
    models: list[str],
    datasets: list[str],
    ks: list[int],
    seeds: list[int],
    voc_size: int = 5000,
    svm_kernel: str = "linear",
    svm_C: float = 1.0,
    split_mode: str = "official",
    test_size: float = 0.2,
) -> list[ClassificationRunResult]:
    """Flat, per-(model, dataset, k, seed) result list - one run per
    combination. Aggregation (mean/std/CI across seeds) is a reporting
    concern - see classification_report.py::aggregate_classification_results,
    which consumes exactly this flat list. Every individual seed's result
    is still in the returned list and gets persisted.

    split_mode="official" (default, unchanged) uses run_single()'s
    hicot_*-only official train/test split. split_mode="random" uses
    run_single_random_split() instead - any dataset id, a stratified
    80/20 split drawn per seed. `ks` may contain `None` in random mode
    (meaning "auto-derive K from the dataset's own num_classes") -
    official mode requires real ints, since ECRTM-style K is never
    dataset-derived.

    Prints a one-line status per combination AS IT FINISHES, so progress
    is visible during a long run and partial results survive even if a
    later combination is interrupted."""
    if split_mode not in ("official", "random"):
        raise ValueError(f"Unknown split_mode '{split_mode}'. Expected 'official' or 'random'.")

    results = []
    total = len(ks) * len(datasets) * len(models) * len(seeds)
    count = 0
    for k in ks:
        for dataset_id in datasets:
            for model_name in models:
                for seed in seeds:
                    count += 1
                    if split_mode == "official":
                        result = run_single(model_name, dataset_id, k, seed=seed, voc_size=voc_size, svm_kernel=svm_kernel, svm_C=svm_C)
                    else:
                        result = run_single_random_split(
                            model_name, dataset_id, k, seed=seed, voc_size=voc_size,
                            test_size=test_size, svm_kernel=svm_kernel, svm_C=svm_C,
                        )
                    results.append(result)
                    k_label = k if k is not None else "auto"
                    if result.status == "ok":
                        print(f"[{count}/{total}] model={model_name} dataset={dataset_id} k={k_label} seed={seed}: ok "
                              f"accuracy={result.accuracy} f1={result.f1}", flush=True)
                    else:
                        print(f"[{count}/{total}] model={model_name} dataset={dataset_id} k={k_label} seed={seed}: ERROR "
                              f"{result.error.splitlines()[0]}", flush=True)
    return results
