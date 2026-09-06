"""Downstream text-classification experiment (--experiment classification):

    document -> model representation (theta or mu) -> SVM -> labels

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
    raise ValueError(f"Unknown representation_source '{representation_source}'")


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
    from sklearn.svm import SVC

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

        clf = SVC(kernel=svm_kernel, C=svm_C, random_state=seed)
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


def run_sweep(
    models: list[str],
    datasets: list[str],
    ks: list[int],
    seeds: list[int],
    voc_size: int = 5000,
    svm_kernel: str = "linear",
    svm_C: float = 1.0,
) -> list[ClassificationRunResult]:
    """Flat, per-(model, dataset, k, seed) result list - one run per
    combination. Aggregation (mean/std/CI across seeds) is a reporting
    concern - see classification_report.py::aggregate_classification_results,
    which consumes exactly this flat list. Every individual seed's result
    is still in the returned list and gets persisted.

    Prints a one-line status per combination AS IT FINISHES, so progress
    is visible during a long run and partial results survive even if a
    later combination is interrupted."""
    results = []
    total = len(ks) * len(datasets) * len(models) * len(seeds)
    count = 0
    for k in ks:
        for dataset_id in datasets:
            for model_name in models:
                for seed in seeds:
                    count += 1
                    result = run_single(model_name, dataset_id, k, seed=seed, voc_size=voc_size, svm_kernel=svm_kernel, svm_C=svm_C)
                    results.append(result)
                    if result.status == "ok":
                        print(f"[{count}/{total}] model={model_name} dataset={dataset_id} k={k} seed={seed}: ok "
                              f"accuracy={result.accuracy} f1={result.f1}", flush=True)
                    else:
                        print(f"[{count}/{total}] model={model_name} dataset={dataset_id} k={k} seed={seed}: ERROR "
                              f"{result.error.splitlines()[0]}", flush=True)
    return results
