"""Lightweight interface/unit tests for the cluster experiment's model
registry - proves every registered model (including fastopic/glocom,
which this project's own instructions say NOT to actually train during
this task) can be CONSTRUCTED and exposes the two capability methods the
generic runner relies on (`fit`, `get_document_clusters`) - no training,
no dataset download, no network access.
"""

from vaebm_benchmark.experiment.cluster_runner import CLUSTER_MODEL_BUILDERS, list_cluster_models


def test_all_four_models_are_registered():
    assert set(list_cluster_models()) == {"vaebm", "bertopic", "fastopic", "glocom"}


def test_registry_is_a_plain_dict_not_branching_logic():
    """The generic runner (run_single) only ever does a dict lookup by
    model name - this test documents/pins that contract."""
    assert isinstance(CLUSTER_MODEL_BUILDERS, dict)
    for name, builder in CLUSTER_MODEL_BUILDERS.items():
        assert callable(builder), f"builder for '{name}' is not callable"


def test_vaebm_builder_exposes_required_capabilities():
    from vaebm_benchmark.experiment.cluster_runner import _build_vaebm

    model = _build_vaebm(k=8, seed=42, voc_size=2000)
    assert hasattr(model, "fit") and callable(model.fit)
    assert hasattr(model, "get_document_clusters") and callable(model.get_document_clusters)
    assert model.n_clusters == 8


def test_bertopic_builder_exposes_required_capabilities():
    from vaebm_benchmark.experiment.cluster_runner import _build_bertopic

    model = _build_bertopic(k=8, seed=42, voc_size=2000)
    assert hasattr(model, "fit") and callable(model.fit)
    assert hasattr(model, "get_document_clusters") and callable(model.get_document_clusters)
    assert model.n_clusters == 8


def test_fastopic_builder_exposes_required_capabilities_without_training():
    """Constructs the adapter only - never calls .fit(), per this task's
    own instruction not to actually run FASTopic here."""
    from vaebm_benchmark.experiment.cluster_runner import _build_fastopic

    model = _build_fastopic(k=8, seed=42, voc_size=2000)
    assert hasattr(model, "fit") and callable(model.fit)
    assert hasattr(model, "get_document_clusters") and callable(model.get_document_clusters)
    assert model.num_topics == 8
    # Generic (non-protocol-pinned) path: no released artifact injected.
    assert model.released_train_bow is None
    assert model.released_vocab is None


def test_glocom_builder_exposes_required_capabilities_without_training():
    """Constructs the adapter only - never calls .fit(), per this task's
    own instruction not to actually run GloCOM here."""
    from vaebm_benchmark.experiment.cluster_runner import _build_glocom

    model = _build_glocom(k=8, seed=42, voc_size=2000)
    assert hasattr(model, "fit") and callable(model.fit)
    assert hasattr(model, "get_document_clusters") and callable(model.get_document_clusters)
    assert model.num_topics == 8


def test_unknown_model_raises_in_run_single_not_at_import_time():
    from vaebm_benchmark.experiment.cluster_runner import run_single

    result = run_single("not_a_real_model", "search_snippets", seed=42)
    assert result.status == "error"
    assert "Unknown model" in result.error
