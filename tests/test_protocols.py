import pytest

from vaebm_benchmark.evaluation.registry import get_protocol, list_protocols
from vaebm_benchmark.protocols.base import BaselineProtocol


def test_list_protocols():
    assert set(list_protocols()) == {"fastopic", "glocom"}


def test_unknown_protocol_raises():
    with pytest.raises(KeyError):
        get_protocol("not_a_real_protocol")


@pytest.mark.parametrize("name", ["fastopic", "glocom"])
def test_protocol_is_baseline_protocol(name):
    protocol = get_protocol(name)
    assert isinstance(protocol, BaselineProtocol)
    assert protocol.name == name
    assert protocol.paper
    assert protocol.official_repository.startswith("https://github.com/")


@pytest.mark.parametrize("name", ["fastopic", "glocom"])
def test_smoke_test_reduces_epochs_below_paper_default(name):
    smoke = get_protocol(name, smoke_test=True)
    full = get_protocol(name, smoke_test=False)
    assert smoke.epochs < full.epochs
    assert full.epochs == 200


@pytest.mark.parametrize("name", ["fastopic", "glocom"])
def test_verify_report_has_required_sections(name):
    protocol = get_protocol(name)
    report = protocol.verify()
    for key in ["paper", "official_repository", "dataset", "preprocessing", "vocabulary", "K", "seeds", "metrics"]:
        assert key in report, f"{name} protocol.verify() is missing '{key}'"


@pytest.mark.parametrize("name", ["fastopic", "glocom"])
def test_k_never_tuned_independently_for_vaebm(name):
    """The core rule this repo exists to enforce: VAE-BM's K always
    matches the baseline's K for the same dataset, per protocol."""
    protocol = get_protocol(name)
    for dataset_id, k in protocol.topic_count.items():
        baseline = protocol.build_baseline(dataset_id, seed=42)
        vaebm = protocol.build_vaebm(dataset_id, seed=42)
        baseline_core = getattr(baseline, "_inner", baseline)  # unwrap _GloCOMOfficialArtifactAdapter, if present
        assert baseline_core.num_topics == k
        assert vaebm.n_clusters == k
