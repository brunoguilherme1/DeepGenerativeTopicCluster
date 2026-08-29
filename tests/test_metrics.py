from vaebm_benchmark.metrics.clustering_quality import ari, nmi, purity
from vaebm_benchmark.metrics.topic_quality import (
    irbo,
    topic_diversity,
    topic_diversity_glocom,
)


def test_purity_perfect():
    assert purity([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0


def test_purity_worst_case_still_bounded():
    # every predicted cluster is a singleton -> purity is trivially 1.0,
    # which is exactly why this metric is never reported alone (see its
    # docstring) - this test documents that behavior, not a bug.
    assert purity([0, 1, 2, 3], [0, 0, 1, 1]) == 1.0


def test_nmi_identical_partitions():
    assert nmi([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0


def test_ari_random_partition_bounds():
    value = ari([0, 1, 0, 1], [0, 0, 1, 1])
    assert -1.0 <= value <= 1.0


def test_topic_diversity_no_overlap_is_one():
    topics = [["a", "b"], ["c", "d"]]
    assert topic_diversity(topics, top_n=2) == 1.0


def test_topic_diversity_full_overlap_is_low():
    topics = [["a", "b"], ["a", "b"]]
    assert topic_diversity(topics, top_n=2) == 0.5


def test_topic_diversity_glocom_differs_from_standard_on_repeats():
    # "a" repeats across both topics: standard TD still counts it once
    # (contributes to the unique-word numerator); glocom_td excludes it
    # entirely (frequency != 1) - the two must NOT agree here.
    topics = [["a", "b"], ["a", "c"]]
    standard = topic_diversity(topics, top_n=2)
    glocom = topic_diversity_glocom(topics, top_n=2)
    assert standard == 3 / 4
    assert glocom == 2 / 4
    assert standard != glocom


def test_irbo_identical_topics_is_zero():
    topics = [["a", "b", "c"], ["a", "b", "c"]]
    assert irbo(topics, top_n=3) == 0.0
