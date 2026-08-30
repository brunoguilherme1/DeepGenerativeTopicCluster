"""Regression test for the vocabulary-equivalence fix in vaebm.py's
fit_predict(): when a baseline's exact vocabulary is pinned via
`vocabulary=`, the vectorizer must ALSO use simple whitespace tokenization
(matching how official artifacts like GloCOM's texts.txt/vocab.txt or
FASTopic's topmost train_texts.txt/vocab.txt were themselves built) -
not sklearn's default regex tokenizer, which silently drops
single-character tokens and could undercount relative to the official
artifact even with the same vocabulary list.

This mirrors exactly the vectorizer construction in
models/vaebm.py::VaeBmKMeansFit.fit_predict when `vocabulary` is given,
without needing to run the full (TensorFlow-dependent) training pipeline.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer


def _released_artifact_vectorizer(vocabulary):
    return CountVectorizer(vocabulary=vocabulary, tokenizer=str.split, lowercase=False, token_pattern=None)


def test_single_character_vocabulary_word_is_not_silently_dropped():
    # sklearn's default token_pattern (\b\w\w+\b) requires >=2 word
    # characters and would drop the single-char word "a" entirely.
    vocab = ["a", "cat", "sat"]
    texts = ["a cat sat", "a a cat"]

    vectorizer = _released_artifact_vectorizer(vocab)
    counts = vectorizer.fit_transform(texts).toarray()

    # Manually count via the exact same rule the official artifacts use:
    # split on whitespace, no case-folding, no length filtering.
    expected = [[sum(1 for tok in text.split() if tok == word) for word in vocab] for text in texts]
    assert counts.tolist() == expected


def test_default_sklearn_tokenizer_would_have_undercounted(monkeypatch=None):
    """Documents the exact failure mode being guarded against: the
    DEFAULT sklearn tokenizer (no override) drops single-char tokens,
    which the fixed configuration above does not."""
    vocab = ["a", "cat"]
    texts = ["a cat"]

    default_vectorizer = CountVectorizer(vocabulary=vocab)  # no tokenizer override - the old, buggy configuration
    default_counts = default_vectorizer.fit_transform(texts).toarray()
    assert default_counts.tolist() == [[0, 1]]  # "a" silently undercounted to 0

    fixed_vectorizer = _released_artifact_vectorizer(vocab)
    fixed_counts = fixed_vectorizer.fit_transform(texts).toarray()
    assert fixed_counts.tolist() == [[1, 1]]  # "a" correctly counted


def test_case_is_preserved_not_refolded():
    vocab = ["Cat", "cat"]
    texts = ["Cat cat Cat"]

    vectorizer = _released_artifact_vectorizer(vocab)
    counts = vectorizer.fit_transform(texts).toarray()

    assert counts.tolist() == [[2, 1]]
