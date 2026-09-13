"""Raw-text preprocessing utilities shared across experiment runners -
kept separate from any single model adapter since it's applied BEFORE a
document list reaches `model.fit()`, not as part of a model's own
internal vectorizer.

`remove_english_stopwords()` is used by experiment/cluster_runner.py for
exactly one case: `model=hicot` on a NON-`hicot_*` dataset (HiCOTAdapter's
own generic/self-fit path builds its vocabulary via a plain
`CountVectorizer(max_features=voc_size)` with no stopword filtering at
all - see models/hicot_adapter.py's own module docstring - unlike
lda_adapter.py/vaebm.py, which already pass `stop_words="english"` to
their own vectorizers). It is NEVER applied to `hicot_*` datasets (those
use HiCOT's own official artifacts verbatim, no extra preprocessing
pass) and NEVER to any other model (sbert_kmeans/BERTopic/FASTopic keep
their own existing text handling unchanged).

Uses scikit-learn's own `ENGLISH_STOP_WORDS` list (already a project
dependency via every vectorizer-based adapter) rather than adding NLTK or
another stopword corpus as a new dependency.
"""

from __future__ import annotations


def remove_english_stopwords(documents: list[str]) -> list[str]:
    """Returns a NEW list, same length and order as `documents` (never
    drops a document, even if removing its stopwords leaves it empty -
    an empty string in that slot preserves alignment with labels/index),
    with English stopwords removed word-by-word."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    cleaned = []
    for doc in documents:
        words = [w for w in doc.split() if w.lower() not in ENGLISH_STOP_WORDS]
        cleaned.append(" ".join(words))
    return cleaned
