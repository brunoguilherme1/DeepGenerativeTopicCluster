"""Shared BoW+embedding preprocessing for the VAE-BM variants
(vaebm_poe.py, vaebm_dec.py) - NOT imported by models/vaebm.py itself,
whose own `VaeBmKMeansFit.fit_predict()` keeps its own inline copy of
this exact logic untouched, per that module's own "copied here verbatim"
convention. This is a fresh re-implementation matching it 1:1 (same
TfidfVectorizer(norm=None) + log1p, same CountVectorizer fallback, same
SentenceTransformer embedding call) so the two new variants preprocess
identically to the original, without editing the original file.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore


def build_bow_and_embeddings(
    texts: Sequence[str],
    voc_size: int,
    vectorizer_type: str = "tfidf",
    embedder: Union[str, np.ndarray] = "all-MiniLM-L6-v2",
    vocabulary: Optional[Sequence[str]] = None,
):
    """Returns (X_bow [N,V] float32, E [N,d] float32, vectorizer, embedder_obj_or_None).
    Mirrors vaebm.py::VaeBmKMeansFit.fit_predict()'s own preprocessing
    exactly - see that module's docstring for why `vocabulary`, when
    given, forces `tokenizer=str.split, lowercase=False`."""
    if vocabulary is not None:
        vectorizer_kwargs = {"vocabulary": list(vocabulary), "tokenizer": str.split, "lowercase": False, "token_pattern": None}
    else:
        vectorizer_kwargs = {"max_features": voc_size}

    if vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(norm=None, **vectorizer_kwargs)
        X = vectorizer.fit_transform(texts)
        X = X.tocoo(copy=False)
        X.data = np.log1p(X.data)
        X = X.tocsr()
    else:
        if vocabulary is None:
            vectorizer_kwargs["stop_words"] = "english"
        vectorizer = CountVectorizer(**vectorizer_kwargs)
        X = vectorizer.fit_transform(texts)
    X_bow = X.toarray().astype(np.float32)

    embedder_obj = None
    if isinstance(embedder, str):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not available.")
        embedder_obj = SentenceTransformer(embedder)
        E = embedder_obj.encode(texts, batch_size=32, convert_to_numpy=True).astype(np.float32)
    else:
        E = np.asarray(embedder, dtype=np.float32)

    return X_bow, E, vectorizer, embedder_obj


def transform_bow_and_embeddings(texts: Sequence[str], vectorizer, embedder_obj) -> tuple[np.ndarray, np.ndarray]:
    """predict()-time counterpart of build_bow_and_embeddings() - transforms
    new documents through an already-fitted vectorizer/embedder."""
    X = vectorizer.transform(texts)
    if isinstance(vectorizer, TfidfVectorizer):
        X = X.tocoo(copy=False)
        X.data = np.log1p(X.data)
        X = X.tocsr()
    X_bow = X.toarray().astype(np.float32)

    if embedder_obj is None:
        raise RuntimeError("Embedder was not initialized.")
    E = embedder_obj.encode(texts, batch_size=128, convert_to_numpy=True).astype(np.float32)
    return X_bow, E
