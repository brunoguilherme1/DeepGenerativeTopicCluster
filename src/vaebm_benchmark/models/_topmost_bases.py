"""Shared plumbing for topmost-based training (used by both the GloCOM
adapter and, indirectly, FASTopic via its own dependency on topmost's
Preprocess). topmost.BasicDataset loads bow/vocab arrays from a directory
of files on disk; we already hold preprocessed in-memory bow/vocab from
topmost.Preprocess.preprocess(), so _BowDataset below duck-types the three
attributes topmost.BasicTrainer.train()/test()/get_top_words() actually
touch (`.train_dataloader`, `.train_data`, `.vocab`) and skips disk I/O.
"""

from __future__ import annotations

import numpy as np

DEFAULT_VOCAB_SIZE_CAP = 10_000


def run_preprocess(
    documents: list[str],
    vocab_size_cap: int = DEFAULT_VOCAB_SIZE_CAP,
    seed: int = 42,
    verbose: bool = False,
):
    """Runs topmost's own preprocessing (tokenize, frequency-capped vocab,
    BOW-vectorize) once at fit time. Returns the Preprocess instance itself
    (needed to vectorize new documents against the same fixed vocab later),
    the vocab, the dense train BOW, and the tokenized/vocab-filtered
    train texts."""
    from topmost import Preprocess

    preprocess = Preprocess(vocab_size=vocab_size_cap, seed=seed, verbose=verbose)
    rst = preprocess.preprocess(documents)
    train_bow = rst["train_bow"]
    if hasattr(train_bow, "toarray"):
        train_bow = train_bow.toarray()
    return preprocess, rst["vocab"], train_bow, rst["train_texts"]


def vectorize_against_vocab(preprocess, documents: list[str], vocab: list[str]) -> np.ndarray:
    """Same tokenizer/vocab used at fit time, applied to (possibly unseen)
    documents - never re-fit through Preprocess.preprocess() itself, which
    would leak the new documents' own vocabulary into the model."""
    _, sparse_bow = preprocess.parse(documents, vocab)
    return sparse_bow.toarray()


def resolve_device(explicit: str = None):
    import torch

    if explicit is not None:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BowDataset:
    """Minimal duck-typed stand-in for topmost.BasicDataset."""

    def __init__(self, train_bow: np.ndarray, vocab: list[str], device, batch_size: int = 200):
        import torch
        from torch.utils.data import DataLoader

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.train_data = torch.from_numpy(train_bow.astype("float32")).to(device)
        self.train_dataloader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
