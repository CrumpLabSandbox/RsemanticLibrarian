"""Fast context and permutation-order sums via sparse matrix products.

Context information is linear in the environment vectors: a word's context memory is
the sum, over its occurrences, of the other words in the sentence. Writing the corpus as
a word-by-sentence incidence matrix ``A`` and a sentence-by-word matrix ``B`` of
contributing (non-stop) words, the whole context memory is ``A @ (B @ E)`` minus each
word's own contribution. BEAGLE-RP's order information is linear too: for each offset
``k`` it is ``M_k @ roll(E, k)``, where ``M_k`` counts how often word ``v`` occurs ``k``
positions after word ``w``. Both replace a Python loop over every token.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp


def _eligible(sentence_ids: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [np.asarray(s, dtype=np.int64) for s in sentence_ids if len(s) >= 2]


def _dense(m: object) -> np.ndarray:
    if sp.issparse(m):
        m = m.toarray()  # type: ignore[attr-defined]
    return np.asarray(m, dtype=np.float32)


def context_sums(
    sentence_ids: Sequence[np.ndarray],
    env: np.ndarray | sp.csr_matrix,
    keep: np.ndarray,
    chunk: int = 4096,
) -> np.ndarray:
    """Context memory: for every occurrence of a word in a sentence of two or more words,
    add the environment vectors of the sentence's other kept words (paper Eq. 1).

    Args:
        sentence_ids: Vocabulary indices of each sentence.
        env: ``(vocab, dim)`` environment vectors, dense or sparse.
        keep: Boolean mask over the vocabulary; ``False`` words (stop words) receive
            context but do not contribute to anyone's.
        chunk: Sentences processed per sparse product (bounds memory).
    """
    n_vocab, dim = env.shape
    out = np.zeros((n_vocab, dim), dtype=np.float32)
    occurrences = np.zeros(n_vocab, dtype=np.float64)
    keep_f = keep.astype(np.float32)
    sents = _eligible(sentence_ids)
    for start in range(0, len(sents), chunk):
        part = sents[start : start + chunk]
        cols = np.concatenate(part)
        rows = np.repeat(np.arange(len(part)), [len(s) for s in part])
        recipients = sp.csr_matrix(
            (np.ones(len(cols), dtype=np.float32), (cols, rows)), shape=(n_vocab, len(part))
        )
        contributors = sp.csr_matrix((keep_f[cols], (rows, cols)), shape=(len(part), n_vocab))
        totals = _dense(contributors @ env)
        out += recipients @ totals
        occurrences += np.bincount(cols, minlength=n_vocab)
    self_weight = (occurrences * keep).astype(np.float32)
    out -= _dense(sp.diags(self_weight) @ env) if sp.issparse(env) else self_weight[:, None] * env
    return out


def roll_columns(env: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """``np.roll(env, k, axis=1)`` for a sparse matrix."""
    rolled = env.copy()
    rolled.indices = (rolled.indices + k) % env.shape[1]
    rolled.has_sorted_indices = False
    return rolled


def permutation_order_sums(
    sentence_ids: Sequence[np.ndarray], env: sp.csr_matrix, window: int
) -> np.ndarray:
    """BEAGLE-RP order memory: each word accumulates the environment vectors of the words
    within ``window`` positions in its sentence, rotated by their offset."""
    n_vocab, dim = env.shape
    out = np.zeros((n_vocab, dim), dtype=np.float32)
    sents = _eligible(sentence_ids)
    if not sents:
        return out
    flat = np.concatenate(sents)
    lengths = np.array([len(s) for s in sents])
    starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
    pos = np.arange(len(flat)) - starts
    length_of = np.repeat(lengths, lengths)
    for k in range(-window, window + 1):
        if k == 0:
            continue
        valid = (pos + k >= 0) & (pos + k < length_of)
        idx = np.flatnonzero(valid)
        pairs = sp.csr_matrix(
            (np.ones(len(idx), dtype=np.float32), (flat[idx], flat[idx + k])),
            shape=(n_vocab, n_vocab),
        )
        out += _dense(pairs @ roll_columns(env, k))
    return out
