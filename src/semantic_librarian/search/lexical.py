"""Keyword scoring over a library's own tokens.

These scorers use the same tokenization and vocabulary as the semantic models, so a
query means the same thing to both, and every document gets a score (needed to fuse
rankings and to evaluate methods side by side).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from semantic_librarian.space.vocabulary import Vocabulary


def doc_term_matrix(doc_tokens: Sequence[Sequence[str]], vocab: Vocabulary) -> sp.csr_matrix:
    """Documents x vocabulary matrix of token counts."""
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for r, tokens in enumerate(doc_tokens):
        ids = vocab.ids(tokens)
        rows.append(np.full(len(ids), r, dtype=np.int64))
        cols.append(ids)
    row = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    col = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    m = sp.csr_matrix(
        (np.ones(len(row), dtype=np.float32), (row, col)), shape=(len(doc_tokens), len(vocab))
    )
    m.sum_duplicates()
    return m


class BM25:
    """Okapi BM25 (Robertson et al.) over a document-term count matrix.

    ``idf = log(1 + (N - df + 0.5) / (df + 0.5))``; each distinct query term contributes
    ``idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len))``.
    """

    def __init__(self, doc_terms: sp.spmatrix | sp.sparray, k1: float = 1.2, b: float = 0.75):
        self.doc_terms = sp.csc_matrix(doc_terms, dtype=np.float32)
        self.k1 = k1
        self.b = b
        n_docs = self.doc_terms.shape[0]
        lengths = np.asarray(self.doc_terms.sum(axis=1)).ravel()
        avg = lengths.mean() if n_docs and lengths.mean() > 0 else 1.0
        self.norm = k1 * (1 - b + b * lengths / avg)
        df = np.diff(self.doc_terms.indptr)
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))

    @property
    def n_docs(self) -> int:
        return int(self.doc_terms.shape[0])

    def scores(self, term_ids: Sequence[int] | np.ndarray) -> np.ndarray:
        """BM25 score of every document for the given (vocabulary-index) query terms."""
        out = np.zeros(self.n_docs, dtype=np.float64)
        for t in np.unique(np.asarray(term_ids, dtype=np.int64)):
            start, stop = self.doc_terms.indptr[t], self.doc_terms.indptr[t + 1]
            docs = self.doc_terms.indices[start:stop]
            tf = self.doc_terms.data[start:stop].astype(np.float64)
            out[docs] += self.idf[t] * tf * (self.k1 + 1) / (tf + self.norm[docs])
        return out

    def save(self, path: Path) -> None:
        sp.save_npz(path, sp.csr_matrix(self.doc_terms))

    @classmethod
    def load(cls, path: Path) -> BM25:
        return cls(sp.load_npz(path))


def word_match_scores(
    doc_terms: sp.spmatrix | sp.sparray,
    term_ids: np.ndarray,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """The paper's word-match control: the document ranked first has "the largest word
    overlap" with the query, i.e. the most distinct query words.

    Args:
        doc_terms: Documents x vocabulary counts.
        term_ids: Vocabulary indices of the query words (repeats are ignored).
        exclude: Vocabulary indices to ignore, normally stop words. Without this the
            count is dominated by "the", "of" and "and", which favors long documents
            (median rank of about 200-370 on the APA corpus instead of about 1).
            A query made only of excluded words keeps them.
    """
    n_vocab = doc_terms.shape[1]
    ids = np.unique(np.asarray(term_ids, dtype=np.int64))
    if exclude is not None and len(exclude):
        content = np.setdiff1d(ids, exclude)
        ids = content if len(content) else ids
    q = np.zeros(n_vocab, dtype=np.float64)
    q[ids] = 1.0
    present = sp.csr_matrix(doc_terms, copy=True)
    present.data[:] = 1.0
    return np.asarray(present @ q).ravel()
