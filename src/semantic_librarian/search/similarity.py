"""Cosine similarity and the three query styles of the original interface.

``compound``
    Sum the query's word vectors and rank by cosine with that sum (paper Eq. 9-10).
``and``
    Rank by the product of the cosines with each query word: items must be close to
    every word.
``or``
    Divide each word's cosines by the absolute value of their maximum, then take the
    maximum over words: items close to any one word rank highly.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

QueryMode = Literal["compound", "and", "or"]
QUERY_MODES: tuple[str, ...] = ("compound", "and", "or")


def cosine_scores(
    query: np.ndarray, matrix: np.ndarray, matrix_norms: np.ndarray | None = None
) -> np.ndarray:
    """Cosine between ``query`` (one vector, or several as rows) and every row of ``matrix``.

    Zero vectors get a similarity of 0 instead of NaN.
    """
    q = np.atleast_2d(np.asarray(query, dtype=np.float64))
    if matrix_norms is None:
        matrix_norms = np.linalg.norm(matrix, axis=1)
    qn = np.linalg.norm(q, axis=1)
    dots = q @ np.asarray(matrix, dtype=np.float64).T
    denom = np.outer(qn, matrix_norms)
    with np.errstate(invalid="ignore", divide="ignore"):
        sims = np.where(denom > 0, dots / np.where(denom > 0, denom, 1.0), 0.0)
    return sims[0] if np.ndim(query) == 1 else sims


def combine_term_scores(term_scores: np.ndarray, mode: QueryMode) -> np.ndarray:
    """Combine per-term cosines, shaped ``(n_terms, n_items)``, for ``and`` / ``or`` queries.

    A single-term query returns that term's cosines unchanged; the R package returned
    nothing in that case.
    """
    term_scores = np.atleast_2d(term_scores)
    if term_scores.shape[0] == 1:
        return term_scores[0].copy()
    if mode == "and":
        return np.prod(term_scores, axis=0)
    if mode == "or":
        scale = np.abs(term_scores.max(axis=1, keepdims=True))
        scale[scale == 0] = 1.0
        return (term_scores / scale).max(axis=0)
    raise ValueError(f"mode must be 'and' or 'or' here, got {mode!r}")


def rank(scores: np.ndarray, k: int | None = None, mask: np.ndarray | None = None) -> np.ndarray:
    """Indices of the highest scores, best first (stable for ties), optionally masked."""
    candidates = np.arange(len(scores)) if mask is None else np.flatnonzero(mask)
    order = candidates[np.argsort(-scores[candidates], kind="stable")]
    return order if k is None else order[:k]
