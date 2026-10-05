"""Vector spaces for grouping variables (authors, years, chapters...)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from semantic_librarian.space.vector_space import VectorSpace


def factor_space(
    name: str,
    doc_levels: Sequence[Sequence[str]],
    doc_vectors: np.ndarray,
    how: Literal["sum", "mean"] = "sum",
) -> VectorSpace:
    """Build a space whose items are the levels of a factor.

    Each level's vector is the sum (paper Eq. 11) or mean of the vectors of the documents
    carrying that level. Levels are matched exactly; the legacy R code used a regular
    expression, so "Smith, J." also matched "Smith, John".

    Args:
        name: Name of the resulting space (e.g. ``"author"``).
        doc_levels: For each document row, the levels it carries (possibly none).
        doc_vectors: Document vectors, one row per document.
        how: ``"sum"`` (paper) or ``"mean"`` (insensitive to how prolific a level is).
    """
    if len(doc_levels) != doc_vectors.shape[0]:
        raise ValueError("doc_levels must have one entry per document vector")
    level_index: dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    for doc_row, levels in enumerate(doc_levels):
        for level in dict.fromkeys(levels):
            j = level_index.setdefault(level, len(level_index))
            rows.append(j)
            cols.append(doc_row)
    out = np.zeros((len(level_index), doc_vectors.shape[1]), dtype=np.float64)
    if rows:
        np.add.at(out, np.asarray(rows), np.asarray(doc_vectors[np.asarray(cols)], np.float64))
    if how == "mean" and rows:
        counts = np.bincount(np.asarray(rows), minlength=len(level_index))
        out /= counts[:, None]
    return VectorSpace(name, list(level_index), out)
