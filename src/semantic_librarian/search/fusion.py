"""Combining rankings from different scorers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

RRF_K = 60


def ranks(scores: np.ndarray) -> np.ndarray:
    """1-based rank of every item by descending score (ties broken by position)."""
    order = np.argsort(-scores, kind="stable")
    out = np.empty(len(scores), dtype=np.float64)
    out[order] = np.arange(1, len(scores) + 1)
    return out


def reciprocal_rank_fusion(
    scores: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
    eligible: Sequence[np.ndarray | None] | None = None,
    k: int = RRF_K,
) -> np.ndarray:
    """Weighted reciprocal rank fusion (Cormack, Clarke & Buettcher, 2009).

    Each scorer contributes ``weight / (k + rank)`` to an item it ranks. Items a scorer
    considers ineligible (e.g. documents sharing no word with a keyword query) get no
    contribution from it, so they cannot be pulled up by a meaningless rank.
    """
    weights = list(weights) if weights is not None else [1.0] * len(scores)
    eligible = list(eligible) if eligible is not None else [None] * len(scores)
    if not (len(weights) == len(eligible) == len(scores)):
        raise ValueError("scores, weights and eligible must have the same length")
    fused = np.zeros(len(scores[0]), dtype=np.float64)
    for s, w, ok in zip(scores, weights, eligible, strict=True):
        contribution = w / (k + ranks(np.asarray(s, dtype=np.float64)))
        if ok is not None:
            contribution = np.where(ok, contribution, 0.0)
        fused += contribution
    return fused
