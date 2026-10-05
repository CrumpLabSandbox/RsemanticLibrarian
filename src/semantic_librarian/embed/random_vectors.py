"""Random environment vectors."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def ternary_vectors(
    n: int, dim: int, nonzeros: int, rng: np.random.Generator, chunk: int = 4096
) -> np.ndarray:
    """Sparse ternary random index vectors (R: ``sl_create_riv``).

    Each row has ``nonzeros / 2`` elements equal to +1 and as many equal to -1 at random
    positions; all other elements are 0. Returned as ``int8`` to save memory.
    """
    if nonzeros % 2 or nonzeros <= 0:
        raise ValueError("nonzeros must be a positive even integer")
    if nonzeros > dim:
        raise ValueError("nonzeros cannot exceed dim")
    out = np.zeros((n, dim), dtype=np.int8)
    half = nonzeros // 2
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        positions = np.argpartition(rng.random((stop - start, dim)), nonzeros - 1, axis=1)
        positions = positions[:, :nonzeros]
        rows = np.arange(stop - start)[:, None]
        out[start + rows, positions[:, :half]] = 1
        out[start + rows, positions[:, half:]] = -1
    return out


def ternary_sparse(n: int, dim: int, nonzeros: int, rng: np.random.Generator) -> sp.csr_matrix:
    """The same kind of vectors as :func:`ternary_vectors`, stored sparsely (float32)."""
    if nonzeros % 2 or nonzeros <= 0:
        raise ValueError("nonzeros must be a positive even integer")
    if nonzeros > dim:
        raise ValueError("nonzeros cannot exceed dim")
    positions = np.empty((n, nonzeros), dtype=np.int64)
    for start in range(0, n, 4096):
        stop = min(start + 4096, n)
        draws = rng.random((stop - start, dim))
        positions[start:stop] = np.argpartition(draws, nonzeros - 1, axis=1)[:, :nonzeros]
    signs = np.tile(np.r_[np.ones(nonzeros // 2), -np.ones(nonzeros // 2)], n)
    indptr = np.arange(0, n * nonzeros + 1, nonzeros)
    m = sp.csr_matrix((signs.astype(np.float32), positions.ravel(), indptr), shape=(n, dim))
    m.sort_indices()
    return m


def gaussian_vectors(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Dense random vectors with elements drawn from N(0, 1/dim) (BEAGLE's environment)."""
    return rng.normal(0.0, 1.0 / np.sqrt(dim), size=(n, dim)).astype(np.float32)


def environment_vectors(
    kind: str, n: int, dim: int, nonzeros: int, rng: np.random.Generator
) -> np.ndarray:
    if kind == "gaussian":
        return gaussian_vectors(n, dim, rng)
    if kind == "ternary":
        return ternary_vectors(n, dim, nonzeros, rng)
    raise ValueError(f"unknown environment vector kind {kind!r}; use 'gaussian' or 'ternary'")
