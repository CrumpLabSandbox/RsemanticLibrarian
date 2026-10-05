"""Classical multidimensional scaling and k-means, as used for the interface's plots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semantic_librarian.search.similarity import cosine_scores


@dataclass
class Projection:
    """2-D coordinates and cluster assignments for a set of items."""

    labels: list[str]
    coords: np.ndarray  # (n, 2)
    clusters: np.ndarray  # (n,), 0-based


def classical_mds(vectors: np.ndarray, k: int = 2) -> np.ndarray:
    """Classical (Torgerson) MDS of cosine distances, matching R's
    ``cmdscale(1 - cosine(t(vectors)), k = k)``.

    Eigenvector signs are arbitrary, so coordinates may be mirrored relative to R.
    """
    n = vectors.shape[0]
    if n == 0:
        return np.zeros((0, k))
    if n == 1:
        return np.zeros((1, k))
    sims = cosine_scores(vectors, vectors)
    d2 = (1.0 - sims) ** 2
    centre = np.eye(n) - np.full((n, n), 1.0 / n)
    b = -0.5 * centre @ d2 @ centre
    values, vecs = np.linalg.eigh((b + b.T) / 2)
    top = np.argsort(values)[::-1][:k]
    values = np.clip(values[top], 0.0, None)
    coords = vecs[:, top] * np.sqrt(values)
    if coords.shape[1] < k:
        coords = np.pad(coords, ((0, 0), (0, k - coords.shape[1])))
    return coords


def cluster(coords: np.ndarray, n_clusters: int, seed: int = 0) -> np.ndarray:
    """k-means on the projected coordinates (as the original interface did)."""
    n = coords.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    n_clusters = max(1, min(n_clusters, n))
    if n_clusters == 1:
        return np.zeros(n, dtype=np.int64)
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    return np.asarray(model.fit_predict(coords), dtype=np.int64)


def project(
    labels: list[str], vectors: np.ndarray, n_clusters: int = 3, seed: int = 0
) -> Projection:
    """MDS map plus k-means clusters for the given items."""
    coords = classical_mds(np.asarray(vectors, dtype=np.float64))
    return Projection(list(labels), coords, cluster(coords, n_clusters, seed))
