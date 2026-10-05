"""A labelled matrix of vectors living in a shared coordinate system."""

from __future__ import annotations

import json
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path

import numpy as np


class VectorSpace:
    """Rows of ``vectors`` are items (words, documents, authors...) named by ``labels``.

    All spaces produced by one build share coordinates, so any item can be compared with
    any other item, in the same or in a different space.
    """

    def __init__(self, name: str, labels: Sequence[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors)
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2-D matrix")
        if len(labels) != vectors.shape[0]:
            raise ValueError(
                f"{len(labels)} labels for {vectors.shape[0]} vectors in space {name!r}"
            )
        self.name = name
        self.labels: list[str] = [str(label) for label in labels]
        self.vectors = vectors

    def __len__(self) -> int:
        return len(self.labels)

    def __repr__(self) -> str:
        return f"VectorSpace({self.name!r}, items={len(self)}, dim={self.dim})"

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @cached_property
    def index(self) -> dict[str, int]:
        return {label: i for i, label in enumerate(self.labels)}

    def __contains__(self, label: object) -> bool:
        return label in self.index

    def row(self, label: str) -> int:
        try:
            return self.index[label]
        except KeyError:
            raise KeyError(f"{label!r} is not in the {self.name!r} space") from None

    def vector(self, label: str) -> np.ndarray:
        return self.vectors[self.row(label)]

    @cached_property
    def norms(self) -> np.ndarray:
        return np.sqrt(np.einsum("ij,ij->i", self.vectors, self.vectors, dtype=np.float64))

    def save(self, directory: Path, dtype: str = "float32") -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / f"{self.name}.npy", self.vectors.astype(dtype, copy=False))
        (directory / f"{self.name}.labels.json").write_text(
            json.dumps(self.labels, ensure_ascii=False)
        )

    @classmethod
    def load(cls, directory: Path, name: str, mmap: bool = True) -> VectorSpace:
        vectors = np.load(directory / f"{name}.npy", mmap_mode="r" if mmap else None)
        labels = json.loads((directory / f"{name}.labels.json").read_text())
        return cls(name, labels, vectors)
