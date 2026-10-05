"""Token vocabulary with stable row indices."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np


class Vocabulary:
    """Maps tokens to row indices in first-occurrence order, with corpus counts.

    First-occurrence order matches R's ``unique()`` and so the legacy dictionary.
    """

    def __init__(self, tokens: Sequence[str], counts: Sequence[int] | None = None) -> None:
        self.tokens: list[str] = list(tokens)
        self.index: dict[str, int] = {t: i for i, t in enumerate(self.tokens)}
        if len(self.index) != len(self.tokens):
            raise ValueError("vocabulary tokens must be unique")
        self.counts = np.asarray(
            counts if counts is not None else np.zeros(len(self.tokens)), dtype=np.int64
        )

    @classmethod
    def build(cls, token_lists: Iterable[Sequence[str]]) -> Vocabulary:
        index: dict[str, int] = {}
        counts: list[int] = []
        for tokens in token_lists:
            for t in tokens:
                i = index.get(t)
                if i is None:
                    index[t] = len(counts)
                    counts.append(1)
                else:
                    counts[i] += 1
        return cls(list(index), counts)

    def __len__(self) -> int:
        return len(self.tokens)

    def __contains__(self, token: object) -> bool:
        return token in self.index

    def ids(self, tokens: Iterable[str]) -> np.ndarray:
        """Row indices of the tokens that are in the vocabulary (unknown ones are dropped)."""
        return np.fromiter((self.index[t] for t in tokens if t in self.index), dtype=np.int64)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"tokens": self.tokens, "counts": self.counts.tolist()}))

    @classmethod
    def load(cls, path: Path) -> Vocabulary:
        data = json.loads(path.read_text())
        return cls(data["tokens"], data["counts"])
