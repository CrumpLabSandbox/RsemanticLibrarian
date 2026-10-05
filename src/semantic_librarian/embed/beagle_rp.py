"""BEAGLE-RP: BEAGLE with random permutations (Sahlgren, Holst & Kanerva, 2008;
Recchia, Sahlgren, Kanerva & Jones, 2015).

Environment vectors are sparse and ternary. Context information is the sum of the other
non-stop words' environment vectors in the sentence. Order information sums the
environment vectors of the words within ``window`` positions of the target, each shifted
(rotated) by its offset from the target, so "dog bit" and "bit dog" differ. As in the
paper, order windows do not cross sentence boundaries. The final vector is the sum of the
unit-normalized context and order vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from semantic_librarian.embed.base import WordModelConfig, WordSumEmbedder, unit_rows
from semantic_librarian.embed.cooccurrence import context_sums, permutation_order_sums
from semantic_librarian.embed.random_vectors import ternary_sparse
from semantic_librarian.space.vocabulary import Vocabulary


@dataclass
class BeagleRPConfig(WordModelConfig):
    dim: int = 3000
    nonzeros: int = 60
    window: int = 2
    context: bool = True
    order: bool = True


class BeagleRP(WordSumEmbedder):
    name: ClassVar[str] = "beagle-rp"

    def __init__(self, cfg: BeagleRPConfig | None = None, **params: Any) -> None:
        super().__init__()
        self.cfg: BeagleRPConfig = cfg or BeagleRPConfig(**params)
        if not (self.cfg.context or self.cfg.order):
            raise ValueError("at least one of context and order must be enabled")
        if self.cfg.window < 1:
            raise ValueError("window must be at least 1")

    def learn_word_vectors(self, sentence_ids: list[np.ndarray], vocab: Vocabulary) -> np.ndarray:
        rng = np.random.default_rng(self.cfg.seed)
        env = ternary_sparse(len(vocab), self.cfg.dim, self.cfg.nonzeros, rng)
        stop = self.stopword_set
        keep = np.array([t not in stop for t in vocab.tokens], dtype=bool)
        parts = []
        if self.cfg.context:
            parts.append(unit_rows(context_sums(sentence_ids, env, keep)))
        if self.cfg.order:
            parts.append(unit_rows(permutation_order_sums(sentence_ids, env, self.cfg.window)))
        return np.asarray(sum(parts), dtype=np.float32)
