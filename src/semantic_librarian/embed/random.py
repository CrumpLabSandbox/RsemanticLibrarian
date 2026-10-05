"""The paper's non-semantic control: each word is just its random environment vector.

Documents are still sums of word vectors, so this baseline isolates what the learned
semantic structure adds (paper Simulation 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np

from semantic_librarian.embed.base import WordModelConfig, WordSumEmbedder
from semantic_librarian.embed.random_vectors import environment_vectors
from semantic_librarian.space.vocabulary import Vocabulary


@dataclass
class RandomConfig(WordModelConfig):
    environment: Literal["gaussian", "ternary"] = "gaussian"
    nonzeros: int = 30


class RandomVectors(WordSumEmbedder):
    name: ClassVar[str] = "random"

    def __init__(self, cfg: RandomConfig | None = None, **params: Any) -> None:
        super().__init__()
        self.cfg: RandomConfig = cfg or RandomConfig(**params)

    def learn_word_vectors(self, sentence_ids: list[np.ndarray], vocab: Vocabulary) -> np.ndarray:
        rng = np.random.default_rng(self.cfg.seed)
        env = environment_vectors(
            self.cfg.environment, len(vocab), self.cfg.dim, self.cfg.nonzeros, rng
        )
        return env.astype(np.float32)
