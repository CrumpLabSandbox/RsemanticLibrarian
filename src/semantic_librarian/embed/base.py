"""The embedder interface.

Every embedder turns a corpus into a document matrix and can embed a free-text query
into the same space. Word-based models (BEAGLE, BEAGLE-RP, the random baseline) also
produce a word space and build documents and queries by summing word vectors, as in the
paper (Eqs. 8 and 9). Future transformer-based embedders implement the same interface
without a word space.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

import numpy as np

from semantic_librarian.space.vector_space import VectorSpace
from semantic_librarian.space.vocabulary import Vocabulary
from semantic_librarian.text.stopwords import resolve_stopwords


@dataclass
class Corpus:
    """Tokenized corpus handed to an embedder.

    Attributes:
        doc_ids: Document identifiers, one per document row.
        sentences: Token lists for every sentence, in reading order across documents.
        doc_tokens: All tokens of each document, one list per document row.
        vocab: Vocabulary over all tokens.
        chunk_ids: Labels of the chunks of chunked documents (PDFs), in document order.
        chunk_tokens: All tokens of each chunk, one list per chunk.
    """

    doc_ids: list[str]
    sentences: list[list[str]]
    doc_tokens: list[list[str]]
    vocab: Vocabulary
    chunk_ids: list[str] = field(default_factory=list)
    chunk_tokens: list[list[str]] = field(default_factory=list)

    def chunks(self) -> Corpus:
        """The chunks as a corpus of their own, so an embedder can embed them like documents."""
        return Corpus(self.chunk_ids, [], self.chunk_tokens, self.vocab)

    def sentence_ids(self) -> list[np.ndarray]:
        return [self.vocab.ids(s) for s in self.sentences]


class Embedder(ABC):
    """Base class for all embedders."""

    name: ClassVar[str]

    @abstractmethod
    def fit(self, corpus: Corpus) -> None:
        """Learn whatever the model needs from the corpus."""

    @abstractmethod
    def embed_documents(self, corpus: Corpus) -> np.ndarray:
        """Return one row per document, in ``corpus.doc_ids`` order."""

    @abstractmethod
    def embed_query(self, tokens: Sequence[str], unique: bool = True) -> np.ndarray | None:
        """Embed a tokenized query, or return ``None`` if nothing in it is known.

        With ``unique=False`` repeated words count once per occurrence (paper Eq. 9);
        the default counts each distinct word once, as the R package did.
        """

    def word_space(self) -> VectorSpace | None:
        """The word space, for models that have one."""
        return None

    @abstractmethod
    def config(self) -> dict[str, Any]:
        """Parameters needed to reproduce this embedder."""


@dataclass
class WordModelConfig:
    """Options shared by all word-sum models."""

    dim: int = 1024
    seed: int = 0
    stopwords: str | list[str] | None = "english"
    # Exclude stop words when summing word vectors into documents and queries.
    compose_without_stopwords: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WordSumEmbedder(Embedder):
    """A model that learns word vectors; documents and queries are sums of word vectors."""

    cfg: WordModelConfig

    def __init__(self) -> None:
        self.vocab: Vocabulary | None = None
        self.word_vectors: np.ndarray | None = None

    @abstractmethod
    def learn_word_vectors(self, sentence_ids: list[np.ndarray], vocab: Vocabulary) -> np.ndarray:
        """Return a ``(len(vocab), dim)`` matrix of word vectors."""

    @property
    def stopword_set(self) -> frozenset[str]:
        return resolve_stopwords(self.cfg.stopwords)

    def fit(self, corpus: Corpus) -> None:
        self.vocab = corpus.vocab
        self.word_vectors = self.learn_word_vectors(corpus.sentence_ids(), corpus.vocab)

    def _compose_ids(self, tokens: Sequence[str]) -> np.ndarray:
        assert self.vocab is not None
        if self.cfg.compose_without_stopwords:
            stop = self.stopword_set
            kept = [t for t in tokens if t not in stop]
            # A document made only of stop words still gets a vector.
            tokens = kept or list(tokens)
        return self.vocab.ids(tokens)

    def embed_documents(self, corpus: Corpus) -> np.ndarray:
        if self.word_vectors is None:
            raise RuntimeError("fit() must be called before embed_documents()")
        return compose_documents(
            [self._compose_ids(toks) for toks in corpus.doc_tokens], self.word_vectors
        )

    def embed_query(self, tokens: Sequence[str], unique: bool = True) -> np.ndarray | None:
        if self.word_vectors is None or self.vocab is None:
            raise RuntimeError("the model has no word vectors; fit or load it first")
        ids = self._compose_ids(tokens)
        if unique:
            ids = np.unique(ids)
        if ids.size == 0:
            return None
        return np.asarray(self.word_vectors[ids].sum(axis=0), dtype=np.float64)

    def query_terms(self, tokens: Sequence[str]) -> list[str]:
        """The distinct query tokens used for search, in query order.

        Unknown tokens are dropped, and so are stop words when the model composes without
        them (unless the query is nothing but stop words).
        """
        assert self.vocab is not None
        known = [t for t in dict.fromkeys(tokens) if t in self.vocab]
        if self.cfg.compose_without_stopwords:
            stop = self.stopword_set
            content = [t for t in known if t not in stop]
            return content or known
        return known

    def term_vectors(self, terms: Sequence[str]) -> np.ndarray:
        """Word vectors of the given (known) terms, one row each."""
        assert self.vocab is not None and self.word_vectors is not None
        return np.asarray(self.word_vectors[self.vocab.ids(terms)], dtype=np.float64)

    def word_space(self) -> VectorSpace | None:
        if self.word_vectors is None or self.vocab is None:
            return None
        return VectorSpace("word", self.vocab.tokens, self.word_vectors)

    def load_word_space(self, space: VectorSpace, vocab: Vocabulary) -> None:
        """Restore a fitted model from a saved word space (enough for querying)."""
        if space.labels != vocab.tokens:
            raise ValueError("word space labels do not match the vocabulary")
        self.vocab = vocab
        self.word_vectors = space.vectors

    def config(self) -> dict[str, Any]:
        return {"name": self.name, **self.cfg.to_dict()}


def compose_documents(doc_ids: Sequence[np.ndarray], word_vectors: np.ndarray) -> np.ndarray:
    """Sum word vectors into document vectors (paper Eq. 8). Empty documents get zeros."""
    out = np.zeros((len(doc_ids), word_vectors.shape[1]), dtype=np.float64)
    for row, ids in enumerate(doc_ids):
        if len(ids):
            out[row] = word_vectors[ids].sum(axis=0, dtype=np.float64)
    return out


def unit_rows(m: np.ndarray) -> np.ndarray:
    """Scale each row to unit length; all-zero rows stay zero."""
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms
