"""A Semantic Librarian library: a directory with documents, config and vector spaces.

::

    mylib/
    ├── config.toml       how to build (text mode, fields, factors, embedder)
    ├── library.sqlite    documents, factors, sources, build history, keyword index
    └── vectors/          one .npy + .labels.json per space, vocab.json, build.json

Spaces: ``document``, ``word``, one per factor, and ``chunk`` when the library holds
chunked documents (PDFs).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from semantic_librarian.config import LibraryConfig
from semantic_librarian.document import Chunk, Document, chunk_label
from semantic_librarian.embed.base import Corpus, WordSumEmbedder
from semantic_librarian.embed.registry import embedder_from_config, make_embedder
from semantic_librarian.ingest import FieldMapping, detect_format, read_documents
from semantic_librarian.ingest.plaintext import Split
from semantic_librarian.project.maps import Projection, project
from semantic_librarian.search.fusion import reciprocal_rank_fusion
from semantic_librarian.search.lexical import BM25, doc_term_matrix
from semantic_librarian.search.similarity import (
    QUERY_MODES,
    QueryMode,
    combine_term_scores,
    cosine_scores,
    rank,
)
from semantic_librarian.space.factors import factor_space
from semantic_librarian.space.vector_space import VectorSpace
from semantic_librarian.space.vocabulary import Vocabulary
from semantic_librarian.store.sqlite import LibraryStore, file_sha256
from semantic_librarian.text.chunk import DEFAULT_CHUNK_WORDS
from semantic_librarian.text.clean import TextProcessor

CONFIG_FILE = "config.toml"
DB_FILE = "library.sqlite"
VECTORS_DIR = "vectors"
DOC_TERMS_FILE = "doc_terms.npz"
RESERVED_SPACES = ("word", "document", "chunk")
SEARCH_METHODS = ("semantic", "lexical", "hybrid")


class LibraryError(RuntimeError):
    """A library operation that cannot proceed (not built, missing space...)."""


@dataclass
class Hit:
    """One ranked item."""

    rank: int
    label: str
    score: float
    document: Document | None = None
    # the passage itself for hits in the chunk space; ``document`` is then its parent
    chunk: Chunk | None = None
    # component scores for fused rankings, e.g. {"semantic": 0.61, "lexical": 7.2}
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"rank": self.rank, "label": self.label, "score": self.score}
        if self.extra:
            d["components"] = self.extra
        if self.document is not None:
            d["title"] = self.document.title()
            d["meta"] = self.document.meta
            d["factors"] = self.document.factors
        if self.chunk is not None:
            d["text"] = self.chunk.text
            d["chunk"] = self.chunk.meta
        return d


@dataclass
class SearchResult:
    query: str
    mode: str
    space: str
    terms: list[str]
    unknown: list[str]
    hits: list[Hit] = field(default_factory=list)
    method: str = "semantic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "method": self.method,
            "mode": self.mode,
            "space": self.space,
            "terms": self.terms,
            "unknown": self.unknown,
            "hits": [h.to_dict() for h in self.hits],
        }


@dataclass
class AddReport:
    path: str
    format: str
    added: int
    skipped: bool = False


@dataclass
class BuildReport:
    embedder: str
    n_documents: int
    vocab_size: int | None
    spaces: dict[str, int]
    seconds: float


class Library:
    """A library on disk. Open one with ``Library.open`` or make one with ``Library.create``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        if not (self.root / CONFIG_FILE).exists():
            raise LibraryError(f"{self.root} is not a library (no {CONFIG_FILE}); run `sl init`")
        self.config = LibraryConfig.load(self.root / CONFIG_FILE)
        self.store = LibraryStore(self.root / DB_FILE)

    # -- lifecycle ----------------------------------------------------------------------

    @classmethod
    def create(cls, root: Path | str, name: str | None = None, **settings: Any) -> Library:
        """Create a new, empty library directory. ``settings`` are LibraryConfig fields."""
        root = Path(root)
        if (root / CONFIG_FILE).exists():
            raise LibraryError(f"{root} already contains a library")
        root.mkdir(parents=True, exist_ok=True)
        LibraryConfig(name=name or root.name, **settings).save(root / CONFIG_FILE)
        return cls(root)

    @classmethod
    def open(cls, root: Path | str) -> Library:
        return cls(Path(root))

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Library:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def save_config(self) -> None:
        self.config.save(self.root / CONFIG_FILE)

    # -- adding documents ---------------------------------------------------------------

    def add_documents(
        self, docs: Iterable[Document], source: str | None = None, kind: str = "python"
    ) -> int:
        """Add documents directly. Re-adding with the same ``source`` replaces them."""
        source_id = None
        if source is not None:
            source_id = self.store.replace_source(source, kind, None)
        n = self.store.add_documents(docs, source_id)
        self._invalidate()
        return n

    def add_file(
        self,
        path: Path | str,
        fmt: str | None = None,
        mapping: FieldMapping | None = None,
        split: Split = "document",
        force: bool = False,
        chunk_words: int = DEFAULT_CHUNK_WORDS,
    ) -> AddReport:
        """Ingest a file. Unchanged files that were added before are skipped.

        ``split`` applies to text files and ``chunk_words`` to PDFs.
        """
        path = Path(path).resolve()
        fmt = fmt or detect_format(path)
        sha = file_sha256(path)
        key = str(path)
        if not force and self.store.source_sha(key) == sha:
            return AddReport(key, fmt, 0, skipped=True)
        docs = list(read_documents(path, fmt, mapping, split, chunk_words))
        options: dict[str, Any] = {"split": split}
        if fmt == "pdf":
            options = {"chunk_words": chunk_words}
        if mapping is not None:
            options.update({"text": mapping.text, "factors": mapping.factors, "id": mapping.id})
        source_id = self.store.replace_source(key, fmt, sha, options)
        n = self.store.add_documents(docs, source_id)
        self._invalidate()
        return AddReport(key, fmt, n)

    # -- building -----------------------------------------------------------------------

    def corpus(self, documents: Sequence[Document] | None = None) -> Corpus:
        """Tokenize the documents according to the config."""
        docs = list(documents) if documents is not None else self.documents
        processor = TextProcessor(self.config.text_mode)  # type: ignore[arg-type]
        fields = self.config.fields or None
        sentences: list[list[str]] = []
        doc_tokens: list[list[str]] = []
        chunk_ids: list[str] = []
        chunk_tokens: list[list[str]] = []
        for doc in docs:
            tokens: list[str] = []
            for n, text in enumerate(doc.embedded_texts(fields), start=1):
                piece: list[str] = []
                for sentence in processor.sentences(text):
                    sentences.append(sentence)
                    piece.extend(sentence)
                tokens.extend(piece)
                if doc.chunks:
                    chunk_ids.append(chunk_label(doc.id, n))
                    chunk_tokens.append(piece)
            doc_tokens.append(tokens)
        vocab = Vocabulary.build(doc_tokens)
        return Corpus([d.id for d in docs], sentences, doc_tokens, vocab, chunk_ids, chunk_tokens)

    def build(
        self,
        embedder: str | None = None,
        params: dict[str, Any] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> BuildReport:
        """Embed every document and write all vector spaces.

        ``embedder`` and ``params`` override (and are saved into) the config.
        """
        say = log or (lambda _msg: None)
        start = datetime.now(UTC)
        if embedder is not None:
            self.config.embedder = embedder
        if params:
            self.config.embedder_params.update(params)
        self.save_config()

        docs = self.documents
        if not docs:
            raise LibraryError("the library has no documents; add some with `sl add`")
        say(f"tokenizing {len(docs)} documents ({self.config.text_mode} text mode)")
        corpus = self.corpus(docs)
        model = make_embedder(self.config.embedder, **self.config.embedder_params)
        if hasattr(model, "log"):
            model.log = say
        say(
            f"fitting {self.config.embedder} on {len(corpus.sentences)} sentences, "
            f"{len(corpus.vocab)} word types"
        )
        model.fit(corpus)
        say("embedding documents")
        doc_vectors = model.embed_documents(corpus)

        spaces: dict[str, VectorSpace] = {
            "document": VectorSpace("document", corpus.doc_ids, doc_vectors)
        }
        word = model.word_space()
        if word is not None:
            spaces["word"] = word
        if corpus.chunk_ids:
            say(f"embedding {len(corpus.chunk_ids)} chunks")
            spaces["chunk"] = VectorSpace(
                "chunk", corpus.chunk_ids, model.embed_documents(corpus.chunks())
            )
        wanted = self.config.factors or self.store.factor_names()
        for name in wanted:
            levels = [d.factors.get(name, []) for d in docs]
            if not any(levels):
                continue
            space_name = f"factor-{name}" if name in RESERVED_SPACES else name
            say(f"building factor space {space_name!r}")
            spaces[space_name] = factor_space(
                space_name,
                levels,
                doc_vectors,
                how=self.config.factor_aggregate,  # type: ignore[arg-type]
            )

        tmp = self.root / f"{VECTORS_DIR}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        for space in spaces.values():
            space.save(tmp)
        corpus.vocab.save(tmp / "vocab.json")
        BM25(doc_term_matrix(corpus.doc_tokens, corpus.vocab)).save(tmp / DOC_TERMS_FILE)
        seconds = (datetime.now(UTC) - start).total_seconds()
        build_id = self.store.record_build(
            self.config.embedder, model.config(), len(docs), len(corpus.vocab)
        )
        space_sizes = {name: len(s) for name, s in spaces.items()}
        info: dict[str, Any] = {
            "build_id": build_id,
            "created_at": start.isoformat(timespec="seconds"),
            "embedder": model.config(),
            "text_mode": self.config.text_mode,
            "fields": self.config.fields,
            "spaces": space_sizes,
            "n_documents": len(docs),
            "vocab_size": len(corpus.vocab),
            "seconds": seconds,
        }
        (tmp / "build.json").write_text(json.dumps(info, indent=2))
        final = self.root / VECTORS_DIR
        old = self.root / f"{VECTORS_DIR}.old"
        if final.exists():
            final.rename(old)
        tmp.rename(final)
        if old.exists():
            shutil.rmtree(old)
        self._invalidate()
        say(f"built in {seconds:.1f}s")
        return BuildReport(self.config.embedder, len(docs), len(corpus.vocab), space_sizes, seconds)

    # -- loading ------------------------------------------------------------------------

    def _invalidate(self) -> None:
        for attr in (
            "documents",
            "_doc_index",
            "_chunk_index",
            "build_info",
            "spaces",
            "embedder",
            "bm25",
        ):
            self.__dict__.pop(attr, None)

    @cached_property
    def documents(self) -> list[Document]:
        return list(self.store.documents())

    @cached_property
    def _doc_index(self) -> dict[str, Document]:
        return {d.id: d for d in self.documents}

    @cached_property
    def _chunk_index(self) -> dict[str, tuple[Document, Chunk]]:
        return {
            chunk_label(d.id, n): (d, c)
            for d in self.documents
            for n, c in enumerate(d.chunks, start=1)
        }

    def chunk(self, label: str) -> tuple[Document, Chunk]:
        """A chunk and the document it belongs to, by chunk label (``<document id>#<n>``)."""
        try:
            return self._chunk_index[label]
        except KeyError:
            raise KeyError(f"no chunk labelled {label!r}") from None

    def document(self, doc_id: str) -> Document:
        try:
            return self._doc_index[doc_id]
        except KeyError:
            raise KeyError(f"no document with id {doc_id!r}") from None

    @property
    def is_built(self) -> bool:
        return (self.root / VECTORS_DIR / "build.json").exists()

    @cached_property
    def build_info(self) -> dict[str, Any]:
        if not self.is_built:
            raise LibraryError("the library has not been built yet; run `sl build`")
        info: dict[str, Any] = json.loads((self.root / VECTORS_DIR / "build.json").read_text())
        return info

    @cached_property
    def spaces(self) -> dict[str, VectorSpace]:
        directory = self.root / VECTORS_DIR
        return {name: VectorSpace.load(directory, name) for name in self.build_info["spaces"]}

    def space(self, name: str) -> VectorSpace:
        try:
            return self.spaces[name]
        except KeyError:
            known = ", ".join(self.spaces)
            raise LibraryError(f"no space named {name!r}; available: {known}") from None

    @cached_property
    def embedder(self) -> WordSumEmbedder:
        model = embedder_from_config(self.build_info["embedder"])
        if not isinstance(model, WordSumEmbedder):  # pragma: no cover - future embedders
            raise LibraryError("only word-based embedders can be restored in this version")
        vocab = Vocabulary.load(self.root / VECTORS_DIR / "vocab.json")
        model.load_word_space(self.space("word"), vocab)
        return model

    @cached_property
    def bm25(self) -> BM25:
        path = self.root / VECTORS_DIR / DOC_TERMS_FILE
        if not path.exists():
            raise LibraryError("this build has no keyword index; run `sl build` again")
        return BM25.load(path)

    @property
    def is_stale(self) -> bool:
        """True when documents were added after the last build."""
        return self.is_built and self.build_info["n_documents"] != self.store.count_documents()

    # -- querying -----------------------------------------------------------------------

    def _hits(
        self,
        space: VectorSpace,
        scores: np.ndarray,
        k: int | None,
        mask: np.ndarray | None,
        components: dict[str, np.ndarray] | None = None,
    ) -> list[Hit]:
        hits = []
        for r, i in enumerate(rank(scores, k, mask), start=1):
            label = space.labels[i]
            extra = {name: float(v[i]) for name, v in (components or {}).items()}
            doc, chunk = None, None
            if space.name == "document":
                doc = self._doc_index.get(label)
            elif space.name == "chunk":
                doc, chunk = self._chunk_index.get(label, (None, None))
            hits.append(Hit(r, label, float(scores[i]), doc, chunk, extra))
        return hits

    def _mask(
        self,
        space: VectorSpace,
        where: Callable[[Document], bool] | None,
        include_stopwords: bool = False,
    ) -> np.ndarray | None:
        mask = None
        if where is not None:
            if space.name == "document":
                owners = [self._doc_index[label] for label in space.labels]
            elif space.name == "chunk":
                owners = [self._chunk_index[label][0] for label in space.labels]
            else:
                raise LibraryError("filters apply to the document and chunk spaces only")
            mask = np.array([where(doc) for doc in owners], dtype=bool)
        if space.name == "word" and not include_stopwords:
            stop = self.embedder.stopword_set
            if stop:
                mask = np.array([label not in stop for label in space.labels], dtype=bool)
        return mask

    def search(
        self,
        query: str,
        mode: QueryMode = "compound",
        space: str = "document",
        k: int | None = 10,
        where: Callable[[Document], bool] | None = None,
        method: str = "semantic",
        lexical_weight: float = 0.5,
        include_stopwords: bool = False,
    ) -> SearchResult:
        """Rank the items of ``space`` against a free-text query.

        Args:
            query: Free text; words not in the vocabulary are ignored (see ``unknown``).
            mode: How semantic scores combine query words: ``compound`` (sum of words),
                ``and`` or ``or``; see ``semantic_librarian.search.similarity``.
            space: Which space to rank: ``document``, ``word``, ``chunk`` (passages of
                PDFs) or any factor space.
            k: Number of hits (``None`` for all).
            where: Optional document filter, e.g. ``lambda d: d.meta.get("year", 0) > 1990``.
                In the chunk space it is applied to each chunk's document.
            method: ``semantic`` (vector space), ``lexical`` (BM25 keywords) or ``hybrid``
                (reciprocal rank fusion of the two). Lexical and hybrid rank documents.
            lexical_weight: Share of the lexical ranking in a hybrid search (0 to 1).
            include_stopwords: Show stop words when ranking the word space.
        """
        if mode not in QUERY_MODES:
            raise ValueError(f"mode must be one of {QUERY_MODES}")
        if method not in SEARCH_METHODS:
            raise ValueError(f"method must be one of {SEARCH_METHODS}")
        if method != "semantic" and space != "document":
            raise LibraryError(f"{method} search ranks documents; use space='document'")
        if not 0.0 <= lexical_weight <= 1.0:
            raise ValueError("lexical_weight must be between 0 and 1")
        target = self.space(space)
        processor = TextProcessor(self.config.text_mode)  # type: ignore[arg-type]
        tokens = processor.tokens(query)
        model = self.embedder
        assert model.vocab is not None
        known = [t for t in dict.fromkeys(tokens) if t in model.vocab]
        unknown = [t for t in dict.fromkeys(tokens) if t not in model.vocab]
        terms = known if method == "lexical" else model.query_terms(tokens)
        result = SearchResult(query, mode, space, terms, unknown, method=method)
        if not terms:
            return result
        mask = self._mask(target, where, include_stopwords)

        semantic = None
        if method in ("semantic", "hybrid"):
            if mode == "compound":
                q = model.embed_query(terms)
                assert q is not None
                semantic = cosine_scores(q, target.vectors, target.norms)
            else:
                term_scores = cosine_scores(model.term_vectors(terms), target.vectors, target.norms)
                semantic = combine_term_scores(term_scores, mode)
            if method == "semantic":
                result.hits = self._hits(target, semantic, k, mask)
                return result

        lexical = self.bm25.scores(model.vocab.ids(known))
        matched = lexical > 0
        if method == "lexical":
            mask = matched if mask is None else mask & matched
            result.hits = self._hits(target, lexical, k, mask)
            return result

        assert semantic is not None
        fused = reciprocal_rank_fusion(
            [semantic, lexical],
            weights=[1.0 - lexical_weight, lexical_weight],
            eligible=[None, matched],
        )
        result.hits = self._hits(
            target, fused, k, mask, components={"semantic": semantic, "lexical": lexical}
        )
        return result

    def similar(
        self,
        label: str,
        space: str = "document",
        target: str | None = None,
        k: int | None = 10,
        where: Callable[[Document], bool] | None = None,
        include_stopwords: bool = False,
    ) -> list[Hit]:
        """Items of ``target`` (default: the same space) closest to an existing item.

        Covers the original interface's article-article, author-author and
        article-to-author searches, and any other pairing of spaces.
        """
        source = self.space(space)
        tgt = self.space(target or space)
        scores = cosine_scores(source.vector(label), tgt.vectors, tgt.norms)
        return self._hits(tgt, scores, k, self._mask(tgt, where, include_stopwords))

    def project(
        self, labels: Sequence[str], space: str = "document", n_clusters: int = 3, seed: int = 0
    ) -> Projection:
        """2-D MDS map with k-means clusters for the given items of a space."""
        s = self.space(space)
        rows = [s.row(label) for label in labels]
        return project(list(labels), np.asarray(s.vectors[rows]), n_clusters, seed)
