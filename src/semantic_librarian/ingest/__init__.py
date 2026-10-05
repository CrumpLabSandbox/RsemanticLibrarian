"""Read files into Documents."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from semantic_librarian.document import Document
from semantic_librarian.ingest.bibtex import read_bibtex
from semantic_librarian.ingest.mapping import FieldMapping, record_to_document
from semantic_librarian.ingest.pdf import read_pdf
from semantic_librarian.ingest.plaintext import Split, read_text
from semantic_librarian.ingest.tabular import read_csv, read_json, read_jsonl
from semantic_librarian.text.chunk import DEFAULT_CHUNK_WORDS

FORMATS: dict[str, tuple[str, ...]] = {
    "json": (".json",),
    "jsonl": (".jsonl", ".ndjson"),
    "csv": (".csv", ".tsv"),
    "bibtex": (".bib", ".bibtex"),
    "text": (".txt", ".md", ".markdown", ".text"),
    "pdf": (".pdf",),
}


def detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    for fmt, suffixes in FORMATS.items():
        if suffix in suffixes:
            return fmt
    raise ValueError(f"cannot tell the format of {path.name}; pass --format")


def read_documents(
    path: Path,
    fmt: str | None = None,
    mapping: FieldMapping | None = None,
    split: Split = "document",
    chunk_words: int = DEFAULT_CHUNK_WORDS,
) -> Iterator[Document]:
    """Yield the documents in ``path``.

    Args:
        path: The file to read.
        fmt: One of ``json``, ``jsonl``, ``csv``, ``bibtex``, ``text``, ``pdf``; detected
            from the extension when omitted.
        mapping: Field mapping for ``json``, ``jsonl`` and ``csv``.
        split: For ``text``: one document per file, paragraph or sentence.
        chunk_words: For ``pdf``: the approximate length of a chunk, in words.
    """
    path = Path(path)
    fmt = fmt or detect_format(path)
    mapping = mapping or FieldMapping()
    if fmt == "json":
        yield from read_json(path, mapping)
    elif fmt == "jsonl":
        yield from read_jsonl(path, mapping)
    elif fmt == "csv":
        yield from read_csv(path, mapping)
    elif fmt == "bibtex":
        yield from read_bibtex(path)
    elif fmt == "text":
        yield from read_text(path, split)
    elif fmt == "pdf":
        yield from read_pdf(path, chunk_words)
    else:
        raise ValueError(f"unknown format {fmt!r}; use one of {', '.join(FORMATS)}")


__all__ = ["FORMATS", "FieldMapping", "detect_format", "read_documents", "record_to_document"]
