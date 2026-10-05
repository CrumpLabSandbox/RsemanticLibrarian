"""The source-agnostic document record every ingester produces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """One passage of a long document, e.g. a few paragraphs of a PDF.

    Attributes:
        text: The passage.
        meta: Where it came from, e.g. ``{"page": 3, "page_end": 4}``.
    """

    text: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "meta": self.meta}


def chunk_label(doc_id: str, position: int) -> str:
    """The label of a document's ``position``-th chunk (1-based) in the chunk space."""
    return f"{doc_id}#{position}"


@dataclass
class Document:
    """One searchable unit: an article, a book chapter, a sentence, a whole PDF...

    Attributes:
        id: Stable identifier, unique within a library.
        fields: Named text fields (``title``, ``abstract``, ``body``...). These are what
            gets embedded.
        meta: Anything else worth displaying or filtering on (year, DOI, source path).
        factors: Grouping variables, each with one or more levels. Every factor becomes its
            own vector space whose items are the sums of their documents' vectors; e.g.
            ``{"author": ["Crump, Matthew J. C.", "Logan, Gordon D."], "year": ["2010"]}``.
        chunks: Passages of a long document, in reading order. A document with chunks is
            embedded from its chunks (its vector is the sum of theirs) and its fields are
            only displayed; each chunk is also searchable on its own in the ``chunk``
            space.
    """

    id: str
    fields: dict[str, str]
    meta: dict[str, Any] = field(default_factory=dict)
    factors: dict[str, list[str]] = field(default_factory=dict)
    chunks: list[Chunk] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Document.id must be a non-empty string")
        self.fields = {k: v for k, v in self.fields.items() if isinstance(v, str) and v.strip()}
        self.factors = {
            k: [str(level) for level in levels if str(level).strip()]
            for k, levels in self.factors.items()
        }
        self.factors = {k: v for k, v in self.factors.items() if v}
        self.chunks = [c for c in self.chunks if c.text.strip()]

    def texts(self, field_names: Iterable[str] | None = None) -> list[str]:
        """The text of the requested fields (all fields if ``None``), in field order."""
        if field_names is None:
            return list(self.fields.values())
        return [self.fields[f] for f in field_names if f in self.fields]

    def embedded_texts(self, field_names: Iterable[str] | None = None) -> list[str]:
        """The text that gets embedded: the chunks if there are any, else the fields."""
        if self.chunks:
            return [c.text for c in self.chunks]
        return self.texts(field_names)

    def title(self) -> str:
        for key in ("title", "name", "heading"):
            if key in self.fields:
                return self.fields[key]
        first = next(iter(self.fields.values()), "")
        return first[:120] or self.id

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "fields": self.fields,
            "meta": self.meta,
            "factors": self.factors,
        }
        if self.chunks:
            d["chunks"] = [c.to_dict() for c in self.chunks]
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Document:
        return cls(
            id=str(data["id"]),
            fields=dict(data.get("fields", {})),
            meta=dict(data.get("meta", {})),
            factors={k: list(v) for k, v in data.get("factors", {}).items()},
            chunks=[Chunk(c["text"], dict(c.get("meta", {}))) for c in data.get("chunks", [])],
        )
