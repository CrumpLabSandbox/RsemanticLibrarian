"""Turning generic records (JSON objects, CSV rows) into Documents."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from semantic_librarian.document import Document

DEFAULT_TEXT_FIELDS: tuple[str, ...] = (
    "title",
    "abstract",
    "keywords",
    "text",
    "body",
    "content",
    "sentence",
    "sentences",
    "summary",
    "description",
)
DEFAULT_ID_FIELDS: tuple[str, ...] = ("id", "doc_id", "key", "doi")
# factor name -> candidate record keys; used when no factors are given explicitly
DEFAULT_FACTORS: dict[str, tuple[str, ...]] = {
    "author": ("author", "authors", "authorlist"),
    "year": ("year",),
}
DEFAULT_SEPARATOR = r"\s*;\s*|\s+and\s+"


@dataclass
class FieldMapping:
    """Which record keys are text, which are factors, and which is the id.

    Attributes:
        text: Keys whose values are embedded. ``None`` picks the keys in
            ``DEFAULT_TEXT_FIELDS`` that are present.
        factors: Factor name -> record key. ``None`` auto-detects ``author`` and ``year``.
        separators: Factor name -> regular expression splitting a string value into
            levels. Lists are used as given. Default for author-like keys: ``;`` or ``and``.
        id: Key holding the document id. ``None`` tries ``DEFAULT_ID_FIELDS``, then numbers
            the records.
    """

    text: list[str] | None = None
    factors: dict[str, str] | None = None
    separators: dict[str, str] = field(default_factory=dict)
    id: str | None = None

    @classmethod
    def from_options(
        cls,
        text: Iterable[str] | None = None,
        factors: Iterable[str] | None = None,
        id: str | None = None,
    ) -> FieldMapping:
        """Build from CLI-style options.

        ``factors`` items look like ``author`` (key and factor share a name),
        ``author=authors`` (factor name = record key) and may end in ``:SEP`` with a
        literal separator, e.g. ``author=authorlist:;``.
        """
        fmap: dict[str, str] | None = None
        seps: dict[str, str] = {}
        if factors is not None:
            fmap = {}
            for spec in factors:
                spec, _, sep = spec.partition(":")
                name, _, key = spec.partition("=")
                name = name.strip()
                fmap[name] = (key or name).strip()
                if sep:
                    seps[name] = rf"\s*{re.escape(sep)}\s*" if sep.strip() else r"\s+"
        return cls(
            text=list(text) if text is not None else None, factors=fmap, separators=seps, id=id
        )


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list | tuple):
        return "; ".join(str(v) for v in value if v is not None)
    s = str(value)
    return s if s.strip() else None


def _levels(value: Any, separator: str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    if not s:
        return []
    if separator:
        return [p.strip() for p in re.split(separator, s) if p.strip()]
    return [s]


def record_to_document(
    record: Mapping[str, Any], mapping: FieldMapping, fallback_id: str
) -> Document:
    keys = list(record)
    text_keys = (
        mapping.text
        if mapping.text is not None
        else [k for k in keys if k.lower() in DEFAULT_TEXT_FIELDS]
    )
    if not text_keys:
        raise ValueError(
            f"no text fields found among {keys}; name them explicitly (e.g. --text title,abstract)"
        )
    fields = {}
    for k in text_keys:
        t = _as_text(record.get(k))
        if t is not None:
            fields[k] = t

    if mapping.factors is None:
        lower = {k.lower(): k for k in keys}
        factor_keys = {}
        for name, candidates in DEFAULT_FACTORS.items():
            for c in candidates:
                if c in lower:
                    factor_keys[name] = lower[c]
                    break
    else:
        factor_keys = mapping.factors
    factors = {}
    for name, key in factor_keys.items():
        sep = mapping.separators.get(name)
        if sep is None and name in ("author", "authors", "editor", "keyword", "keywords"):
            sep = DEFAULT_SEPARATOR
        levels = _levels(record.get(key), sep)
        if levels:
            factors[name] = levels

    doc_id = None
    id_keys = [mapping.id] if mapping.id else [k for k in keys if k.lower() in DEFAULT_ID_FIELDS]
    for k in id_keys:
        v = record.get(k)
        if v is not None and str(v).strip():
            doc_id = str(v).strip()
            break
    meta = {k: v for k, v in record.items() if k not in text_keys and v is not None}
    return Document(doc_id or fallback_id, fields, meta, factors)
