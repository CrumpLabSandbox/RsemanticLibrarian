"""BibTeX / BibLaTeX ingestion (e.g. a Zotero or Mendeley export)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode

from semantic_librarian.document import Document

TEXT_FIELDS = ("title", "abstract", "keywords")
_BRACES = re.compile(r"[{}]")
_SPACES = re.compile(r"\s+")


def _clean(value: str) -> str:
    return _SPACES.sub(" ", _BRACES.sub("", value)).strip()


def read_bibtex(path: Path) -> Iterator[Document]:
    """One document per entry. Title, abstract and keywords are embedded; author, year
    and venue become factors; every other field is kept as metadata."""
    parser = BibTexParser(common_strings=True)
    parser.customization = convert_to_unicode
    parser.ignore_nonstandard_types = False
    with path.open(encoding="utf-8") as f:
        db = bibtexparser.load(f, parser=parser)
    for entry in db.entries:
        e = {k.lower(): _clean(v) for k, v in entry.items() if isinstance(v, str)}
        fields = {k: e[k] for k in TEXT_FIELDS if e.get(k)}
        if not fields:
            continue
        factors: dict[str, list[str]] = {}
        if e.get("author"):
            factors["author"] = [
                a.strip() for a in re.split(r"\s+and\s+", e["author"]) if a.strip()
            ]
        year = e.get("year") or (e.get("date", "")[:4] if e.get("date", "")[:4].isdigit() else "")
        if year:
            factors["year"] = [year]
        venue = e.get("journal") or e.get("journaltitle") or e.get("booktitle")
        if venue:
            factors["venue"] = [venue]
        meta = {k: v for k, v in e.items() if k not in TEXT_FIELDS and k not in ("id",)}
        meta["entrytype"] = e.get("entrytype", "")
        yield Document(e.get("id") or entry["ID"], fields, meta, factors)
