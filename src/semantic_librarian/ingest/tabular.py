"""JSON, JSON Lines and CSV ingestion."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from semantic_librarian.document import Document
from semantic_librarian.ingest.mapping import FieldMapping, record_to_document


def _records_from_json(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("documents", "records", "data", "items", "articles"):
            if isinstance(data.get(key), list):
                return list(data[key])
    raise ValueError("JSON must be a list of objects or contain a 'documents' list")


def _documents(
    records: Iterator[dict[str, Any]] | list[dict[str, Any]], mapping: FieldMapping, stem: str
) -> Iterator[Document]:
    for n, rec in enumerate(records, start=1):
        if not isinstance(rec, dict):
            raise ValueError(f"record {n} is not an object")
        yield record_to_document(rec, mapping, f"{stem}-{n:06d}")


def read_json(path: Path, mapping: FieldMapping) -> Iterator[Document]:
    records = _records_from_json(json.loads(path.read_text(encoding="utf-8")))
    yield from _documents(records, mapping, path.stem)


def read_jsonl(path: Path, mapping: FieldMapping) -> Iterator[Document]:
    def records() -> Iterator[dict[str, Any]]:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    yield from _documents(records(), mapping, path.stem)


def read_csv(path: Path, mapping: FieldMapping) -> Iterator[Document]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        dialect = csv.Sniffer().sniff(f.read(64 * 1024), delimiters=",\t;|")
        f.seek(0)
        reader = csv.DictReader(f, dialect=dialect)
        yield from _documents(
            ({k: (v if v != "" else None) for k, v in row.items()} for row in reader),
            mapping,
            path.stem,
        )
