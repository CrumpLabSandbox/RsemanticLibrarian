"""SQLite storage for documents, factors, sources and build records.

Vectors are kept outside the database as ``.npy`` files so they can be memory-mapped.
The database also holds an FTS5 index for lexical (BM25) search.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from semantic_librarian.document import Chunk, Document

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    sha256 TEXT,
    options TEXT NOT NULL DEFAULT '{}',
    added_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    row INTEGER PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    source_id INTEGER REFERENCES sources(id) ON DELETE CASCADE,
    fields TEXT NOT NULL,
    meta TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS factors (
    doc_row INTEGER NOT NULL REFERENCES documents(row) ON DELETE CASCADE,
    factor TEXT NOT NULL,
    level TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (doc_row, factor, level)
);
CREATE INDEX IF NOT EXISTS factors_by_level ON factors(factor, level);
CREATE TABLE IF NOT EXISTS chunks (
    doc_row INTEGER NOT NULL REFERENCES documents(row) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    text TEXT NOT NULL,
    meta TEXT NOT NULL,
    PRIMARY KEY (doc_row, position)
);
CREATE TABLE IF NOT EXISTS builds (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    embedder TEXT NOT NULL,
    config TEXT NOT NULL,
    n_documents INTEGER NOT NULL,
    vocab_size INTEGER
);
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(doc_id UNINDEXED, body);
"""


class DuplicateDocumentError(ValueError):
    """Raised when a document id is already present in the library."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class LibraryStore:
    """Document storage backed by one SQLite file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_SCHEMA)
        version = self.get_meta("schema_version")
        if version is not None and int(version) > SCHEMA_VERSION:
            raise RuntimeError(
                f"{self.path} uses schema {version}; this version supports {SCHEMA_VERSION}"
            )
        # Version 2 added the chunks table, which the script above creates when missing.
        if version != str(SCHEMA_VERSION):
            self.set_meta("schema_version", str(SCHEMA_VERSION))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> LibraryStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- meta ---------------------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    # -- sources ------------------------------------------------------------------------

    def source_sha(self, path: str) -> str | None:
        row = self.conn.execute("SELECT sha256 FROM sources WHERE path = ?", (path,)).fetchone()
        return None if row is None else row[0]

    def replace_source(
        self, path: str, kind: str, sha256: str | None, options: dict[str, Any] | None = None
    ) -> int:
        """Register a source, removing any documents previously ingested from it."""
        self.conn.execute(
            "DELETE FROM documents_fts WHERE doc_id IN "
            "(SELECT d.id FROM documents d JOIN sources s ON d.source_id = s.id "
            "WHERE s.path = ?)",
            (path,),
        )
        self.conn.execute("DELETE FROM sources WHERE path = ?", (path,))
        cur = self.conn.execute(
            "INSERT INTO sources(path, kind, sha256, options, added_at) VALUES (?, ?, ?, ?, ?)",
            (path, kind, sha256, json.dumps(options or {}), _now()),
        )
        assert cur.lastrowid is not None
        return int(cur.lastrowid)

    def sources(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT s.id, s.path, s.kind, s.sha256, s.added_at, COUNT(d.row) "
            "FROM sources s LEFT JOIN documents d ON d.source_id = s.id "
            "GROUP BY s.id ORDER BY s.id"
        ).fetchall()
        keys = ("id", "path", "kind", "sha256", "added_at", "documents")
        return [dict(zip(keys, r, strict=True)) for r in rows]

    # -- documents ----------------------------------------------------------------------

    def add_documents(self, docs: Iterable[Document], source_id: int | None = None) -> int:
        """Insert documents; returns how many were added. Ids must be new."""
        n = 0
        try:
            for doc in docs:
                try:
                    cur = self.conn.execute(
                        "INSERT INTO documents(id, source_id, fields, meta) VALUES (?, ?, ?, ?)",
                        (
                            doc.id,
                            source_id,
                            json.dumps(doc.fields, ensure_ascii=False),
                            json.dumps(doc.meta, ensure_ascii=False, default=str),
                        ),
                    )
                except sqlite3.IntegrityError:
                    raise DuplicateDocumentError(
                        f"a document with id {doc.id!r} is already in the library"
                    ) from None
                row = cur.lastrowid
                self.conn.executemany(
                    "INSERT OR IGNORE INTO factors(doc_row, factor, level, position) "
                    "VALUES (?, ?, ?, ?)",
                    [
                        (row, f, level, i)
                        for f, levels in doc.factors.items()
                        for i, level in enumerate(levels)
                    ],
                )
                self.conn.executemany(
                    "INSERT INTO chunks(doc_row, position, text, meta) VALUES (?, ?, ?, ?)",
                    [
                        (row, i, c.text, json.dumps(c.meta, ensure_ascii=False, default=str))
                        for i, c in enumerate(doc.chunks, start=1)
                    ],
                )
                body = [*doc.fields.values(), *(c.text for c in doc.chunks)]
                self.conn.execute(
                    "INSERT INTO documents_fts(doc_id, body) VALUES (?, ?)",
                    (doc.id, "\n".join(body)),
                )
                n += 1
        except BaseException:
            self.conn.rollback()
            raise
        self.conn.commit()
        return n

    def count_documents(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def count_chunks(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def documents(self) -> Iterator[Document]:
        """All documents in insertion order."""
        chunks: dict[int, list[Chunk]] = {}
        for row, text, meta in self.conn.execute(
            "SELECT doc_row, text, meta FROM chunks ORDER BY doc_row, position"
        ):
            chunks.setdefault(row, []).append(Chunk(text, json.loads(meta)))
        factors: dict[int, dict[str, list[str]]] = {}
        for row, factor, level in self.conn.execute(
            "SELECT doc_row, factor, level FROM factors ORDER BY doc_row, factor, position"
        ):
            factors.setdefault(row, {}).setdefault(factor, []).append(level)
        for row, doc_id, fields, meta in self.conn.execute(
            "SELECT row, id, fields, meta FROM documents ORDER BY row"
        ):
            yield Document(
                doc_id,
                json.loads(fields),
                json.loads(meta),
                factors.get(row, {}),
                chunks.get(row, []),
            )

    def get_document(self, doc_id: str) -> Document | None:
        r = self.conn.execute(
            "SELECT row, fields, meta FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if r is None:
            return None
        factors: dict[str, list[str]] = {}
        for factor, level in self.conn.execute(
            "SELECT factor, level FROM factors WHERE doc_row = ? ORDER BY factor, position",
            (r[0],),
        ):
            factors.setdefault(factor, []).append(level)
        chunks = [
            Chunk(text, json.loads(meta))
            for text, meta in self.conn.execute(
                "SELECT text, meta FROM chunks WHERE doc_row = ? ORDER BY position", (r[0],)
            )
        ]
        return Document(doc_id, json.loads(r[1]), json.loads(r[2]), factors, chunks)

    def factor_names(self) -> list[str]:
        return [
            r[0] for r in self.conn.execute("SELECT DISTINCT factor FROM factors ORDER BY factor")
        ]

    def lexical_search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """BM25 keyword search. Returns ``(doc_id, score)`` with higher scores better."""
        terms = [t for t in query.replace('"', " ").split() if t]
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        rows = self.conn.execute(
            "SELECT doc_id, bm25(documents_fts) AS s FROM documents_fts "
            "WHERE documents_fts MATCH ? ORDER BY s LIMIT ?",
            (match, k),
        ).fetchall()
        return [(r[0], -float(r[1])) for r in rows]

    # -- builds -------------------------------------------------------------------------

    def record_build(
        self, embedder: str, config: dict[str, Any], n_documents: int, vocab_size: int | None
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO builds(created_at, embedder, config, n_documents, vocab_size) "
            "VALUES (?, ?, ?, ?, ?)",
            (_now(), embedder, json.dumps(config), n_documents, vocab_size),
        )
        self.conn.commit()
        assert cur.lastrowid is not None
        return int(cur.lastrowid)

    def builds(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, created_at, embedder, config, n_documents, vocab_size FROM builds "
            "ORDER BY id"
        ).fetchall()
        keys = ("id", "created_at", "embedder", "config", "n_documents", "vocab_size")
        out = [dict(zip(keys, r, strict=True)) for r in rows]
        for b in out:
            b["config"] = json.loads(b["config"])
        return out
