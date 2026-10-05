"""Write a built library as a static web bundle.

The bundle is the web app (``index.html`` and ``assets/``) plus a ``data/`` folder, and
needs no server-side code: the browser loads the vectors and ranks them itself. Any
static file host can serve it, as can ``sl serve``.

::

    out/
    ├── index.html, assets/      the web app
    └── data/
        ├── manifest.json        spaces, text mode, stop words, counts
        ├── spaces/              per space: vectors (.bin), row scales, labels (.json)
        ├── documents.json       titles, years and factor levels of every document
        ├── docs/                full records, a few hundred documents per file
        ├── chunks.json          the document and pages of every chunk
        └── chunks/              chunk text, a few hundred chunks per file

Vectors are stored as 8-bit integers by default: each row is divided by its largest
absolute value and scaled to -127..127, and the divisor is kept in a ``scale`` file so
that rows can still be added together (queries are sums of word vectors). This takes a
quarter of the space of 32-bit floats and changes cosines by well under 0.01.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from semantic_librarian.library import RESERVED_SPACES, Library, LibraryError
from semantic_librarian.space.vector_space import VectorSpace

EXPORT_FORMAT = 1
Precision = Literal["int8", "float32"]
PRECISIONS: tuple[str, ...] = ("int8", "float32")
SHARD = 500
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
_BLOCK = 4096


@dataclass
class ExportReport:
    path: Path
    spaces: dict[str, int]
    megabytes: float


def frontend_dir() -> Path:
    """The built web app shipped with the package."""
    if not (WEB_DIR / "index.html").exists():
        raise LibraryError(
            f"the web app has not been built ({WEB_DIR} has no index.html); "
            "run `npm install && npm run build` in the repository's web/ folder"
        )
    return WEB_DIR


def app_version() -> str:
    """A fingerprint of the built web app, so `sl serve` notices when it has changed."""
    digest = hashlib.sha256()
    for file in sorted(frontend_dir().rglob("*")):
        if file.is_file() and file.suffix != ".py":
            digest.update(file.read_bytes())
    return digest.hexdigest()[:12]


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))


def _write_vectors(space: VectorSpace, path: Path, precision: Precision) -> Path | None:
    """Write the matrix row by row; returns the scale file for 8-bit exports."""
    scale_path = path.with_suffix(".scale.bin") if precision == "int8" else None
    scales = np.zeros(len(space), dtype="<f4")
    with path.open("wb") as out:
        for start in range(0, len(space), _BLOCK):
            block = np.asarray(space.vectors[start : start + _BLOCK], dtype=np.float32)
            if precision == "float32":
                out.write(block.astype("<f4").tobytes())
                continue
            largest = np.abs(block).max(axis=1, keepdims=True)
            largest[largest == 0] = 1.0
            out.write(np.rint(block / largest * 127.0).astype(np.int8).tobytes())
            scales[start : start + _BLOCK] = largest[:, 0] / 127.0
    if scale_path is not None:
        scale_path.write_bytes(scales.tobytes())
    return scale_path


def _prepare(out: Path) -> None:
    """Create ``out``, or clear a previous export. Anything else is left alone."""
    if out.exists() and not out.is_dir():
        raise LibraryError(f"{out} is a file, not a folder")
    if out.exists() and any(out.iterdir()):
        if not (out / "data" / "manifest.json").exists():
            raise LibraryError(
                f"{out} is not empty and is not an earlier export; choose another folder"
            )
        for name in ("data", "assets"):
            shutil.rmtree(out / name, ignore_errors=True)
        (out / "index.html").unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)


def export_library(
    lib: Library,
    out: Path | str,
    precision: Precision = "int8",
    log: Callable[[str], None] | None = None,
) -> ExportReport:
    """Write the web app and the library's data to ``out``.

    Args:
        lib: A built library.
        out: Folder to write; it must be new, empty, or an earlier export (replaced).
        precision: ``int8`` (default, compact) or ``float32`` (exact, four times larger).
    """
    say = log or (lambda _msg: None)
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    info = lib.build_info
    if lib.is_stale:
        raise LibraryError("documents changed since the last build; run `sl build` first")
    frontend = frontend_dir()
    out = Path(out)
    _prepare(out)
    shutil.copytree(frontend, out, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.py"))
    data = out / "data"
    (data / "spaces").mkdir(parents=True)

    spaces: list[dict[str, Any]] = []
    for i, (name, space) in enumerate(lib.spaces.items()):
        say(f"writing space {name!r} ({len(space)} items)")
        stem = f"spaces/{i:02d}"
        scale = _write_vectors(space, data / f"{stem}.bin", precision)
        _write_json(data / f"{stem}.labels.json", space.labels)
        spaces.append(
            {
                "name": name,
                "kind": name if name in RESERVED_SPACES else "factor",
                "n": len(space),
                "dim": space.dim,
                "dtype": precision,
                "vectors": f"{stem}.bin",
                "scale": f"{stem}.scale.bin" if scale is not None else None,
                "labels": f"{stem}.labels.json",
            }
        )

    docs = lib.documents
    if [d.id for d in docs] != lib.space("document").labels:  # pragma: no cover - defensive
        raise LibraryError("the document space does not match the stored documents")
    say(f"writing {len(docs)} documents")
    factor_levels: dict[str, list[list[int]]] = {}
    for space_name, space in lib.spaces.items():
        if space_name in RESERVED_SPACES:
            continue
        # a factor named like a built-in space is stored as "factor-<name>"
        bare = space_name.removeprefix("factor-")
        factor = bare if bare in RESERVED_SPACES else space_name
        index = space.index
        factor_levels[space_name] = [
            [index[level] for level in d.factors.get(factor, []) if level in index] for d in docs
        ]
    years = [d.factors.get("year", [d.meta.get("year")])[0] for d in docs]
    _write_json(
        data / "documents.json",
        {
            "title": [" ".join(d.title().split()) for d in docs],
            "year": [None if y is None else str(y) for y in years],
            "factors": factor_levels,
        },
    )
    (data / "docs").mkdir()
    chunk_doc: list[int] = []
    chunk_pages: list[list[Any]] = []
    chunk_texts: list[str] = []
    for start in range(0, len(docs), SHARD):
        records = []
        for row, d in enumerate(docs[start : start + SHARD], start=start):
            record: dict[str, Any] = {
                "id": d.id,
                "fields": d.fields,
                "meta": d.meta,
                "factors": d.factors,
            }
            if d.chunks:
                record["chunks"] = [len(chunk_doc), len(d.chunks)]
            for c in d.chunks:
                chunk_doc.append(row)
                chunk_pages.append([c.meta.get("page"), c.meta.get("page_end")])
                chunk_texts.append(c.text)
            records.append(record)
        (data / "docs" / f"{start // SHARD:05d}.json").write_text(
            json.dumps(records, ensure_ascii=False, separators=(",", ":"), default=str)
        )
    if chunk_doc:
        _write_json(data / "chunks.json", {"doc": chunk_doc, "pages": chunk_pages})
        (data / "chunks").mkdir()
        for start in range(0, len(chunk_texts), SHARD):
            _write_json(
                data / "chunks" / f"{start // SHARD:05d}.json", chunk_texts[start : start + SHARD]
            )

    model = lib.embedder
    _write_json(
        data / "manifest.json",
        {
            "format": EXPORT_FORMAT,
            "app": app_version(),
            "name": lib.config.name,
            "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "build_id": info.get("build_id"),
            "built_at": info["created_at"],
            "embedder": info["embedder"],
            "text_mode": info.get("text_mode", lib.config.text_mode),
            "stopwords": sorted(model.stopword_set),
            "compose_without_stopwords": bool(model.cfg.compose_without_stopwords),
            "n_documents": len(docs),
            "n_chunks": len(chunk_doc),
            "shard": SHARD,
            "spaces": spaces,
        },
    )
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    say(f"wrote {size:.1f} MB to {out}")
    return ExportReport(out, {s["name"]: s["n"] for s in spaces}, size)
