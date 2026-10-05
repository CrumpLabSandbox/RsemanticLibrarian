"""Plain text and Markdown ingestion.

A file can become one document, one document per paragraph (blank-line separated), or
one document per sentence (the "Alice in Wonderland" example). Each document gets a
``source`` factor naming its file, so files themselves become a searchable space.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from semantic_librarian.document import Document
from semantic_librarian.text.clean import split_sentences

Split = Literal["document", "paragraph", "sentence"]
_PARAGRAPH = re.compile(r"\n\s*\n")


def read_text(path: Path, split: Split = "document") -> Iterator[Document]:
    text = path.read_text(encoding="utf-8", errors="replace")
    stem = path.stem
    if split == "document":
        units = [text.strip()]
    elif split == "paragraph":
        units = [" ".join(p.split()) for p in _PARAGRAPH.split(text)]
    elif split == "sentence":
        units = split_sentences(" ".join(text.split()))
    else:
        raise ValueError(f"split must be document, paragraph or sentence, not {split!r}")
    units = [u for u in units if u.strip()]
    for n, unit in enumerate(units, start=1):
        doc_id = stem if split == "document" else f"{stem}-{split[0]}{n:06d}"
        yield Document(
            doc_id,
            {"text": unit},
            {"source": path.name, "position": n},
            {"source": [stem]},
        )
