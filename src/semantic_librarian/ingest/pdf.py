"""PDF ingestion: full text, cleaned and cut into chunks.

Each PDF becomes one document. Text is read with pymupdf block by block, in the order the
file stores it (which follows the columns of most articles). Running headers, footers and
page numbers are dropped, words hyphenated across lines are rejoined, and paragraphs that
continue over a column or page break are merged. The text is then packed into chunks of a
few paragraphs, each remembering its pages.

Scanned PDFs without a text layer are rejected; there is no OCR. Tables, captions and
reference lists are kept as ordinary text.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semantic_librarian.document import Document
from semantic_librarian.text.chunk import DEFAULT_CHUNK_WORDS, Paragraph, chunk_paragraphs

# Share of the page height, at the top and at the bottom, where headers and footers live.
MARGIN = 0.1


@dataclass
class Block:
    """A block of text on a page, with its vertical extent as fractions of the page height."""

    page: int
    text: str
    top: float
    bottom: float

    @property
    def in_margin(self) -> bool:
        return self.bottom <= MARGIN or self.top >= 1.0 - MARGIN


# -- cleaning (no pymupdf needed) --------------------------------------------------------

_DIGITS = re.compile(r"\d+")
_PAGE_NUMBER = re.compile(r"(page )?#( ?(of|/) ?#)?")
_LINE_HYPHEN = re.compile(r"(?<=[^\W\d_])-\n(?=[^\W\d_])")
_ENDS_HYPHENATED = re.compile(r"[^\W\d_]-$")
_ENDS_SENTENCE = re.compile(r"[.!?:;][\"'”’)\]]*$")


def _margin_key(text: str) -> str:
    """Margin text with numbers masked, so "Smith 12" and "Smith 13" count as the same."""
    return " ".join(_DIGITS.sub("#", text.casefold()).split())


def strip_headers_footers(pages: Sequence[Sequence[Block]]) -> list[list[Block]]:
    """Drop page numbers and text that repeats in the top or bottom margin of many pages.

    A margin block is a running header or footer when the same text, numbers aside,
    appears in the margins of at least 30% of the pages (and at least two).
    """
    seen: Counter[str] = Counter()
    for blocks in pages:
        seen.update({_margin_key(b.text) for b in blocks if b.in_margin})
    threshold = max(2, math.ceil(0.3 * len(pages)))
    kept: list[list[Block]] = []
    for blocks in pages:
        keep = []
        for b in blocks:
            key = _margin_key(b.text)
            if b.in_margin and (seen[key] >= threshold or _PAGE_NUMBER.fullmatch(key)):
                continue
            keep.append(b)
        kept.append(keep)
    return kept


def dehyphenate(text: str) -> str:
    """Join the lines of a block, rejoining words split by a hyphen at a line end.

    "move-\\nment" becomes "movement". A hyphen is kept when the next line starts with a
    capital ("Stroop-\\nSimon"). Compounds broken at their own hyphen ("well-\\nknown") lose
    it, which is the usual price of this rule.
    """
    text = unicodedata.normalize("NFKC", text).replace("­", "")

    def join(match: re.Match[str]) -> str:
        return "" if text[match.end()].islower() else "-"

    return " ".join(_LINE_HYPHEN.sub(join, text).split())


def paragraphs_from_blocks(pages: Sequence[Sequence[Block]]) -> list[Paragraph]:
    """Turn blocks into paragraphs, merging those that continue across a break.

    A block continues the previous one when it starts with a lower-case letter and the
    previous block ends in a hyphenated word or without sentence-final punctuation.
    """
    paragraphs: list[Paragraph] = []
    for blocks in pages:
        for block in blocks:
            text = dehyphenate(block.text)
            if not text:
                continue
            last = paragraphs[-1] if paragraphs else None
            if last is not None and text[0].islower():
                if _ENDS_HYPHENATED.search(last.text):
                    last.text = last.text[:-1] + text
                    last.page_end = block.page
                    continue
                if not _ENDS_SENTENCE.search(last.text):
                    last.text = f"{last.text} {text}"
                    last.page_end = block.page
                    continue
            paragraphs.append(Paragraph(text, block.page, block.page))
    return paragraphs


_NOT_A_TITLE = re.compile(
    r"\.(pdf|docx?|tex|dvi|indd|qxd|ps|rtf)$|^(microsoft word|untitled)", re.I
)
_AUTHOR_SEPARATORS = re.compile(r"\s*;\s*|\s+and\s+|\s*&\s*")


def plausible_title(title: str | None) -> str | None:
    """The metadata title, unless it is empty or a file name left behind by the software."""
    title = " ".join((title or "").split())
    if len(title) < 4 or _NOT_A_TITLE.search(title):
        return None
    return title


def largest_text(spans: Sequence[tuple[float, str]]) -> str | None:
    """The text set in the largest font among ``(size, text)`` spans, in reading order.

    Returns ``None`` when nothing stands out: all text is one size, or the largest text is
    a single word or longer than a title.
    """
    sized = [(size, " ".join(text.split())) for size, text in spans]
    sized = [(size, text) for size, text in sized if sum(c.isalpha() for c in text) >= 2]
    if not sized:
        return None
    largest = max(size for size, _ in sized)
    if all(size >= largest - 0.5 for size, _ in sized):
        return None
    title = " ".join(text for size, text in sized if size >= largest - 0.5)
    return title if 4 <= len(title) <= 300 and " " in title else None


def split_authors(author: str | None) -> list[str]:
    """Split a metadata author string on ";", "and" and "&" (commas may be inside names)."""
    return [a for a in (p.strip(" ,") for p in _AUTHOR_SEPARATORS.split(author or "")) if a]


# -- reading -----------------------------------------------------------------------------


def _pymupdf() -> Any:
    try:
        import pymupdf
    except ImportError:  # pragma: no cover - depends on the installation
        raise ValueError(
            "reading PDFs needs pymupdf; install it with `pip install 'semantic-librarian[pdf]'`"
        ) from None
    # Problems are reported as exceptions; MuPDF's own messages on stderr are noise.
    pymupdf.TOOLS.mupdf_display_errors(False)
    pymupdf.TOOLS.mupdf_display_warnings(False)
    return pymupdf


def read_pdf(path: Path, chunk_words: int = DEFAULT_CHUNK_WORDS) -> Iterator[Document]:
    """Yield one chunked document for the PDF at ``path``.

    The document id is the file name without its extension. The title comes from the PDF
    metadata when that looks like a title, otherwise from the largest text on the first
    page, otherwise from the file name; ``meta["title_source"]`` says which. Metadata
    authors become the ``author`` factor.
    """
    pymupdf = _pymupdf()
    try:
        pdf = pymupdf.open(path)
    except Exception as exc:
        raise ValueError(f"cannot read {path.name} as a PDF: {exc}") from None
    with pdf:
        if pdf.needs_pass:
            raise ValueError(f"{path.name} is password-protected")
        info = pdf.metadata or {}
        pages: list[list[Block]] = []
        first_page_spans: list[tuple[float, str]] = []
        for number, page in enumerate(pdf, start=1):
            height = page.rect.height or 1.0
            pages.append(
                [
                    Block(number, text, y0 / height, y1 / height)
                    for _x0, y0, _x1, y1, text, _n, kind in page.get_text("blocks")
                    if kind == 0
                ]
            )
            if number == 1:
                first_page_spans = [
                    (span["size"], span["text"])
                    for block in page.get_text("dict")["blocks"]
                    for line in block.get("lines", [])
                    for span in line["spans"]
                ]

    chunks = chunk_paragraphs(paragraphs_from_blocks(strip_headers_footers(pages)), chunk_words)
    if not chunks:
        raise ValueError(
            f"{path.name} has no extractable text (a scanned PDF?); OCR is not supported"
        )
    title, title_source = plausible_title(info.get("title")), "metadata"
    if title is None:
        title, title_source = largest_text(first_page_spans), "largest text on page 1"
    if title is None:
        title, title_source = path.stem, "file name"
    meta: dict[str, Any] = {
        "source": path.name,
        "pages": len(pages),
        "title_source": title_source,
    }
    created = re.match(r"D:(\d{4})", info.get("creationDate") or "")
    if created:
        # When the file was made, which is often not the year of publication.
        meta["file_year"] = int(created.group(1))
    yield Document(
        path.stem,
        {"title": title},
        meta,
        {"author": split_authors(info.get("author"))},
        chunks,
    )
