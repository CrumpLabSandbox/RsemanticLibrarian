"""Cut a long text into passages ("chunks") of roughly equal length.

Chunks never overlap and together hold every word of the text once, so the sum of the
chunk vectors of a word-sum model is the vector of the whole document.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from semantic_librarian.document import Chunk
from semantic_librarian.text.clean import split_sentences

DEFAULT_CHUNK_WORDS = 200


@dataclass
class Paragraph:
    """A paragraph and the pages (1-based) it starts and ends on."""

    text: str
    page: int
    page_end: int


def _units(paragraphs: Iterable[Paragraph], max_words: int) -> list[Paragraph]:
    """Paragraphs, with overlong ones cut at sentence ends (or, failing that, mid-sentence)."""
    units: list[Paragraph] = []
    for para in paragraphs:
        if len(para.text.split()) <= max_words:
            units.append(para)
            continue
        for sentence in split_sentences(para.text):
            words = sentence.split()
            for start in range(0, len(words), max_words):
                piece = " ".join(words[start : start + max_words])
                units.append(Paragraph(piece, para.page, para.page_end))
    return units


def chunk_paragraphs(
    paragraphs: Iterable[Paragraph], target_words: int = DEFAULT_CHUNK_WORDS
) -> list[Chunk]:
    """Pack paragraphs into chunks of about ``target_words`` words.

    Paragraphs are kept whole when they fit. A chunk is closed once it reaches the target,
    and never grows beyond one and a half times the target. A short remainder at the end
    joins the last chunk. Paragraphs inside a chunk are separated by blank lines, which the
    sentence splitter treats as sentence ends.
    """
    if target_words < 1:
        raise ValueError("target_words must be at least 1")
    max_words = max(target_words, target_words * 3 // 2)
    groups: list[list[Paragraph]] = []
    current: list[Paragraph] = []
    size = 0
    for unit in _units(paragraphs, max_words):
        n = len(unit.text.split())
        if n == 0:
            continue
        if current and size + n > max_words:
            groups.append(current)
            current, size = [], 0
        current.append(unit)
        size += n
        if size >= target_words:
            groups.append(current)
            current, size = [], 0
    if current:
        if groups and size < target_words // 4:
            groups[-1].extend(current)
        else:
            groups.append(current)
    return [
        Chunk(
            "\n\n".join(u.text for u in group),
            {"page": group[0].page, "page_end": group[-1].page_end},
        )
        for group in groups
    ]
