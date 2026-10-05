"""Make a library from any plain text, with your own factors.

The Python version of the R package's "Semantic Librarian for any text (Alice in
Wonderland)". Every sentence becomes a document. Each sentence carries a ``chapter``
factor, so chapters get vectors too and can be compared with sentences and words.

    python examples/any_text.py TEXT_FILE [LIBRARY_FOLDER] [QUERY]

Chapters are found from lines that start with "CHAPTER"; a text without such lines is
treated as one chapter.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from semantic_librarian import Document, Library
from semantic_librarian.text.clean import split_sentences

CHAPTER = re.compile(r"^\s*chapter\b.*$", re.IGNORECASE | re.MULTILINE)


def sentences_by_chapter(text: str) -> list[Document]:
    """One document per sentence, labelled with the chapter it comes from."""
    headings = CHAPTER.findall(text)
    bodies = CHAPTER.split(text)
    # text before the first heading (a title page, say) belongs to no chapter
    chapters = list(zip(headings, bodies[1:], strict=True)) if headings else [("1", text)]
    docs = []
    for number, (heading, body) in enumerate(chapters, start=1):
        name = " ".join(heading.split()) if headings else "1"
        for n, sentence in enumerate(split_sentences(" ".join(body.split())), start=1):
            docs.append(
                Document(
                    id=f"c{number:02d}-s{n:04d}",
                    fields={"text": sentence},
                    meta={"position": n},
                    factors={"chapter": [name]},
                )
            )
    return docs


def main(text_file: str, folder: str = "libraries/any-text", query: str = "") -> None:
    if Path(folder, "config.toml").exists():
        lib = Library.open(folder)
    else:
        lib = Library.create(folder)
        text = Path(text_file).read_text(encoding="utf-8", errors="replace")
        lib.add_documents(sentences_by_chapter(text), source=text_file)
        lib.build()
    print({name: len(space) for name, space in lib.spaces.items()})

    if query:
        print(f"\nsentences closest to: {query}")
        for hit in lib.search(query, k=5).hits:
            chapter = hit.document.factors["chapter"][0]
            print(f"{hit.rank:>3}  {hit.score:.3f}  [{chapter}] {hit.document.title()[:70]}")
        print(f"\nchapters closest to: {query}")
        for hit in lib.search(query, space="chapter", k=3).hits:
            print(f"{hit.rank:>3}  {hit.score:.3f}  {hit.label}")

    # Any space can be compared with any other: here, chapters with chapters.
    first = lib.space("chapter").labels[0]
    print(f"\nchapters closest to: {first}")
    for hit in lib.similar(first, space="chapter", k=4):
        print(f"{hit.rank:>3}  {hit.score:.3f}  {hit.label}")
    lib.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(*sys.argv[1:4])
