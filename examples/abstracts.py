"""Search a library of journal abstracts: text search, a map, and neighbourhoods.

The Python version of the R package's "Semantic Librarian Tutorial". It uses the same
100 abstracts from the Canadian Journal of Experimental Psychology (1947-1949).

    python examples/abstracts.py [LIBRARY_FOLDER]
"""

from __future__ import annotations

import sys
from pathlib import Path

from semantic_librarian import Library

SAMPLE = Path(__file__).parent.parent / "tests" / "fixtures" / "apa_sample" / "articles.json"


def main(folder: str = "libraries/abstracts") -> None:
    if Path(folder, "config.toml").exists():
        lib = Library.open(folder)
    else:
        lib = Library.create(folder)
        # title, abstract and keywords are embedded; authorlist and year become factors
        lib.add_file(SAMPLE)
        lib.build()

    # 1. Search with free text. Words outside the library's vocabulary are reported.
    result = lib.search("president of psychology", mode="compound", k=25)
    print("query words used:", result.terms, "| ignored:", result.unknown)
    for hit in result.hits[:5]:
        print(f"{hit.rank:>3}  {hit.score:.3f}  {hit.document.title()}")

    # 2. A 2-D map of those results: classical MDS, then k-means clusters.
    projection = lib.project([hit.label for hit in result.hits], n_clusters=3)
    print("\nmap coordinates and cluster of the first five results")
    for label, (x, y), cluster in zip(
        projection.labels[:5], projection.coords, projection.clusters, strict=False
    ):
        print(f"{x:>7.3f} {y:>7.3f}  cluster {cluster}  {lib.document(label).title()[:50]}")

    # 3. Documents similar to a document (the first one in the library).
    first = lib.documents[0]
    print(f"\nclosest to: {first.title()}")
    for hit in lib.similar(first.id, space="document", k=5):
        print(f"{hit.rank:>3}  {hit.score:.3f}  {hit.document.title()}")

    # 4. Authors similar to an author, and the documents closest to that author.
    author = lib.space("author").labels[0]
    print(f"\nauthors closest to: {author}")
    for hit in lib.similar(author, space="author", k=5):
        print(f"{hit.rank:>3}  {hit.score:.3f}  {hit.label}")
    print(f"\ndocuments closest to: {author}")
    for hit in lib.similar(author, space="author", target="document", k=3):
        print(f"{hit.rank:>3}  {hit.score:.3f}  {hit.document.title()}")
    lib.close()


if __name__ == "__main__":
    main(*sys.argv[1:2])
