"""Turn a BibTeX file (for example a Zotero export) into a searchable library.

The Python version of the R package's "Semantic Librarian for bib files".

    python examples/bibliography.py [BIB_FILE] [LIBRARY_FOLDER]
"""

from __future__ import annotations

import sys
from pathlib import Path

from semantic_librarian import Library

CRUMP_BIB = Path(__file__).parent.parent / "tests" / "fixtures" / "crump.bib"


def main(bib: str = str(CRUMP_BIB), folder: str = "libraries/bibliography") -> None:
    if Path(folder, "config.toml").exists():
        lib = Library.open(folder)
    else:
        lib = Library.create(folder)
        # every entry becomes a document; author, year and venue become factors
        lib.add_file(bib)
        lib.build()
    print({name: len(space) for name, space in lib.spaces.items()})

    # A map of the whole bibliography, as in the R vignette.
    labels = lib.space("document").labels
    projection = lib.project(labels, n_clusters=3)
    for cluster in range(3):
        members = [lab for lab, c in zip(labels, projection.clusters, strict=True) if c == cluster]
        print(f"\ncluster {cluster}: {len(members)} documents, for example")
        for label in members[:3]:
            print("   ", lib.document(label).title()[:75])

    print("\nwords closest to 'typing'")
    for hit in lib.search("typing", space="word", k=6).hits:
        print(f"{hit.rank:>3}  {hit.score:.3f}  {hit.label}")
    lib.close()


if __name__ == "__main__":
    main(*sys.argv[1:3])
