# Python reference

Everything the command line does is available from Python.

```python
from semantic_librarian import Document, Library

lib = Library.create("mylib")  # or Library.open("mylib")
lib.add_file("references.bib")
lib.add_documents([Document("note-1", {"text": "Some text."}, factors={"tag": ["idea"]})])
lib.build()

result = lib.search("attention and memory", mode="or", k=20)
for hit in result.hits:
    print(hit.rank, round(hit.score, 3), hit.document.title())

lib.similar("Crump, Matthew J. C.", space="author", target="document")
lib.project([hit.label for hit in result.hits], n_clusters=3)
lib.close()
```

A library can also be used as a context manager: `with Library.open("mylib") as lib:`.

## Library

::: semantic_librarian.library.Library
    options:
      members:
        - create
        - open
        - add_file
        - add_documents
        - build
        - search
        - similar
        - project
        - document
        - chunk
        - space
        - is_built
        - is_stale

## Records

::: semantic_librarian.document.Document
    options:
      members: [title]

::: semantic_librarian.document.Chunk
    options:
      members: []

## Results

::: semantic_librarian.library.SearchResult
    options:
      members: []

::: semantic_librarian.library.Hit
    options:
      members: []

::: semantic_librarian.project.maps.Projection
    options:
      members: []

## Vectors

`lib.space("author")` returns a `VectorSpace`: `labels` names the rows of the numpy
matrix `vectors`. Use it when you want the numbers themselves.

::: semantic_librarian.space.vector_space.VectorSpace
    options:
      members: [vector, row, norms]

## Export

::: semantic_librarian.export.bundle.export_library
