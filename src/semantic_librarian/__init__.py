"""Semantic Librarian: searchable vector-space libraries from your own documents."""

from semantic_librarian.document import Chunk, Document
from semantic_librarian.library import Hit, Library, LibraryError, SearchResult

__version__ = "0.1.0.dev0"

__all__ = ["Chunk", "Document", "Hit", "Library", "LibraryError", "SearchResult", "__version__"]
