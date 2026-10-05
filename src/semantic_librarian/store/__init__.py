"""Persistent storage for a library's documents."""

from semantic_librarian.store.sqlite import DuplicateDocumentError, LibraryStore

__all__ = ["DuplicateDocumentError", "LibraryStore"]
