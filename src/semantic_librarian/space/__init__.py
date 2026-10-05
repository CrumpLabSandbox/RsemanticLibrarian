"""Vocabularies, labelled vector spaces, factor spaces and the Library object."""

from semantic_librarian.space.factors import factor_space
from semantic_librarian.space.vector_space import VectorSpace
from semantic_librarian.space.vocabulary import Vocabulary

__all__ = ["VectorSpace", "Vocabulary", "factor_space"]
