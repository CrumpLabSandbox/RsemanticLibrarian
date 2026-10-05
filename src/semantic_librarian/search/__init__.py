"""Similarity search over vector spaces."""

from semantic_librarian.search.fusion import reciprocal_rank_fusion
from semantic_librarian.search.lexical import BM25, doc_term_matrix, word_match_scores
from semantic_librarian.search.similarity import (
    QUERY_MODES,
    QueryMode,
    combine_term_scores,
    cosine_scores,
    rank,
)

__all__ = [
    "BM25",
    "QUERY_MODES",
    "QueryMode",
    "combine_term_scores",
    "cosine_scores",
    "doc_term_matrix",
    "rank",
    "reciprocal_rank_fusion",
    "word_match_scores",
]
