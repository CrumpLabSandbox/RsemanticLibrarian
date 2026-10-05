"""Embedding models."""

from semantic_librarian.embed.base import Corpus, Embedder, WordSumEmbedder, compose_documents
from semantic_librarian.embed.beagle import Beagle, BeagleConfig, bind, order_vectors
from semantic_librarian.embed.beagle_rp import BeagleRP, BeagleRPConfig
from semantic_librarian.embed.random import RandomVectors
from semantic_librarian.embed.registry import EMBEDDERS, embedder_from_config, make_embedder

__all__ = [
    "EMBEDDERS",
    "Beagle",
    "BeagleConfig",
    "BeagleRP",
    "BeagleRPConfig",
    "Corpus",
    "Embedder",
    "RandomVectors",
    "WordSumEmbedder",
    "bind",
    "compose_documents",
    "embedder_from_config",
    "make_embedder",
    "order_vectors",
]
