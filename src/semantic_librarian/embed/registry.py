"""Look up embedders by name."""

from __future__ import annotations

from typing import Any

from semantic_librarian.embed.base import Embedder
from semantic_librarian.embed.beagle import LEGACY_PRESET, Beagle
from semantic_librarian.embed.beagle_rp import BeagleRP
from semantic_librarian.embed.random import RandomVectors

EMBEDDERS: dict[str, str] = {
    "beagle": "BEAGLE with context and order information (Jones & Mewhort, 2007). Default.",
    "beagle-legacy": "Context-only BEAGLE exactly as in the R package RsemanticLibrarian.",
    "beagle-rp": "BEAGLE with random permutations (Sahlgren et al., 2008).",
    "random": "Non-semantic control: words are their random environment vectors.",
}


def make_embedder(name: str, **params: Any) -> Embedder:
    """Create an embedder from its registered name and parameter overrides."""
    key = name.lower()
    if key == "beagle":
        return Beagle(**params)
    if key == "beagle-legacy":
        return Beagle(**{**LEGACY_PRESET, **params})
    if key == "beagle-rp":
        return BeagleRP(**params)
    if key == "random":
        return RandomVectors(**params)
    known = ", ".join(EMBEDDERS)
    raise ValueError(f"unknown embedder {name!r}; available: {known}")


def embedder_from_config(config: dict[str, Any]) -> Embedder:
    """Recreate an embedder from the dictionary returned by ``Embedder.config()``."""
    params = dict(config)
    name = params.pop("name")
    if name == "beagle" and params.get("variant") == "legacy":
        name = "beagle-legacy"
    return make_embedder(name, **params)
