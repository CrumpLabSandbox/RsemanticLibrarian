"""Library configuration stored as ``config.toml`` in the library directory."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from semantic_librarian.store.sqlite import SCHEMA_VERSION


@dataclass
class LibraryConfig:
    """Settings that decide how a library is built.

    Attributes:
        name: Display name.
        text_mode: ``"unicode"`` (default) or ``"legacy"`` for exact R-package cleaning.
        fields: Document fields to embed, in order. Empty means every field.
        factors: Factors that get their own vector space. Empty means every factor.
        factor_aggregate: ``"sum"`` (paper Eq. 11) or ``"mean"``.
        embedder: Registered embedder name (see ``sl embedders``).
        embedder_params: Overrides for the embedder's defaults, e.g. ``{"dim": 512}``.
    """

    name: str
    text_mode: str = "unicode"
    fields: list[str] = field(default_factory=list)
    factors: list[str] = field(default_factory=list)
    factor_aggregate: str = "sum"
    embedder: str = "beagle"
    embedder_params: dict[str, Any] = field(default_factory=dict)

    def to_toml(self) -> str:
        lines = [
            "# Semantic Librarian library configuration.",
            "# Edit and run `sl build` again to apply changes.",
            "",
            "[library]",
            f"name = {_toml_value(self.name)}",
            f"schema_version = {SCHEMA_VERSION}",
            "",
            "[text]",
            "# 'unicode' (default) or 'legacy' (exactly the R package's cleaning)",
            f"mode = {_toml_value(self.text_mode)}",
            "# document fields to embed, in order; empty = all fields",
            f"fields = {_toml_value(self.fields)}",
            "",
            "[factors]",
            "# factors that get their own vector space; empty = all factors",
            f"include = {_toml_value(self.factors)}",
            "# 'sum' (as in the paper) or 'mean'",
            f"aggregate = {_toml_value(self.factor_aggregate)}",
            "",
            "[embedder]",
            "# beagle | beagle-legacy | beagle-rp | random",
            f"name = {_toml_value(self.embedder)}",
            "",
            "[embedder.params]",
            '# e.g. dim = 1024, seed = 0, max_ngram = 7, stopwords = "english"',
        ]
        lines += [f"{k} = {_toml_value(v)}" for k, v in sorted(self.embedder_params.items())]
        return "\n".join(lines) + "\n"

    def save(self, path: Path) -> None:
        path.write_text(self.to_toml())

    @classmethod
    def load(cls, path: Path) -> LibraryConfig:
        data = tomllib.loads(path.read_text())
        lib = data.get("library", {})
        text = data.get("text", {})
        factors = data.get("factors", {})
        emb = data.get("embedder", {})
        return cls(
            name=lib.get("name", path.parent.name),
            text_mode=text.get("mode", "unicode"),
            fields=list(text.get("fields", [])),
            factors=list(factors.get("include", [])),
            factor_aggregate=factors.get("aggregate", "sum"),
            embedder=emb.get("name", "beagle"),
            embedder_params=dict(emb.get("params", {})),
        )


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float):
        return repr(v)
    if isinstance(v, str):
        escaped = v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    if isinstance(v, list | tuple):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if v is None:
        raise ValueError("TOML has no null; omit the key instead")
    raise TypeError(f"cannot write {type(v).__name__} to TOML")
