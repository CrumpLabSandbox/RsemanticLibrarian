from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
LEGACY = FIXTURES / "legacy"
APA = FIXTURES / "apa_sample"


def as_list(value: Any) -> list[Any]:
    """jsonlite's auto_unbox turns length-1 vectors into scalars; undo that."""
    return value if isinstance(value, list) else [value]


@pytest.fixture(scope="session")
def cleaning_ref() -> dict[str, Any]:
    return json.loads((LEGACY / "cleaning.json").read_text())


@pytest.fixture(scope="session")
def beagle_ref() -> dict[str, Any]:
    ref = json.loads((LEGACY / "beagle_parity.json").read_text())
    ref["sentences"] = [as_list(s) for s in ref["sentences"]]
    ref["author_doc_rows"] = [as_list(r) for r in ref["author_doc_rows"]]
    arrays = np.load(LEGACY / "beagle_parity.npz")
    ref["arrays"] = {k: arrays[k] for k in arrays.files}
    return ref


@pytest.fixture(scope="session")
def search_ref() -> dict[str, Any]:
    return json.loads((LEGACY / "search.json").read_text())


@pytest.fixture(scope="session")
def apa_sample() -> dict[str, Any]:
    vectors = np.load(APA / "vectors.npz")
    return {
        "articles": json.loads((APA / "articles.json").read_text()),
        "dictionary": json.loads((APA / "dictionary.json").read_text()),
        "authors": json.loads((APA / "authors.json").read_text()),
        **{k: vectors[k] for k in vectors.files},
    }


@pytest.fixture()
def crump_bib() -> Path:
    return FIXTURES / "crump.bib"
