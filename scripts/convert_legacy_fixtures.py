"""Convert legacy R data into Python test fixtures.

Run from the repository root after ``scripts/legacy_reference.R``::

    python scripts/convert_legacy_fixtures.py

It writes:

* ``tests/fixtures/apa_sample/``: the 100-article APA sample bundled with the R package
  (metadata as JSON, word/article/author vectors as a compressed ``.npz``).
* ``tests/fixtures/legacy/beagle_parity.npz``: the raw float64 matrices written by the R
  reference script, packed into one compressed file (the ``.f64`` files are removed).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyreadr

ROOT = Path(__file__).resolve().parents[1]
RDA = ROOT / "legacy" / "R" / "data"
LEGACY = ROOT / "tests" / "fixtures" / "legacy"
SAMPLE = ROOT / "tests" / "fixtures" / "apa_sample"


def read_rda(name: str):  # type: ignore[no-untyped-def]
    return next(iter(pyreadr.read_r(str(RDA / f"{name}.rda")).values()))


def convert_sample() -> None:
    SAMPLE.mkdir(parents=True, exist_ok=True)
    articles = read_rda("article_df")
    keep = [
        "index",
        "journal",
        "year",
        "title",
        "abstract",
        "keywords",
        "authorlist",
        "pages",
        "issue",
        "volume",
    ]
    records = []
    for row in articles[keep].itertuples(index=False):
        rec = {}
        for key, value in zip(keep, row, strict=True):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                rec[key] = None
            elif key in ("index", "year"):
                rec[key] = int(value)
            else:
                rec[key] = str(value)
        records.append(rec)
    (SAMPLE / "articles.json").write_text(json.dumps(records, indent=1, ensure_ascii=False))

    dictionary = [str(w) for w in read_rda("dictionary_words").iloc[:, 0]]
    authors = [str(a) for a in read_rda("author_list").iloc[:, 0]]
    (SAMPLE / "dictionary.json").write_text(json.dumps(dictionary))
    (SAMPLE / "authors.json").write_text(json.dumps(authors, ensure_ascii=False))

    np.savez_compressed(
        SAMPLE / "vectors.npz",
        word_vectors=read_rda("WordVectors").to_numpy(dtype=np.float64),
        article_vectors=read_rda("article_vectors").to_numpy(dtype=np.float64),
        author_vectors=read_rda("AuthorVectors").to_numpy(dtype=np.float64),
    )


def pack_beagle() -> None:
    shapes = json.loads((LEGACY / "beagle_shapes.json").read_text())
    arrays = {}
    for name, shape in shapes.items():
        path = LEGACY / f"beagle_{name}.f64"
        arrays[name] = np.fromfile(path, dtype="<f8").reshape(shape)
    env = arrays["environment"]
    assert np.array_equal(env, np.round(env)), "environment vectors should be ternary"
    arrays["environment"] = env.astype(np.int8)
    np.savez_compressed(LEGACY / "beagle_parity.npz", **arrays)
    for name in shapes:
        (LEGACY / f"beagle_{name}.f64").unlink()
    (LEGACY / "beagle_shapes.json").unlink()


if __name__ == "__main__":
    convert_sample()
    pack_beagle()
    print("fixtures written")
