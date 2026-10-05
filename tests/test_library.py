from __future__ import annotations

import json

import numpy as np
import pytest

from semantic_librarian import Document, Library, LibraryError
from semantic_librarian.config import LibraryConfig


@pytest.fixture()
def built(tmp_path, crump_bib):
    lib = Library.create(tmp_path / "lib", embedder_params={"dim": 256})
    lib.add_file(crump_bib)
    lib.build()
    yield lib
    lib.close()


def test_create_and_open_errors(tmp_path):
    with pytest.raises(LibraryError):
        Library.open(tmp_path / "nothing")
    Library.create(tmp_path / "lib").close()
    with pytest.raises(LibraryError):
        Library.create(tmp_path / "lib")
    with Library.open(tmp_path / "lib") as lib:
        assert not lib.is_built
        with pytest.raises(LibraryError, match="no documents"):
            lib.build()
        with pytest.raises(LibraryError, match="not been built"):
            lib.search("x")


def test_build_creates_spaces(built):
    info = built.build_info
    assert set(info["spaces"]) == {"document", "word", "author", "year", "venue"}
    assert info["spaces"]["document"] == 39
    assert built.space("document").dim == 256
    assert not built.is_stale
    assert (built.root / "vectors" / "vocab.json").exists()
    with pytest.raises(LibraryError, match="no space"):
        built.space("nope")


def test_search_finds_the_keyboard_paper(built):
    result = built.search("keyboard typing skill", k=3)
    assert result.terms == ["keyboard", "typing", "skill"]
    assert "crumpWarningThisKeyboard2010" in [h.label for h in result.hits]
    assert result.hits[0].document is not None
    assert [h.rank for h in result.hits] == [1, 2, 3]
    assert result.hits[0].score >= result.hits[1].score


def test_search_modes_unknown_words_and_filters(built):
    for mode in ("compound", "and", "or"):
        assert len(built.search("attention memory", mode=mode, k=5).hits) == 5
    res = built.search("zebra attention", k=2)
    assert res.unknown == ["zebra"] and res.terms == ["attention"]
    assert built.search("zebra").hits == []

    def recent(doc):  # one entry's year is "In press"
        year = doc.factors["year"][0]
        return year.isdigit() and int(year) >= 2016

    hits = built.search("attention", k=None, where=recent).hits
    assert hits and all(recent(h.document) for h in hits)
    assert len(hits) < len(built.documents)
    with pytest.raises(ValueError):
        built.search("x", mode="xor")  # type: ignore[arg-type]
    with pytest.raises(LibraryError):
        built.search("attention", space="author", where=lambda d: True)


def test_lexical_and_hybrid_search(built):
    lexical = built.search("keyboard", method="lexical", k=None)
    assert lexical.hits and all(
        "keyboard" in " ".join(h.document.fields.values()).lower() for h in lexical.hits
    )
    hybrid = built.search("keyboard typing", method="hybrid", k=5)
    assert hybrid.method == "hybrid" and set(hybrid.hits[0].extra) == {"semantic", "lexical"}
    assert hybrid.to_dict()["hits"][0]["components"]
    semantic_only = built.search("keyboard typing", method="hybrid", lexical_weight=0.0, k=5)
    plain = built.search("keyboard typing", k=5)
    assert [h.label for h in semantic_only.hits] == [h.label for h in plain.hits]
    with pytest.raises(LibraryError):
        built.search("typing", method="lexical", space="author")
    with pytest.raises(ValueError):
        built.search("typing", method="fuzzy")
    with pytest.raises(ValueError):
        built.search("typing", method="hybrid", lexical_weight=2)


def test_word_space_hides_stopwords(built):
    hidden = [h.label for h in built.search("the typing", space="word", k=30).hits]
    shown = [
        h.label for h in built.search("typing", space="word", k=None, include_stopwords=True).hits
    ]
    assert "the" not in hidden and "and" not in hidden
    assert "the" in shown
    assert "the" not in [h.label for h in built.similar("typing", space="word", k=30)]


def test_missing_keyword_index(built):
    (built.root / "vectors" / "doc_terms.npz").unlink()
    built._invalidate()
    with pytest.raises(LibraryError, match="keyword index"):
        built.search("typing", method="lexical")


def test_similar_and_cross_space(built):
    authors = built.similar("Crump, Matthew J. C.", space="author", k=3)
    assert authors[0].label == "Crump, Matthew J. C." and authors[0].score == pytest.approx(1)
    docs = built.similar("Crump, Matthew J. C.", space="author", target="document", k=2)
    assert all(h.document is not None for h in docs)
    words = built.search("typing", space="word", k=1)
    assert words.hits[0].label == "typing"
    with pytest.raises(KeyError):
        built.similar("Nobody", space="author")


def test_project(built):
    labels = [h.label for h in built.search("memory", k=10).hits]
    p = built.project(labels, n_clusters=2)
    assert p.coords.shape == (10, 2) and set(p.clusters) <= {0, 1}


def test_add_file_skips_unchanged_and_detects_staleness(built, crump_bib, tmp_path):
    assert built.add_file(crump_bib).skipped
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps([{"id": "new", "title": "A new paper on attention"}]))
    built.add_file(extra)
    assert built.is_stale
    built.build()
    assert not built.is_stale and built.build_info["n_documents"] == 40
    assert len(built.store.builds()) == 2


def test_reopen_and_reproducible(built):
    first = np.array(built.space("document").vectors)
    root = built.root
    built.close()
    with Library.open(root) as again:
        np.testing.assert_array_equal(np.array(again.space("document").vectors), first)
        again.build()
        np.testing.assert_allclose(np.array(again.space("document").vectors), first, rtol=1e-6)


def test_other_embedders_and_legacy_mode(tmp_path, apa_sample):
    docs = [
        Document(
            str(a["index"]),
            {"title": a["title"], "abstract": a["abstract"]},
            {"year": a["year"]},
            {"author": a["authorlist"].split(";"), "year": [a["year"]]},
        )
        for a in apa_sample["articles"][:30]
    ]
    with Library.create(tmp_path / "apa", text_mode="legacy") as lib:
        assert lib.add_documents(docs, source="apa") == 30
        for name, params in [
            ("beagle-legacy", {"dim": 128, "nonzeros": 8}),
            ("beagle-rp", {"dim": 256, "nonzeros": 16}),
            ("random", {"dim": 64}),
            ("beagle", {"dim": 64, "max_ngram": 3}),
        ]:
            lib.build(embedder=name, params=params)
            assert lib.config.embedder == name
            assert lib.search("psychology in canada", k=3).hits
        assert LibraryConfig.load(lib.root / "config.toml").embedder_params["max_ngram"] == 3


def test_reserved_factor_names(tmp_path):
    with Library.create(tmp_path / "lib", embedder_params={"dim": 32}) as lib:
        lib.add_documents(
            [
                Document("a", {"text": "one two three"}, factors={"word": ["x"]}),
                Document("b", {"text": "two three four"}, factors={"word": ["y"]}),
            ]
        )
        lib.build()
        assert "factor-word" in lib.spaces


def test_config_roundtrip(tmp_path):
    cfg = LibraryConfig(name='my "lib"', fields=["title"], embedder_params={"dim": 8, "x": True})
    cfg.save(tmp_path / "c.toml")
    assert LibraryConfig.load(tmp_path / "c.toml") == cfg
