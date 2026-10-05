from __future__ import annotations

import pytest

from semantic_librarian.text import (
    TextProcessor,
    breakdown,
    legacy_sentences,
    legacy_tokens,
    resolve_stopwords,
    split_sentences,
    unicode_tokens,
)


def test_breakdown_matches_r(cleaning_ref):
    for text, expected in zip(cleaning_ref["inputs"], cleaning_ref["breakdown"], strict=True):
        assert breakdown(text) == expected, text


def test_legacy_tokens_match_sl_clean(cleaning_ref):
    for text, expected in zip(cleaning_ref["inputs"], cleaning_ref["sl_clean"], strict=True):
        assert legacy_tokens(text) == expected, text


def test_legacy_dictionary_matches_sl_corpus_dictionary(cleaning_ref):
    tokens = [t for text in cleaning_ref["inputs"] for t in legacy_tokens(text)]
    assert list(dict.fromkeys(tokens)) == cleaning_ref["sl_corpus_dictionary"]


def test_legacy_sentences_match_sl_clean_vector(cleaning_ref):
    got = [s for text in cleaning_ref["sl_clean_vector_inputs"] for s in legacy_sentences(text)]
    assert got == cleaning_ref["sl_clean_vector"]


def test_legacy_sentences_edge_cases():
    assert legacy_sentences("") == []
    assert legacy_sentences("a.") == [["a"]]
    assert legacy_sentences("a..b") == [["a"], [], ["b"]]


def test_unicode_tokens():
    assert unicode_tokens("Don't split Köhler's N400 in 1990 naïve STRASSE straße") == [
        "dont",
        "split",
        "kohlers",
        "n400",
        "in",
        "naive",
        "strasse",
        "strasse",
    ]
    assert unicode_tokens("日本語 текст") == ["日本語", "текст"]
    assert unicode_tokens("3.14 -- 42") == []


def test_split_sentences_protects_abbreviations():
    text = (
        "Smith et al. found effects (e.g., N400). J. Doe disagreed! "
        "The value was 3.14 here.\n\nNew paragraph without a stop"
    )
    assert split_sentences(text) == [
        "Smith et al. found effects (e.g., N400).",
        "J. Doe disagreed!",
        "The value was 3.14 here.",
        "New paragraph without a stop",
    ]


def test_text_processor_modes():
    legacy = TextProcessor("legacy")
    modern = TextProcessor("unicode")
    text = "Typing is fast, e.g. 100 wpm. Errors are rare."
    # the R package splits on every period, so "e.g." and the number create breaks
    assert legacy.sentences(text) == [
        ["typing", "is", "fast", "e"],
        ["g"],
        ["wpm"],
        ["errors", "are", "rare"],
    ]
    assert modern.sentences(text) == [
        ["typing", "is", "fast", "e", "g", "wpm"],
        ["errors", "are", "rare"],
    ]
    with pytest.raises(ValueError):
        TextProcessor("klingon")  # type: ignore[arg-type]


def test_resolve_stopwords():
    assert resolve_stopwords(None) == frozenset()
    assert "the" in resolve_stopwords("english")
    assert resolve_stopwords(["x"]) == frozenset({"x"})
    with pytest.raises(ValueError):
        resolve_stopwords("klingon")
