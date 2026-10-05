from __future__ import annotations

import math

import numpy as np
import pytest

from semantic_librarian.search import (
    BM25,
    doc_term_matrix,
    reciprocal_rank_fusion,
    word_match_scores,
)
from semantic_librarian.search.fusion import ranks
from semantic_librarian.space import Vocabulary

DOCS = [["cat", "sat", "cat"], ["dog", "sat"], ["bird"], []]


@pytest.fixture()
def vocab():
    return Vocabulary.build(DOCS)


def test_doc_term_matrix(vocab):
    m = doc_term_matrix(DOCS, vocab).toarray()
    assert m.tolist() == [[2, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]


def test_bm25_matches_formula(vocab, tmp_path):
    bm25 = BM25(doc_term_matrix(DOCS, vocab), k1=1.2, b=0.75)
    scores = bm25.scores(vocab.ids(["cat", "cat"]))  # repeated terms count once
    avg = 6 / 4
    idf = math.log1p((4 - 1 + 0.5) / (1 + 0.5))
    expected = idf * 2 * 2.2 / (2 + 1.2 * (1 - 0.75 + 0.75 * 3 / avg))
    assert scores[0] == pytest.approx(expected)
    assert scores[1:].tolist() == [0, 0, 0]
    both = bm25.scores(vocab.ids(["sat"]))
    assert both[0] > 0 and both[1] > both[0]  # shorter document wins
    bm25.save(tmp_path / "dt.npz")
    np.testing.assert_allclose(BM25.load(tmp_path / "dt.npz").scores(vocab.ids(["sat"])), both)


def test_word_match_counts_distinct_overlap(vocab):
    m = doc_term_matrix(DOCS, vocab)
    assert word_match_scores(m, vocab.ids(["cat", "sat", "sat"])).tolist() == [2, 1, 0, 0]
    sat = vocab.ids(["sat"])
    assert word_match_scores(m, vocab.ids(["cat", "sat"]), exclude=sat).tolist() == [1, 0, 0, 0]
    # a query made only of excluded words keeps them
    assert word_match_scores(m, sat, exclude=sat).tolist() == [1, 1, 0, 0]


def test_rrf():
    a = np.array([0.9, 0.5, 0.1])
    b = np.array([0.0, 2.0, 1.0])
    assert ranks(a).tolist() == [1, 2, 3]
    fused = reciprocal_rank_fusion([a, b], eligible=[None, b > 0], k=1)
    assert fused == pytest.approx([1 / 2, 1 / 3 + 1 / 2, 1 / 4 + 1 / 3])
    only_a = reciprocal_rank_fusion([a, b], weights=[1, 0])
    assert np.argsort(-only_a).tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([a, b], weights=[1])
