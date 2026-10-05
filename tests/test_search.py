from __future__ import annotations

import numpy as np
import pytest

from semantic_librarian.project import classical_mds, cluster, project
from semantic_librarian.search import combine_term_scores, cosine_scores, rank
from semantic_librarian.text import legacy_tokens

# ---- parity with the R search functions on the bundled 100-article sample ----------


def _terms(query, dictionary):
    return [t for t in legacy_tokens(query) if t in dictionary]


def _by_index(r_out):
    return dict(zip(r_out["index"], r_out["similarity"], strict=True))


def test_query_terms_match_get_search_terms(search_ref, apa_sample):
    assert _terms(search_ref["query"], apa_sample["dictionary"]) == search_ref["terms"]


@pytest.mark.parametrize(("query_type", "mode"), [("1", "compound"), ("2", "and"), ("3", "or")])
def test_search_matches_get_search_article_similarities(search_ref, apa_sample, query_type, mode):
    dictionary = apa_sample["dictionary"]
    words = apa_sample["word_vectors"]
    articles = apa_sample["article_vectors"]
    rows = [i for i, w in enumerate(dictionary) if w in search_ref["terms"]]
    if mode == "compound":
        scores = cosine_scores(words[rows].sum(axis=0), articles)
        tol = 1e-12
    else:
        scores = combine_term_scores(cosine_scores(words[rows], articles), mode)
        tol = 5.1e-5  # R rounds AND/OR scores to 4 decimals
    expected = _by_index(search_ref["search"][query_type])
    indices = [a["index"] for a in apa_sample["articles"]]
    for row, idx in enumerate(indices):
        assert scores[row] == pytest.approx(expected[idx], abs=tol)
    if mode == "compound":
        assert [indices[i] for i in rank(scores)] == search_ref["search"][query_type]["index"]


def test_article_similarity_matches_r(search_ref, apa_sample):
    articles = apa_sample["article_vectors"]
    scores = cosine_scores(articles[0], articles)
    expected = _by_index(search_ref["article_article"])
    for row, art in enumerate(apa_sample["articles"]):
        assert scores[row] == pytest.approx(expected[art["index"]], abs=5.1e-5)


def test_author_similarity_matches_r(search_ref, apa_sample):
    authors = apa_sample["author_vectors"]
    scores = cosine_scores(authors[0], authors)
    expected = _by_index(search_ref["author_author"])
    for row in range(len(apa_sample["authors"])):
        assert scores[row] == pytest.approx(expected[row + 1], abs=5.1e-5)


def _assert_same_up_to_sign(coords, x, y):
    ref = np.column_stack([x, y])
    for j in range(2):
        sign = np.sign(coords[:, j] @ ref[:, j])
        np.testing.assert_allclose(coords[:, j] * sign, ref[:, j], atol=1e-9)


def test_mds_matches_cmdscale_for_articles(search_ref, apa_sample):
    r = search_ref["mds_articles"]
    vectors = apa_sample["article_vectors"][np.asarray(r["index"]) - 1]
    _assert_same_up_to_sign(classical_mds(vectors), r["X"], r["Y"])


def test_mds_matches_cmdscale_for_authors(search_ref, apa_sample):
    r = search_ref["mds_authors"]
    vectors = apa_sample["author_vectors"][np.asarray(r["index"]) - 1]
    _assert_same_up_to_sign(classical_mds(vectors), r["X"], r["Y"])


# ---- behaviour ---------------------------------------------------------------------


def test_cosine_handles_zero_vectors():
    m = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    s = cosine_scores(np.array([1.0, 0.0]), m)
    assert s == pytest.approx([1.0, 0.0, np.sqrt(0.5)])
    assert (cosine_scores(np.zeros(2), m) == 0).all()


def test_single_term_and_or_queries_return_scores():
    # the R package returned nothing for single-word AND/OR queries
    s = np.array([[0.2, 0.9, -0.1]])
    assert combine_term_scores(s, "and") == pytest.approx([0.2, 0.9, -0.1])
    assert combine_term_scores(s, "or") == pytest.approx([0.2, 0.9, -0.1])
    with pytest.raises(ValueError):
        combine_term_scores(np.ones((2, 3)), "compound")  # type: ignore[arg-type]


def test_rank_with_mask_and_ties():
    scores = np.array([0.5, 0.9, 0.5, 0.1])
    assert rank(scores).tolist() == [1, 0, 2, 3]
    assert rank(scores, k=2).tolist() == [1, 0]
    assert rank(scores, mask=np.array([True, False, True, True])).tolist() == [0, 2, 3]


def test_projection_and_clusters():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(3, 20)) * 5
    vectors = np.vstack([c + rng.normal(size=(10, 20)) for c in centres])
    p = project([str(i) for i in range(30)], vectors, n_clusters=3)
    assert p.coords.shape == (30, 2)
    assert len(set(p.clusters[:10])) == 1 and len(set(p.clusters)) == 3
    assert classical_mds(vectors[:1]).shape == (1, 2)
    assert sorted(cluster(np.array([[0.0, 0.0], [1.0, 1.0]]), 5).tolist()) == [0, 1]
    assert cluster(np.zeros((0, 2)), 3).shape == (0,)
