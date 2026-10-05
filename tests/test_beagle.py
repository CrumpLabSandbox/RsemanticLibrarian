from __future__ import annotations

import numpy as np
import pytest

from semantic_librarian.embed import (
    Beagle,
    BeagleRP,
    Corpus,
    RandomVectors,
    bind,
    compose_documents,
    make_embedder,
    order_vectors,
)
from semantic_librarian.embed.beagle import LEGACY_PRESET
from semantic_librarian.embed.random_vectors import ternary_vectors
from semantic_librarian.space import Vocabulary, factor_space
from semantic_librarian.text import legacy_sentences, legacy_tokens

# ---- parity with RsemanticLibrarian::sl_beagle_vectors -----------------------------


def _legacy_inputs(beagle_ref):
    docs = beagle_ref["documents"]
    # sl_semantic_space() unlists the columns, so all titles come before all abstracts
    texts = [d["title"] for d in docs] + [d["abstract"] for d in docs]
    sentences = [s for t in texts for s in legacy_sentences(t)]
    vocab = Vocabulary.build(legacy_tokens(t) for t in texts)
    return docs, sentences, vocab


def test_legacy_preprocessing_matches_r(beagle_ref):
    _, sentences, vocab = _legacy_inputs(beagle_ref)
    assert sentences == beagle_ref["sentences"]
    assert vocab.tokens == beagle_ref["dictionary"]


def test_legacy_word_vectors_match_r(beagle_ref):
    _, sentences, vocab = _legacy_inputs(beagle_ref)
    arrays = beagle_ref["arrays"]
    dim, nonzeros = beagle_ref["riv"]
    model = Beagle(**{**LEGACY_PRESET, "dim": dim, "nonzeros": nonzeros})
    words = model.learn_word_vectors(
        [vocab.ids(s) for s in sentences], vocab, environment=arrays["environment"]
    )
    np.testing.assert_allclose(words, arrays["word_vectors"], rtol=0, atol=1e-12)


def test_legacy_document_and_author_vectors_match_r(beagle_ref):
    docs, _, vocab = _legacy_inputs(beagle_ref)
    arrays = beagle_ref["arrays"]
    words = arrays["word_vectors"]
    doc_ids = [vocab.ids(legacy_tokens(d["title"] + "." + d["abstract"])) for d in docs]
    doc_vectors = compose_documents(doc_ids, words)
    np.testing.assert_allclose(doc_vectors, arrays["doc_vectors"], rtol=1e-12, atol=1e-9)

    levels = [[a.strip() for a in d["authorlist"].split(";")] for d in docs]
    authors = factor_space("author", levels, doc_vectors)
    assert authors.labels == beagle_ref["authors"]
    # R matched authors with a regular expression; compare where that equals exact match
    compared = 0
    for i, name in enumerate(authors.labels):
        exact = [r + 1 for r, lv in enumerate(levels) if name in lv]
        if exact == beagle_ref["author_doc_rows"][i]:
            np.testing.assert_allclose(
                authors.vectors[i], arrays["author_vectors"][i], rtol=1e-12, atol=1e-9
            )
            compared += 1
    assert compared >= len(authors.labels) - 2


# ---- order information -------------------------------------------------------------


def _naive_order(x, phi, p1, p2, lam):
    length = len(x)
    out = np.zeros_like(x)
    for p in range(length):
        for a in range(length):
            for b in range(a + 1, length):
                if not a <= p <= b or b - a + 1 > lam:
                    continue
                seq = [phi if i == p else x[i] for i in range(a, b + 1)]
                chain = seq[0]
                for v in seq[1:]:
                    chain = bind(chain, v, p1, p2)
                out[p] += chain
    return out


@pytest.mark.parametrize(("length", "lam"), [(2, 7), (5, 7), (5, 3), (9, 4), (12, 7), (1, 7)])
def test_order_vectors_match_naive_definition(length, lam):
    rng = np.random.default_rng(length * 10 + lam)
    d = 128
    p1, p2 = rng.permutation(d), rng.permutation(d)
    x = rng.normal(0, 1 / np.sqrt(d), (length, d))
    phi = rng.normal(0, 1 / np.sqrt(d), d)
    np.testing.assert_allclose(
        order_vectors(x, phi, p1, p2, lam), _naive_order(x, phi, p1, p2, lam), atol=1e-12
    )


def test_order_example_from_paper_has_seven_terms():
    # "a dog bit the mailman": with all-ones vectors under identity permutations every
    # bound chain is a constant vector, so the count of terms is recoverable.
    d = 16
    ident = np.arange(d)
    x = np.full((5, d), 1.0 / d)
    phi = np.full(d, 1.0 / d)
    out = order_vectors(x, phi, ident, ident, 7)
    # each n-gram chain of constant 1/d vectors convolves to the constant 1/d
    assert out[1] == pytest.approx(np.full(d, 7.0 / d))


def test_parallel_order_memory_is_identical_to_serial(monkeypatch):
    import semantic_librarian.embed.beagle as beagle

    monkeypatch.setattr(beagle, "CHUNK_TOKENS", 40)  # force many chunks
    rng = np.random.default_rng(5)
    d = 64
    env = rng.normal(0, 1 / 8, (50, d)).astype(np.float32)
    sentences = [rng.integers(0, 50, size=rng.integers(1, 12)) for _ in range(60)]
    phi = rng.normal(0, 1 / 8, d)
    p1, p2 = rng.permutation(d), rng.permutation(d)
    messages = []
    serial = beagle.order_memory(sentences, env, phi, p1, p2, 4, workers=1, log=messages.append)
    parallel = beagle.order_memory(sentences, env, phi, p1, p2, 4, workers=2)
    np.testing.assert_array_equal(serial, parallel)
    assert messages and messages[-1].startswith("order information")
    # and equal to summing the per-sentence definition directly
    direct = np.zeros_like(serial, dtype=np.float64)
    for s in sentences:
        if len(s) > 1:
            np.add.at(direct, s, order_vectors(env[s].astype(np.float64), phi, p1, p2, 4))
    np.testing.assert_allclose(serial, direct, rtol=1e-5, atol=1e-6)


def test_resolve_workers():
    from semantic_librarian.embed.beagle import PARALLEL_MIN_TOKENS, resolve_workers

    assert resolve_workers(3, 10) == 3
    assert resolve_workers(0, 10) == 1
    assert resolve_workers(0, PARALLEL_MIN_TOKENS) >= 1


def test_bind_is_not_commutative():
    rng = np.random.default_rng(0)
    d = 64
    p1, p2 = rng.permutation(d), rng.permutation(d)
    a, b = rng.normal(size=d), rng.normal(size=d)
    assert not np.allclose(bind(a, b, p1, p2), bind(b, a, p1, p2))


# ---- composite BEAGLE, BEAGLE-RP, random -------------------------------------------


def _toy_corpus() -> Corpus:
    sentences = [
        "the dog chased the cat".split(),
        "the cat chased the mouse".split(),
        "a dog bit the mailman".split(),
        "stocks fell as markets crashed".split(),
        "markets rallied and stocks rose".split(),
        "the dog and the cat slept".split(),
    ] * 3
    vocab = Vocabulary.build(sentences)
    return Corpus([f"d{i}" for i in range(len(sentences))], sentences, sentences, vocab)


@pytest.mark.parametrize(
    "model",
    [
        Beagle(dim=256, seed=1),
        Beagle(dim=256, seed=1, order=False),
        Beagle(dim=256, seed=1, context=False, environment="ternary", nonzeros=8),
        BeagleRP(dim=512, nonzeros=16, seed=1),
        RandomVectors(dim=128, seed=1),
    ],
    ids=["beagle", "beagle-context", "beagle-order", "beagle-rp", "random"],
)
def test_models_produce_finite_deterministic_vectors(model):
    corpus = _toy_corpus()
    model.fit(corpus)
    words = model.word_vectors
    assert words.shape == (len(corpus.vocab), model.cfg.dim)
    assert np.isfinite(words).all()
    docs = model.embed_documents(corpus)
    assert docs.shape == (len(corpus.doc_ids), model.cfg.dim)
    again = type(model)(cfg=model.cfg)
    again.fit(corpus)
    np.testing.assert_array_equal(again.word_vectors, words)


def test_beagle_learns_related_words():
    corpus = _toy_corpus()
    model = Beagle(dim=512, seed=3)
    model.fit(corpus)
    space = model.word_space()
    from semantic_librarian.search import cosine_scores

    dog = cosine_scores(space.vector("dog"), space.vectors)
    assert dog[space.row("cat")] > dog[space.row("stocks")]
    stocks = cosine_scores(space.vector("stocks"), space.vectors)
    assert stocks[space.row("markets")] > stocks[space.row("dog")]


def test_query_embedding_and_terms():
    corpus = _toy_corpus()
    model = Beagle(dim=128, seed=0)
    model.fit(corpus)
    assert model.query_terms(["the", "dog", "zebra", "dog"]) == ["dog"]
    assert model.query_terms(["the"]) == ["the"]  # only stop words: keep them
    assert model.embed_query(["zebra"]) is None
    q = model.embed_query(["dog", "cat"])
    np.testing.assert_allclose(q, model.term_vectors(["dog", "cat"]).sum(axis=0), rtol=1e-6)


def test_beagle_validation():
    with pytest.raises(ValueError):
        Beagle(variant="legacy", order=True)
    with pytest.raises(ValueError):
        Beagle(context=False, order=False)
    with pytest.raises(ValueError):
        Beagle(max_ngram=1)
    with pytest.raises(ValueError):
        make_embedder("word2vec")


def test_ternary_vectors():
    rng = np.random.default_rng(0)
    m = ternary_vectors(50, 100, 10, rng, chunk=7)
    assert m.dtype == np.int8
    assert ((m == 1).sum(axis=1) == 5).all()
    assert ((m == -1).sum(axis=1) == 5).all()
    with pytest.raises(ValueError):
        ternary_vectors(1, 10, 3, rng)
