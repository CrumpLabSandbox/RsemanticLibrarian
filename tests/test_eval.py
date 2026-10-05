from __future__ import annotations

import numpy as np
import pytest

from semantic_librarian import Library
from semantic_librarian.embed import Beagle
from semantic_librarian.eval import EvalConfig, evaluate
from semantic_librarian.eval.simulations import (
    _substitute,
    associate_candidates,
    nearest_associates,
    target_ranks,
)


def test_target_ranks_share_ties():
    scores = np.array([[0.9, 0.5, 0.5, 0.1], [0.2, 0.2, 0.2, 0.2]])
    assert target_ranks(scores, [1, 3]).tolist() == [2.5, 2.5]
    assert target_ranks(scores[:1], [0]).tolist() == [1.0]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    from pathlib import Path

    bib = Path(__file__).parent / "fixtures" / "crump.bib"
    lib = Library.create(tmp_path_factory.mktemp("eval") / "lib")
    lib.add_file(bib)
    yield lib.corpus()
    lib.close()


SMALL = {
    "beagle": {"dim": 128, "max_ngram": 3},
    "random": {"dim": 64},
    "beagle-rp": {"dim": 256, "nonzeros": 16},
}


def test_nearest_associates(corpus):
    model = Beagle(dim=128, max_ngram=3)
    model.fit(corpus)
    assoc = nearest_associates(model, ["typing", "keyboard", "typing"])
    assert set(assoc) == {"typing", "keyboard"}
    for word, neighbor in assoc.items():
        assert neighbor != word and neighbor not in model.stopword_set


def test_associate_candidates_and_substitution(corpus):
    model = Beagle(dim=128, max_ngram=3)
    model.fit(corpus)
    cands = associate_candidates(model, ["typing"], k=5)["typing"]
    assert len(cands) == 5 and cands[0] == nearest_associates(model, ["typing"])["typing"]
    space = model.word_space()
    from semantic_librarian.search import cosine_scores

    sims = cosine_scores(space.vector("typing"), space.vectors)
    assert [sims[space.row(c)] for c in cands] == sorted(
        [sims[space.row(c)] for c in cands], reverse=True
    )
    table = {"a": ["x", "y"], "b": ["z"]}
    assert _substitute(["a", "b", "c"], table, None) == ["x", "z", "c"]
    assert _substitute(["a", "b"], table, {"x", "z"}) == ["y", "z"]  # z: no alternative


def test_outside_document_associates_and_prefit(corpus):
    model = Beagle(dim=128, max_ngram=3)
    model.fit(corpus)
    cfg = EvalConfig(
        methods=["beagle", "bm25"],
        percents=[50],
        n_targets=6,
        simulations=[2],
        associates="outside-document",
        params=SMALL,
    )
    logs = []
    res = evaluate(corpus, cfg, log=logs.append, prefit={"beagle": model})
    assert "using the library's built beagle vectors" in logs
    assert {r["simulation"] for r in res.ranks} == {2}
    with pytest.raises(ValueError):
        evaluate(corpus, EvalConfig(associates="synonyms"))


def test_evaluate_runs_all_simulations(corpus):
    cfg = EvalConfig(percents=[10, 100], n_targets=12, params=SMALL)
    res = evaluate(corpus, cfg)
    assert res.n_targets == 12 and "12 target documents" in res.format_text()
    s1 = res.rank_summary(1)
    assert {r["method"] for r in s1} == set(cfg.methods)
    by = {(r["method"], r["percent"]): r for r in s1}
    # the full document is its own best match for every vector method
    for m in ("beagle", "beagle-rp", "random"):
        assert by[(m, 100)]["median_rank"] == 1.0
    assert len(res.rank_summary(2)) == len(s1)
    pairs = res.agreement_summary()
    assert len(pairs) == 15 * 2
    assert all(-1 <= p["mean_rho"] <= 1 for p in pairs)
    text = res.format_text()
    assert "Simulation 3" in text and "beagle / beagle-rp" in text
    assert res.to_dict()["summary"]["simulation_1"] == s1


def test_evaluate_is_reproducible_and_validates(corpus):
    cfg = EvalConfig(
        methods=["random", "wordmatch"],
        percents=[25],
        n_targets=5,
        simulations=[1, 3],
        params=SMALL,
    )
    a, b = evaluate(corpus, cfg), evaluate(corpus, cfg)
    assert a.ranks == b.ranks and a.agreement == b.agreement
    assert {r["simulation"] for r in a.ranks} == {1}
    with pytest.raises(ValueError):
        evaluate(corpus, EvalConfig(methods=[]))
    with pytest.raises(ValueError):
        evaluate(corpus, EvalConfig(methods=["bm25", "bm25"]))
    with pytest.raises(ValueError):
        evaluate(corpus, EvalConfig(methods=["bm25"], min_tokens=10**6))
