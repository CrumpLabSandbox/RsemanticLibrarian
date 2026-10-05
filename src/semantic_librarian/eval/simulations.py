"""The paper's three simulations, generalized to any set of retrieval methods.

Simulation 1, target recovery
    Sample a document, sample a percentage of its words, query with them, and record the
    rank of the sampled document among all documents (paper Fig. 2).
Simulation 2, associate substitution
    As Simulation 1, but every sampled word is replaced by its nearest neighbor in a
    reference semantic word space, so the query shares meaning but not words with the
    target (paper Fig. 3). All methods receive the same substituted queries.
Simulation 3, agreement
    For the Simulation 1 queries, the Spearman correlation between the complete document
    rankings produced by every pair of methods (paper Fig. 4).

Methods are embedder names (``beagle``, ``beagle-rp``, ``random``, ...) plus three
lexical baselines: ``wordmatch`` (the paper's word-match control: documents ranked by
how many distinct non-stop query words they contain), ``bm25``, and ``hybrid``
(reciprocal rank fusion of the reference semantic model and BM25, as in
``Library.search(method="hybrid")``).

Ranks are computed with ties shared: a target tied with ``t`` other documents is
credited with the average of the tied positions. Queries are built with word
multiplicity (paper Eq. 9), so a query made of 100% of a document's words is that
document's own vector.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
from scipy.stats import rankdata

from semantic_librarian.embed.base import Corpus, WordSumEmbedder
from semantic_librarian.embed.registry import make_embedder
from semantic_librarian.search.fusion import reciprocal_rank_fusion
from semantic_librarian.search.lexical import BM25, doc_term_matrix, word_match_scores
from semantic_librarian.text.stopwords import ENGLISH_STOPWORDS

LEXICAL_METHODS = ("wordmatch", "bm25", "hybrid")
DEFAULT_METHODS = ("beagle", "beagle-rp", "random", "wordmatch", "bm25", "hybrid")


@dataclass
class EvalConfig:
    """Settings for :func:`evaluate`.

    Attributes:
        methods: Methods to compare (embedder names and/or lexical baselines).
        percents: Percentages of a document's words used as the query.
        n_targets: Documents sampled as targets (the same targets at every percentage).
        simulations: Which simulations to run (1, 2, 3).
        seed: Seed for sampling targets and query words.
        reference: Semantic model that supplies associates (Simulation 2) and the
            semantic half of ``hybrid``.
        params: Per-embedder parameter overrides, e.g. ``{"beagle": {"dim": 512}}``.
        min_tokens: Documents with fewer tokens are not used as targets.
        lexical_weight: Share of BM25 in ``hybrid``.
        associates: How Simulation 2 replaces words. ``"nearest"`` uses each word's
            nearest neighbor (as in the paper). ``"outside-document"`` uses the nearest
            neighbor that does not occur in the target document, so the substituted
            query shares no replaced word with the target.
    """

    methods: list[str] = field(default_factory=lambda: list(DEFAULT_METHODS))
    percents: list[int] = field(default_factory=lambda: [5, 10, 25, 50, 100])
    n_targets: int = 200
    simulations: list[int] = field(default_factory=lambda: [1, 2, 3])
    seed: int = 0
    reference: str = "beagle"
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    min_tokens: int = 10
    lexical_weight: float = 0.5
    associates: str = "nearest"


ASSOCIATE_MODES = ("nearest", "outside-document")


# ---- scorers ------------------------------------------------------------------------


class _Scorer:
    name: str

    def scores(self, queries: Sequence[Sequence[str]]) -> np.ndarray:
        """``(n_queries, n_documents)`` scores, higher is better."""
        raise NotImplementedError


class _EmbedderScorer(_Scorer):
    def __init__(
        self, name: str, model: WordSumEmbedder, corpus: Corpus, fitted: bool = False
    ) -> None:
        self.name = name
        self.model = model
        if not fitted:
            model.fit(corpus)
        docs = np.asarray(model.embed_documents(corpus), dtype=np.float64)
        norms = np.linalg.norm(docs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.docs = docs / norms

    def scores(self, queries: Sequence[Sequence[str]]) -> np.ndarray:
        dim = self.docs.shape[1]
        q = np.zeros((len(queries), dim), dtype=np.float64)
        for i, tokens in enumerate(queries):
            v = self.model.embed_query(tokens, unique=False)
            if v is not None:
                n = np.linalg.norm(v)
                q[i] = v / n if n > 0 else v
        return q @ self.docs.T


class _WordMatchScorer(_Scorer):
    name = "wordmatch"

    def __init__(self, corpus: Corpus) -> None:
        self.vocab = corpus.vocab
        self.doc_terms = doc_term_matrix(corpus.doc_tokens, corpus.vocab)
        self.doc_terms.data[:] = 1.0  # presence, not counts
        self.stop_ids = self.vocab.ids(ENGLISH_STOPWORDS)

    def scores(self, queries: Sequence[Sequence[str]]) -> np.ndarray:
        return np.vstack(
            [
                word_match_scores(self.doc_terms, self.vocab.ids(q), exclude=self.stop_ids)
                for q in queries
            ]
        )


class _BM25Scorer(_Scorer):
    name = "bm25"

    def __init__(self, corpus: Corpus) -> None:
        self.vocab = corpus.vocab
        self.bm25 = BM25(doc_term_matrix(corpus.doc_tokens, corpus.vocab))

    def scores(self, queries: Sequence[Sequence[str]]) -> np.ndarray:
        return np.vstack([self.bm25.scores(self.vocab.ids(q)) for q in queries])


class _HybridScorer(_Scorer):
    name = "hybrid"

    def __init__(self, semantic: _EmbedderScorer, bm25: _BM25Scorer, weight: float) -> None:
        self.semantic = semantic
        self.bm25 = bm25
        self.weight = weight

    def scores(self, queries: Sequence[Sequence[str]]) -> np.ndarray:
        sem = self.semantic.scores(queries)
        lex = self.bm25.scores(queries)
        return np.vstack(
            [
                reciprocal_rank_fusion(
                    [s, x], weights=[1 - self.weight, self.weight], eligible=[None, x > 0]
                )
                for s, x in zip(sem, lex, strict=True)
            ]
        )


# ---- results ------------------------------------------------------------------------


@dataclass
class EvalResults:
    """Raw per-query results plus summaries.

    ``ranks`` rows: simulation, percent, target (document id), method, rank.
    ``agreement`` rows: percent, target, method_a, method_b, rho.
    """

    config: EvalConfig
    n_documents: int
    vocab_size: int
    n_targets: int = 0
    ranks: list[dict[str, Any]] = field(default_factory=list)
    agreement: list[dict[str, Any]] = field(default_factory=list)

    def rank_summary(self, simulation: int) -> list[dict[str, Any]]:
        """Median rank, mean reciprocal rank, and hit rates per method and percentage."""
        groups: dict[tuple[str, int], list[float]] = {}
        for row in self.ranks:
            if row["simulation"] == simulation:
                groups.setdefault((row["method"], row["percent"]), []).append(row["rank"])
        out = []
        for method in self.config.methods:
            for pct in self.config.percents:
                r = groups.get((method, pct))
                if not r:
                    continue
                arr = np.asarray(r)
                out.append(
                    {
                        "method": method,
                        "percent": pct,
                        "n": len(r),
                        "median_rank": float(np.median(arr)),
                        "mrr": float(np.mean(1.0 / arr)),
                        "top1": float(np.mean(arr <= 1)),
                        "top10": float(np.mean(arr <= 10)),
                    }
                )
        return out

    def agreement_summary(self) -> list[dict[str, Any]]:
        """Mean and standard deviation of Spearman's rho per method pair and percentage."""
        groups: dict[tuple[str, str, int], list[float]] = {}
        for row in self.agreement:
            key = (row["method_a"], row["method_b"], row["percent"])
            groups.setdefault(key, []).append(row["rho"])
        return [
            {
                "method_a": a,
                "method_b": b,
                "percent": p,
                "n": len(v),
                "mean_rho": float(np.mean(v)),
                "sd_rho": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            }
            for (a, b, p), v in groups.items()
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "n_documents": self.n_documents,
            "vocab_size": self.vocab_size,
            "n_targets": self.n_targets,
            "summary": {
                "simulation_1": self.rank_summary(1),
                "simulation_2": self.rank_summary(2),
                "simulation_3": self.agreement_summary(),
            },
            "ranks": self.ranks,
            "agreement": self.agreement,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)

    def format_text(self) -> str:
        """Plain-text tables of the summaries, one row per method."""
        lines = [
            f"{self.n_documents} documents, vocabulary {self.vocab_size}, "
            f"{self.n_targets} target documents per percentage, seed {self.config.seed}",
        ]
        pcts = self.config.percents
        header = f"{'method':<12}" + "".join(f"{str(p) + '%':>13}" for p in pcts)
        titles = {
            1: "Simulation 1: target recovery, median rank (top-10 rate)",
            2: "Simulation 2: semantic associates, median rank (top-10 rate)",
        }
        for sim in (1, 2):
            rows = self.rank_summary(sim)
            if not rows:
                continue
            lines += ["", titles[sim], header]
            by = {(r["method"], r["percent"]): r for r in rows}
            for m in self.config.methods:
                cells = []
                for p in pcts:
                    r = by.get((m, p))
                    cells.append(f"{r['median_rank']:.1f} ({r['top10']:.2f})" if r else "")
                lines.append(f"{m:<12}" + "".join(f"{c:>13}" for c in cells))
        agreement = self.agreement_summary()
        if agreement:
            lines += [
                "",
                "Simulation 3: rank agreement, mean Spearman rho",
                f"{'pair':<24}" + "".join(f"{str(p) + '%':>9}" for p in pcts),
            ]
            by3 = {(r["method_a"], r["method_b"], r["percent"]): r for r in agreement}
            for a, b in combinations(self.config.methods, 2):
                pair = [by3.get((a, b, p)) for p in pcts]
                lines.append(
                    f"{a + ' / ' + b:<24}"
                    + "".join(f"{c['mean_rho']:>9.3f}" if c else f"{'':>9}" for c in pair)
                )
        return "\n".join(lines)


# ---- the simulations ----------------------------------------------------------------


def target_ranks(scores: np.ndarray, targets: Sequence[int] | np.ndarray) -> np.ndarray:
    """Rank of each row's target, ties sharing the average position."""
    t = np.asarray(targets)
    target_scores = scores[np.arange(len(t)), t][:, None]
    greater = (scores > target_scores).sum(axis=1)
    ties = (scores == target_scores).sum(axis=1) - 1
    return 1.0 + greater + ties / 2.0


def associate_candidates(
    model: WordSumEmbedder, tokens: Sequence[str], k: int = 20, chunk: int = 512
) -> dict[str, list[str]]:
    """Each token's ``k`` nearest neighbors in the model's word space, best first,
    excluding itself and stop words. Tokens whose vector is zero map to themselves."""
    assert model.vocab is not None and model.word_vectors is not None
    words = np.asarray(model.word_vectors, dtype=np.float32)
    norms = np.linalg.norm(words, axis=1)
    unit = words / np.where(norms > 0, norms, 1.0)[:, None]
    stop = model.stopword_set
    banned = np.array([t in stop for t in model.vocab.tokens], dtype=bool)
    k = max(1, min(k, int((~banned).sum()) - 1))
    unique = list(dict.fromkeys(tokens))
    out: dict[str, list[str]] = {}
    for start in range(0, len(unique), chunk):
        part = unique[start : start + chunk]
        ids = model.vocab.ids(part)
        sims = unit[ids] @ unit.T
        sims[:, banned] = -np.inf
        sims[np.arange(len(ids)), ids] = -np.inf
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1, kind="stable")
        top = np.take_along_axis(top, order, axis=1)
        for word, i, row in zip(part, ids, top, strict=True):
            out[word] = [model.vocab.tokens[j] for j in row] if norms[i] > 0 else [word]
    return out


def nearest_associates(model: WordSumEmbedder, tokens: Sequence[str]) -> dict[str, str]:
    """Each token's nearest neighbor (see :func:`associate_candidates`)."""
    return {w: c[0] for w, c in associate_candidates(model, tokens, k=1).items()}


def _substitute(
    query: Sequence[str], candidates: dict[str, list[str]], exclude: set[str] | None
) -> list[str]:
    out = []
    for t in query:
        options = candidates.get(t, [t])
        if exclude is not None:
            options = [o for o in options if o not in exclude] or options
        out.append(options[0])
    return out


def _build_scorers(
    corpus: Corpus,
    cfg: EvalConfig,
    log: Callable[[str], None],
    prefit: dict[str, WordSumEmbedder],
) -> tuple[dict[str, _Scorer], _EmbedderScorer | None]:
    scorers: dict[str, _Scorer] = {}
    semantic: dict[str, _EmbedderScorer] = {}

    def embedder(name: str) -> _EmbedderScorer:
        ready = prefit.get(name)
        if name not in semantic and ready is not None:
            if ready.vocab is not None and ready.vocab.tokens == corpus.vocab.tokens:
                log(f"using the library's built {name} vectors")
                semantic[name] = _EmbedderScorer(name, ready, corpus, fitted=True)
        if name not in semantic:
            log(f"fitting {name}")
            model = make_embedder(name, **cfg.params.get(name, {}))
            if not isinstance(model, WordSumEmbedder):  # pragma: no cover
                raise ValueError(f"{name} is not a word-based embedder")
            semantic[name] = _EmbedderScorer(name, model, corpus)
        return semantic[name]

    needs_reference = 2 in cfg.simulations or "hybrid" in cfg.methods
    reference = embedder(cfg.reference) if needs_reference else None
    bm25 = None
    for name in cfg.methods:
        if name == "wordmatch":
            scorers[name] = _WordMatchScorer(corpus)
        elif name in ("bm25", "hybrid"):
            bm25 = bm25 or _BM25Scorer(corpus)
            if name == "bm25":
                scorers[name] = bm25
            else:
                assert reference is not None
                scorers[name] = _HybridScorer(reference, bm25, cfg.lexical_weight)
        else:
            scorers[name] = embedder(name)
    return scorers, reference


def evaluate(
    corpus: Corpus,
    config: EvalConfig | None = None,
    log: Callable[[str], None] | None = None,
    prefit: dict[str, WordSumEmbedder] | None = None,
) -> EvalResults:
    """Run the simulations on a tokenized corpus (see :meth:`Library.corpus`).

    ``prefit`` maps method names to already-fitted models (e.g. a built library's
    embedder); they are used instead of refitting when their vocabulary matches.
    """
    cfg = config or EvalConfig()
    say = log or (lambda _m: None)
    if not cfg.methods:
        raise ValueError("no methods to evaluate")
    if cfg.associates not in ASSOCIATE_MODES:
        raise ValueError(f"associates must be one of {ASSOCIATE_MODES}")
    if len(set(cfg.methods)) != len(cfg.methods):
        raise ValueError("methods must be distinct")
    lengths = np.array([len(t) for t in corpus.doc_tokens])
    candidates = np.flatnonzero(lengths >= cfg.min_tokens)
    if len(candidates) == 0:
        raise ValueError(f"no document has at least {cfg.min_tokens} tokens")
    rng = np.random.default_rng(cfg.seed)
    n_targets = min(cfg.n_targets, len(candidates))
    targets = rng.choice(candidates, size=n_targets, replace=False)

    scorers, reference = _build_scorers(corpus, cfg, say, prefit or {})
    results = EvalResults(cfg, len(corpus.doc_ids), len(corpus.vocab), n_targets)

    neighbors: dict[str, list[str]] = {}
    if 2 in cfg.simulations:
        assert reference is not None
        needed = [t for i in targets for t in corpus.doc_tokens[i]]
        say("finding semantic associates")
        k = 1 if cfg.associates == "nearest" else 50
        neighbors = associate_candidates(reference.model, needed, k=k)

    for pct in cfg.percents:
        queries = []
        for i in targets:
            tokens = corpus.doc_tokens[i]
            m = max(1, round(len(tokens) * pct / 100))
            picks = np.sort(rng.choice(len(tokens), size=m, replace=False))
            queries.append([tokens[j] for j in picks])
        say(f"{pct}% queries")
        ids = [corpus.doc_ids[i] for i in targets]

        full_scores: dict[str, np.ndarray] = {}
        for name, scorer in scorers.items():
            s = scorer.scores(queries)
            full_scores[name] = s
            if 1 in cfg.simulations:
                for doc_id, r in zip(ids, target_ranks(s, targets), strict=True):
                    results.ranks.append(
                        {
                            "simulation": 1,
                            "percent": pct,
                            "target": doc_id,
                            "method": name,
                            "rank": float(r),
                        }
                    )
        if 2 in cfg.simulations:
            swapped = [
                _substitute(
                    q,
                    neighbors,
                    set(corpus.doc_tokens[i]) if cfg.associates == "outside-document" else None,
                )
                for q, i in zip(queries, targets, strict=True)
            ]
            for name, scorer in scorers.items():
                s = scorer.scores(swapped)
                for doc_id, r in zip(ids, target_ranks(s, targets), strict=True):
                    results.ranks.append(
                        {
                            "simulation": 2,
                            "percent": pct,
                            "target": doc_id,
                            "method": name,
                            "rank": float(r),
                        }
                    )
        if 3 in cfg.simulations and len(scorers) > 1:
            ranked = {n: rankdata(-s, axis=1) for n, s in full_scores.items()}
            for a, b in combinations(cfg.methods, 2):
                ra, rb = ranked[a], ranked[b]
                ra_c = ra - ra.mean(axis=1, keepdims=True)
                rb_c = rb - rb.mean(axis=1, keepdims=True)
                denom = np.sqrt((ra_c**2).sum(axis=1) * (rb_c**2).sum(axis=1))
                rho = np.where(
                    denom > 0, (ra_c * rb_c).sum(axis=1) / np.where(denom > 0, denom, 1.0), 0.0
                )
                for doc_id, value in zip(ids, rho, strict=True):
                    results.agreement.append(
                        {
                            "percent": pct,
                            "target": doc_id,
                            "method_a": a,
                            "method_b": b,
                            "rho": float(value),
                        }
                    )
    return results
