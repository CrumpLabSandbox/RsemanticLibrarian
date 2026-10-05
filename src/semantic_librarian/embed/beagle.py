"""BEAGLE: Bound Encoding of the Aggregate Language Environment (Jones & Mewhort, 2007).

Two variants are implemented.

``legacy`` (preset ``beagle-legacy``)
    Exactly what ``RsemanticLibrarian::sl_beagle_vectors()`` computes: context information
    only, sparse ternary environment vectors, memory initialised to the environment
    vectors, and after every update each vector is divided by the absolute value of its
    maximum element. Given the same environment vectors and sentences, the output matches
    the R function to floating-point precision.

``composite`` (preset ``beagle``, the default)
    The model described in the paper (Eqs. 1-5). Separate context and order memories:

    * context: each word accumulates the environment vectors of the other non-stop words
      in its sentence (Eq. 1);
    * order: each word accumulates every n-gram (2 <= n <= ``max_ngram``) in its sentence
      that contains it, with the word replaced by the placeholder vector Phi, bound with
      non-commutative circular convolution (Eqs. 2-4);

    and the final word vector is the sum of the unit-normalized context and order vectors.
    Non-commutative convolution permutes the two operands with fixed random permutations
    before ordinary circular convolution, which is computed with real FFTs.
"""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
import scipy.fft as sfft

from semantic_librarian.embed.base import WordModelConfig, WordSumEmbedder, unit_rows
from semantic_librarian.embed.cooccurrence import context_sums
from semantic_librarian.embed.random_vectors import environment_vectors
from semantic_librarian.space.vocabulary import Vocabulary


@dataclass
class BeagleConfig(WordModelConfig):
    variant: Literal["composite", "legacy"] = "composite"
    environment: Literal["gaussian", "ternary"] = "gaussian"
    nonzeros: int = 30
    context: bool = True
    order: bool = True
    max_ngram: int = 7
    # Worker processes for the order computation: 0 = automatic, 1 = no parallelism.
    # Results are identical for any number of workers.
    workers: int = 0


LEGACY_PRESET: dict[str, Any] = {
    "variant": "legacy",
    "environment": "ternary",
    "dim": 1024,
    "nonzeros": 30,
    "order": False,
    "stopwords": None,
    "compose_without_stopwords": False,
}


class Beagle(WordSumEmbedder):
    name: ClassVar[str] = "beagle"

    def __init__(self, cfg: BeagleConfig | None = None, **params: Any) -> None:
        super().__init__()
        self.cfg: BeagleConfig = cfg or BeagleConfig(**params)
        if self.cfg.variant == "legacy" and self.cfg.order:
            raise ValueError("the legacy variant has no order information; set order=False")
        if not (self.cfg.context or self.cfg.order):
            raise ValueError("at least one of context and order must be enabled")
        if self.cfg.max_ngram < 2:
            raise ValueError("max_ngram must be at least 2")
        self.environment: np.ndarray | None = None
        self.context_vectors: np.ndarray | None = None
        self.order_vectors: np.ndarray | None = None
        self.log: Callable[[str], None] | None = None

    # -- public -----------------------------------------------------------------------

    def learn_word_vectors(
        self,
        sentence_ids: list[np.ndarray],
        vocab: Vocabulary,
        environment: np.ndarray | None = None,
    ) -> np.ndarray:
        """Learn word vectors.

        Args:
            sentence_ids: Vocabulary row indices for each sentence.
            vocab: The vocabulary.
            environment: Optional ``(len(vocab), dim)`` environment matrix. When omitted it
                is generated from ``cfg.seed``. Passing one is how the parity test feeds
                the R-generated random vectors in.
        """
        rng = np.random.default_rng(self.cfg.seed)
        if environment is None:
            environment = environment_vectors(
                self.cfg.environment, len(vocab), self.cfg.dim, self.cfg.nonzeros, rng
            )
        if environment.shape != (len(vocab), self.cfg.dim):
            raise ValueError(
                f"environment must be {(len(vocab), self.cfg.dim)}, got {environment.shape}"
            )
        self.environment = environment
        if self.cfg.variant == "legacy":
            return _legacy_beagle(sentence_ids, environment)

        stop = self.stopword_set
        keep = np.array([t not in stop for t in vocab.tokens], dtype=bool)
        dim = self.cfg.dim
        memory_ctx = context_sums(sentence_ids, environment, keep) if self.cfg.context else None
        memory_ord = None
        if self.cfg.order:
            phi = environment_vectors(self.cfg.environment, 1, dim, self.cfg.nonzeros, rng)[0]
            perm1 = rng.permutation(dim)
            perm2 = rng.permutation(dim)
            self._phi, self._perm1, self._perm2 = phi.astype(np.float64), perm1, perm2
            memory_ord = order_memory(
                sentence_ids,
                environment,
                self._phi,
                perm1,
                perm2,
                self.cfg.max_ngram,
                workers=self.cfg.workers,
                log=self.log,
            )

        self.context_vectors = memory_ctx
        self.order_vectors = memory_ord
        parts = [unit_rows(m) for m in (memory_ctx, memory_ord) if m is not None]
        return np.asarray(sum(parts), dtype=np.float32)


def _legacy_beagle(sentence_ids: Sequence[np.ndarray], environment: np.ndarray) -> np.ndarray:
    """Port of ``sl_beagle_vectors()``. Keeps its normalisation quirk: ``x / abs(max(x))``."""
    env = np.asarray(environment, dtype=np.float64)
    memory = env.copy()
    for ids in sentence_ids:
        n = len(ids)
        if n <= 1:
            continue
        x = env[ids]
        total = x.sum(axis=0)
        if len(np.unique(ids)) == n:
            ctx = total - x
            memory[ids] += ctx / _abs_max(ctx)[:, None]
            memory[ids] /= _abs_max(memory[ids])[:, None]
        else:  # repeated words update sequentially, exactly as the R loop does
            for w in range(n):
                ctx = total - x[w]
                row = memory[ids[w]] + ctx / _abs_max(ctx[None])[0]
                memory[ids[w]] = row / _abs_max(row[None])[0]
    return memory


def _abs_max(m: np.ndarray) -> np.ndarray:
    """``abs(max(x))`` per row, guarding against division by zero (R would yield Inf)."""
    d = np.abs(m.max(axis=1))
    d[d == 0] = 1.0
    return d


def bind(a: np.ndarray, b: np.ndarray, perm1: np.ndarray, perm2: np.ndarray) -> np.ndarray:
    """Non-commutative circular convolution of (rows of) ``a`` and ``b``."""
    dim = a.shape[-1]
    return sfft.irfft(sfft.rfft(a[..., perm1]) * sfft.rfft(b[..., perm2]), n=dim)


def order_vectors(
    x: np.ndarray,
    phi: np.ndarray,
    perm1: np.ndarray,
    perm2: np.ndarray,
    max_ngram: int,
) -> np.ndarray:
    """Order information for every position of one sentence (paper Eqs. 3-4).

    For the word at position ``p`` this sums, over every contiguous window of length
    2..``max_ngram`` that contains ``p``, the left-to-right chain of bindings of the
    window's environment vectors with ``p`` replaced by ``phi``. "a dog bit the mailman"
    gives seven terms for "dog", as in the paper.

    Chains are advanced one word per step for all windows at once. A window's chain is
    the same for every target until it reaches the placeholder, so those shared prefixes
    are computed once and branch off when the placeholder is bound in.

    Args:
        x: ``(L, dim)`` environment vectors of the sentence's words.
        phi: The placeholder vector.
        perm1, perm2: Fixed permutations that make the binding non-commutative.
        max_ngram: Longest window considered.
    """
    x = np.asarray(x, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    length, dim = x.shape
    out = np.zeros((length, dim), dtype=np.float64)
    if length < 2:
        return out
    lam = min(max_ngram, length)
    fx = sfft.rfft(x[:, perm2])
    fphi = sfft.rfft(phi[perm2])

    # Plain prefixes: plain[a] = x[a] (*) x[a+1] (*) ... (*) x[a+t] for the current t.
    plain_start = np.arange(length)
    plain = x.copy()
    # Placeholder chains: one per (start, target) whose placeholder is already bound in.
    # A window starting at its own target begins as phi.
    ph_start = np.arange(length)
    ph_target = np.arange(length)
    ph = np.broadcast_to(phi, (length, dim)).copy()

    targets_done: list[np.ndarray] = []
    chains_done: list[np.ndarray] = []
    for t in range(1, lam):
        # extend placeholder chains with the next real word
        ends = ph_start + t
        alive = ends < length
        ph_start, ph_target, ends = ph_start[alive], ph_target[alive], ends[alive]
        ext = sfft.irfft(sfft.rfft(ph[alive][:, perm1]) * fx[ends], n=dim)
        # plain prefixes whose next position becomes the placeholder branch off here
        p_ends = plain_start + t
        p_alive = p_ends < length
        plain_start, p_ends = plain_start[p_alive], p_ends[p_alive]
        plain_f = sfft.rfft(plain[p_alive][:, perm1])
        branched = sfft.irfft(plain_f * fphi, n=dim)
        plain = sfft.irfft(plain_f * fx[p_ends], n=dim)

        ph = np.concatenate([ext, branched])
        ph_target = np.concatenate([ph_target, p_ends])
        ph_start = np.concatenate([ph_start, plain_start])
        targets_done.append(ph_target)
        chains_done.append(ph)

    if targets_done:
        targets = np.concatenate(targets_done)
        order = np.argsort(targets, kind="stable")
        sorted_targets = targets[order]
        starts_at = np.flatnonzero(np.r_[True, sorted_targets[1:] != sorted_targets[:-1]])
        sums = np.add.reduceat(np.concatenate(chains_done)[order], starts_at, axis=0)
        out[sorted_targets[starts_at]] += sums
    return out


# ---- order memory over a whole corpus, optionally in parallel -----------------------

# Sentences are grouped into chunks of about this many tokens. Chunk boundaries depend
# only on the corpus, and partial sums are added in chunk order, so the result does not
# depend on the number of workers.
CHUNK_TOKENS = 20_000
# Below this many tokens, starting worker processes costs more than it saves.
PARALLEL_MIN_TOKENS = 50_000


@dataclass
class _OrderChunk:
    rows: np.ndarray  # vocabulary rows touched by this chunk
    sentences: list[np.ndarray]  # each sentence as indices into ``rows``
    env: np.ndarray  # environment vectors of ``rows``
    phi: np.ndarray
    perm1: np.ndarray
    perm2: np.ndarray
    max_ngram: int


def _single_threaded_worker() -> None:
    """Each worker uses one BLAS thread; otherwise workers oversubscribe the CPU."""
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except ImportError:  # pragma: no cover - threadpoolctl ships with scikit-learn
        pass


def _order_chunk(chunk: _OrderChunk) -> np.ndarray:
    out = np.zeros((len(chunk.rows), chunk.env.shape[1]), dtype=np.float64)
    for local in chunk.sentences:
        x = np.asarray(chunk.env[local], dtype=np.float64)
        np.add.at(
            out, local, order_vectors(x, chunk.phi, chunk.perm1, chunk.perm2, chunk.max_ngram)
        )
    return out


def _chunks(
    sentence_ids: Sequence[np.ndarray],
    environment: np.ndarray,
    phi: np.ndarray,
    perm1: np.ndarray,
    perm2: np.ndarray,
    max_ngram: int,
) -> Iterator[_OrderChunk]:
    batch: list[np.ndarray] = []
    size = 0

    def emit() -> _OrderChunk:
        rows, inverse = np.unique(np.concatenate(batch), return_inverse=True)
        bounds = np.cumsum([len(b) for b in batch])[:-1]
        return _OrderChunk(
            rows,
            np.split(inverse, bounds),
            np.asarray(environment[rows]),
            phi,
            perm1,
            perm2,
            max_ngram,
        )

    for ids in sentence_ids:
        if len(ids) < 2:
            continue
        batch.append(np.asarray(ids, dtype=np.int64))
        size += len(ids)
        if size >= CHUNK_TOKENS:
            yield emit()
            batch, size = [], 0
    if batch:
        yield emit()


def resolve_workers(workers: int, n_tokens: int) -> int:
    """How many processes to use: ``workers`` if positive, else automatic."""
    if workers > 0:
        return workers
    if n_tokens < PARALLEL_MIN_TOKENS:
        return 1
    return max(1, min(os.cpu_count() or 1, 8))


def order_memory(
    sentence_ids: Sequence[np.ndarray],
    environment: np.ndarray,
    phi: np.ndarray,
    perm1: np.ndarray,
    perm2: np.ndarray,
    max_ngram: int,
    workers: int = 0,
    log: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Sum the order vectors of every word occurrence in the corpus.

    Work is split into chunks of sentences; with more than one worker the chunks run in
    separate processes, at most ``2 * workers`` at a time to bound memory.
    """
    n_vocab, dim = environment.shape
    memory = np.zeros((n_vocab, dim), dtype=np.float32)
    n_tokens = sum(len(s) for s in sentence_ids if len(s) >= 2)
    n_workers = resolve_workers(workers, n_tokens)
    total_chunks = max(1, -(-n_tokens // CHUNK_TOKENS))
    chunks = _chunks(sentence_ids, environment, phi, perm1, perm2, max_ngram)
    done = 0

    def add(rows: np.ndarray, part: np.ndarray) -> None:
        nonlocal done
        memory[rows] += part.astype(np.float32)
        done += 1
        if log is not None and done % 10 == 0:
            log(f"order information: about {100 * done // total_chunks}% done")

    if n_workers == 1:
        for chunk in chunks:
            add(chunk.rows, _order_chunk(chunk))
        if log is not None:
            log(f"order information: {done} chunks done")
        return memory

    if log is not None:
        log(f"order information: {n_tokens} tokens on {n_workers} processes")
    pending: deque[tuple[np.ndarray, Future[np.ndarray]]] = deque()
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_single_threaded_worker) as pool:
        for chunk in chunks:
            pending.append((chunk.rows, pool.submit(_order_chunk, chunk)))
            if len(pending) >= 2 * n_workers:
                rows, fut = pending.popleft()
                add(rows, fut.result())
        while pending:
            rows, fut = pending.popleft()
            add(rows, fut.result())
    if log is not None:
        log(f"order information: {done} chunks done")
    return memory
