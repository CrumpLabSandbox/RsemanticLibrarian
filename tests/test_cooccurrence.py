from __future__ import annotations

import numpy as np
import pytest

from semantic_librarian.embed.cooccurrence import (
    context_sums,
    permutation_order_sums,
    roll_columns,
)
from semantic_librarian.embed.random_vectors import ternary_sparse


def _sentences(rng, n_vocab=30, n=40):
    return [rng.integers(0, n_vocab, size=rng.integers(1, 9)) for _ in range(n)]


def _naive_context(sentences, env, keep):
    out = np.zeros(env.shape)
    for ids in sentences:
        if len(ids) < 2:
            continue
        x = env[ids]
        total = x[keep[ids]].sum(axis=0)
        for i, w in enumerate(ids):
            out[w] += total - (x[i] if keep[w] else 0)
    return out


def _naive_rp_order(sentences, env, window):
    out = np.zeros(env.shape)
    for ids in sentences:
        if len(ids) < 2:
            continue
        for p, w in enumerate(ids):
            for k in range(-window, window + 1):
                if k and 0 <= p + k < len(ids):
                    out[w] += np.roll(env[ids[p + k]], k)
    return out


@pytest.mark.parametrize("sparse", [False, True])
def test_context_sums_match_loop(sparse):
    rng = np.random.default_rng(0)
    sentences = _sentences(rng)
    env_sparse = ternary_sparse(30, 64, 8, rng)
    env = env_sparse.toarray()
    keep = rng.random(30) > 0.2
    got = context_sums(sentences, env_sparse if sparse else env, keep, chunk=7)
    np.testing.assert_allclose(got, _naive_context(sentences, env, keep), atol=1e-5)


def test_permutation_order_sums_match_loop():
    rng = np.random.default_rng(1)
    sentences = _sentences(rng)
    env = ternary_sparse(30, 64, 8, rng)
    got = permutation_order_sums(sentences, env, window=2)
    np.testing.assert_allclose(got, _naive_rp_order(sentences, env.toarray(), 2), atol=1e-5)
    assert permutation_order_sums([np.array([3])], env, 2).sum() == 0


def test_roll_columns_and_ternary_sparse():
    rng = np.random.default_rng(2)
    env = ternary_sparse(5, 20, 6, rng)
    assert (env.getnnz(axis=1) == 6).all()
    assert (env.toarray().sum(axis=1) == 0).all()
    for k in (-3, 1, 7):
        np.testing.assert_array_equal(
            roll_columns(env, k).toarray(), np.roll(env.toarray(), k, axis=1)
        )
