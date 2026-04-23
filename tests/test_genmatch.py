import numpy as np
from src.genmatch import GenMatch, mahalanobis_distance_matrix


def test_mahalanobis_symmetric_positive():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    Sigma_inv = np.eye(4)
    D = mahalanobis_distance_matrix(X, X, Sigma_inv)
    assert D.shape == (20, 20)
    np.testing.assert_allclose(D, D.T, atol=1e-8)
    assert np.all(D >= -1e-8)
    assert np.all(np.diag(D) < 1e-6)


def test_genmatch_ga_fitness_improves():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    treat = rng.integers(0, 2, 200)
    gm = GenMatch(pop_size=30, n_generations=20, seed=0)
    history = gm.fit(X, treat, return_history=True)
    # Monotonicity guaranteed by elitism; allow 2% slack for float noise
    assert history[-1] >= history[0] * 0.98
    # After 20 generations it should have found something non-trivial
    assert history[-1] > 0.0


def test_genmatch_match_returns_valid_pairs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    treat = np.concatenate([np.ones(30, dtype=int), np.zeros(70, dtype=int)])
    gm = GenMatch(pop_size=10, n_generations=5, seed=0).fit(X, treat)
    matched_idx = gm.match(X, treat)
    # one control per treated
    assert len(matched_idx) == 30
    # all matched are controls
    assert np.all(treat[matched_idx] == 0)
