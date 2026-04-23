"""
Tests for Thompson sampling on Gaussian Process bandit.
"""
import numpy as np
import pytest
from src.gp import GaussianProcess
from src.kernels import rbf
from src.thompson_gp import (
    sample_from_gp_posterior,
    thompson_sampling_loop,
    random_baseline,
)


def test_joint_sample_shape():
    """Sample from GP posterior should return shape (n_points,) or (n_points, n_samples)."""
    # Create a simple GP
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0.0, 1.0, 0.5])
    gp = GaussianProcess(kernel=lambda A, B: rbf(A, B, length=1.0, var=1.0), noise=1e-4)
    gp.fit(X_train, y_train)

    # Test point
    X_test = np.array([[0.5], [1.5]])

    # Single sample
    sample = sample_from_gp_posterior(gp, X_test, n_samples=1, jitter=1e-6)
    assert sample.shape == (len(X_test),), f"Expected shape {(len(X_test),)}, got {sample.shape}"

    # Multiple samples
    samples = sample_from_gp_posterior(gp, X_test, n_samples=5, jitter=1e-6)
    assert samples.shape == (len(X_test), 5), f"Expected shape {(len(X_test), 5)}, got {samples.shape}"


def test_joint_sample_respects_mean():
    """With very small variance, sample should be close to posterior mean."""
    # Create a GP with small noise
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0.0, 1.0, 0.5])
    gp = GaussianProcess(kernel=lambda A, B: rbf(A, B, length=1.0, var=0.01), noise=1e-6)
    gp.fit(X_train, y_train)

    X_test = np.array([[0.5], [1.5], [0.3]])
    mu, _ = gp.predict(X_test)

    # Draw many samples and check mean is close to posterior mean
    samples = sample_from_gp_posterior(gp, X_test, n_samples=100, jitter=1e-8)
    sample_mean = samples.mean(axis=1)
    np.testing.assert_allclose(sample_mean, mu, atol=0.15)


def test_thompson_beats_random_on_synthetic():
    """Thompson sampling should beat random baseline on a synthetic 1D problem with clear optimum."""
    rng = np.random.default_rng(42)

    # Create synthetic problem: f(x) = -(x-0.5)^2 (optimum at x=0.5, value=0)
    # Pool of 20 arms uniformly distributed in [0, 1]
    X_pool = np.sort(rng.uniform(0, 1, 20)).reshape(-1, 1)
    y_pool = -(X_pool - 0.5) ** 2
    y_pool = y_pool.flatten()

    # Seed set: 3 random points
    seed_idx = np.array([2, 10, 18])
    X_seen = X_pool[seed_idx]
    y_seen = y_pool[seed_idx]

    # Define GP kernel
    def kernel(A, B):
        return rbf(A, B, length=0.3, var=1.0)

    gp = GaussianProcess(kernel=kernel, noise=0.01)
    gp.fit(X_seen, y_seen)

    # Thompson sampling
    thompson_chosen, thompson_rewards, thompson_regret = thompson_sampling_loop(
        gp, X_pool, y_pool, X_seen, y_seen, n_rounds=15, seed=42
    )

    # Random baseline
    random_chosen, random_rewards, random_regret = random_baseline(
        X_pool, y_pool, n_rounds=15, seed=42
    )

    # Check shapes
    assert len(thompson_chosen) == 15
    assert len(thompson_rewards) == 15
    assert len(thompson_regret) == 15

    # Thompson should accumulate less regret than random
    # (Thompson should be better on average after 15 rounds)
    assert thompson_regret[-1] < random_regret[-1], \
        f"Thompson regret {thompson_regret[-1]:.4f} >= Random regret {random_regret[-1]:.4f}"


def test_thompson_sampling_loop_outputs():
    """Check thompson_sampling_loop returns correct output format."""
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0.0, 1.0, 0.5])
    gp = GaussianProcess(kernel=lambda A, B: rbf(A, B, length=1.0, var=1.0), noise=0.01)
    gp.fit(X_train, y_train)

    X_pool = np.array([[0.3], [0.7], [1.5], [2.5]])
    y_pool = np.array([0.2, 0.8, 0.9, 0.3])

    chosen_idx, rewards, regrets = thompson_sampling_loop(
        gp, X_pool, y_pool, X_train, y_train, n_rounds=5, seed=42
    )

    assert len(chosen_idx) == 5, "Should have 5 arm selections"
    assert len(rewards) == 5, "Should have 5 reward observations"
    assert len(regrets) == 5, "Should have 5 regret values"
    assert all(0 <= idx < len(X_pool) for idx in chosen_idx), "All chosen indices must be in pool"
    assert all(regrets[i] >= regrets[i-1] if i > 0 else True for i in range(len(regrets))), \
        "Cumulative regret should be non-decreasing"


def test_random_baseline_format():
    """Check random_baseline returns correct output format."""
    X_pool = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_pool = np.array([0.1, 0.8, 0.5, 0.3])

    chosen_idx, rewards, regrets = random_baseline(X_pool, y_pool, n_rounds=8, seed=123)

    assert len(chosen_idx) == 8, "Should have 8 arm selections"
    assert len(rewards) == 8, "Should have 8 reward observations"
    assert len(regrets) == 8, "Should have 8 regret values"
    assert all(0 <= idx < len(X_pool) for idx in chosen_idx), "All chosen indices must be in pool"
    assert all(regrets[i] >= regrets[i-1] if i > 0 else True for i in range(len(regrets))), \
        "Cumulative regret should be non-decreasing"
