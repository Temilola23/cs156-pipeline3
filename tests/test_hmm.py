import numpy as np
from src.hmm import HMM


def test_forward_likelihood_valid():
    np.random.seed(0)
    hmm = HMM(n_states=2, n_obs_bins=3, seed=0)
    obs = np.array([0, 1, 2, 1, 0])
    log_px = hmm.log_likelihood(obs)
    assert np.isfinite(log_px)
    assert log_px < 0  # log-prob


def test_viterbi_returns_valid_path():
    hmm = HMM(n_states=3, n_obs_bins=5, seed=0)
    obs = np.array([0, 2, 4, 3, 1, 2, 4])
    path = hmm.viterbi(obs)
    assert len(path) == len(obs)
    assert path.min() >= 0 and path.max() < 3


def test_baum_welch_increases_likelihood():
    rng = np.random.default_rng(0)
    obs = rng.integers(0, 4, 60)
    hmm = HMM(n_states=2, n_obs_bins=4, seed=0)
    ll_before = hmm.log_likelihood(obs)
    hmm.baum_welch(obs, n_iter=15)
    ll_after = hmm.log_likelihood(obs)
    assert ll_after >= ll_before - 1e-6
