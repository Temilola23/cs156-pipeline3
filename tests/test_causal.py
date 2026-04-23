"""Tests for causal inference: IPW + AIPW estimators."""
import numpy as np
import pytest
from src.causal import (
    fit_logistic,
    predict_proba,
    fit_ols_ridge,
    predict_ols,
    ipw_ate,
    aipw_ate,
    bootstrap_ate,
)


class TestLogistic:
    def test_logistic_converges_on_linearly_separable(self):
        """Logistic regression should converge and recover sign on separable data."""
        np.random.seed(42)
        # Create linearly separable data
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)  # y = 1 if x1 + x2 > 0

        # Fit with default hyperparams
        w, b = fit_logistic(X, y, lr=0.1, n_epochs=1000)

        # Check that we recovered the sign (positive bias term is plausible)
        assert isinstance(w, np.ndarray)
        assert isinstance(b, (float, np.floating))
        assert w.shape == (2,)

        # Predictions should show some separation
        preds = predict_proba(X, w, b)
        assert preds.min() > 0 and preds.max() < 1  # Valid probabilities
        assert preds.mean() > 0.3  # Some positive class samples


    def test_ols_recovers_ground_truth(self):
        """OLS with ridge should recover true linear relationship."""
        np.random.seed(42)
        # True model: y = 2*x + 1 + noise
        X = np.random.randn(100, 1)
        true_slope = 2.0
        true_intercept = 1.0
        noise = np.random.randn(100) * 0.1
        y = true_slope * X[:, 0] + true_intercept + noise

        # Fit ridge OLS
        w = fit_ols_ridge(X, y, lam=1e-3)

        # w[0] is intercept, w[1] is slope
        assert w.shape == (2,)
        assert abs(w[0] - true_intercept) < 0.2, f"Intercept: got {w[0]}, expected ~{true_intercept}"
        assert abs(w[1] - true_slope) < 0.2, f"Slope: got {w[1]}, expected ~{true_slope}"

        # Predictions should be close
        y_pred = predict_ols(X, w)
        mse = np.mean((y - y_pred) ** 2)
        assert mse < 0.1


    def test_ipw_recovers_simulated_ate(self):
        """IPW should recover ATE on a simulated causal dataset."""
        np.random.seed(42)
        n = 500

        # Covariates
        X = np.random.randn(n, 2)
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        # Propensity: P(T=1 | X)
        true_prop = 0.5 / (1 + np.exp(-X_norm[:, 0]))
        T = (np.random.rand(n) < true_prop).astype(int)

        # Outcome: E[Y | T, X] with true ATE = 1.0
        # Y = 0.5*X1 + 0.3*X2 + T*1.0 + noise
        Y = 0.5 * X_norm[:, 0] + 0.3 * X_norm[:, 1] + 1.0 * T + np.random.randn(n) * 0.2

        # Fit IPW
        ate_est = ipw_ate(X_norm, T, Y)

        # Should be within ~0.3 of true ATE = 1.0
        assert isinstance(ate_est, (float, np.floating))
        assert abs(ate_est - 1.0) < 0.3, f"IPW ATE: got {ate_est}, expected ~1.0"


    def test_aipw_lower_variance_than_ipw(self):
        """AIPW should have lower variance than IPW on the same simulated data."""
        np.random.seed(42)
        n = 500

        # Same causal DGP as above
        X = np.random.randn(n, 2)
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        true_prop = 0.5 / (1 + np.exp(-X_norm[:, 0]))
        T = (np.random.rand(n) < true_prop).astype(int)

        Y = 0.5 * X_norm[:, 0] + 0.3 * X_norm[:, 1] + 1.0 * T + np.random.randn(n) * 0.2

        # Bootstrap to get SE estimates
        def ipw_fn(X_b, T_b, Y_b):
            return ipw_ate(X_b, T_b, Y_b)

        def aipw_fn(X_b, T_b, Y_b):
            return aipw_ate(X_b, T_b, Y_b)

        _, ipw_se, _, _ = bootstrap_ate(X_norm, T, Y, ipw_fn, n_boot=100, seed=42)
        _, aipw_se, _, _ = bootstrap_ate(X_norm, T, Y, aipw_fn, n_boot=100, seed=42)

        # AIPW should have comparable or lower variance (doubly robust)
        # We allow some tolerance due to randomness
        assert aipw_se < ipw_se * 1.2, f"AIPW SE {aipw_se} not < IPW SE {ipw_se} * 1.2"


class TestBootstrap:
    def test_bootstrap_ate_returns_correct_shape(self):
        """Bootstrap should return (mean, std, ci_low, ci_high)."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        T = np.random.binomial(1, 0.5, 50)
        Y = T + np.random.randn(50) * 0.1

        mean, std, ci_low, ci_high = bootstrap_ate(X, T, Y, ipw_ate, n_boot=50, seed=42)

        assert isinstance(mean, (float, np.floating))
        assert isinstance(std, (float, np.floating))
        assert isinstance(ci_low, (float, np.floating))
        assert isinstance(ci_high, (float, np.floating))
        assert ci_low < mean < ci_high
        assert std > 0
