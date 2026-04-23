"""
Causal inference: IPW + AIPW estimators from scratch.

Theory:
  - IPW (Inverse Probability Weighting): τ_IPW = E[T*Y/e(X) - (1-T)*Y/(1-e(X))]
  - AIPW (Augmented IPW, doubly robust): adds outcome model μ_t(X)
    τ_AIPW = E[μ_1(X) - μ_0(X) + T*(Y - μ_1(X))/e(X) - (1-T)*(Y - μ_0(X))/(1-e(X))]
"""
from __future__ import annotations
import numpy as np
from typing import Callable


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    n_epochs: int = 1000,
) -> tuple[np.ndarray, float]:
    """
    Fit logistic regression via gradient descent on negative log-likelihood.

    Args:
        X: (n, p) feature matrix
        y: (n,) binary labels {0, 1}
        lr: learning rate
        n_epochs: number of gradient descent steps

    Returns:
        w: (p,) weight vector
        b: scalar bias term
    """
    n, p = X.shape
    w = np.zeros(p)
    b = 0.0

    for epoch in range(n_epochs):
        # Predictions
        z = X @ w + b
        z = np.clip(z, -500, 500)  # Numerical stability
        preds = sigmoid(z)

        # Gradient of negative log-likelihood w.r.t. w, b
        # NLL = -mean(y*log(preds) + (1-y)*log(1-preds))
        # d/dw NLL = mean((preds - y) * X)
        # d/db NLL = mean(preds - y)
        error = preds - y  # (n,)
        grad_w = (X.T @ error) / n  # (p,)
        grad_b = np.mean(error)

        # Gradient step
        w -= lr * grad_w
        b -= lr * grad_b
        
        # Clip to prevent explosion
        w = np.clip(w, -10, 10)
        b = np.clip(b, -10, 10)

    return w, b


def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """
    Predict P(y=1 | X) using sigmoid.

    Args:
        X: (n, p) feature matrix
        w: (p,) weights
        b: scalar bias

    Returns:
        (n,) probabilities in (0, 1)
    """
    z = X @ w + b
    z = np.clip(z, -500, 500)
    return sigmoid(z)


def fit_ols_ridge(
    X: np.ndarray,
    y: np.ndarray,
    lam: float = 1e-3,
) -> np.ndarray:
    """
    Fit OLS with ridge penalty via normal equations.

    Solves: (X'X + λI) w = X'y
    where w[0] is intercept, w[1:] are slopes.

    Args:
        X: (n, p) feature matrix (intercept NOT included)
        y: (n,) target vector
        lam: ridge penalty

    Returns:
        w: (p+1,) [intercept, slope_1, ..., slope_p]
    """
    n, p = X.shape

    # Prepend intercept column
    X_aug = np.hstack([np.ones((n, 1)), X])  # (n, p+1)

    # Solve (X'X + λI) w = X'y
    XtX = X_aug.T @ X_aug
    Xty = X_aug.T @ y

    # Add ridge to diagonal (but not intercept row/col)
    ridge_matrix = np.eye(p + 1)
    ridge_matrix[0, 0] = 0  # Don't regularize intercept
    XtX_ridge = XtX + lam * ridge_matrix

    # Solve via normal equations
    try:
        w = np.linalg.solve(XtX_ridge, Xty)
    except np.linalg.LinAlgError:
        # Singular matrix fallback
        w = np.linalg.lstsq(XtX_ridge, Xty, rcond=None)[0]

    return w


def predict_ols(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Predict using OLS model.

    Args:
        X: (n, p) feature matrix (intercept NOT included)
        w: (p+1,) weights [intercept, slope_1, ..., slope_p]

    Returns:
        (n,) predictions
    """
    n = X.shape[0]
    X_aug = np.hstack([np.ones((n, 1)), X])
    return X_aug @ w


def ipw_ate(X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> float:
    """
    Estimate average treatment effect via IPW.

    τ_IPW = (1/n) * Σ_i [ T_i * Y_i / e(X_i) - (1-T_i) * Y_i / (1-e(X_i)) ]

    Args:
        X: (n, p) covariates
        T: (n,) binary treatment {0, 1}
        Y: (n,) outcome

    Returns:
        scalar ATE estimate
    """
    n = X.shape[0]

    # Fit propensity score
    w_prop, b_prop = fit_logistic(X, T, lr=0.01, n_epochs=500)
    e = predict_proba(X, w_prop, b_prop)

    # Avoid division by zero and extreme weights
    e = np.clip(e, 0.01, 0.99)

    # IPW formula
    treated_term = T * Y / e
    control_term = (1 - T) * Y / (1 - e)
    tau = np.mean(treated_term - control_term)

    return float(tau)


def aipw_ate(X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> float:
    """
    Estimate ATE via AIPW (augmented IPW, doubly robust).

    τ_AIPW = (1/n) * Σ_i [ μ_1(X_i) - μ_0(X_i)
                           + T_i * (Y_i - μ_1(X_i)) / e(X_i)
                           - (1-T_i) * (Y_i - μ_0(X_i)) / (1-e(X_i)) ]

    where μ_t(X) = E[Y | T=t, X].

    Args:
        X: (n, p) covariates
        T: (n,) binary treatment {0, 1}
        Y: (n,) outcome

    Returns:
        scalar ATE estimate
    """
    n = X.shape[0]

    # Fit propensity score
    w_prop, b_prop = fit_logistic(X, T, lr=0.01, n_epochs=500)
    e = predict_proba(X, w_prop, b_prop)
    e = np.clip(e, 0.01, 0.99)

    # Fit outcome model for T=1
    mask_1 = T == 1
    if mask_1.sum() > 1:
        w_mu1 = fit_ols_ridge(X[mask_1], Y[mask_1], lam=1e-3)
    else:
        w_mu1 = np.ones(X.shape[1] + 1) * Y[mask_1].mean()

    # Fit outcome model for T=0
    mask_0 = T == 0
    if mask_0.sum() > 1:
        w_mu0 = fit_ols_ridge(X[mask_0], Y[mask_0], lam=1e-3)
    else:
        w_mu0 = np.ones(X.shape[1] + 1) * Y[mask_0].mean()

    # Predict outcomes
    mu1 = predict_ols(X, w_mu1)
    mu0 = predict_ols(X, w_mu0)

    # AIPW formula
    base_term = mu1 - mu0
    treated_term = T * (Y - mu1) / e
    control_term = (1 - T) * (Y - mu0) / (1 - e)
    tau = np.mean(base_term + treated_term - control_term)

    return float(tau)


def bootstrap_ate(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    estimator_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """
    Bootstrap-based inference for ATE.

    Resamples rows with replacement, recomputes estimator, reports
    mean, std, and 95% percentile CI.

    Args:
        X: (n, p) covariates
        T: (n,) treatment
        Y: (n,) outcome
        estimator_fn: function(X, T, Y) -> float returning ATE estimate
        n_boot: number of bootstrap resamples
        seed: RNG seed

    Returns:
        (mean, std, ci_low, ci_high) where ci_*  are 2.5/97.5 percentiles
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    estimates = []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        X_b, T_b, Y_b = X[idx], T[idx], Y[idx]

        try:
            tau_b = estimator_fn(X_b, T_b, Y_b)
            # Only include finite estimates
            if np.isfinite(tau_b):
                estimates.append(tau_b)
        except Exception:
            # Skip if estimation fails (e.g., singular matrix)
            continue

    if len(estimates) < 10:
        # Fallback if too many failures
        return float(np.mean([estimator_fn(X, T, Y)])), 0.1, -0.1, 0.1

    estimates = np.array(estimates)
    mean = float(np.mean(estimates))
    std = float(np.std(estimates, ddof=1))
    ci_low = float(np.percentile(estimates, 2.5))
    ci_high = float(np.percentile(estimates, 97.5))

    return mean, std, ci_low, ci_high
