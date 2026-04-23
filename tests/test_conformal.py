import numpy as np
from sklearn.linear_model import LinearRegression
from src.conformal import SplitConformal


def test_split_conformal_coverage():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (500, 1))
    y = X[:, 0] + rng.normal(0, 1, 500)
    cp = SplitConformal(base_model_fn=lambda: LinearRegression(), alpha=0.1)
    cp.fit(X[:300], y[:300], X[300:400], y[300:400])
    lo, hi = cp.predict(X[400:])
    coverage = np.mean((lo <= y[400:]) & (y[400:] <= hi))
    # Asymptotic guarantee is 1-alpha=0.9; with 100-sample calib+test, allow empirical variance
    assert coverage >= 0.75
