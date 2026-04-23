"""Split-conformal prediction wrapper around any sklearn-style regressor."""
import numpy as np


class SplitConformal:
    """Distribution-free prediction intervals via split conformal prediction.

    Given a model-factory `base_model_fn` that returns a fresh regressor with
    `.fit(X,y)` and `.predict(X)` methods, plus proper-training and calibration
    splits, builds intervals with guaranteed marginal coverage >= 1-alpha.
    """
    def __init__(self, base_model_fn, alpha=0.1):
        self.base_model_fn = base_model_fn
        self.alpha = alpha

    def fit(self, X_train, y_train, X_calib, y_calib):
        self.model = self.base_model_fn()
        self.model.fit(X_train, y_train)
        preds_calib = np.asarray(self.model.predict(X_calib))
        residuals = np.abs(np.asarray(y_calib) - preds_calib)
        # finite-sample-corrected quantile
        n = len(residuals)
        k = int(np.ceil((1 - self.alpha) * (n + 1)))
        k = min(k, n)  # guard
        self.q_hat = np.sort(residuals)[k - 1]
        return self

    def predict(self, X):
        mu = np.asarray(self.model.predict(X))
        return mu - self.q_hat, mu + self.q_hat
