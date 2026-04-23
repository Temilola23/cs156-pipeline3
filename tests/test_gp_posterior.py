import numpy as np
from src.gp import GaussianProcess
from src.kernels import rbf

def test_posterior_interpolates_training_points():
    X = np.array([[0.],[1.],[2.]])
    y = np.array([0.0, 1.0, 0.5])
    gp = GaussianProcess(kernel=lambda A,B: rbf(A,B, length=1.0, var=1.0), noise=1e-6)
    gp.fit(X, y)
    mu, _ = gp.predict(X)
    np.testing.assert_allclose(mu, y, atol=1e-3)
