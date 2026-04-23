import numpy as np
from src.kernels import rbf, periodic, string_kernel

def test_rbf_symmetric_and_psd():
    X = np.random.randn(10, 3)
    K = rbf(X, X, length=1.0, var=1.0)
    np.testing.assert_allclose(K, K.T, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(K) >= -1e-8)

def test_periodic_period_identity():
    x = np.array([[0.0]]); xp = np.array([[2*np.pi]])
    K = periodic(x, xp, length=1.0, period=2*np.pi, var=1.0)
    assert np.isclose(K[0,0], 1.0)

def test_string_kernel_symmetric():
    a = ["action scifi", "drama romance"]
    b = ["action scifi", "drama romance"]
    K = string_kernel(a, b)
    np.testing.assert_allclose(K, K.T, atol=1e-8)
