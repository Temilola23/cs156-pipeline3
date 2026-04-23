"""Gaussian process regression via Cholesky (from scratch)."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

class GaussianProcess:
    def __init__(self, kernel, noise=1e-4):
        self.kernel = kernel
        self.noise = noise
    def fit(self, X, y):
        self.X, self.y = X, y
        K = self.kernel(X, X) + self.noise * np.eye(len(X))
        self.L = cho_factor(K, lower=True)
        self.alpha = cho_solve(self.L, y)
        return self
    def predict(self, Xs):
        Ks = self.kernel(Xs, self.X)
        Kss = self.kernel(Xs, Xs)
        mu = Ks @ self.alpha
        v = cho_solve(self.L, Ks.T)
        cov = Kss - Ks @ v + self.noise * np.eye(len(Xs))
        return mu, np.diag(cov)
    def log_marginal_likelihood(self):
        n = len(self.y)
        return -0.5 * self.y @ self.alpha - np.log(np.diag(self.L[0])).sum() - 0.5*n*np.log(2*np.pi)
