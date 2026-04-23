"""GenMatch: genetic-algorithm search over covariate weights for Mahalanobis-based
matching. Fitness = minimum (over covariates) KS-test p-value of treated-vs-matched-control
distributions. Based on Diamond & Sekhon (2013).

From-scratch implementation:
- Mahalanobis distance with diagonal weight scaling
- Tournament-selection GA with real-valued weights, elitism
- Fitness via worst-case covariate KS p-value on matched sample
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp


# --------------------------------------------------------------------------- #
# Distance
# --------------------------------------------------------------------------- #
def mahalanobis_distance_matrix(A: np.ndarray, B: np.ndarray, Sigma_inv: np.ndarray) -> np.ndarray:
    """Pairwise Mahalanobis distances: D[i,j] = sqrt((a_i - b_j)^T Sigma_inv (a_i - b_j)).

    Uses the factorization Sigma_inv = L L^T so Mahalanobis in the original space
    equals Euclidean after transforming x -> L^T x.
    """
    A = np.atleast_2d(A); B = np.atleast_2d(B)
    # symmetric PSD factor
    try:
        L = np.linalg.cholesky(Sigma_inv + 1e-12 * np.eye(Sigma_inv.shape[0]))
    except np.linalg.LinAlgError:
        # fall back to eig for near-singular
        w, V = np.linalg.eigh(Sigma_inv)
        w = np.clip(w, 0.0, None)
        L = V @ np.diag(np.sqrt(w))
    A_t = A @ L
    B_t = B @ L
    d2 = (np.sum(A_t**2, 1)[:, None]
          + np.sum(B_t**2, 1)[None, :]
          - 2.0 * A_t @ B_t.T)
    return np.sqrt(np.clip(d2, 0.0, None))


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def _weighted_sigma_inv(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Scale the standardised inverse covariance by per-covariate weights w >= 0.

    Effectively this inflates the importance of covariates with larger w in the
    Mahalanobis distance: Sigma_inv_weighted = diag(sqrt(w)) Sigma_inv diag(sqrt(w)).
    """
    Sigma = np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1])
    Sigma_inv = np.linalg.inv(Sigma)
    sw = np.sqrt(np.clip(w, 0.0, None))
    return (sw[:, None] * Sigma_inv) * sw[None, :]


def _nearest_control(treated_idx: np.ndarray,
                     control_idx: np.ndarray,
                     D: np.ndarray,
                     with_replacement: bool = True) -> np.ndarray:
    """For each treated row return the index (into the original X) of the closest control."""
    sub = D[np.ix_(treated_idx, control_idx)]
    if with_replacement:
        chosen = sub.argmin(axis=1)
        return control_idx[chosen]
    # greedy without replacement: pick smallest remaining each iteration
    taken = np.zeros(len(control_idx), dtype=bool)
    out = np.zeros(len(treated_idx), dtype=int)
    order = np.argsort(sub.min(axis=1))
    for i in order:
        row = np.where(taken, np.inf, sub[i])
        j = int(np.argmin(row))
        taken[j] = True
        out[i] = control_idx[j]
    return out


# --------------------------------------------------------------------------- #
# Fitness
# --------------------------------------------------------------------------- #
def _fitness(w: np.ndarray, X: np.ndarray, treat: np.ndarray) -> float:
    """Return minimum (worst) KS-test p-value across covariates after matching.

    Higher is better — when all covariates balance well, p-values are large
    (failure to reject equal distributions).
    """
    treated_idx = np.where(treat == 1)[0]
    control_idx = np.where(treat == 0)[0]
    if len(treated_idx) == 0 or len(control_idx) == 0:
        return 0.0
    Sigma_inv = _weighted_sigma_inv(X, w)
    D = mahalanobis_distance_matrix(X, X, Sigma_inv)
    matched = _nearest_control(treated_idx, control_idx, D, with_replacement=True)
    worst_p = 1.0
    for j in range(X.shape[1]):
        _, p = ks_2samp(X[treated_idx, j], X[matched, j])
        if p < worst_p:
            worst_p = float(p)
    return worst_p


# --------------------------------------------------------------------------- #
# GA
# --------------------------------------------------------------------------- #
class GenMatch:
    """Genetic-algorithm covariate-weight search for Mahalanobis matching."""
    def __init__(self,
                 pop_size: int = 100,
                 n_generations: int = 50,
                 mutation_rate: float = 0.1,
                 mutation_sigma: float = 0.5,
                 tournament_k: int = 3,
                 elitism: int = 2,
                 seed: int = 0):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.tournament_k = tournament_k
        self.elitism = elitism
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    def _init_population(self, d: int) -> np.ndarray:
        """Draw initial weights uniformly in [0, 2]."""
        return self.rng.uniform(0.0, 2.0, size=(self.pop_size, d))

    def _tournament(self, pop: np.ndarray, fits: np.ndarray) -> np.ndarray:
        idx = self.rng.integers(0, len(pop), size=self.tournament_k)
        best = idx[np.argmax(fits[idx])]
        return pop[best].copy()

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Uniform per-gene crossover with random alpha in [0,1]."""
        alpha = self.rng.uniform(0.0, 1.0, size=a.shape)
        return alpha * a + (1.0 - alpha) * b

    def _mutate(self, w: np.ndarray) -> np.ndarray:
        mask = self.rng.uniform(size=w.shape) < self.mutation_rate
        noise = self.rng.normal(0.0, self.mutation_sigma, size=w.shape)
        w_new = np.where(mask, w + noise, w)
        return np.clip(w_new, 0.0, None)

    # ------------------------------------------------------------------ #
    def fit(self,
            X: np.ndarray,
            treat: np.ndarray,
            return_history: bool = False):
        """Run the GA. Stores best weight vector in self.best_weights_."""
        X = np.asarray(X, dtype=float)
        treat = np.asarray(treat).astype(int)
        d = X.shape[1]
        pop = self._init_population(d)
        fits = np.array([_fitness(w, X, treat) for w in pop])
        history = [float(fits.max())]
        best_w = pop[int(fits.argmax())].copy()
        best_f = float(fits.max())

        for _ in range(self.n_generations):
            # elitism: carry top-k
            elite_idx = np.argsort(-fits)[:self.elitism]
            new_pop = [pop[i].copy() for i in elite_idx]
            while len(new_pop) < self.pop_size:
                a = self._tournament(pop, fits)
                b = self._tournament(pop, fits)
                child = self._mutate(self._crossover(a, b))
                new_pop.append(child)
            pop = np.array(new_pop)
            fits = np.array([_fitness(w, X, treat) for w in pop])

            gen_best = float(fits.max())
            if gen_best > best_f:
                best_f = gen_best
                best_w = pop[int(fits.argmax())].copy()
            # guard monotonic-ish progress via best-so-far
            history.append(best_f)

        self.best_weights_ = best_w
        self.best_fitness_ = best_f
        self.fitness_history_ = history
        return history if return_history else self

    # ------------------------------------------------------------------ #
    def match(self, X: np.ndarray, treat: np.ndarray, with_replacement: bool = True) -> np.ndarray:
        """Return array of control indices matched to treated rows (one per treated)."""
        if not hasattr(self, 'best_weights_'):
            raise RuntimeError("Call .fit() before .match()")
        X = np.asarray(X, dtype=float); treat = np.asarray(treat).astype(int)
        treated_idx = np.where(treat == 1)[0]
        control_idx = np.where(treat == 0)[0]
        Sigma_inv = _weighted_sigma_inv(X, self.best_weights_)
        D = mahalanobis_distance_matrix(X, X, Sigma_inv)
        return _nearest_control(treated_idx, control_idx, D, with_replacement=with_replacement)

    def balance_report(self, X: np.ndarray, treat: np.ndarray) -> dict:
        """Before/after KS p-values per covariate."""
        treat = np.asarray(treat).astype(int)
        t_idx = np.where(treat == 1)[0]; c_idx = np.where(treat == 0)[0]
        before = [float(ks_2samp(X[t_idx, j], X[c_idx, j])[1]) for j in range(X.shape[1])]
        matched = self.match(X, treat)
        after = [float(ks_2samp(X[t_idx, j], X[matched, j])[1]) for j in range(X.shape[1])]
        return {'before_p': before, 'after_p': after}
