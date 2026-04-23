"""From-scratch HMM with forward-backward, Baum-Welch EM, and Viterbi."""
import numpy as np

LOG_EPS = 1e-300


def _log(x):
    return np.log(np.clip(x, LOG_EPS, None))


def _logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m_safe, axis=axis) + np.log(
        np.sum(np.exp(a - m_safe), axis=axis)
    )


class HMM:
    """Discrete-observation HMM. K hidden states, B observation bins."""

    def __init__(self, n_states, n_obs_bins, seed=0):
        self.K, self.B = n_states, n_obs_bins
        rng = np.random.default_rng(seed)
        self.A = rng.dirichlet(np.ones(n_states), n_states)
        self.E = rng.dirichlet(np.ones(n_obs_bins), n_states)
        self.pi = rng.dirichlet(np.ones(n_states))

    # ---------- forward / backward in log space ----------
    def _log_alpha(self, obs):
        T = len(obs)
        log_A = _log(self.A)
        log_E = _log(self.E)
        log_pi = _log(self.pi)
        la = np.full((T, self.K), -np.inf)
        la[0] = log_pi + log_E[:, obs[0]]
        for t in range(1, T):
            la[t] = log_E[:, obs[t]] + _logsumexp(
                la[t - 1][:, None] + log_A, axis=0
            )
        return la

    def _log_beta(self, obs):
        T = len(obs)
        log_A = _log(self.A)
        log_E = _log(self.E)
        lb = np.full((T, self.K), -np.inf)
        lb[T - 1] = 0.0
        for t in range(T - 2, -1, -1):
            lb[t] = _logsumexp(
                log_A + log_E[:, obs[t + 1]][None, :] + lb[t + 1][None, :],
                axis=1,
            )
        return lb

    def log_likelihood(self, obs):
        la = self._log_alpha(obs)
        return _logsumexp(la[-1])

    # ---------- Viterbi ----------
    def viterbi(self, obs):
        T = len(obs)
        log_A = _log(self.A)
        log_E = _log(self.E)
        log_pi = _log(self.pi)
        delta = np.full((T, self.K), -np.inf)
        psi = np.zeros((T, self.K), dtype=int)
        delta[0] = log_pi + log_E[:, obs[0]]
        for t in range(1, T):
            cand = delta[t - 1][:, None] + log_A
            psi[t] = np.argmax(cand, axis=0)
            delta[t] = np.max(cand, axis=0) + log_E[:, obs[t]]
        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    # ---------- Baum-Welch EM ----------
    def baum_welch(self, obs, n_iter=30, tol=1e-4):
        T = len(obs)
        history = []
        for it in range(n_iter):
            la = self._log_alpha(obs)
            lb = self._log_beta(obs)
            log_px = _logsumexp(la[-1])
            history.append(log_px)

            # posterior gamma, xi in log space
            log_gamma = la + lb - log_px  # (T, K)
            gamma = np.exp(log_gamma)

            log_A = _log(self.A)
            log_E = _log(self.E)

            # recompute log_xi cleanly
            log_xi = np.full((T - 1, self.K, self.K), -np.inf)
            for t in range(T - 1):
                log_xi[t] = (la[t][:, None] + log_A
                             + log_E[:, obs[t + 1]][None, :]
                             + lb[t + 1][None, :]
                             - log_px)
            xi = np.exp(log_xi)

            # M-step
            self.pi = gamma[0] / gamma[0].sum()
            A_new = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, None]
            self.A = A_new / A_new.sum(axis=1, keepdims=True)
            E_new = np.zeros_like(self.E)
            for b in range(self.B):
                mask = (obs == b)
                E_new[:, b] = gamma[mask].sum(axis=0)
            self.E = E_new / (E_new.sum(axis=1, keepdims=True) + 1e-12)

            if it > 0 and abs(history[-1] - history[-2]) < tol:
                break
        return history

    def posterior_states(self, obs):
        la = self._log_alpha(obs)
        lb = self._log_beta(obs)
        log_px = _logsumexp(la[-1])
        return np.exp(la + lb - log_px)
