"""From-scratch state-space filters: Kalman, RTS smoother, EKF, UKF (Merwe sigma points)."""
import numpy as np


class KalmanFilter:
    def __init__(self, F, H, Q, R):
        self.F, self.H, self.Q, self.R = F, H, Q, R

    def init(self, x0, P0):
        self.x, self.P = x0.copy(), P0.copy()

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, y):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (y - self.H @ self.x)
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P
        return K


class RTSSmoother:
    """Rauch-Tung-Striebel backward pass given filtered means/covariances."""
    def smooth(self, xs_f, Ps_f, F):
        n = len(xs_f)
        xs = [x.copy() for x in xs_f]
        Ps = [P.copy() for P in Ps_f]
        eye = np.eye(len(xs[0]))
        for t in range(n - 2, -1, -1):
            P_pred = F @ Ps[t] @ F.T
            G = Ps[t] @ F.T @ np.linalg.inv(P_pred + 1e-8 * eye)
            xs[t] = xs[t] + G @ (xs[t + 1] - F @ xs[t])
            Ps[t] = Ps[t] + G @ (Ps[t + 1] - P_pred) @ G.T
        return xs, Ps


class ExtendedKalmanFilter:
    """EKF with finite-difference Jacobians. f, h are callables on x."""
    def __init__(self, f, h, Q, R, eps=1e-5):
        self.f, self.h, self.Q, self.R, self.eps = f, h, Q, R, eps

    def init(self, x0, P0):
        self.x, self.P = x0.copy(), P0.copy()

    def _jac(self, fn, x):
        n = len(x)
        fx = fn(x)
        J = np.zeros((len(fx), n))
        for i in range(n):
            dx = np.zeros(n); dx[i] = self.eps
            J[:, i] = (fn(x + dx) - fx) / self.eps
        return J

    def predict(self):
        F = self._jac(self.f, self.x)
        self.x = self.f(self.x)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, y):
        H = self._jac(self.h, self.x)
        yhat = self.h(self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (y - yhat)
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P


class UKF:
    """Unscented Kalman filter with Merwe scaled sigma points."""
    def __init__(self, f, h, Q, R, n, alpha=1e-3, beta=2.0, kappa=0.0):
        self.f, self.h, self.Q, self.R, self.n = f, h, Q, R, n
        lam = alpha**2 * (n + kappa) - n
        self.c = n + lam
        self.Wm = np.full(2 * n + 1, 1.0 / (2 * self.c))
        self.Wc = self.Wm.copy()
        self.Wm[0] = lam / self.c
        self.Wc[0] = lam / self.c + (1 - alpha**2 + beta)

    def init(self, x0, P0):
        self.x, self.P = x0.copy(), P0.copy()

    def _sigmas(self, x, P):
        U = np.linalg.cholesky(self.c * P + 1e-9 * np.eye(self.n))
        pts = [x.copy()]
        for i in range(self.n):
            pts.append(x + U[i])
        for i in range(self.n):
            pts.append(x - U[i])
        return np.array(pts)

    def predict(self):
        sp = self._sigmas(self.x, self.P)
        sp_f = np.array([self.f(s) for s in sp])
        self.x = (self.Wm[:, None] * sp_f).sum(0)
        self.P = self.Q + sum(
            self.Wc[i] * np.outer(sp_f[i] - self.x, sp_f[i] - self.x)
            for i in range(len(sp))
        )
        self.sp_f = sp_f

    def update(self, y):
        sp_h = np.array([self.h(s) for s in self.sp_f])
        if sp_h.ndim == 1:
            sp_h = sp_h[:, None]
        yhat = (self.Wm[:, None] * sp_h).sum(0)
        Pyy = self.R + sum(
            self.Wc[i] * np.outer(sp_h[i] - yhat, sp_h[i] - yhat)
            for i in range(len(sp_h))
        )
        Pxy = sum(
            self.Wc[i] * np.outer(self.sp_f[i] - self.x, sp_h[i] - yhat)
            for i in range(len(sp_h))
        )
        K = Pxy @ np.linalg.inv(Pyy)
        self.x = self.x + K @ (y - yhat)
        self.P = self.P - K @ Pyy @ K.T
