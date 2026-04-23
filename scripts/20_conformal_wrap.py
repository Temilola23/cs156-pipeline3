"""Wrap the Task-1.1 GP with split-conformal intervals over 324 ratings (year feature)."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.data_io import load_324_ratings
from src.kernels import rbf, periodic
from src.gp import GaussianProcess
from src.conformal import SplitConformal

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
(ART / "plots").mkdir(parents=True, exist_ok=True)

df = load_324_ratings().dropna(subset=['year']).reset_index(drop=True)
X = df[['year']].values.astype(float)
y = df['rating'].values.astype(float)

rng = np.random.default_rng(42)
idx = rng.permutation(len(df))
n = len(df)
n_train = int(0.5 * n); n_calib = int(0.25 * n)
i_tr = idx[:n_train]; i_cal = idx[n_train:n_train + n_calib]; i_te = idx[n_train + n_calib:]


def combined(A, B):
    return rbf(A, B, length=5.0, var=0.3) + periodic(A, B, length=3.0, period=10.0, var=0.1)


class GPRegressor:
    """Thin adapter so GP fits the sklearn-style API SplitConformal expects."""
    def fit(self, X, y):
        self.y_mean = float(np.mean(y))
        self.gp = GaussianProcess(kernel=combined, noise=0.25)
        self.gp.fit(X, y - self.y_mean)
        return self

    def predict(self, X):
        mu, _ = self.gp.predict(X)
        return mu + self.y_mean


cp = SplitConformal(base_model_fn=lambda: GPRegressor(), alpha=0.1)
cp.fit(X[i_tr], y[i_tr], X[i_cal], y[i_cal])
lo, hi = cp.predict(X[i_te])
mu = cp.model.predict(X[i_te])

coverage = float(np.mean((lo <= y[i_te]) & (y[i_te] <= hi)))
avg_width = float(np.mean(hi - lo))

np.savez(ART / "conformal_intervals.npz",
         X_test=X[i_te], y_test=y[i_te], mu=mu, lo=lo, hi=hi,
         q_hat=cp.q_hat, alpha=cp.alpha, coverage=coverage, avg_width=avg_width)

fig, ax = plt.subplots(figsize=(9, 4.5))
order = np.argsort(X[i_te].flatten())
xs = X[i_te].flatten()[order]
ax.fill_between(xs, lo[order], hi[order], alpha=0.25, label=f'90% conformal band (width={avg_width:.2f})')
ax.scatter(X[i_te], y[i_te], s=18, c='red', label='held-out ratings', zorder=3)
ax.plot(xs, mu[order], c='black', lw=1, label='GP mean')
ax.set_xlabel('Year'); ax.set_ylabel('Rating')
ax.set_title(f'Split conformal prediction over GP — empirical coverage {coverage:.2%}')
ax.legend()
plt.tight_layout()
plt.savefig(ART / "plots" / "conformal_bands.png", dpi=130)

print(f"[Conformal] alpha=0.1, q_hat={cp.q_hat:.3f}")
print(f"[Conformal] empirical coverage on {len(i_te)} held-out = {coverage:.2%}, avg width = {avg_width:.3f}")
print("[Conformal] saved artifacts/conformal_intervals.npz + plot")
