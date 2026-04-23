"""Fit GP over 324 ratings → predict a rating surface over the TMDB catalog (82 movies)."""
import numpy as np, json, pickle
from pathlib import Path
from src.data_io import load_324_ratings, load_movie_meta
from src.kernels import rbf, periodic, string_kernel
from src.gp import GaussianProcess
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"; ART.mkdir(exist_ok=True)
(ART / "plots").mkdir(exist_ok=True)

df = load_324_ratings()
meta = load_movie_meta()

# features: year (float); drop rows missing year
df = df.dropna(subset=['year'])
X_train = df[['year']].values.astype(float)
y_train = df['rating'].values.astype(float)
y_mean = y_train.mean(); y_train = y_train - y_mean  # centre

# combined kernel: RBF over year + Periodic (decade cycle)
def combined(A, B):
    yearA, yearB = A[:, :1], B[:, :1]
    return rbf(yearA, yearB, length=5.0, var=0.3) + periodic(yearA, yearB, length=3.0, period=10.0, var=0.1)

gp = GaussianProcess(kernel=combined, noise=0.25)
gp.fit(X_train, y_train)

# predict on TMDB catalog (82 movies)
cat_items = [(k, m) for k, m in meta.items() if m.get('year') is not None]
cat_keys = [k for k, _ in cat_items]
cat_years = np.array([[m['year']] for _, m in cat_items], dtype=float)
mu, var = gp.predict(cat_years)
mu = mu + y_mean

np.savez(ART / "gp_posterior.npz", mu=mu, var=var, cat_keys=np.array(cat_keys))
print(f"[GP] lml={gp.log_marginal_likelihood():.3f}, predicted {len(mu)} movies")

# plot: rating surface vs year
fig, ax = plt.subplots(figsize=(9,4))
years_sorted_idx = np.argsort(cat_years.flatten())
ax.plot(cat_years[years_sorted_idx], mu[years_sorted_idx], label='GP mean')
ax.fill_between(cat_years[years_sorted_idx].flatten(),
                mu[years_sorted_idx]-2*np.sqrt(var[years_sorted_idx]),
                mu[years_sorted_idx]+2*np.sqrt(var[years_sorted_idx]), alpha=0.2)
ax.scatter(X_train, y_train + y_mean, s=8, c='red', label='324 ratings')
ax.set_xlabel('Year'); ax.set_ylabel('Predicted rating'); ax.legend()
plt.tight_layout(); plt.savefig(ART / "plots" / "gp_rating_surface.png", dpi=130)
print("[GP] Saved artifacts/gp_posterior.npz + plot")
