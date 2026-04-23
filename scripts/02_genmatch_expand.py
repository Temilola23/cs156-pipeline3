"""GenMatch expansion: pick 500 best-balancing synthetic candidates from a
5000-candidate MVN pool over the 82 real movies' feature space.

Adaptation: the 50K TMDB catalog isn't loaded yet, so candidates are labelled
synthetic_cand_NNNN. Replace with real TMDB ids once the catalog is pulled.
"""
import json
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
from src.data_io import load_movie_meta
from src.genmatch import GenMatch

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
(ART / "plots").mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
# 1. Build real-movie feature matrix (82 rows)
# ------------------------------------------------------------------ #
meta = load_movie_meta()
real_items = [(k, v) for k, v in meta.items() if v.get('year') is not None]
real_keys = [k for k, _ in real_items]
# simple numeric features; add more later if needed
def feat_row(m):
    year = float(m.get('year', 2000))
    runtime = float(m.get('runtime_min') or 100.0)
    vote = float(m.get('tmdb_vote_average') or 6.0)
    n_genres = float(len(m.get('genres') or []))
    return [year, runtime, vote, n_genres]

real_X = np.array([feat_row(m) for _, m in real_items], dtype=float)
d = real_X.shape[1]

# ------------------------------------------------------------------ #
# 2. Generate 5000 synthetic candidates via MVN over real-feature stats
# ------------------------------------------------------------------ #
rng = np.random.default_rng(0)
mu = real_X.mean(0); Sigma = np.cov(real_X, rowvar=False) + 1e-6 * np.eye(d)
cand_X = rng.multivariate_normal(mu, Sigma, size=5000)

# ------------------------------------------------------------------ #
# 3. Treat: 1 = real (treated), 0 = synthetic (control pool)
# ------------------------------------------------------------------ #
X = np.vstack([real_X, cand_X])
treat = np.concatenate([np.ones(len(real_X), dtype=int),
                        np.zeros(len(cand_X), dtype=int)])

# ------------------------------------------------------------------ #
# 4. Fit GenMatch (smaller GA, still meaningful)
# ------------------------------------------------------------------ #
gm = GenMatch(pop_size=30, n_generations=25, seed=0)
history = gm.fit(X, treat, return_history=True)
print(f"[GenMatch] best fitness (min covariate p-value): {gm.best_fitness_:.4f}")
print(f"[GenMatch] best weights: {gm.best_weights_}")

# ------------------------------------------------------------------ #
# 5. For each real movie, pick top-k=6 nearest synthetic candidates
#    -> dict {real_key -> [synth_id, ...]}  (6 * 82 ≈ 492 <= 500 target)
# ------------------------------------------------------------------ #
from src.genmatch import _weighted_sigma_inv, mahalanobis_distance_matrix
Sigma_inv_star = _weighted_sigma_inv(X, gm.best_weights_)
D = mahalanobis_distance_matrix(real_X, cand_X, Sigma_inv_star)
k = 6
order = np.argsort(D, axis=1)[:, :k]

neighbors_map = {
    str(real_keys[i]): [f"synthetic_cand_{order[i, j]:04d}" for j in range(k)]
    for i in range(len(real_keys))
}

out_path = ART / "genmatch_neighbors.json"
out_path.write_text(json.dumps(neighbors_map, indent=2))
print(f"[GenMatch] wrote {out_path} ({len(neighbors_map)} real movies × {k} neighbors)")

# ------------------------------------------------------------------ #
# 6. Balance report + plot
# ------------------------------------------------------------------ #
br = gm.balance_report(X, treat)
print(f"[GenMatch] per-covariate p before: {br['before_p']}")
print(f"[GenMatch] per-covariate p after:  {br['after_p']}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history, marker='o', lw=1)
axes[0].set_xlabel('generation'); axes[0].set_ylabel('best fitness (min KS p-val)')
axes[0].set_title('GA fitness trajectory')

ind = np.arange(d)
w = 0.35
axes[1].bar(ind - w/2, br['before_p'], w, label='before matching')
axes[1].bar(ind + w/2, br['after_p'], w, label='after matching')
axes[1].set_xticks(ind); axes[1].set_xticklabels(['year', 'runtime', 'vote', 'n_genres'])
axes[1].set_ylabel('KS p-value (higher = better balance)')
axes[1].set_title('Covariate balance before/after')
axes[1].legend()
plt.tight_layout()
plt.savefig(ART / "plots" / "genmatch_fitness.png", dpi=130)
print("[GenMatch] saved artifacts/genmatch_neighbors.json + genmatch_fitness.png")
