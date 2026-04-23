"""Fit 3-regime HMM to chronologically-sorted binned ratings."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.data_io import load_324_ratings
from src.hmm import HMM

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
(ART / "plots").mkdir(parents=True, exist_ok=True)

df = load_324_ratings().sort_values('timestamp').reset_index(drop=True)
# bin ratings to 5 levels: 0..4 (ratings are 1-5)
obs = np.clip(df['rating'].astype(int).values - 1, 0, 4)

hmm = HMM(n_states=3, n_obs_bins=5, seed=0)
history = hmm.baum_welch(obs, n_iter=40)
print(f"[HMM] Baum-Welch converged after {len(history)} iterations")
print(f"[HMM] log-likelihood trajectory: {history[0]:.2f} -> {history[-1]:.2f}")

gamma = hmm.posterior_states(obs)   # (T, 3)
path = hmm.viterbi(obs)

np.savez(ART / "hmm_regimes.npz",
         gamma=gamma, viterbi=path, A=hmm.A, E=hmm.E, pi=hmm.pi,
         ll_history=np.array(history), obs=obs)

# Stacked-area plot of posterior state occupancy over time
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].stackplot(range(len(obs)), gamma.T,
                  labels=['regime 0', 'regime 1', 'regime 2'], alpha=0.75)
axes[0].legend(loc='upper right')
axes[0].set_ylabel('posterior occupancy')
axes[0].set_title('HMM 3-regime posterior state occupancy (chronological)')

axes[1].plot(path, drawstyle='steps-post', lw=1.0)
axes[1].set_yticks([0, 1, 2])
axes[1].set_ylabel('Viterbi state')
axes[1].set_xlabel('rating index (chrono)')
axes[1].set_title('Viterbi most-likely regime path')
plt.tight_layout()
plt.savefig(ART / "plots" / "hmm_regimes.png", dpi=130)
print("[HMM] saved artifacts/hmm_regimes.npz + plot")
