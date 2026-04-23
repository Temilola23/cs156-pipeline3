"""Bootstrap particle filter (N=500) + FFBSi smoother on 324 chronological ratings."""
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from src.data_io import load_324_ratings
from src.kalman import BootstrapParticleFilter, FFBSi

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
(ART / "plots").mkdir(parents=True, exist_ok=True)

df = load_324_ratings().sort_values('timestamp')
y = df['rating'].values.astype(float)
y_c = y - y.mean()

pf = BootstrapParticleFilter(f=lambda x: x, h=lambda x: x, Q=0.02, R=0.4, N=500, seed=0)
pf.init(0.0, 1.0)
means = []
for yi in y_c:
    pf.step(yi)
    means.append(pf.mean())

# FFBSi: draw 200 smoothed trajectories
ffbsi = FFBSi(f=lambda x: x, Q=0.02, seed=0)
traj = ffbsi.sample(pf.history, pf.weights, M=200)

np.savez(ART / "pf_samples.npz",
         particles_final=pf.particles,
         weights_final=pf.weights,
         filtered_mean=np.array(means),
         smoothed_trajectories=traj,
         ess=np.array(pf.ess_history),
         y=y_c)

# Plot particle cloud evolution + smoothed mean
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
particles_over_time = np.array(pf.history[1:])  # drop initial prior
T = particles_over_time.shape[0]
for t in range(0, T, max(1, T // 60)):
    axes[0].scatter([t] * particles_over_time.shape[1], particles_over_time[t],
                    s=2, alpha=0.15, c='steelblue')
axes[0].plot(means, c='red', lw=1.5, label='PF filtered mean')
axes[0].scatter(range(len(y_c)), y_c, s=4, c='black', alpha=0.4, label='observations')
axes[0].legend(); axes[0].set_ylabel('latent taste')
axes[0].set_title('Particle cloud evolution (N=500)')

axes[1].plot(traj.T, color='green', alpha=0.05)
axes[1].plot(traj.mean(0), c='darkgreen', lw=1.5, label='FFBSi posterior mean')
axes[1].scatter(range(len(y_c)), y_c, s=4, c='black', alpha=0.4)
axes[1].legend(); axes[1].set_xlabel('rating index (chronological)')
axes[1].set_ylabel('latent taste')
axes[1].set_title('FFBSi smoothed trajectories (M=200)')
plt.tight_layout()
plt.savefig(ART / "plots" / "pf_cloud.png", dpi=130)

print(f"[PF] final ESS={pf.ess_history[-1]:.1f}/{pf.N}, filtered mean last={means[-1]:.3f}")
print(f"[PF] saved artifacts/pf_samples.npz + pf_cloud.png")
