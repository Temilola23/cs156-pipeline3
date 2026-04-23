"""Kalman ladder on chronologically-sorted 324 ratings."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, matplotlib.pyplot as plt
from src.data_io import load_324_ratings
from src.kalman import KalmanFilter, RTSSmoother, UKF

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
(ART / "plots").mkdir(parents=True, exist_ok=True)

df = load_324_ratings().sort_values('timestamp')
y = df['rating'].values.astype(float)
y_c = y - y.mean()

F = np.array([[1.0]]); H = np.array([[1.0]]); Q = np.array([[0.02]]); R = np.array([[0.4]])
kf = KalmanFilter(F, H, Q, R); kf.init(np.zeros(1), np.eye(1))
kalman_xs, kalman_Ps = [], []
for yi in y_c:
    kf.predict(); kf.update(np.array([yi]))
    kalman_xs.append(kf.x.copy()); kalman_Ps.append(kf.P.copy())

rts_xs, rts_Ps = RTSSmoother().smooth(kalman_xs, kalman_Ps, F)

ukf = UKF(f=lambda x: x, h=lambda x: np.tanh(x), Q=Q, R=R, n=1)
ukf.init(np.zeros(1), np.eye(1)); ukf_xs = []
for yi in y_c:
    ukf.predict(); ukf.update(np.array([np.tanh(yi)]))
    ukf_xs.append(ukf.x.copy())

np.savez(ART / "ukf_latent.npz",
         kalman=np.array(kalman_xs).squeeze(),
         rts=np.array(rts_xs).squeeze(),
         ukf=np.array(ukf_xs).squeeze(),
         y=y_c)

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
for ax, (name, x) in zip(axes.flat, [('Kalman', kalman_xs), ('RTS', rts_xs), ('UKF', ukf_xs)]):
    ax.plot(np.array(x).squeeze(), label=name)
    ax.scatter(range(len(y_c)), y_c, s=4, alpha=0.3)
    ax.legend()
axes[1, 1].axis('off')
plt.tight_layout()
plt.savefig(ART / "plots" / "kalman_ladder.png", dpi=130)
print(f"[Kalman] ladder fit complete ({len(y_c)} observations)")
