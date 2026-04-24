#!/usr/bin/env python3
"""
Rebuild anova_forest.png from the existing anova_trace.nc without re-running MCMC.

Fixes the axis-label bug where modality names (`all`, `metadata`, `poster`,
`synopsis`) were placed on the x-axis by `ax.set_xticklabels(...)` after
`az.plot_forest`, leaving generic indices `a_m[0..3]` on the y-axis.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
PLOTS = ART / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

idata = az.from_netcdf(ART / "anova_trace.nc")
a_m_samples = idata.posterior['a_m'].values  # (n_chain, n_draw, 4)
modality_names = ['all', 'metadata', 'poster', 'synopsis']

means, hdi_lows, hdi_highs = [], [], []
for i in range(len(modality_names)):
    samples_m = a_m_samples[:, :, i].flatten()
    means.append(float(np.mean(samples_m)))
    hdi = az.hdi(samples_m, hdi_prob=0.94)
    hdi_lows.append(float(hdi[0]))
    hdi_highs.append(float(hdi[1]))
means = np.array(means)
hdi_lows = np.array(hdi_lows)
hdi_highs = np.array(hdi_highs)

fig, ax = plt.subplots(figsize=(10, 6))
y_positions = np.arange(len(modality_names))[::-1]  # 'all' on top

ax.axvline(0.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1)

for y, m, lo, hi in zip(y_positions, means, hdi_lows, hdi_highs):
    ax.plot([lo, hi], [y, y], color='#1f4e79', linewidth=2.2, zorder=2)
    ax.scatter([lo, hi], [y, y], marker='|', s=120, color='#1f4e79', zorder=3)
    ax.scatter([m], [y], marker='o', s=90, color='#c0392b',
               edgecolor='black', linewidth=0.7, zorder=4)

for y, m in zip(y_positions, means):
    ax.text(m, y + 0.18, f'{m:+.3f}', ha='center', va='bottom',
            fontsize=10, color='#222', zorder=5)

ax.set_yticks(y_positions)
ax.set_yticklabels(modality_names, fontsize=12)
ax.set_ylabel('Modality')
ax.set_xlabel(r'Posterior offset $a_m$ relative to grand mean $\mu$')
ax.set_title('Hierarchical ANOVA: modality offsets (dot = posterior mean, bar = 94% HDI)')
ax.grid(axis='x', alpha=0.3)
ax.set_ylim(y_positions.min() - 0.6, y_positions.max() + 0.6)

fig.tight_layout()
forest_path = PLOTS / "anova_forest.png"
fig.savefig(forest_path, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"Saved {forest_path}")
print("Modality posterior means / 94% HDIs:")
for name, m, lo, hi in zip(modality_names, means, hdi_lows, hdi_highs):
    print(f"  {name:10s}  mean={m:+.4f}  HDI=[{lo:+.4f}, {hi:+.4f}]")
