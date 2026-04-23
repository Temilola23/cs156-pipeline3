#!/usr/bin/env python3
"""
Task 1.12: Fit Bayesian hierarchical one-way ANOVA (PyMC) on Temilola's 324 ratings.

Model:
  μ ~ Normal(3.5, 1)            # Grand mean (rating scale 1-5)
  σ_modality ~ HalfNormal(1)     # Between-modality sd (partial pooling)
  a_m ~ Normal(0, σ_modality)    # Modality offset for m ∈ {all, metadata, poster, synopsis}
  σ_movie ~ HalfNormal(1)        # Between-movie sd
  b_i ~ Normal(0, σ_movie)       # Movie-level random effect
  σ_obs ~ HalfNormal(1)          # Observation noise
  y_ij ~ Normal(μ + a_m[j] + b_i[j], σ_obs)

Outputs:
  - artifacts/anova_trace.nc: NetCDF trace
  - artifacts/anova_summary.csv: Summary statistics
  - artifacts/plots/anova_forest.png: Forest plot of modality offsets
  - artifacts/plots/anova_variance_decomp.png: Variance decomposition bar chart
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_io import load_324_ratings
from src.hierarchical_anova import fit_hierarchical_anova

# Paths
ART = ROOT / "artifacts"
PLOTS = ART / "plots"
(PLOTS).mkdir(parents=True, exist_ok=True)

print("[ANOVA-19] Loading 324 ratings...")
df = load_324_ratings()

print(f"[ANOVA-19] Data shape: {df.shape}")
print(f"[ANOVA-19] Modalities: {df['modality'].value_counts().to_dict()}")
print(f"[ANOVA-19] Unique movies: {df['tmdb_id'].nunique()}")

# Fit model
print("[ANOVA-19] Fitting hierarchical ANOVA...")
idata = fit_hierarchical_anova(
    df,
    n_tune=1000,
    n_draw=1000,
    n_chains=2,
    seed=42,
    target_accept=0.95,
)

print("[ANOVA-19] Sampling complete!")

# Save trace
trace_path = ART / "anova_trace.nc"
print(f"[ANOVA-19] Saving trace to {trace_path}...")
az.to_netcdf(idata, trace_path)

# Summary statistics
summary_df = az.summary(idata, var_names=['mu', 'a_m', 'sigma_modality', 'sigma_movie', 'sigma_obs'])
summary_csv = ART / "anova_summary.csv"
print(f"[ANOVA-19] Saving summary to {summary_csv}...")
summary_df.to_csv(summary_csv)

print("\n" + "="*70)
print("POSTERIOR SUMMARY")
print("="*70)
print(summary_df)

# ============================================================================
# Extract key posterior statistics
# ============================================================================
post = idata.posterior

mu_samples = post['mu'].values.flatten()
a_m_samples = post['a_m'].values  # (n_chain, n_draw, 4)
sigma_modality_samples = post['sigma_modality'].values.flatten()
sigma_movie_samples = post['sigma_movie'].values.flatten()
sigma_obs_samples = post['sigma_obs'].values.flatten()

# Compute HDI
def hdi_94(samples):
    """Compute 94% highest density interval."""
    samples_2d = np.atleast_1d(samples)
    if samples_2d.ndim > 1:
        samples_2d = samples_2d.flatten()
    hdi_result = az.hdi(samples_2d, hdi_prob=0.94)
    return f"[{hdi_result[0]:.4f}, {hdi_result[1]:.4f}]"

print("\n" + "="*70)
print("POSTERIOR MEANS & 94% HDI")
print("="*70)
print(f"μ (grand mean):           {np.mean(mu_samples):.4f}  [94% HDI: {hdi_94(mu_samples)}]")
print(f"σ_modality:               {np.mean(sigma_modality_samples):.4f}  [94% HDI: {hdi_94(sigma_modality_samples)}]")
print(f"σ_movie:                  {np.mean(sigma_movie_samples):.4f}  [94% HDI: {hdi_94(sigma_movie_samples)}]")
print(f"σ_obs:                    {np.mean(sigma_obs_samples):.4f}  [94% HDI: {hdi_94(sigma_obs_samples)}]")

modality_names = ['all', 'metadata', 'poster', 'synopsis']
print("\nModality offsets a_m:")
for i, name in enumerate(modality_names):
    samples_m = a_m_samples[:, :, i].flatten()
    print(f"  {name:10s}:  {np.mean(samples_m):6.4f}  [94% HDI: {hdi_94(samples_m)}]")

# ============================================================================
# Diagnostics
# ============================================================================
print("\n" + "="*70)
print("DIAGNOSTICS")
print("="*70)
r_hat = az.rhat(idata, var_names=['mu', 'a_m', 'sigma_modality', 'sigma_movie', 'sigma_obs'])
print("R̂ (Gelman-Rubin, should be < 1.01):")
print(r_hat)

# Count divergences
n_divergences = idata.sample_stats['diverging'].sum().values
print(f"\nTotal divergences: {n_divergences} (0 is ideal, <10 is acceptable)")

# ============================================================================
# Plot 1: Forest plot of modality offsets
# ============================================================================
print("\n[ANOVA-19] Creating forest plot...")
fig, ax = plt.subplots(figsize=(10, 6))

# Extract posterior for a_m
a_m_var = idata.posterior['a_m']
az.plot_forest(
    [a_m_var],
    var_names=['a_m'],
    combined=True,
    hdi_prob=0.94,
    ax=ax,
)
ax.set_xticklabels(modality_names)
ax.set_ylabel('Modality')
ax.set_xlabel('Posterior offset a_m (relative to grand mean)')
ax.set_title('Bayesian Hierarchical ANOVA: Modality Effects')
fig.tight_layout()
forest_path = PLOTS / "anova_forest.png"
fig.savefig(forest_path, dpi=150, bbox_inches='tight')
print(f"[ANOVA-19] Saved forest plot to {forest_path}")
plt.close(fig)

# ============================================================================
# Plot 2: Variance decomposition
# ============================================================================
print("[ANOVA-19] Creating variance decomposition plot...")
fig, ax = plt.subplots(figsize=(8, 6))

variance_names = ['σ_modality\n(between-modality)', 'σ_movie\n(between-movie)', 'σ_obs\n(observation)']
variance_means = [
    np.mean(sigma_modality_samples),
    np.mean(sigma_movie_samples),
    np.mean(sigma_obs_samples),
]
variance_std = [
    np.std(sigma_modality_samples),
    np.std(sigma_movie_samples),
    np.std(sigma_obs_samples),
]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = ax.bar(variance_names, variance_means, yerr=variance_std, capsize=5, color=colors, alpha=0.7, edgecolor='black')

ax.set_ylabel('Posterior mean (with std error)')
ax.set_title('Variance Decomposition: Which factor dominates?')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, mean in zip(bars, variance_means):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

fig.tight_layout()
variance_path = PLOTS / "anova_variance_decomp.png"
fig.savefig(variance_path, dpi=150, bbox_inches='tight')
print(f"[ANOVA-19] Saved variance decomposition to {variance_path}")
plt.close(fig)

print("\n" + "="*70)
print("Task 1.12 complete!")
print("="*70)
print(f"Trace:               {trace_path}")
print(f"Summary CSV:         {summary_csv}")
print(f"Forest plot:         {forest_path}")
print(f"Variance plot:       {variance_path}")
