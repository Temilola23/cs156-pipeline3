#!/usr/bin/env python3
"""
Task 1.13: IPW + AIPW causal effect of modality (synopsis vs metadata) on rating.

Binary contrast: T=1 if synopsis modality, T=0 if metadata modality.
Drop 'all' and 'poster' conditions.
Outcome: Temilola's rating.
Covariates: year, runtime, vote_average, n_genres (standardized).

Outputs:
  - Propensity score overlap histogram
  - JSON summary with ATE estimates + bootstrap CIs
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setup path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_io import load_324_ratings, load_movie_meta
from src.causal import ipw_ate, aipw_ate, bootstrap_ate, predict_proba, fit_logistic


def standardize(x: np.ndarray) -> np.ndarray:
    """Z-score standardization."""
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def main():
    # Load data
    print("[*] Loading 324 ratings...")
    df = load_324_ratings()

    # Filter to binary contrast: synopsis vs metadata
    df_binary = df[df['modality'].isin(['synopsis', 'metadata'])].copy()
    print(f"[*] Filtered to synopsis/metadata: {len(df_binary)} observations")

    # Create treatment: T=1 if synopsis, T=0 if metadata
    df_binary['T'] = (df_binary['modality'] == 'synopsis').astype(int)
    Y = df_binary['rating'].values
    T = df_binary['T'].values

    # Extract covariates
    # Year: convert to numeric, handle missing by imputing median
    year = pd.to_numeric(df_binary['year'], errors='coerce')
    year = year.fillna(year.median()).values

    # Runtime: handle missing by imputing median
    runtime = df_binary['runtime_min'].fillna(df_binary['runtime_min'].median()).values.astype(float)

    # Vote average: handle missing by imputing median
    vote_avg = df_binary['vote_average'].fillna(df_binary['vote_average'].median()).values.astype(float)

    # Number of genres
    n_genres = np.array([len(g) if isinstance(g, list) else 0 for g in df_binary['genres']], dtype=float)

    # Standardize covariates
    year_norm = standardize(year)
    runtime_norm = standardize(runtime)
    vote_norm = standardize(vote_avg)
    n_genres_norm = standardize(n_genres)

    X = np.column_stack([year_norm, runtime_norm, vote_norm, n_genres_norm])

    print(f"[*] Covariates shape: {X.shape}")
    print(f"[*] Y shape: {Y.shape}, T shape: {T.shape}")

    # Summary statistics
    n_treated = (T == 1).sum()
    n_control = (T == 0).sum()
    print(f"[*] Treated (synopsis): {n_treated}, Control (metadata): {n_control}")

    # Naive difference in means
    naive_diff = Y[T == 1].mean() - Y[T == 0].mean()
    print(f"[*] Naive diff in means: {naive_diff:.4f}")

    # Fit propensity score for overlap plot
    print("[*] Fitting propensity score...")
    w_prop, b_prop = fit_logistic(X, T, lr=0.01, n_epochs=500)
    e = predict_proba(X, w_prop, b_prop)

    # IPW estimation
    print("[*] Computing IPW ATE...")
    ipw_mean, ipw_std, ipw_ci_low, ipw_ci_high = bootstrap_ate(
        X, T, Y, ipw_ate, n_boot=500, seed=42
    )
    print(f"    IPW ATE: {ipw_mean:.4f} ± {ipw_std:.4f}")
    print(f"    95% CI: [{ipw_ci_low:.4f}, {ipw_ci_high:.4f}]")

    # AIPW estimation
    print("[*] Computing AIPW ATE...")
    aipw_mean, aipw_std, aipw_ci_low, aipw_ci_high = bootstrap_ate(
        X, T, Y, aipw_ate, n_boot=500, seed=42
    )
    print(f"    AIPW ATE: {aipw_mean:.4f} ± {aipw_std:.4f}")
    print(f"    95% CI: [{aipw_ci_low:.4f}, {aipw_ci_high:.4f}]")

    # Propensity score overlap plot
    print("[*] Creating propensity overlap plot...")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Separate by treatment
    e_treated = e[T == 1]
    e_control = e[T == 0]

    # Count distribution
    unique_vals = np.unique(np.round(e, 4))
    
    if len(unique_vals) <= 5:
        # Use bar chart for discrete propensity scores
        treated_counts = pd.Series(e_treated).value_counts().sort_index()
        control_counts = pd.Series(e_control).value_counts().sort_index()
        
        x_pos = np.arange(len(unique_vals))
        width = 0.35
        
        treated_bar = [treated_counts.get(v, 0) for v in unique_vals]
        control_bar = [control_counts.get(v, 0) for v in unique_vals]
        
        ax.bar(x_pos - width/2, control_bar, width, label=f"Control (n={n_control})", color='blue', alpha=0.7, edgecolor='black')
        ax.bar(x_pos + width/2, treated_bar, width, label=f"Treated (n={n_treated})", color='red', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Propensity Score P(T=1|X)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{v:.4f}' for v in unique_vals], rotation=45, ha='right')
    else:
        # Use histogram for continuous propensity scores
        ax.hist(e_control, bins=15, alpha=0.6, label=f"Control (n={n_control})", color='blue', edgecolor='black')
        ax.hist(e_treated, bins=15, alpha=0.6, label=f"Treated (n={n_treated})", color='red', edgecolor='black')
        ax.set_xlabel('Propensity Score P(T=1|X)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)

    ax.set_title('Propensity Score Overlap (IPW Weighting Quality)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')

    # Save
    plot_path = ROOT / "artifacts" / "plots" / "propensity_overlap.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"    Saved to {plot_path}")
    plt.close(fig)

    # Check for overlap issues
    prop_min = e.min()
    prop_max = e.max()
    print(f"[*] Propensity score range: [{prop_min:.6f}, {prop_max:.6f}]")
    
    # Assess overlap quality
    if prop_max - prop_min < 0.01:
        print("    INFO: Excellent overlap (narrow range). Treatment appears well-balanced by covariates.")
    elif prop_min < 0.05 or prop_max > 0.95:
        print("    WARNING: Poor overlap detected (scores near 0 or 1). IPW variance may be inflated.")

    # Save results to JSON
    results = {
        "n": len(df_binary),
        "n_treated": int(n_treated),
        "n_control": int(n_control),
        "naive_diff": float(naive_diff),
        "ipw_ate": float(ipw_mean),
        "ipw_se": float(ipw_std),
        "ipw_ci": [float(ipw_ci_low), float(ipw_ci_high)],
        "aipw_ate": float(aipw_mean),
        "aipw_se": float(aipw_std),
        "aipw_ci": [float(aipw_ci_low), float(aipw_ci_high)],
    }

    json_path = ROOT / "artifacts" / "causal_ipw_aipw.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[*] Results saved to {json_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Causal Effect of Modality (Synopsis vs Metadata)")
    print("=" * 70)
    print(f"Sample size: {len(df_binary)}")
    print(f"  - Treated (synopsis):  {n_treated}")
    print(f"  - Control (metadata):  {n_control}")
    print(f"\nNaive difference in means (no covariate adjustment): {naive_diff:7.4f}")
    print(f"\nIPW ATE (Inverse Probability Weighting):")
    print(f"  Estimate: {ipw_mean:7.4f}")
    print(f"  Std Err:  {ipw_std:7.4f}")
    print(f"  95% CI:   [{ipw_ci_low:7.4f}, {ipw_ci_high:7.4f}]")
    print(f"\nAIPW ATE (Augmented IPW, Doubly Robust):")
    print(f"  Estimate: {aipw_mean:7.4f}")
    print(f"  Std Err:  {aipw_std:7.4f}")
    print(f"  95% CI:   [{aipw_ci_low:7.4f}, {aipw_ci_high:7.4f}]")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
