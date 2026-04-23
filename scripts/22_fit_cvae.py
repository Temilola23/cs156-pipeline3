"""
Train Conditional VAE on tabular rating-feature data and generate 500 samples per rating bin.

Feature table columns (same as Task 1.8 and 1.11):
  - year_norm, runtime_norm, vote_norm, n_genres_norm (normalized numeric features)
  - modality_0, modality_1, modality_2, modality_3 (one-hot encoded: all, metadata, poster, synopsis)
  - rating (1-5, binned into 5 classes as conditions)
"""
from __future__ import annotations
import sys
from pathlib import Path

# Add parent to path for src imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.data_io import load_324_ratings, load_movie_meta
from src.cvae import CVAE

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
(ART / "plots").mkdir(parents=True, exist_ok=True)

print("[CVAE] Loading data...")
df = load_324_ratings()

# ============================================================================
# Build feature table (same as Task 1.8)
# ============================================================================
# Start with basic numeric features
feature_df = pd.DataFrame(index=df.index)

# 1. Year (numeric, impute median if NaN)
year_median = pd.to_numeric(df['year'], errors='coerce').median()
feature_df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(year_median).values

# 2. Runtime (numeric, impute median if NaN)
runtime_median = df['runtime_min'].median()
feature_df['runtime_min'] = df['runtime_min'].fillna(runtime_median).values

# 3. TMDB vote average (numeric, join from meta, impute median)
meta = load_movie_meta()
tmdb_votes = []
for tid in df['tmdb_id']:
    m = meta.get(int(tid), {})
    vote = m.get('tmdb_vote_average')
    if vote is not None:
        tmdb_votes.append(float(vote))
    else:
        tmdb_votes.append(np.nan)
feature_df['vote_average'] = np.array(tmdb_votes)
vote_median = feature_df['vote_average'].median()
feature_df['vote_average'] = feature_df['vote_average'].fillna(vote_median).values

# 4. Number of genres
feature_df['n_genres'] = df['genres'].apply(lambda g: len(g) if isinstance(g, list) else 0).values

# 5. One-hot encode modality (4 categories: all, metadata, poster, synopsis)
modalities = ['all', 'metadata', 'poster', 'synopsis']
for i, mod in enumerate(modalities):
    feature_df[f'modality_{i}'] = (df['modality'] == mod).astype(int).values

# 6. Rating (target for conditioning)
feature_df['rating'] = df['rating'].values
# Round to nearest integer 1-5 for condition bins
feature_df['rating_bin'] = np.round(df['rating']).astype(int).clip(1, 5)

print(f"[CVAE] Feature table shape: {feature_df.shape}")
print(f"[CVAE] Columns: {feature_df.columns.tolist()}")

# ============================================================================
# Standardize numeric features only (keep one-hot as is)
# ============================================================================
numeric_cols = ['year', 'runtime_min', 'vote_average', 'n_genres']
onehot_cols = [c for c in feature_df.columns if c.startswith('modality_')]

X_numeric = feature_df[numeric_cols].values.astype(np.float32)
X_onehot = feature_df[onehot_cols].values.astype(np.float32)

# Standardize numeric features
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric).astype(np.float32)

# Combine scaled numeric + one-hot
X_scaled = np.concatenate([X_numeric_scaled, X_onehot], axis=1)
print(f"[CVAE] Scaled feature matrix shape: {X_scaled.shape}")

# Extract rating bins for conditioning
rating_bins = feature_df['rating_bin'].values - 1  # Convert to 0-4 for one-hot indexing
print(f"[CVAE] Rating bins distribution:")
for b in range(5):
    count = (rating_bins == b).sum()
    print(f"  Bin {b+1}: {count} samples")

# ============================================================================
# Train CVAE with KL annealing
# ============================================================================
torch.manual_seed(42)
np.random.seed(42)

d = X_scaled.shape[1]  # 8 = 4 numeric + 4 one-hot
n_cond = 5  # 5 rating bins
h = 32
z_dim = 4
batch_size = 32
n_epochs = 1000
lr = 1e-3

device = torch.device('cpu')
cvae = CVAE(d=d, n_cond=n_cond, h=h, z_dim=z_dim).to(device)
optimizer = torch.optim.Adam(cvae.parameters(), lr=lr)

X_tensor = torch.from_numpy(X_scaled).to(device)
rating_bins_tensor = torch.from_numpy(rating_bins).to(device)

print(f"\n[CVAE] Training config: d={d}, n_cond={n_cond}, h={h}, z_dim={z_dim}, epochs={n_epochs}, batch_size={batch_size}")
print(f"[CVAE] Using KL annealing schedule (0 -> 1.0 over 1000 epochs)")

losses = []
recon_losses = []
kl_losses = []
cvae.train()

for epoch in range(n_epochs):
    # KL annealing: ramp from 0 to 1.0 over training
    beta = min(1.0, epoch / (n_epochs * 0.3))  # reach 1.0 at 30% of training

    # Shuffle data
    idx = torch.randperm(len(X_tensor))
    X_shuffled = X_tensor[idx]
    rating_bins_shuffled = rating_bins_tensor[idx]

    epoch_loss = 0.0
    epoch_recon = 0.0
    epoch_kl = 0.0
    n_batches = 0

    for i in range(0, len(X_shuffled), batch_size):
        x_batch = X_shuffled[i:i+batch_size]
        rating_batch = rating_bins_shuffled[i:i+batch_size]

        # Convert rating bins to one-hot
        c_batch = torch.zeros(len(rating_batch), n_cond, device=device)
        c_batch.scatter_(1, rating_batch.unsqueeze(1), 1.0)

        optimizer.zero_grad()
        x_recon, mu, log_var = cvae.forward(x_batch, c_batch)
        loss = cvae.elbo_loss(x_batch, x_recon, mu, log_var, beta=beta)
        loss.backward()
        optimizer.step()

        # Track components for diagnostics
        recon_term = F.mse_loss(x_recon, x_batch, reduction="mean")
        kl_term = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

        epoch_loss += loss.item()
        epoch_recon += recon_term.item()
        epoch_kl += kl_term.item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    avg_recon = epoch_recon / n_batches
    avg_kl = epoch_kl / n_batches
    losses.append(avg_loss)
    recon_losses.append(avg_recon)
    kl_losses.append(avg_kl)

    if (epoch + 1) % 100 == 0:
        print(f"  Epoch {epoch+1}/{n_epochs}: ELBO={avg_loss:.4f}, recon={avg_recon:.4f}, KL={avg_kl:.4f}, beta={beta:.2f}")

print(f"\n[CVAE] Training complete.")
print(f"[CVAE] Initial ELBO: {losses[0]:.4f}")
print(f"[CVAE] Final ELBO: {losses[-1]:.4f}")

# ============================================================================
# Generate synthetic samples: 500 per rating bin
# ============================================================================
cvae.eval()
n_per_bin = 500
n_total = n_per_bin * n_cond

print(f"\n[CVAE] Generating {n_total} synthetic samples ({n_per_bin} per rating bin)...")

all_synthetic = []
all_cond_ratings = []

with torch.no_grad():
    for rating_bin in range(n_cond):
        # Create one-hot condition for this bin
        cond = torch.zeros(n_cond, device=device)
        cond[rating_bin] = 1.0

        # Generate samples
        X_synth_bin = cvae.sample_conditional(n_per_bin, cond)
        X_synth_bin = X_synth_bin.cpu().numpy().astype(np.float32)
        
        all_synthetic.append(X_synth_bin)
        all_cond_ratings.extend([rating_bin + 1] * n_per_bin)  # Store 1-5 for readability

X_synthetic = np.vstack(all_synthetic)
print(f"[CVAE] Synthetic matrix shape: {X_synthetic.shape}")

# ============================================================================
# Inverse-transform (unstandardize) numeric features
# ============================================================================
X_synthetic_numeric = X_synthetic[:, :len(numeric_cols)]
X_synthetic_onehot = X_synthetic[:, len(numeric_cols):]

X_synthetic_numeric_unscaled = scaler.inverse_transform(X_synthetic_numeric).astype(np.float32)

# Clip values to reasonable ranges
X_synthetic_numeric_unscaled[:, 0] = np.clip(X_synthetic_numeric_unscaled[:, 0], 1950, 2030)  # year
X_synthetic_numeric_unscaled[:, 1] = np.clip(X_synthetic_numeric_unscaled[:, 1], 30, 200)  # runtime
X_synthetic_numeric_unscaled[:, 2] = np.clip(X_synthetic_numeric_unscaled[:, 2], 1, 10)  # vote
X_synthetic_numeric_unscaled[:, 3] = np.clip(X_synthetic_numeric_unscaled[:, 3], 0, 15)  # n_genres

# Build synthetic dataframe
synth_df = pd.DataFrame(X_synthetic_numeric_unscaled, columns=numeric_cols)
for i, col in enumerate(onehot_cols):
    synth_df[col] = X_synthetic_onehot[:, i]

synth_df['cond_rating'] = all_cond_ratings

print(f"\n[CVAE] Synthetic data stats (numeric columns only):")
print(synth_df[numeric_cols].describe())

# ============================================================================
# Save synthetic data
# ============================================================================
synth_path = ART / "cvae_conditional_synth.parquet"
synth_df.to_parquet(synth_path)
print(f"\n[CVAE] Saved {len(synth_df)} synthetic rows to {synth_path}")

# ============================================================================
# Sanity check: mean vote_average by cond_rating
# ============================================================================
print(f"\n[CVAE] Mean vote_average by condition rating:")
for rating_bin in range(1, 6):
    mask = synth_df['cond_rating'] == rating_bin
    mean_vote = synth_df.loc[mask, 'vote_average'].mean()
    print(f"  Cond rating {rating_bin}: mean vote_average = {mean_vote:.3f}")

# ============================================================================
# PCA visualization: real vs synthetic by rating
# ============================================================================
print("\n[CVAE] Creating PCA visualization...")

# Use only numeric features for PCA
X_real_numeric = X_numeric_scaled  # Real data (standardized)
X_synth_numeric = X_synthetic[:, :len(numeric_cols)]  # Synthetic data (before inverse transform)

# Fit PCA on combined real + synthetic data
pca = PCA(n_components=2, random_state=42)
X_combined = np.vstack([X_real_numeric, X_synth_numeric])
X_pca_combined = pca.fit_transform(X_combined)

X_pca_real = X_pca_combined[:len(X_real_numeric)]
X_pca_synth = X_pca_combined[len(X_real_numeric):]

# Create color maps for ratings
real_ratings = feature_df['rating_bin'].values
synth_ratings = np.array(all_cond_ratings)

# Print PCA variance and overlap stats
print(f"[CVAE] PCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"[CVAE] Real data in PCA space:")
print(f"  - mean: {X_pca_real.mean(axis=0)}")
print(f"  - std:  {X_pca_real.std(axis=0)}")
print(f"[CVAE] Synthetic data in PCA space:")
print(f"  - mean: {X_pca_synth.mean(axis=0)}")
print(f"  - std:  {X_pca_synth.std(axis=0)}")

# Plot
fig, ax = plt.subplots(figsize=(12, 9))

# Plot real data colored by rating bin
cmap = plt.cm.get_cmap('RdYlGn', 5)
for rating in range(1, 6):
    mask = real_ratings == rating
    ax.scatter(X_pca_real[mask, 0], X_pca_real[mask, 1], 
               c=[cmap(rating-1)], alpha=0.7, s=60, label=f'Real rating {rating}', edgecolors='black', linewidth=0.5)

# Plot synthetic data colored by condition rating (more transparent, smaller)
for rating in range(1, 6):
    mask = synth_ratings == rating
    ax.scatter(X_pca_synth[mask, 0], X_pca_synth[mask, 1], 
               c=[cmap(rating-1)], alpha=0.3, s=15, label=f'Synthetic cond={rating}', marker='x')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax.set_title('CVAE: synthetic vs real rating-conditioned feature distribution')
ax.legend(loc='best', fontsize=10, ncol=2)
ax.grid(alpha=0.3)

plot_path = ART / "plots" / "cvae_conditional.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=130)
print(f"\n[CVAE] Saved plot to {plot_path}")

print("\n[CVAE] Task 1.15 complete!")
