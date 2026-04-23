"""
Train TVAE on tabular rating-feature data and generate 5000 synthetic samples.

Feature table columns:
  - rating (1-5)
  - year (numeric)
  - runtime_min (numeric)
  - tmdb_vote_average (numeric)
  - n_genres (count)
  - modality_0, modality_1, modality_2, modality_3 (one-hot encoded: all, metadata, poster, synopsis)
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.data_io import load_324_ratings, load_movie_meta
from src.tvae import TVAE

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
(ART / "plots").mkdir(parents=True, exist_ok=True)

print("[TVAE] Loading data...")
df = load_324_ratings()

# ============================================================================
# Build feature table
# ============================================================================
# Start with basic numeric features
feature_df = pd.DataFrame(index=df.index)

# 1. Rating (1-5, already continuous)
feature_df['rating'] = df['rating'].values

# 2. Year (numeric, impute median if NaN)
year_median = pd.to_numeric(df['year'], errors='coerce').median()
feature_df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(year_median).values

# 3. Runtime (numeric, impute median if NaN)
runtime_median = df['runtime_min'].median()
feature_df['runtime_min'] = df['runtime_min'].fillna(runtime_median).values

# 4. TMDB vote average (numeric, join from meta, impute median)
meta = load_movie_meta()
tmdb_votes = []
for tid in df['tmdb_id']:
    m = meta.get(int(tid), {})
    vote = m.get('tmdb_vote_average')
    if vote is not None:
        tmdb_votes.append(float(vote))
    else:
        tmdb_votes.append(np.nan)
feature_df['tmdb_vote_average'] = np.array(tmdb_votes)
vote_median = feature_df['tmdb_vote_average'].median()
feature_df['tmdb_vote_average'] = feature_df['tmdb_vote_average'].fillna(vote_median).values

# 5. Number of genres
feature_df['n_genres'] = df['genres'].apply(lambda g: len(g) if isinstance(g, list) else 0).values

# 6. One-hot encode modality (4 categories: all, metadata, poster, synopsis)
modalities = ['all', 'metadata', 'poster', 'synopsis']
for i, mod in enumerate(modalities):
    feature_df[f'modality_{i}'] = (df['modality'] == mod).astype(int).values

print(f"[TVAE] Feature table shape: {feature_df.shape}")
print(f"[TVAE] Columns: {feature_df.columns.tolist()}")

# ============================================================================
# Standardize numeric features only (keep one-hot as is)
# ============================================================================
numeric_cols = ['rating', 'year', 'runtime_min', 'tmdb_vote_average', 'n_genres']
onehot_cols = [c for c in feature_df.columns if c.startswith('modality_')]

X_numeric = feature_df[numeric_cols].values.astype(np.float32)
X_onehot = feature_df[onehot_cols].values.astype(np.float32)

# Standardize numeric features
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric).astype(np.float32)

# Combine scaled numeric + one-hot
X_scaled = np.concatenate([X_numeric_scaled, X_onehot], axis=1)
print(f"[TVAE] Scaled feature matrix shape: {X_scaled.shape}")

# ============================================================================
# Train TVAE with KL annealing
# ============================================================================
torch.manual_seed(42)
np.random.seed(42)

d = X_scaled.shape[1]
h = 32
z_dim = 4
batch_size = 32
n_epochs = 1500
lr = 1e-3

device = torch.device('cpu')
tvae = TVAE(d=d, h=h, z_dim=z_dim).to(device)
optimizer = torch.optim.Adam(tvae.parameters(), lr=lr)

X_tensor = torch.from_numpy(X_scaled).to(device)

print(f"\n[TVAE] Training config: d={d}, h={h}, z_dim={z_dim}, epochs={n_epochs}, batch_size={batch_size}")
print(f"[TVAE] Using KL annealing schedule (0 -> 1.0 over 1500 epochs)")

losses = []
recon_losses = []
kl_losses = []
tvae.train()

for epoch in range(n_epochs):
    # KL annealing: ramp from 0 to 1.0 over training
    beta = min(1.0, epoch / (n_epochs * 0.3))  # reach 1.0 at 30% of training
    
    # Shuffle data
    idx = torch.randperm(len(X_tensor))
    X_shuffled = X_tensor[idx]

    epoch_loss = 0.0
    epoch_recon = 0.0
    epoch_kl = 0.0
    n_batches = 0

    for i in range(0, len(X_shuffled), batch_size):
        x_batch = X_shuffled[i:i+batch_size]

        optimizer.zero_grad()
        x_recon, mu, log_var = tvae.forward(x_batch)
        loss = tvae.elbo_loss(x_batch, x_recon, mu, log_var, beta=beta)
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

    if (epoch + 1) % 200 == 0:
        print(f"  Epoch {epoch+1}/{n_epochs}: ELBO={avg_loss:.4f}, recon={avg_recon:.4f}, KL={avg_kl:.4f}, beta={beta:.2f}")

print(f"\n[TVAE] Training complete.")
print(f"[TVAE] Initial loss: {losses[0]:.4f}")
print(f"[TVAE] Final loss: {losses[-1]:.4f}")

# ============================================================================
# Generate synthetic samples
# ============================================================================
tvae.eval()
n_synthetic = 5000

print(f"\n[TVAE] Generating {n_synthetic} synthetic samples...")
with torch.no_grad():
    X_synthetic = tvae.sample(n_synthetic)
X_synthetic = X_synthetic.cpu().numpy().astype(np.float32)

print(f"[TVAE] Synthetic matrix shape: {X_synthetic.shape}")

# ============================================================================
# Inverse-transform (unstandardize) numeric features
# ============================================================================
X_synthetic_numeric = X_synthetic[:, :len(numeric_cols)]
X_synthetic_onehot = X_synthetic[:, len(numeric_cols):]

X_synthetic_numeric_unscaled = scaler.inverse_transform(X_synthetic_numeric).astype(np.float32)

# Clip values to reasonable ranges
X_synthetic_numeric_unscaled[:, 0] = np.clip(X_synthetic_numeric_unscaled[:, 0], 1, 10)  # rating: 1-10
X_synthetic_numeric_unscaled[:, 1] = np.clip(X_synthetic_numeric_unscaled[:, 1], 1950, 2030)  # year
X_synthetic_numeric_unscaled[:, 2] = np.clip(X_synthetic_numeric_unscaled[:, 2], 30, 200)  # runtime
X_synthetic_numeric_unscaled[:, 3] = np.clip(X_synthetic_numeric_unscaled[:, 3], 1, 10)  # tmdb_vote
X_synthetic_numeric_unscaled[:, 4] = np.clip(X_synthetic_numeric_unscaled[:, 4], 0, 15)  # n_genres

# Build synthetic dataframe
synth_df = pd.DataFrame(X_synthetic_numeric_unscaled, columns=numeric_cols)
for i, col in enumerate(onehot_cols):
    synth_df[col] = X_synthetic_onehot[:, i]

print(f"\n[TVAE] Synthetic data stats (numeric columns only):")
print(synth_df[numeric_cols].describe())

# ============================================================================
# Save synthetic data
# ============================================================================
synth_path = ART / "tvae_synth.parquet"
synth_df.to_parquet(synth_path)
print(f"\n[TVAE] Saved {len(synth_df)} synthetic rows to {synth_path}")

# ============================================================================
# PCA visualization: real vs synthetic
# ============================================================================
print("\n[TVAE] Creating PCA visualization...")

# Use only numeric features for PCA
X_real_numeric = X_numeric_scaled  # Real data (standardized)
X_synth_numeric = X_synthetic_numeric  # Synthetic data (before inverse transform)

# Fit PCA on combined real + synthetic data
pca = PCA(n_components=2, random_state=42)
X_combined = np.vstack([X_real_numeric, X_synth_numeric])
X_pca_combined = pca.fit_transform(X_combined)

X_pca_real = X_pca_combined[:len(X_real_numeric)]
X_pca_synth = X_pca_combined[len(X_real_numeric):]

# Print PCA variance and overlap stats
print(f"[TVAE] PCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"[TVAE] Real data in PCA space:")
print(f"  - mean: {X_pca_real.mean(axis=0)}")
print(f"  - std:  {X_pca_real.std(axis=0)}")
print(f"[TVAE] Synthetic data in PCA space:")
print(f"  - mean: {X_pca_synth.mean(axis=0)}")
print(f"  - std:  {X_pca_synth.std(axis=0)}")

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(X_pca_real[:, 0], X_pca_real[:, 1], c='blue', alpha=0.7, s=50, label=f'Real (n={len(X_pca_real)})')
ax.scatter(X_pca_synth[:, 0], X_pca_synth[:, 1], c='orange', alpha=0.3, s=20, label=f'Synthetic (n={len(X_pca_synth)})')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax.set_title('TVAE synthetic vs real rating-feature distribution')
ax.legend(loc='best', fontsize=11)
ax.grid(alpha=0.3)

plot_path = ART / "plots" / "tvae_overlap.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=130)
print(f"\n[TVAE] Saved plot to {plot_path}")

print("\n[TVAE] Task 1.8 complete!")
