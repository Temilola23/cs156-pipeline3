"""
Train MC Dropout BNN on Temilola's 324 ratings, then predict with uncertainty
on the training set and a 4-modality counterfactual grid (82 movies × 4 modalities).

Feature engineering (matching Task 1.8 TVAE):
  - year_norm, runtime_norm, vote_norm, n_genres_norm (5 numeric cols, standardized)
  - modality_0, modality_1, modality_2, modality_3 (one-hot, 4 cols)
  Total: d = 9

Output:
  - artifacts/bnn_mcd_predictions.npz: tmdb_id, modality, mean, std
  - artifacts/plots/bnn_mcd_uncertainty.png: scatter of mean vs std
"""
from __future__ import annotations
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_io import load_324_ratings, load_movie_meta
from src.bnn_mcd import MCDropoutBNN

ART = ROOT / "artifacts"
(ART / "plots").mkdir(parents=True, exist_ok=True)

print("[BNN-MCD] Loading data...")
df = load_324_ratings()

# ============================================================================
# Build feature table (same as Task 1.8)
# ============================================================================
feature_df = pd.DataFrame(index=df.index)

# 1. Rating (target)
feature_df['rating'] = df['rating'].values

# 2. Year
year_median = pd.to_numeric(df['year'], errors='coerce').median()
feature_df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(year_median).values

# 3. Runtime
runtime_median = df['runtime_min'].median()
feature_df['runtime_min'] = df['runtime_min'].fillna(runtime_median).values

# 4. TMDB vote average
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

# 6. One-hot modality
modalities = ['all', 'metadata', 'poster', 'synopsis']
for i, mod in enumerate(modalities):
    feature_df[f'modality_{i}'] = (df['modality'] == mod).astype(int).values

print(f"[BNN-MCD] Feature table shape: {feature_df.shape}")

# Standardize numeric features
numeric_cols = ['rating', 'year', 'runtime_min', 'tmdb_vote_average', 'n_genres']
onehot_cols = [c for c in feature_df.columns if c.startswith('modality_')]

X_numeric = feature_df[numeric_cols].values.astype(np.float32)
X_onehot = feature_df[onehot_cols].values.astype(np.float32)

scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric).astype(np.float32)

# Separate features from target
# For training: use year, runtime, vote, n_genres, modality as features (no rating)
# Target: rating
X_train_numeric = X_numeric_scaled[:, 1:]  # Skip rating (col 0), keep year, runtime, vote, n_genres
X_train_onehot = X_onehot
X_train = np.concatenate([X_train_numeric, X_train_onehot], axis=1).astype(np.float32)
y_train = X_numeric_scaled[:, 0].astype(np.float32).reshape(-1, 1)  # rating (standardized)

print(f"[BNN-MCD] Training X shape: {X_train.shape}, y shape: {y_train.shape}")
n_training_samples = len(X_train)

# ============================================================================
# Train BNN
# ============================================================================
torch.manual_seed(42)
np.random.seed(42)

d = X_train.shape[1]  # Should be 8 (year, runtime, vote, n_genres + 4 modality)
h = 32
batch_size = 32
n_epochs = 300
lr = 1e-3

device = torch.device('cpu')
bnn = MCDropoutBNN(d=d, h=h, p=0.2).to(device)
optimizer = torch.optim.Adam(bnn.parameters(), lr=lr)

X_tensor = torch.from_numpy(X_train).to(device)
y_tensor = torch.from_numpy(y_train).to(device)

print(f"\n[BNN-MCD] Training config: d={d}, h={h}, epochs={n_epochs}, batch_size={batch_size}, lr={lr}")
print(f"[BNN-MCD] Training on {n_training_samples} samples")

losses = []
bnn.train()

for epoch in range(n_epochs):
    # Shuffle
    idx = torch.randperm(len(X_tensor))
    X_shuffled = X_tensor[idx]
    y_shuffled = y_tensor[idx]

    epoch_loss = 0.0
    n_batches = 0

    for i in range(0, len(X_shuffled), batch_size):
        x_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]

        optimizer.zero_grad()
        y_pred = bnn(x_batch)
        loss = F.mse_loss(y_pred, y_batch, reduction='mean')
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    losses.append(avg_loss)

    if (epoch + 1) % 100 == 0:
        print(f"  Epoch {epoch+1}/{n_epochs}: MSE={avg_loss:.4f}")

print(f"\n[BNN-MCD] Training complete.")
print(f"[BNN-MCD] Final train MSE: {losses[-1]:.4f}")

# ============================================================================
# Predict on training set (sanity check)
# ============================================================================
print(f"\n[BNN-MCD] Predicting on training set (sanity check)...")
bnn.eval()
with torch.no_grad():
    y_train_pred = bnn(X_tensor)
train_mse = F.mse_loss(y_train_pred, y_tensor).item()
print(f"[BNN-MCD] Training set MSE (eval mode): {train_mse:.4f}")

# ============================================================================
# Generate counterfactual predictions: all 82 unique movies × 4 modalities
# ============================================================================
print(f"\n[BNN-MCD] Generating counterfactual predictions (82 movies × 4 modalities)...")

# Get unique movies
unique_tmdb_ids = df['tmdb_id'].unique()
print(f"[BNN-MCD] Found {len(unique_tmdb_ids)} unique movies")

# For each unique movie, get the modal features (excluding rating + modality)
# Build a mapping from tmdb_id to (year, runtime, vote, n_genres)
movie_features = {}
for tid in unique_tmdb_ids:
    rows = df[df['tmdb_id'] == tid]
    if len(rows) > 0:
        row = rows.iloc[0]
        year = pd.to_numeric(row['year'], errors='coerce')
        if pd.isna(year):
            year = year_median
        runtime = row['runtime_min']
        if pd.isna(runtime):
            runtime = runtime_median
        m = meta.get(int(tid), {})
        vote = m.get('tmdb_vote_average')
        if vote is None:
            vote = vote_median
        n_gen = len(row['genres']) if isinstance(row['genres'], list) else 0
        movie_features[int(tid)] = [year, runtime, vote, n_gen]

print(f"[BNN-MCD] Extracted features for {len(movie_features)} movies")

# For each movie, create 4 samples (one per modality)
X_counterfactual_list = []
tmdb_ids_list = []
modality_names = ['all', 'metadata', 'poster', 'synopsis']
modality_indices = []

for tid in sorted(unique_tmdb_ids):
    tid_int = int(tid)
    if tid_int not in movie_features:
        continue

    base_features = movie_features[tid_int]  # [year, runtime, vote, n_genres]

    # Standardize these features using the same scaler (matching the numeric_cols order)
    base_features_np = np.array(base_features).reshape(1, -1).astype(np.float32)
    # The scaler was fit on [rating, year, runtime, vote, n_genres]
    # So we need scaler params for cols 1-4 (year, runtime, vote, n_genres)
    scaler_means = scaler.mean_[1:]
    scaler_stds = scaler.scale_[1:]
    base_features_scaled = (base_features_np - scaler_means) / scaler_stds

    for mod_idx in range(4):
        # One-hot modality vector
        onehot = np.zeros(4, dtype=np.float32)
        onehot[mod_idx] = 1.0

        # Combine features + one-hot
        x_sample = np.concatenate([base_features_scaled[0], onehot])
        X_counterfactual_list.append(x_sample)
        tmdb_ids_list.append(tid_int)
        modality_indices.append(mod_idx)

X_counterfactual = np.array(X_counterfactual_list, dtype=np.float32)
print(f"[BNN-MCD] Counterfactual matrix shape: {X_counterfactual.shape}")

# Predict with uncertainty
mean_cf, std_cf = bnn.predict_with_uncertainty(X_counterfactual, T=50)

# ============================================================================
# Save predictions
# ============================================================================
print(f"\n[BNN-MCD] Saving predictions...")

predictions_df = pd.DataFrame({
    'tmdb_id': tmdb_ids_list,
    'modality': [modality_names[i] for i in modality_indices],
    'mean': mean_cf,
    'std': std_cf,
})

npz_path = ART / "bnn_mcd_predictions.npz"
np.savez(
    npz_path,
    tmdb_id=predictions_df['tmdb_id'].values,
    modality=predictions_df['modality'].values,
    mean=predictions_df['mean'].values,
    std=predictions_df['std'].values,
)
print(f"[BNN-MCD] Saved predictions to {npz_path}")

# Also save as CSV for convenience
csv_path = ART / "bnn_mcd_predictions.csv"
predictions_df.to_csv(csv_path, index=False)
print(f"[BNN-MCD] Saved predictions to {csv_path}")

# ============================================================================
# Stats and summary
# ============================================================================
print(f"\n[BNN-MCD] Prediction Statistics (all {len(mean_cf)} counterfactual samples):")
print(f"  - Mean (predictions): {mean_cf.mean():.4f} ± {mean_cf.std():.4f}")
print(f"  - Std (uncertainty): {std_cf.mean():.4f} ± {std_cf.std():.4f}")
print(f"    * Min std: {std_cf.min():.4f}")
print(f"    * Max std: {std_cf.max():.4f}")

# Find most uncertain movies
top_uncertain_idx = np.argsort(std_cf)[-3:][::-1]
print(f"\n[BNN-MCD] Top 3 most uncertain predictions:")
for rank, idx in enumerate(top_uncertain_idx, 1):
    tid = tmdb_ids_list[idx]
    mod = modality_names[modality_indices[idx]]
    m = mean_cf[idx]
    s = std_cf[idx]
    print(f"  {rank}. TMDB ID {tid}, modality={mod}: mean={m:.4f}, std={s:.4f}")

# ============================================================================
# Plot: mean vs std, colored by modality
# ============================================================================
print(f"\n[BNN-MCD] Creating uncertainty plot...")

fig, ax = plt.subplots(figsize=(10, 8))

colors = {'all': 'blue', 'metadata': 'green', 'poster': 'orange', 'synopsis': 'red'}
for mod_name in modality_names:
    mask = predictions_df['modality'] == mod_name
    ax.scatter(
        predictions_df[mask]['mean'],
        predictions_df[mask]['std'],
        c=colors[mod_name],
        label=mod_name,
        alpha=0.6,
        s=50,
    )

ax.set_xlabel('Predicted Mean Rating', fontsize=12)
ax.set_ylabel('Predictive Std Dev (Uncertainty)', fontsize=12)
ax.set_title(f'MC Dropout BNN: Predictive Uncertainty (T=50 forward passes)\n{len(predictions_df)} counterfactual samples (82 movies × 4 modalities)', fontsize=12)
ax.legend(loc='best', fontsize=11)
ax.grid(alpha=0.3)

plot_path = ART / "plots" / "bnn_mcd_uncertainty.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=130)
print(f"[BNN-MCD] Saved plot to {plot_path}")
plt.close()

print(f"\n[BNN-MCD] Task 1.11 complete!")
