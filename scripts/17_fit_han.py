#!/usr/bin/env python3
"""
Script to fit HAN on movie-genre heterogeneous graph.

Loads 82 movies + genres, builds MGM adjacency from genre co-occurrence,
trains HAN to predict whether a user will rate a movie >= 3.5 (using median split).

Outputs:
  - artifacts/han_embeddings.npz: movie embeddings, labels, beta weights
  - artifacts/plots/han_training.png: loss + val accuracy curves
"""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Add src to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.han import HAN
from src.data_io import load_movie_meta, load_324_ratings


def build_movie_features(movie_meta: dict) -> tuple[np.ndarray, list[int]]:
    """
    Build feature matrix and tmdb_id list from movie metadata.
    
    Features: [year_norm, runtime_norm, vote_norm, n_genres_norm, rating_mean]
    
    Args:
        movie_meta: dict from load_movie_meta()
    
    Returns:
        features: (n_movies, 5) array
        tmdb_ids: list of tmdb_ids in same order
    """
    tmdb_ids = sorted(movie_meta.keys())
    n_movies = len(tmdb_ids)
    features = np.zeros((n_movies, 5))
    
    years = []
    runtimes = []
    votes = []
    n_genre_list = []
    
    for i, tmdb_id in enumerate(tmdb_ids):
        meta = movie_meta[tmdb_id]
        year = int(meta.get('year', 2000)) if meta.get('year') else 2000
        runtime = meta.get('runtime_min', 100) or 100
        vote = meta.get('tmdb_vote_average', 5.0) or 5.0
        genres = meta.get('genres', [])
        n_genres = len(genres)
        
        years.append(year)
        runtimes.append(runtime)
        votes.append(vote)
        n_genre_list.append(n_genres)
    
    # Normalize
    years = np.array(years)
    runtimes = np.array(runtimes)
    votes = np.array(votes)
    n_genre_list = np.array(n_genre_list)
    
    year_norm = (years - years.min()) / (years.max() - years.min() + 1e-8)
    runtime_norm = (runtimes - runtimes.min()) / (runtimes.max() - runtimes.min() + 1e-8)
    vote_norm = (votes - votes.min()) / (votes.max() - votes.min() + 1e-8)
    n_genres_norm = (n_genre_list - n_genre_list.min()) / (n_genre_list.max() - n_genre_list.min() + 1e-8)
    
    features[:, 0] = year_norm
    features[:, 1] = runtime_norm
    features[:, 2] = vote_norm
    features[:, 3] = n_genres_norm
    features[:, 4] = 5.0  # placeholder; will fill below
    
    return features, tmdb_ids


def build_labels(tmdb_ids: list[int], ratings_df: pd.DataFrame) -> np.ndarray:
    """
    Build binary labels: 1 if mean rating >= median, 0 otherwise.
    
    Args:
        tmdb_ids: list of tmdb_ids
        ratings_df: DataFrame from load_324_ratings()
    
    Returns:
        labels: (n_movies,) binary array, -1 for unrated
    """
    mean_ratings = ratings_df.groupby('tmdb_id')['rating'].mean()
    median_rating = mean_ratings.median()
    print(f"Median rating threshold: {median_rating:.3f}")
    
    labels = np.full(len(tmdb_ids), -1, dtype=int)
    for i, tmdb_id in enumerate(tmdb_ids):
        if int(tmdb_id) in mean_ratings.index:
            mean_r = mean_ratings[int(tmdb_id)]
            labels[i] = 1 if mean_r >= median_rating else 0
    
    return labels, median_rating


def build_adjacencies(tmdb_ids: list[int], movie_meta: dict) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Build adjacency matrices and genre list.
    
    Args:
        tmdb_ids: list of tmdb_ids
        movie_meta: dict from load_movie_meta()
    
    Returns:
        mgm_adj: (n_movies, n_movies) co-genre adjacency
        mg_adj: (n_movies, n_genres) movie-genre bipartite
        genres: list of unique genres
    """
    n_movies = len(tmdb_ids)
    
    # Collect all genres
    all_genres = set()
    movie_genres = {}
    for i, tmdb_id in enumerate(tmdb_ids):
        genres = movie_meta[tmdb_id].get('genres', [])
        movie_genres[i] = genres
        all_genres.update(genres)
    
    genres = sorted(all_genres)
    n_genres = len(genres)
    genre_to_idx = {g: i for i, g in enumerate(genres)}
    
    print(f"Unique genres: {n_genres}")
    print(f"Genres: {genres}")
    
    # Build movie-genre bipartite adjacency
    mg_adj = torch.zeros(n_movies, n_genres)
    for i, genres_list in movie_genres.items():
        for g in genres_list:
            j = genre_to_idx[g]
            mg_adj[i, j] = 1.0
    
    # Build MGM (movie-genre-movie) adjacency: m1 -> m2 if they share >= 1 genre
    mgm_adj = torch.zeros(n_movies, n_movies)
    for i in range(n_movies):
        for j in range(n_movies):
            # Shared genres
            shared = set(movie_genres[i]) & set(movie_genres[j])
            if len(shared) > 0:
                mgm_adj[i, j] = 1.0
    
    return mgm_adj, mg_adj, genres


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\n=== Loading data ===")
    movie_meta = load_movie_meta()
    ratings_df = load_324_ratings()
    
    print(f"Total movies in meta: {len(movie_meta)}")
    print(f"Total ratings: {len(ratings_df)}")
    print(f"Unique rated movies: {ratings_df['tmdb_id'].nunique()}")
    
    # Build features
    print("\n=== Building features ===")
    features, tmdb_ids = build_movie_features(movie_meta)
    n_movies = len(tmdb_ids)
    print(f"Movie features shape: {features.shape}")
    
    # Build labels
    print("\n=== Building labels ===")
    labels, median_threshold = build_labels(tmdb_ids, ratings_df)
    n_labeled = (labels >= 0).sum()
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    print(f"Labeled movies: {n_labeled}/{n_movies}")
    print(f"Positive labels (>= {median_threshold:.3f}): {n_pos}")
    print(f"Negative labels (< {median_threshold:.3f}): {n_neg}")
    
    # Build adjacencies
    print("\n=== Building adjacencies ===")
    mgm_adj, mg_adj, genres = build_adjacencies(tmdb_ids, movie_meta)
    n_genres = len(genres)
    print(f"MGM adjacency shape: {mgm_adj.shape}")
    print(f"MG adjacency shape: {mg_adj.shape}")
    
    # Train/val split (stratified on labeled samples)
    labeled_mask = labels >= 0
    labeled_indices = np.where(labeled_mask)[0]
    labeled_labels = labels[labeled_mask]
    
    train_idx, val_idx = train_test_split(
        labeled_indices, test_size=0.3, stratify=labeled_labels, random_state=42
    )
    
    print(f"\nTrain size: {len(train_idx)}")
    print(f"Val size: {len(val_idx)}")
    
    # Move to device
    features_t = torch.from_numpy(features).float().to(device)
    labels_t = torch.from_numpy(labels).long().to(device)
    mgm_adj = mgm_adj.to(device)
    mg_adj = mg_adj.to(device)
    
    # Initialize model
    print("\n=== Initializing model ===")
    model = HAN(movie_feat_dim=5, n_genres=n_genres, h=32, n_classes=2).to(device)
    print(model)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    # Training loop
    print("\n=== Training ===")
    n_epochs = 100
    train_losses = []
    val_losses = []
    val_accs = []
    
    for epoch in range(n_epochs):
        # Train
        model.train()
        embeddings, logits = model(features_t, mgm_adj, mg_adj)
        loss = criterion(logits, labels_t)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Eval on val set
        model.eval()
        with torch.no_grad():
            embeddings, logits = model(features_t, mgm_adj, mg_adj)
            val_loss = criterion(logits[val_idx], labels_t[val_idx])
            preds = logits[val_idx].argmax(dim=1)
            val_acc = (preds == labels_t[val_idx]).float().mean().item()
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        val_accs.append(val_acc)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | Train loss: {loss.item():.4f} | Val loss: {val_loss.item():.4f} | Val acc: {val_acc:.4f}")
    
    # Final eval on train set
    model.eval()
    with torch.no_grad():
        embeddings, logits = model(features_t, mgm_adj, mg_adj)
        train_loss_final = criterion(logits[train_idx], labels_t[train_idx])
        train_preds = logits[train_idx].argmax(dim=1)
        train_acc_final = (train_preds == labels_t[train_idx]).float().mean().item()
        
        val_loss_final = criterion(logits[val_idx], labels_t[val_idx])
        val_preds = logits[val_idx].argmax(dim=1)
        val_acc_final = (val_preds == labels_t[val_idx]).float().mean().item()
    
    print(f"\n=== Final Results ===")
    print(f"Train Loss: {train_loss_final.item():.4f}")
    print(f"Train Acc:  {train_acc_final:.4f}")
    print(f"Val Loss:   {val_loss_final.item():.4f}")
    print(f"Val Acc:    {val_acc_final:.4f}")
    
    # Semantic-level attention weights
    print(f"\n=== Semantic-level Attention (β) ===")
    beta_weights = model.beta_last.detach().cpu().numpy()
    print(f"β (MGM):   {beta_weights[0]:.4f}")
    print(f"β (M):     {beta_weights[1]:.4f}")
    
    # Save artifacts
    print("\n=== Saving artifacts ===")
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    embeddings_np = embeddings.detach().cpu().numpy()
    beta_np = beta_weights
    
    np.savez(
        artifacts_dir / "han_embeddings.npz",
        embeddings=embeddings_np,
        labels=labels,
        beta=beta_np,
        tmdb_ids=np.array(tmdb_ids),
        genres=np.array(genres),
        median_threshold=np.array([median_threshold])
    )
    print(f"Saved: {artifacts_dir / 'han_embeddings.npz'}")
    
    # Plot training curves
    plots_dir = artifacts_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(val_accs, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = plots_dir / "han_training.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()
    
    print("\n=== Complete ===")


if __name__ == '__main__':
    main()
