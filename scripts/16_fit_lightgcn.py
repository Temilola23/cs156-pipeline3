#!/usr/bin/env python3
"""
Task 1.9: LightGCN from-scratch PyTorch implementation.

Train a 4-layer LightGCN on the combined bipartite user-item graph:
- Temilola's 324 ratings (temi → tmdb_id)
- MovieLens twin 51K+ ratings (ml_<user_id> → ml_<movielens_movie_id>)

Uses BPR loss and evaluates with AUC on held-out test pairs.

Math reference (He et al., 2020):
- Symmetrically normalized adjacency: A_hat = D^(-1/2) A D^(-1/2)
- Propagation: E^(k+1) = A_hat @ E^(k) (no weights, no activations)
- Final embedding: E_final = mean(E^(0), ..., E^(K))
- Scoring: s(u, v) = <E_u, E_v>
- Loss: BPR = -log(sigmoid(s(u,v+) - s(u,v-))) + weight_decay * (||E_u||^2 + ||E_v||^2)
"""
import os
import sys
from pathlib import Path
import json
import pickle
from typing import Tuple, Dict, List
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add Pipeline 3 src to path
PIPELINE3_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE3_DIR))

from src.data_io import load_324_ratings
from src.lightgcn import LightGCN


def combine_and_encode_interactions(
    ratings_324: pd.DataFrame,
    movielens_twin: pd.DataFrame,
    threshold_324: float = 3.5,
    threshold_ml: float = 3.5,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    """
    Combine Temilola's 324 ratings + MovieLens twin into a single interaction table.
    Encode user_id and item_id as integers.

    Parameters
    ----------
    ratings_324 : pd.DataFrame
        Temilola's 324 ratings with columns: [tmdb_id, rating, ...]
    movielens_twin : pd.DataFrame
        MovieLens twin with columns: [user_id, movielens_movie_id, rating, ...]
    threshold_324 : float
        Rating threshold for 324 ratings (>= threshold treated as positive)
    threshold_ml : float
        Rating threshold for ML twin (>= threshold treated as positive)

    Returns
    -------
    interactions : pd.DataFrame
        Combined interactions with columns: [user_id, item_id, rating]
    user_id_to_idx : dict
        Mapping user_id (str) → integer index
    item_id_to_idx : dict
        Mapping item_id (str) → integer index
    """
    rows = []

    # Add Temilola's 324 ratings (user_id = "temi", item_id = f"tmdb_{tmdb_id}")
    for _, row in ratings_324.iterrows():
        if row['rating'] >= threshold_324:
            rows.append({
                'user_id': 'temi',
                'item_id': f"tmdb_{int(row['tmdb_id'])}",
                'rating': row['rating'],
            })

    # Add MovieLens twin (user_id = f"ml_{ml_user_id}", item_id = f"ml_{ml_movie_id}")
    for _, row in movielens_twin.iterrows():
        if row['rating'] >= threshold_ml:
            rows.append({
                'user_id': f"ml_{int(row['user_id'])}",
                'item_id': f"ml_{int(row['movielens_movie_id'])}",
                'rating': row['rating'],
            })

    interactions = pd.DataFrame(rows)
    print(f"Combined interactions: {len(interactions)} rows")
    print(f"  - From 324 ratings: {(interactions['user_id'] == 'temi').sum()}")
    print(f"  - From ML twin: {(interactions['user_id'] != 'temi').sum()}")

    # Encode user_id and item_id as integers
    unique_users = sorted(interactions['user_id'].unique())
    unique_items = sorted(interactions['item_id'].unique())

    user_id_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    item_id_to_idx = {iid: i for i, iid in enumerate(unique_items)}

    interactions['user_idx'] = interactions['user_id'].map(user_id_to_idx)
    interactions['item_idx'] = interactions['item_id'].map(item_id_to_idx)

    print(f"Encoded: {len(unique_users)} users, {len(unique_items)} items")

    return interactions, user_id_to_idx, item_id_to_idx


def build_sparse_adjacency(
    interactions: pd.DataFrame,
    n_users: int,
    n_items: int,
) -> torch.sparse.FloatTensor:
    """
    Build symmetrically-structured sparse adjacency matrix for bipartite graph.

    The matrix is (n_users + n_items) x (n_users + n_items):
    - Position (u, n_users + v) = 1 if user u rated item v
    - Position (n_users + v, u) = 1 (symmetric)

    Parameters
    ----------
    interactions : pd.DataFrame
        DataFrame with [user_idx, item_idx]
    n_users : int
        Number of users
    n_items : int
        Number of items

    Returns
    -------
    A : torch.sparse.FloatTensor
        Sparse COO tensor of shape (n_users + n_items, n_users + n_items)
    """
    size = n_users + n_items
    edge_list = []

    for _, row in interactions.iterrows():
        u, v = int(row['user_idx']), int(row['item_idx'])
        # Edge from user to item: (u, n_users + v)
        edge_list.append((u, n_users + v))
        # Reverse edge for symmetry: (n_users + v, u)
        edge_list.append((n_users + v, u))

    if len(edge_list) == 0:
        # No interactions
        indices = torch.LongTensor([[], []])
        values = torch.FloatTensor([])
    else:
        indices = torch.LongTensor(edge_list).t().contiguous()
        values = torch.ones(len(edge_list), dtype=torch.float32)

    A = torch.sparse_coo_tensor(indices, values, size=(size, size), dtype=torch.float32)

    return A


def train_lightgcn(
    model: LightGCN,
    A_hat: torch.sparse.FloatTensor,
    train_interactions: pd.DataFrame,
    n_items: int,
    n_epochs: int = 30,
    batch_size: int = 128,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
) -> Tuple[List[float], LightGCN]:
    """
    Train LightGCN with BPR loss.

    For each batch:
    1. Sample (user, positive_item, negative_item) triplets
    2. Forward propagate and compute BPR loss
    3. Backward + optimizer step

    Parameters
    ----------
    model : LightGCN
        Uninitialized LightGCN model
    A_hat : torch.sparse.FloatTensor
        Normalized sparse adjacency matrix
    train_interactions : pd.DataFrame
        Training interactions with [user_idx, item_idx]
    n_items : int
        Total number of items
    n_epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    lr : float
        Learning rate
    weight_decay : float
        L2 regularization

    Returns
    -------
    losses : list
        BPR loss per epoch
    model : LightGCN
        Trained model
    """
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()

    losses_per_epoch = []

    for epoch in range(n_epochs):
        epoch_losses = []

        # Build negative sample pool per user
        user_items = {}
        for _, row in train_interactions.iterrows():
            u, v = int(row['user_idx']), int(row['item_idx'])
            if u not in user_items:
                user_items[u] = set()
            user_items[u].add(v)

        # Sample mini-batches
        n_batches = max(1, len(train_interactions) // batch_size)
        for batch_idx in tqdm(
            range(n_batches),
            desc=f"Epoch {epoch + 1}/{n_epochs}",
            leave=False
        ):
            optimizer.zero_grad()

            batch_loss = 0.0

            # Sample batch_size triplets
            for _ in range(batch_size):
                # Random positive interaction
                pos_idx = np.random.randint(len(train_interactions))
                u = int(train_interactions.iloc[pos_idx]['user_idx'])
                v_pos = int(train_interactions.iloc[pos_idx]['item_idx'])

                # Sample negative item (not in user's positive set)
                v_neg = np.random.randint(n_items)
                while v_neg in user_items.get(u, set()):
                    v_neg = np.random.randint(n_items)

                # Forward propagation
                E_u_final, E_v_final = model.propagate(A_hat)

                # Compute BPR loss
                loss = model.bpr_loss(E_u_final, E_v_final, u, v_pos, v_neg, weight_decay=weight_decay)
                batch_loss += loss

            # Average batch loss
            batch_loss /= batch_size
            batch_loss.backward()
            optimizer.step()

            epoch_losses.append(batch_loss.item())

        epoch_loss = np.mean(epoch_losses)
        losses_per_epoch.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{n_epochs}: Loss = {epoch_loss:.6f}")

    return losses_per_epoch, model


def evaluate_auc(
    model: LightGCN,
    A_hat: torch.sparse.FloatTensor,
    test_interactions: pd.DataFrame,
    n_items: int,
    n_negatives: int = 100,
) -> float:
    """
    Evaluate AUC-like metric on test interactions.

    For each test (user, positive_item) pair:
    - Sample n_negatives random items (that user has NOT rated)
    - Count fraction where s(u, v_pos) > s(u, v_neg)
    - Average across all test pairs

    Parameters
    ----------
    model : LightGCN
        Trained LightGCN model
    A_hat : torch.sparse.FloatTensor
        Normalized sparse adjacency matrix
    test_interactions : pd.DataFrame
        Test interactions with [user_idx, item_idx]
    n_items : int
        Total number of items
    n_negatives : int
        Number of negative samples per test pair

    Returns
    -------
    auc : float
        AUC-like metric (fraction of correct rankings)
    """
    model.eval()

    # Forward propagation (once)
    with torch.no_grad():
        E_u_final, E_v_final = model.propagate(A_hat)

    # Build user → positive items mapping
    user_positives = {}
    for _, row in test_interactions.iterrows():
        u = int(row['user_idx'])
        v = int(row['item_idx'])
        if u not in user_positives:
            user_positives[u] = set()
        user_positives[u].add(v)

    # Evaluate
    scores = []

    for _, row in test_interactions.iterrows():
        u = int(row['user_idx'])
        v_pos = int(row['item_idx'])

        # Score of positive item
        with torch.no_grad():
            s_pos = model.score(E_u_final, E_v_final, u, v_pos).item()

        # Sample negative items
        hits = 0
        for _ in range(n_negatives):
            v_neg = np.random.randint(n_items)
            while v_neg in user_positives.get(u, set()):
                v_neg = np.random.randint(n_items)

            # Score of negative item
            with torch.no_grad():
                s_neg = model.score(E_u_final, E_v_final, u, v_neg).item()

            # Check if positive ranks higher
            if s_pos > s_neg:
                hits += 1

        # Fraction correct for this user-item pair
        score = hits / n_negatives
        scores.append(score)

    auc = np.mean(scores)
    return auc


def plot_training_curves(losses, aucs, output_path: Path):
    """Plot training loss and AUC curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    epochs = list(range(1, len(losses) + 1))
    ax1.plot(epochs, losses, 'o-', linewidth=2, markersize=4, color='#1f77b4')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('BPR Loss', fontsize=12)
    ax1.set_title('LightGCN Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # AUC curve
    ax2.plot(epochs[:len(aucs)], aucs, 's-', linewidth=2, markersize=4, color='#ff7f0e')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Test AUC', fontsize=12)
    ax2.set_title('LightGCN Test AUC', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("Task 1.9: From-Scratch 4-Layer LightGCN")
    print("=" * 80)

    # Paths
    artifacts_dir = PIPELINE3_DIR / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # Step 1: Load data
    print("\n[1] Loading data...")
    ratings_324 = load_324_ratings()
    movielens_twin = pd.read_parquet(artifacts_dir / "movielens_twin_ratings.parquet")

    # Step 2: Combine and encode
    print("\n[2] Combining interactions and encoding...")
    interactions, user_id_to_idx, item_id_to_idx = combine_and_encode_interactions(
        ratings_324, movielens_twin,
        threshold_324=3.5,
        threshold_ml=3.5,
    )

    n_users = len(user_id_to_idx)
    n_items = len(item_id_to_idx)

    print(f"  - Total interactions: {len(interactions)}")
    print(f"  - Users: {n_users}")
    print(f"  - Items: {n_items}")

    # Step 3: Train/test split (80/20)
    print("\n[3] Train/test split (80/20)...")
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(interactions))
    cut = int(len(interactions) * 0.8)
    train_idx = idx[:cut]
    test_idx = idx[cut:]

    train_interactions = interactions.iloc[train_idx].reset_index(drop=True)
    test_interactions = interactions.iloc[test_idx].reset_index(drop=True)

    print(f"  - Train: {len(train_interactions)} interactions")
    print(f"  - Test: {len(test_interactions)} interactions")

    # Step 4: Build sparse adjacency
    print("\n[4] Building sparse bipartite adjacency matrix...")
    A = build_sparse_adjacency(train_interactions, n_users, n_items)
    print(f"  - Adjacency shape: {A.size()}")
    print(f"  - Non-zeros: {A._nnz()}")

    # Step 5: Normalize adjacency
    print("\n[5] Normalizing adjacency (D^(-1/2) A D^(-1/2))...")
    model_init = LightGCN(n_users=n_users, n_items=n_items, d=64, n_layers=4)
    A_hat = model_init.normalize_adjacency(A)
    print(f"  - Normalized adjacency non-zeros: {A_hat._nnz()}")

    # Step 6: Initialize model
    print("\n[6] Initializing LightGCN(d=64, n_layers=4)...")
    model = LightGCN(n_users=n_users, n_items=n_items, d=64, n_layers=4)
    print(f"  - User embeddings: {model.E_u.shape}")
    print(f"  - Item embeddings: {model.E_v.shape}")

    # Step 7: Train
    print("\n[7] Training for 30 epochs...")
    start_time = time.time()
    losses, model = train_lightgcn(
        model, A_hat, train_interactions,
        n_items=n_items,
        n_epochs=30,
        batch_size=128,
        lr=0.001,
        weight_decay=1e-4,
    )
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.1f} seconds")

    print(f"\nLoss: Initial = {losses[0]:.6f}, Final = {losses[-1]:.6f}")

    # Step 8: Evaluate AUC on test set (sample every N epochs)
    print("\n[8] Evaluating AUC on test set...")
    aucs = []
    for epoch_idx in range(0, len(losses), max(1, len(losses) // 5)):
        auc = evaluate_auc(model, A_hat, test_interactions, n_items, n_negatives=100)
        aucs.append(auc)
        print(f"  - Epoch {epoch_idx}: AUC = {auc:.4f}")

    # Final AUC
    final_auc = evaluate_auc(model, A_hat, test_interactions, n_items, n_negatives=100)
    print(f"\nFinal AUC on held-out test set: {final_auc:.4f}")

    # Step 9: Save embeddings
    print("\n[9] Saving embeddings...")
    with torch.no_grad():
        E_u_final, E_v_final = model.propagate(A_hat)

    embeddings_path = artifacts_dir / "lightgcn_embeddings.npz"
    np.savez(
        embeddings_path,
        user_embeddings=E_u_final.numpy(),
        item_embeddings=E_v_final.numpy(),
    )

    # Save mappings as JSON
    mappings_path = artifacts_dir / "lightgcn_mappings.json"
    with open(mappings_path, 'w') as f:
        json.dump({
            'user_id_to_idx': user_id_to_idx,
            'item_id_to_idx': item_id_to_idx,
        }, f, indent=2)

    print(f"  - Embeddings: {embeddings_path}")
    print(f"  - Mappings: {mappings_path}")

    # Step 10: Plot training curves
    print("\n[10] Plotting training curves...")
    plot_path = PIPELINE3_DIR / "artifacts" / "plots" / "lightgcn_training.png"
    plot_path.parent.mkdir(exist_ok=True, parents=True)
    plot_training_curves(losses, aucs, plot_path)

    # Summary
    print("\n" + "=" * 80)
    print("Task 1.9 Summary")
    print("=" * 80)
    print(f"n_users: {n_users}")
    print(f"n_items: {n_items}")
    print(f"n_positive_edges: {len(train_interactions)}")
    print(f"training_loss_initial: {losses[0]:.6f}")
    print(f"training_loss_final: {losses[-1]:.6f}")
    print(f"test_auc: {final_auc:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
