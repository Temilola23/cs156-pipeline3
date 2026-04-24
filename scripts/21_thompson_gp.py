#!/usr/bin/env python3
"""
Task 1.14: Thompson sampling bandit over 82 movies.

Load 324 ratings → aggregate per movie → features (year, runtime, vote, n_genres) → 
split into seed (20 movies) and pool (62 arms) → 
run Thompson sampling for 20 rounds vs random baseline → 
report cumulative regret + save history.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_io import load_324_ratings, load_movie_meta
from src.gp import GaussianProcess
from src.kernels import rbf
from src.thompson_gp import thompson_sampling_loop, random_baseline


def standardize(x: np.ndarray) -> np.ndarray:
    """Z-score standardization."""
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def numpy_to_python(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    return obj


def main():
    print("[*] Loading data...")
    df = load_324_ratings()
    meta = load_movie_meta()
    
    # Aggregate ratings by movie: compute mean rating per movie
    print("[*] Aggregating ratings by movie...")
    movie_ratings = df.groupby('tmdb_id').agg({
        'rating': ['mean', 'count'],
        'year': 'first',
        'runtime_min': 'first',
        'vote_average': 'first',
        'genres': 'first'
    }).reset_index()
    
    movie_ratings.columns = ['tmdb_id', 'rating_mean', 'rating_count', 
                             'year', 'runtime_min', 'vote_average', 'genres']
    
    # Drop rows missing key features
    movie_ratings = movie_ratings.dropna(subset=['year', 'runtime_min', 'vote_average', 'genres'])
    
    print(f"[*] Movies with complete features: {len(movie_ratings)}")
    
    # Extract features
    n_movies = len(movie_ratings)
    years = movie_ratings['year'].values.astype(float)
    runtimes = movie_ratings['runtime_min'].values.astype(float)
    votes = movie_ratings['vote_average'].values.astype(float)
    n_genres = np.array([len(g) if isinstance(g, list) else 0 for g in movie_ratings['genres']], dtype=float)
    
    # Standardize
    year_norm = standardize(years)
    runtime_norm = standardize(runtimes)
    vote_norm = standardize(votes)
    n_genres_norm = standardize(n_genres)
    
    X_full = np.column_stack([year_norm, runtime_norm, vote_norm, n_genres_norm])
    y_full = movie_ratings['rating_mean'].values
    
    print(f"[*] Feature matrix shape: {X_full.shape}")
    print(f"[*] Rating range: [{y_full.min():.2f}, {y_full.max():.2f}]")
    
    # Split: 20 seed, rest are arms
    rng = np.random.default_rng(42)
    all_idx = np.arange(n_movies)
    rng.shuffle(all_idx)
    
    seed_idx = all_idx[:20]
    pool_idx = all_idx[20:]
    
    X_seed = X_full[seed_idx]
    y_seed = y_full[seed_idx]
    X_pool = X_full[pool_idx]
    y_pool = y_full[pool_idx]
    
    print(f"[*] Seed set: {len(seed_idx)} movies")
    print(f"[*] Pool (arms): {len(pool_idx)} movies")
    
    # Define GP kernel: RBF
    def kernel(A, B):
        return rbf(A, B, length=1.0, var=1.0)
    
    # Run Thompson sampling and random baseline over multiple seeds for averaging
    print("[*] Running Thompson sampling and random baseline (10 seeds)...")
    
    n_seeds = 10
    n_rounds = 200
    thompson_regrets_all = []
    random_regrets_all = []
    thompson_history_all = []
    
    for seed in range(n_seeds):
        print(f"  Seed {seed}...")
        
        # Fit initial GP on seed set
        gp = GaussianProcess(kernel=kernel, noise=0.1)
        gp.fit(X_seed, y_seed)
        
        # Thompson sampling
        thompson_chosen, thompson_rewards, thompson_regret = thompson_sampling_loop(
            gp, X_pool, y_pool, X_seed, y_seed, n_rounds=n_rounds, seed=seed
        )
        
        # Random baseline
        random_chosen, random_rewards, random_regret = random_baseline(
            X_pool, y_pool, n_rounds=n_rounds, seed=seed
        )
        
        thompson_regrets_all.append(thompson_regret)
        random_regrets_all.append(random_regret)
        thompson_history_all.append({
            'seed': seed,
            'chosen_idx': [int(x) for x in thompson_chosen],
            'rewards': [float(x) for x in thompson_rewards],
            'regret': [float(x) for x in thompson_regret]
        })
    
    # Compute statistics
    thompson_regrets_all = np.array(thompson_regrets_all)
    random_regrets_all = np.array(random_regrets_all)
    
    thompson_mean = thompson_regrets_all.mean(axis=0)
    thompson_std = thompson_regrets_all.std(axis=0)
    random_mean = random_regrets_all.mean(axis=0)
    random_std = random_regrets_all.std(axis=0)
    
    print(f"\n[*] Results after {n_rounds} rounds:")
    print(f"    Thompson cumulative regret: {thompson_mean[-1]:.4f} ± {thompson_std[-1]:.4f}")
    print(f"    Random cumulative regret:   {random_mean[-1]:.4f} ± {random_std[-1]:.4f}")
    print(f"    Thompson advantage: {random_mean[-1] - thompson_mean[-1]:.4f}")
    
    # Count arm selections for Thompson
    all_chosen = [idx for hist in thompson_history_all for idx in hist['chosen_idx']]
    arm_counts = np.bincount(all_chosen, minlength=len(X_pool))
    top_arms = np.argsort(arm_counts)[-5:][::-1]
    print(f"\n[*] Top 5 most frequently chosen arms by Thompson:")
    for rank, arm_idx in enumerate(top_arms, 1):
        count = arm_counts[arm_idx]
        print(f"    {rank}. Arm {arm_idx}: selected {count} times, true reward {y_pool[arm_idx]:.3f}")
    
    # Create output directories
    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    plots_dir = artifacts_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Plot cumulative regret
    fig, ax = plt.subplots(figsize=(10, 6))
    rounds = np.arange(1, n_rounds + 1)
    
    ax.plot(rounds, thompson_mean, 'b-', linewidth=2, label='Thompson sampling')
    ax.fill_between(rounds, 
                    thompson_mean - thompson_std, 
                    thompson_mean + thompson_std, 
                    alpha=0.2, color='blue')
    
    ax.plot(rounds, random_mean, 'r-', linewidth=2, label='Random baseline')
    ax.fill_between(rounds,
                    random_mean - random_std,
                    random_mean + random_std,
                    alpha=0.2, color='red')
    
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Cumulative regret', fontsize=12)
    ax.set_title('Thompson Sampling vs Random Baseline (Movie Bandit, 20 Rounds)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "thompson_regret.png", dpi=130)
    print(f"\n[*] Saved plot: artifacts/plots/thompson_regret.png")
    plt.close()
    
    # Save history JSON
    history_data = {
        'n_seeds': n_seeds,
        'n_rounds': n_rounds,
        'n_pool_arms': int(len(X_pool)),
        'seed_set_size': int(len(X_seed)),
        'thompson_final_regret_mean': float(thompson_mean[-1]),
        'thompson_final_regret_std': float(thompson_std[-1]),
        'random_final_regret_mean': float(random_mean[-1]),
        'random_final_regret_std': float(random_std[-1]),
        'thompson_advantage': float(random_mean[-1] - thompson_mean[-1]),
        'thompson_history': thompson_history_all,
        'thompson_regret_by_round': thompson_mean.tolist(),
        'random_regret_by_round': random_mean.tolist(),
    }
    
    history_data = numpy_to_python(history_data)
    
    history_path = artifacts_dir / "thompson_history.json"
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"[*] Saved history: artifacts/thompson_history.json")


if __name__ == '__main__':
    main()
