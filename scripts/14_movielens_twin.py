#!/usr/bin/env python3
"""
Task 1.7: MovieLens twin dataset generation.

Find top-K users in MovieLens whose rating patterns correlate with Temilola's 324
real ratings, and borrow their full rating vectors to produce a pseudo-rating
dataset of ≥50K rows that expands beyond the 82-movie catalog.

Uses numpy cosine-similarity (NOT lightfm, which is broken on Python 3.12 macOS).
"""
import os
import sys
from pathlib import Path
from typing import Tuple, Dict
import re
import string

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add Pipeline 3 src to path
PIPELINE3_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE3_DIR))

from src.data_io import load_324_ratings, load_movie_meta


def normalize_title(title: str) -> str:
    """
    Normalize a movie title for matching.
    - Remove year (e.g., "(1995)")
    - Convert to lowercase
    - Remove punctuation except spaces
    - Strip leading/trailing whitespace
    - Collapse multiple spaces
    """
    if not isinstance(title, str):
        return ""

    # Remove year pattern: (YYYY)
    title = re.sub(r'\s*\(\d{4}\)\s*', ' ', title)

    # Convert to lowercase
    title = title.lower()

    # Remove punctuation (keep spaces)
    title = ''.join(c if c.isalnum() or c.isspace() else '' for c in title)

    # Collapse multiple spaces
    title = ' '.join(title.split())

    return title


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    Returns a value in [-1, 1], typically [0, 1] for rating vectors.
    """
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)

    # Handle empty/zero vectors
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))


def load_movielens_data(ml_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load MovieLens movies and ratings.

    Returns:
        movies_df: movieId, title, genres
        ratings_df: userId, movieId, rating, timestamp
    """
    movies_path = ml_dir / "movies.csv"
    ratings_path = ml_dir / "ratings.csv"

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    return movies, ratings


def bridge_titles(
    ratings_324: pd.DataFrame,
    movies_ml: pd.DataFrame,
) -> pd.DataFrame:
    """
    Bridge TMDB titles from 324 ratings to MovieLens movieIds via title matching.

    Args:
        ratings_324: DataFrame with 'title' column (Temilola's 324 ratings)
        movies_ml: MovieLens movies.csv DataFrame with 'title' and 'movieId'

    Returns:
        DataFrame with columns: tmdb_title, movielens_id (may have NaN for unmatched)
    """
    # Normalize both sets of titles
    ratings_324 = ratings_324.copy()
    ratings_324['norm_title'] = ratings_324['title'].apply(normalize_title)

    movies_ml = movies_ml.copy()
    movies_ml['norm_title'] = movies_ml['title'].apply(normalize_title)

    # Create lookup: normalized MovieLens title → movieId
    ml_lookup = dict(zip(movies_ml['norm_title'], movies_ml['movieId']))

    # Match each 324 rating to MovieLens
    ratings_324['movielens_id'] = ratings_324['norm_title'].map(ml_lookup)

    return ratings_324[['title', 'movielens_id']]


def build_rating_vectors(
    ratings_df: pd.DataFrame,
    movie_ids: np.ndarray,
) -> Dict[int, np.ndarray]:
    """
    Build sparse rating vectors for each user.

    Args:
        ratings_df: DataFrame with userId, movieId, rating
        movie_ids: Array of unique movieIds (for indexing)

    Returns:
        Dict mapping userId → rating vector (indexed by position in movie_ids)
    """
    movie_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}

    users_vecs = {}
    for user_id, group in ratings_df.groupby('userId'):
        vec = np.zeros(len(movie_ids), dtype=np.float32)
        for _, row in group.iterrows():
            if row['movieId'] in movie_to_idx:
                idx = movie_to_idx[row['movieId']]
                vec[idx] = row['rating']
        users_vecs[user_id] = vec

    return users_vecs


def select_top_k_users(
    temilola_vec: np.ndarray,
    users_vecs: Dict[int, np.ndarray],
    k: int = 200,
) -> list:
    """
    Select top-K users with highest cosine similarity to Temilola's rating vector.

    Args:
        temilola_vec: Temilola's rating vector
        users_vecs: Dict of user_id → rating vector
        k: Number of top users to select

    Returns:
        List of top-K user IDs, sorted by similarity descending
    """
    similarities = []
    for user_id, user_vec in users_vecs.items():
        sim = cosine_similarity(temilola_vec, user_vec)
        similarities.append((user_id, sim))

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top K user IDs
    top_k = [user_id for user_id, _ in similarities[:k]]
    return top_k


def main():
    """Main pipeline: download MovieLens, bridge titles, select twin users, emit parquet."""
    print("=" * 80)
    print("Task 1.7: MovieLens Twin Dataset Generation")
    print("=" * 80)

    # Paths
    ml_dir = PIPELINE3_DIR / "data" / "ml-latest-small"
    artifacts_dir = PIPELINE3_DIR / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # Step 1: Load MovieLens data
    print("\n[1] Loading MovieLens data...")
    movies_ml, ratings_ml = load_movielens_data(ml_dir)
    print(f"  - MovieLens: {len(movies_ml)} movies, {len(ratings_ml)} ratings")
    print(f"  - Users in ML: {ratings_ml['userId'].nunique()}")

    # Step 2: Load Temilola's 324 ratings
    print("\n[2] Loading Temilola's 324 ratings...")
    ratings_324 = load_324_ratings()
    print(f"  - 324 ratings loaded with {len(ratings_324)} rows")
    print(f"  - Columns: {ratings_324.columns.tolist()}")

    # Step 3: Bridge TMDB titles to MovieLens IDs
    print("\n[3] Bridging TMDB titles to MovieLens IDs...")
    bridged = bridge_titles(ratings_324, movies_ml)
    n_bridged = bridged['movielens_id'].notna().sum()
    print(f"  - Successfully bridged {n_bridged}/{len(ratings_324)} titles")
    if n_bridged < 10:
        print(f"  WARNING: Low bridge count ({n_bridged}) — twin may not be well-grounded in your taste")

    # Step 4: Build rating vectors restricted to bridged movies
    print("\n[4] Building rating vectors (restricted to bridged movies)...")
    bridged_movie_ids = bridged[bridged['movielens_id'].notna()]['movielens_id'].unique()
    print(f"  - {len(bridged_movie_ids)} unique bridged movies")

    # Filter MovieLens ratings to bridged movies only
    ratings_ml_bridged = ratings_ml[ratings_ml['movieId'].isin(bridged_movie_ids)].copy()
    print(f"  - MovieLens ratings restricted to bridged: {len(ratings_ml_bridged)} rows")

    # Build vector for Temilola (restricted to bridged)
    movie_to_idx = {mid: idx for idx, mid in enumerate(sorted(bridged_movie_ids))}
    temilola_vec = np.zeros(len(bridged_movie_ids), dtype=np.float32)
    for idx, row in ratings_324.iterrows():
        ml_id = bridged.iloc[idx]['movielens_id']
        if pd.notna(ml_id):
            vec_idx = movie_to_idx[ml_id]
            # Normalize rating to 1-5 scale (0.5-5.0 in MovieLens → round)
            rating = round(row['rating'])  # Temilola's ratings are already ~1-5 range
            temilola_vec[vec_idx] = rating

    print(f"  - Temilola's vector: {np.count_nonzero(temilola_vec)}/{len(temilola_vec)} non-zero")

    # Build vectors for all MovieLens users (restricted to bridged)
    users_vecs = build_rating_vectors(ratings_ml_bridged, sorted(bridged_movie_ids))
    print(f"  - MovieLens users with bridged ratings: {len(users_vecs)}")

    # Step 5: Select top-K users by cosine similarity
    print("\n[5] Selecting top-K users by cosine similarity...")
    K = 200
    top_k_users = select_top_k_users(temilola_vec, users_vecs, k=K)
    print(f"  - Top {K} users selected")

    # Step 6: Emit full rating vectors of selected users (ALL movies, not restricted)
    print("\n[6] Building output: full rating vectors of twin users (all ML movies)...")
    output_rows = []

    for user_id in tqdm(top_k_users, desc="Collecting twin user ratings"):
        user_ratings = ratings_ml[ratings_ml['userId'] == user_id].copy()
        for _, row in user_ratings.iterrows():
            ml_movie_id = int(row['movieId'])
            # Normalize rating: 0.5-5.0 → round to 1-5
            rating = round(row['rating'])
            # Get title from movies_ml
            movie_title = movies_ml[movies_ml['movieId'] == ml_movie_id]['title'].values
            if len(movie_title) > 0:
                title = movie_title[0]
                output_rows.append({
                    'user_id': user_id,
                    'movielens_movie_id': ml_movie_id,
                    'rating': rating,
                    'title': title,
                })

    output_df = pd.DataFrame(output_rows)
    print(f"  - Output: {len(output_df)} rows")
    print(f"  - Unique movies: {output_df['movielens_movie_id'].nunique()}")
    print(f"  - Unique users: {output_df['user_id'].nunique()}")

    # Save to parquet
    parquet_path = artifacts_dir / "movielens_twin_ratings.parquet"
    output_df.to_parquet(parquet_path, index=False)
    print(f"  - Saved to {parquet_path}")

    # Step 7: Verify and print summary
    print("\n[7] Summary:")
    print(f"  - n_users_selected: {output_df['user_id'].nunique()}")
    print(f"  - n_ratings_total: {len(output_df)}")
    print(f"  - n_unique_movies: {output_df['movielens_movie_id'].nunique()}")
    print(f"  - n_bridged_titles: {n_bridged}")
    print(f"  - min_ratings_per_user: {output_df.groupby('user_id').size().min()}")
    print(f"  - max_ratings_per_user: {output_df.groupby('user_id').size().max()}")
    print(f"  - mean_ratings_per_user: {output_df.groupby('user_id').size().mean():.1f}")

    # Verify constraint: ≥ 50K rows
    if len(output_df) >= 50000:
        print(f"  ✓ Meets constraint: {len(output_df)} >= 50000")
    else:
        print(f"  ⚠ WARNING: Only {len(output_df)} rows (target ≥50K)")

    # Verify disjointness from 82 movies
    temilola_movie_ids = set(bridged[bridged['movielens_id'].notna()]['movielens_id'].values)
    twin_movie_ids = set(output_df['movielens_movie_id'].values)
    overlap = len(temilola_movie_ids & twin_movie_ids)
    print(f"  - Overlap with 82-movie catalog: {overlap} movies")
    print(f"  - Truly new movies from twin: {len(twin_movie_ids - temilola_movie_ids)}")

    print("\n" + "=" * 80)
    print("Task 1.7 complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
