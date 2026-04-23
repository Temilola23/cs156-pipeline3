"""Shared data loaders for Pipeline 3."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # Pipeline 3 dir
RATINGS_PATH = ROOT / "data" / "modality_ratings.jsonl"
META_PATH = ROOT / "data" / "movies_meta.json"


def load_movie_meta() -> dict:
    """Return dict keyed by tmdb_id → metadata dict. Source file is a list."""
    items = json.loads(META_PATH.read_text())
    out = {}
    for it in items:
        tid = it.get('tmdb_id')
        if tid is not None:
            out[int(tid)] = it
    return out


def load_324_ratings() -> pd.DataFrame:
    """Load the 324 ratings, standardize column names, join year from meta."""
    rows = [json.loads(l) for l in RATINGS_PATH.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    assert len(df) == 324, f"Expected 324 ratings, got {len(df)}"
    df = df.rename(columns={'condition': 'modality', 'ts': 'timestamp'})
    # join year + genres from meta
    meta = load_movie_meta()
    df['year'] = df['tmdb_id'].map(lambda x: meta.get(int(x), {}).get('year') if pd.notna(x) else None)
    df['genres'] = df['tmdb_id'].map(lambda x: meta.get(int(x), {}).get('genres', []) if pd.notna(x) else [])
    df['runtime_min'] = df['tmdb_id'].map(lambda x: meta.get(int(x), {}).get('runtime_min') if pd.notna(x) else None)
    df['vote_average'] = df['tmdb_id'].map(lambda x: meta.get(int(x), {}).get('tmdb_vote_average') if pd.notna(x) else None)
    return df


def train_test_split_holdout(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42):
    """Random hold-out split on row index. Deterministic given seed."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[idx[:cut]].copy(), df.iloc[idx[cut:]].copy()
