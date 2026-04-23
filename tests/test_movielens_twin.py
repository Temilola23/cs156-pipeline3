"""Tests for MovieLens twin dataset generation (Task 1.7)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add scripts to path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def test_cosine_similarity_correctness():
    """Cosine similarity of a vector with itself should be ~1.0, with orthogonal ~0."""
    # Import here so we can test locally first
    import importlib.util
    spec = importlib.util.spec_from_file_location("movielens_twin",
                                                   SCRIPTS_DIR / "14_movielens_twin.py")
    movielens_twin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(movielens_twin)
    cosine_similarity = movielens_twin.cosine_similarity

    # Self-similarity should be 1.0
    v = np.array([1.0, 2.0, 3.0])
    sim = cosine_similarity(v, v)
    assert abs(sim - 1.0) < 1e-9, f"Expected ~1.0, got {sim}"

    # Orthogonal vectors should have ~0 similarity
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    sim = cosine_similarity(v1, v2)
    assert abs(sim - 0.0) < 1e-9, f"Expected ~0.0, got {sim}"

    # Test with sparse vectors (containing zeros)
    v1 = np.array([3.0, 0.0, 4.0])
    v2 = np.array([3.0, 0.0, 4.0])
    sim = cosine_similarity(v1, v2)
    assert abs(sim - 1.0) < 1e-9, f"Expected ~1.0, got {sim}"


def test_title_normalization():
    """Title normalization should handle years, punctuation, and case."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("movielens_twin",
                                                   SCRIPTS_DIR / "14_movielens_twin.py")
    movielens_twin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(movielens_twin)
    normalize_title = movielens_twin.normalize_title

    # Same title with and without year
    title1 = "Toy Story (1995)"
    title2 = "toy story"
    assert normalize_title(title1) == normalize_title(title2)

    # Titles with punctuation
    title1 = "Mr. & Mrs. Smith"
    title2 = "Mr and Mrs Smith"
    norm1 = normalize_title(title1)
    norm2 = normalize_title(title2)
    # Both should normalize similarly (alphanumeric + spaces)
    assert norm1 != ""
    assert norm2 != ""

    # Case insensitivity
    title1 = "The Dark Knight"
    title2 = "the dark knight"
    assert normalize_title(title1) == normalize_title(title2)


def test_bridge_tmdb_to_movielens_integration():
    """Integration test: can we successfully bridge some TMDB → MovieLens titles?"""
    # This test requires MovieLens data to be downloaded
    # We'll skip if not available
    import importlib.util
    spec = importlib.util.spec_from_file_location("movielens_twin",
                                                   SCRIPTS_DIR / "14_movielens_twin.py")
    movielens_twin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(movielens_twin)
    load_movielens_data = movielens_twin.load_movielens_data
    bridge_titles = movielens_twin.bridge_titles

    ml_dir = Path(__file__).resolve().parent.parent / "data" / "ml-latest-small"
    if not (ml_dir / "movies.csv").exists():
        pytest.skip("MovieLens data not available")

    movies_ml, _ = load_movielens_data(ml_dir)

    # Create a minimal TMDB rating dataframe
    ratings_324 = pd.DataFrame({
        'title': ['Toy Story', 'Iron Man'],
        'rating': [8.5, 8.2]
    })

    bridged = bridge_titles(ratings_324, movies_ml)

    # Should have matched at least some titles
    assert bridged['movielens_id'].notna().sum() > 0, "Bridge failed to match any titles"
