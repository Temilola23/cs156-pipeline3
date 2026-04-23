"""
Tests for Bayesian hierarchical one-way ANOVA (PyMC).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import pymc as pm

from src.hierarchical_anova import build_hierarchical_anova_model


class TestHierarchicalANOVA:
    """Test Bayesian hierarchical ANOVA model compilation and sampling."""

    @pytest.fixture
    def synthetic_data(self):
        """Create a small synthetic dataset (50 rows) for testing."""
        np.random.seed(42)
        n_rows = 50
        modalities = np.random.choice(['all', 'metadata', 'poster', 'synopsis'], size=n_rows)
        tmdb_ids = np.random.choice(range(1, 11), size=n_rows)  # 10 unique movies
        ratings = np.random.uniform(1, 5, size=n_rows)
        return pd.DataFrame({
            'tmdb_id': tmdb_ids,
            'rating': ratings,
            'modality': modalities
        })

    def test_model_compiles(self, synthetic_data):
        """Test that the model compiles on synthetic data and prior sampling works."""
        # Build model
        model, data_dict = build_hierarchical_anova_model(synthetic_data)

        # Assert model exists and has expected parameters
        assert model is not None
        assert data_dict is not None
        assert 'modality_idx' in data_dict
        assert 'movie_idx' in data_dict
        assert 'ratings' in data_dict

        # Draw prior samples (no observations)
        with model:
            # Prior sample using pm.sample_prior_predictive
            idata_prior = pm.sample_prior_predictive(random_seed=42)

        # Check that we got samples
        assert idata_prior is not None
        assert len(idata_prior.prior.dims) > 0, "Prior should have variables"

    def test_data_encoding(self, synthetic_data):
        """Test that modality and movie indices are correctly encoded."""
        model, data_dict = build_hierarchical_anova_model(synthetic_data)

        modality_idx = data_dict['modality_idx']
        movie_idx = data_dict['movie_idx']

        # Check shapes match
        assert len(modality_idx) == len(synthetic_data)
        assert len(movie_idx) == len(synthetic_data)

        # Check indices are in valid range
        assert np.min(modality_idx) >= 0
        assert np.max(modality_idx) <= 3  # 4 modalities: 0, 1, 2, 3
        assert np.min(movie_idx) >= 0

    def test_ratings_match(self, synthetic_data):
        """Test that ratings are correctly passed through."""
        model, data_dict = build_hierarchical_anova_model(synthetic_data)

        ratings_returned = data_dict['ratings']
        np.testing.assert_array_almost_equal(
            ratings_returned,
            synthetic_data['rating'].values
        )
