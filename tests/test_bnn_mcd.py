"""
Tests for MC Dropout Bayesian Neural Network (BNN).
"""
from __future__ import annotations
import numpy as np
import torch
import pytest

from src.bnn_mcd import MCDropoutBNN


class TestMCDropoutBNN:
    """Test MC Dropout BNN forward pass and uncertainty estimation."""

    @pytest.fixture
    def model(self):
        """Create a small BNN for testing."""
        d = 9  # Feature dimension (matching Task 1.8)
        return MCDropoutBNN(d=d, h=32, p=0.2)

    @pytest.fixture
    def X_batch(self):
        """Create dummy batch of features."""
        batch_size = 8
        d = 9
        return np.random.randn(batch_size, d).astype(np.float32)

    def test_bnn_forward_shape(self, model, X_batch):
        """Test that forward pass returns (batch, 1) for regression."""
        X_tensor = torch.from_numpy(X_batch)
        model.eval()
        with torch.no_grad():
            y_pred = model(X_tensor)

        assert y_pred.shape == (X_batch.shape[0], 1), \
            f"Expected shape ({X_batch.shape[0]}, 1), got {y_pred.shape}"

    def test_mc_uncertainty_shapes(self, model, X_batch):
        """Test that predict_with_uncertainty returns two (n,) arrays."""
        mean, std = model.predict_with_uncertainty(X_batch, T=10)

        assert isinstance(mean, np.ndarray), "mean should be numpy array"
        assert isinstance(std, np.ndarray), "std should be numpy array"
        assert mean.shape == (X_batch.shape[0],), \
            f"Expected mean shape ({X_batch.shape[0]},), got {mean.shape}"
        assert std.shape == (X_batch.shape[0],), \
            f"Expected std shape ({X_batch.shape[0]},), got {std.shape}"

    def test_mc_variance_nonzero(self, model, X_batch):
        """Test that dropout-induced variance is nonzero across T passes."""
        # Run multiple forward passes with dropout active
        mean, std = model.predict_with_uncertainty(X_batch, T=50)

        # With dropout, we should see nonzero variance across samples
        assert np.mean(std) > 0.01, \
            f"Expected nonzero uncertainty; got mean std = {np.mean(std)}"
        assert np.all(std >= 0), "Standard deviation should be non-negative"

    def test_mc_predictions_deterministic(self, model, X_batch):
        """Test that setting model to eval() reduces variance."""
        # In eval mode (no dropout), predictions should be more consistent
        model.eval()

        # Run inference multiple times
        predictions = []
        for _ in range(5):
            with torch.no_grad():
                X_tensor = torch.from_numpy(X_batch)
                y = model(X_tensor)
                predictions.append(y.numpy())

        # All predictions should be identical in eval mode
        for p in predictions[1:]:
            np.testing.assert_array_almost_equal(predictions[0], p, decimal=6)
