"""Tests for Conditional Variational Autoencoder (from-scratch PyTorch implementation)."""
import torch
import numpy as np
import pytest
from src.cvae import CVAE


def test_cvae_forward_shapes():
    """Test that forward pass returns correct shapes: (x_recon, mu, log_var) when conditioned."""
    d = 9
    n_cond = 5  # 5 rating bins
    batch_size = 8
    cvae = CVAE(d=d, n_cond=n_cond, h=32, z_dim=4)

    x = torch.randn(batch_size, d)
    # One-hot condition: batch of [1,0,0,0,0] (rating bin 0)
    cond = torch.zeros(batch_size, n_cond)
    cond[:, 0] = 1.0

    x_recon, mu, log_var = cvae.forward(x, cond)

    assert x_recon.shape == (batch_size, d), f"Expected recon shape {(batch_size, d)}, got {x_recon.shape}"
    assert mu.shape == (batch_size, 4), f"Expected mu shape {(batch_size, 4)}, got {mu.shape}"
    assert log_var.shape == (batch_size, 4), f"Expected log_var shape {(batch_size, 4)}, got {log_var.shape}"


def test_cvae_sample_conditional_shape():
    """Test that sample_conditional(n, cond) returns (n, d) tensor."""
    d = 9
    n_cond = 5
    n_samples = 100
    cvae = CVAE(d=d, n_cond=n_cond, h=32, z_dim=4)

    # One-hot condition: [0,0,1,0,0] (rating bin 2)
    cond = torch.zeros(n_cond)
    cond[2] = 1.0

    samples = cvae.sample_conditional(n_samples, cond)

    assert samples.shape == (n_samples, d), f"Expected sample shape {(n_samples, d)}, got {samples.shape}"


def test_cvae_different_conditions_different_samples():
    """Test that sampling with different conditions yields different mean samples."""
    torch.manual_seed(42)
    np.random.seed(42)

    d = 9
    n_cond = 5
    cvae = CVAE(d=d, n_cond=n_cond, h=32, z_dim=4)

    # Condition 1: [1,0,0,0,0] (rating bin 0)
    cond_1 = torch.zeros(n_cond)
    cond_1[0] = 1.0

    # Condition 2: [0,0,0,0,1] (rating bin 4)
    cond_5 = torch.zeros(n_cond)
    cond_5[4] = 1.0

    # Generate many samples with each condition
    with torch.no_grad():
        samples_1 = cvae.sample_conditional(1000, cond_1)
        samples_5 = cvae.sample_conditional(1000, cond_5)

    mean_1 = samples_1.mean(dim=0)
    mean_5 = samples_5.mean(dim=0)

    # The two conditions should produce different mean samples
    # (with high probability, even if the model is not well-trained)
    # Check that the difference is non-trivial (at least one component differs)
    diff = (mean_1 - mean_5).abs()
    assert (diff > 1e-4).any(), f"Conditions produced nearly identical means: {diff}"
