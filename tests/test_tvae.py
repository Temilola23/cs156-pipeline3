"""Tests for Tabular VAE (from-scratch PyTorch implementation)."""
import torch
import numpy as np
import pytest
from src.tvae import TVAE


def test_tvae_forward_shapes():
    """Test that forward pass returns correct shapes: (x_recon, mu, log_var)."""
    d = 10
    batch_size = 8
    tvae = TVAE(d=d, h=32, z_dim=4)

    x = torch.randn(batch_size, d)
    x_recon, mu, log_var = tvae.forward(x)

    assert x_recon.shape == (batch_size, d), f"Expected recon shape {(batch_size, d)}, got {x_recon.shape}"
    assert mu.shape == (batch_size, 4), f"Expected mu shape {(batch_size, 4)}, got {mu.shape}"
    assert log_var.shape == (batch_size, 4), f"Expected log_var shape {(batch_size, 4)}, got {log_var.shape}"


def test_tvae_sample_shape():
    """Test that sample(n) returns (n, d) tensor."""
    d = 10
    n_samples = 100
    tvae = TVAE(d=d, h=32, z_dim=4)

    samples = tvae.sample(n_samples)

    assert samples.shape == (n_samples, d), f"Expected sample shape {(n_samples, d)}, got {samples.shape}"


def test_tvae_loss_decreases():
    """Train for 200 epochs on synthetic data; verify final loss < initial loss."""
    torch.manual_seed(42)
    np.random.seed(42)

    d = 10
    batch_size = 16
    tvae = TVAE(d=d, h=32, z_dim=4)
    optimizer = torch.optim.Adam(tvae.parameters(), lr=1e-3)

    # Create synthetic data for this test
    x_synth = torch.randn(128, d)

    # Compute initial loss
    tvae.eval()
    with torch.no_grad():
        x_recon, mu, log_var = tvae.forward(x_synth[:batch_size])
        initial_loss = tvae.elbo_loss(x_synth[:batch_size], x_recon, mu, log_var, beta=1.0)

    # Train for 200 epochs
    tvae.train()
    for epoch in range(200):
        for i in range(0, len(x_synth), batch_size):
            x_batch = x_synth[i:i+batch_size]
            optimizer.zero_grad()
            x_recon, mu, log_var = tvae.forward(x_batch)
            loss = tvae.elbo_loss(x_batch, x_recon, mu, log_var, beta=1.0)
            loss.backward()
            optimizer.step()

    # Compute final loss
    tvae.eval()
    with torch.no_grad():
        x_recon, mu, log_var = tvae.forward(x_synth[:batch_size])
        final_loss = tvae.elbo_loss(x_synth[:batch_size], x_recon, mu, log_var, beta=1.0)

    assert final_loss < initial_loss, f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
