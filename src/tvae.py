"""
Tabular VAE (from-scratch PyTorch implementation).

A 2-layer encoder/decoder with Gaussian latent variable and ELBO loss.
Architecture:
  - Encoder: d → h → h → (μ, log_var in R^z_dim)
  - Reparameterize: z = μ + σ * ε, where σ = exp(0.5 * log_var), ε ~ N(0, I)
  - Decoder: z_dim → h → h → d
  - Loss: MSE(x, x_recon) + β * KL(q(z|x) || p(z))
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class TVAE(nn.Module):
    """
    Tabular Variational Autoencoder.

    Parameters
    ----------
    d : int
        Input dimension (number of features).
    h : int, default=32
        Hidden dimension (both encoder and decoder).
    z_dim : int, default=4
        Latent dimension.
    """

    def __init__(self, d: int, h: int = 32, z_dim: int = 4):
        super().__init__()
        self.d = d
        self.h = h
        self.z_dim = z_dim

        # Encoder: d -> h -> h -> z_dim (two heads for mu and log_var)
        self.encoder_fc1 = nn.Linear(d, h)
        self.encoder_fc2 = nn.Linear(h, h)
        self.encoder_mu = nn.Linear(h, z_dim)
        self.encoder_log_var = nn.Linear(h, z_dim)

        # Decoder: z_dim -> h -> h -> d
        self.decoder_fc1 = nn.Linear(z_dim, h)
        self.decoder_fc2 = nn.Linear(h, h)
        self.decoder_out = nn.Linear(h, d)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode x to latent distribution parameters.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, d).

        Returns
        -------
        mu : torch.Tensor
            Mean of q(z|x), shape (batch_size, z_dim).
        log_var : torch.Tensor
            Log-variance of q(z|x), shape (batch_size, z_dim).
        """
        h = F.relu(self.encoder_fc1(x))
        h = F.relu(self.encoder_fc2(h))
        mu = self.encoder_mu(h)
        log_var = self.encoder_log_var(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε, ε ~ N(0, I).

        Parameters
        ----------
        mu : torch.Tensor
            Mean, shape (batch_size, z_dim).
        log_var : torch.Tensor
            Log-variance, shape (batch_size, z_dim).

        Returns
        -------
        z : torch.Tensor
            Sampled latent variable, shape (batch_size, z_dim).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent variable z to reconstruction.

        Parameters
        ----------
        z : torch.Tensor
            Latent variable, shape (batch_size, z_dim).

        Returns
        -------
        x_recon : torch.Tensor
            Reconstructed input, shape (batch_size, d).
        """
        h = F.relu(self.decoder_fc1(z))
        h = F.relu(self.decoder_fc2(h))
        x_recon = self.decoder_out(h)
        return x_recon

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode x, reparameterize, decode.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, d).

        Returns
        -------
        x_recon : torch.Tensor
            Reconstructed input, shape (batch_size, d).
        mu : torch.Tensor
            Latent distribution mean, shape (batch_size, z_dim).
        log_var : torch.Tensor
            Latent distribution log-variance, shape (batch_size, z_dim).
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def sample(self, n: int) -> torch.Tensor:
        """
        Generate n samples from the prior p(z) = N(0, I).

        Parameters
        ----------
        n : int
            Number of samples to generate.

        Returns
        -------
        samples : torch.Tensor
            Generated samples, shape (n, d).
        """
        z = torch.randn(n, self.z_dim, device=next(self.parameters()).device)
        samples = self.decode(z)
        return samples

    def elbo_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute ELBO loss: reconstruction + β * KL divergence.

        ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
        ≈ MSE(x, x_recon) + β * KL

        KL(q||p) = 0.5 * sum(1 + log_var - μ^2 - exp(log_var))

        Parameters
        ----------
        x : torch.Tensor
            Original input, shape (batch_size, d).
        x_recon : torch.Tensor
            Reconstructed input, shape (batch_size, d).
        mu : torch.Tensor
            Latent mean, shape (batch_size, z_dim).
        log_var : torch.Tensor
            Latent log-variance, shape (batch_size, z_dim).
        beta : float, default=1.0
            Weight on KL term (annealing parameter).

        Returns
        -------
        loss : torch.Tensor
            Scalar ELBO loss.
        """
        # Reconstruction loss: MSE
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")

        # KL divergence: KL(q(z|x) || N(0, I))
        # = 0.5 * sum_j (1 + log_var_j - μ_j^2 - exp(log_var_j))
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

        loss = recon_loss + beta * kl_loss
        return loss
