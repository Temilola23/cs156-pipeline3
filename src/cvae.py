"""
Conditional Variational Autoencoder (from-scratch PyTorch implementation).

A 2-layer encoder/decoder with Gaussian latent variable and ELBO loss.
Both encoder and decoder are conditioned on a discrete class label.

Architecture:
  - Conditional encoder: [x; c_onehot] → MLP → (μ, log_var in R^z_dim)
  - Reparameterize: z = μ + σ * ε, where σ = exp(0.5 * log_var), ε ~ N(0, I)
  - Conditional decoder: [z; c_onehot] → MLP → d
  - Loss: MSE(x, x_recon) + β * KL(q(z|x,c) || p(z))
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder.

    Parameters
    ----------
    d : int
        Input dimension (number of features).
    n_cond : int
        Number of condition classes (one-hot encoded).
    h : int, default=32
        Hidden dimension (both encoder and decoder).
    z_dim : int, default=4
        Latent dimension.
    """

    def __init__(self, d: int, n_cond: int, h: int = 32, z_dim: int = 4):
        super().__init__()
        self.d = d
        self.n_cond = n_cond
        self.h = h
        self.z_dim = z_dim

        # Encoder: [d + n_cond] -> h -> h -> z_dim (two heads for mu and log_var)
        encoder_input_dim = d + n_cond
        self.encoder_fc1 = nn.Linear(encoder_input_dim, h)
        self.encoder_fc2 = nn.Linear(h, h)
        self.encoder_mu = nn.Linear(h, z_dim)
        self.encoder_log_var = nn.Linear(h, z_dim)

        # Decoder: [z_dim + n_cond] -> h -> h -> d
        decoder_input_dim = z_dim + n_cond
        self.decoder_fc1 = nn.Linear(decoder_input_dim, h)
        self.decoder_fc2 = nn.Linear(h, h)
        self.decoder_out = nn.Linear(h, d)

    def encode(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode [x, c] to latent distribution parameters.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, d).
        c : torch.Tensor
            Condition (one-hot encoded) of shape (batch_size, n_cond).

        Returns
        -------
        mu : torch.Tensor
            Mean of q(z|x,c), shape (batch_size, z_dim).
        log_var : torch.Tensor
            Log-variance of q(z|x,c), shape (batch_size, z_dim).
        """
        xc = torch.cat([x, c], dim=1)
        h = F.relu(self.encoder_fc1(xc))
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

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Decode [z, c] to reconstruction.

        Parameters
        ----------
        z : torch.Tensor
            Latent variable, shape (batch_size, z_dim).
        c : torch.Tensor
            Condition (one-hot encoded), shape (batch_size, n_cond).

        Returns
        -------
        x_recon : torch.Tensor
            Reconstructed input, shape (batch_size, d).
        """
        zc = torch.cat([z, c], dim=1)
        h = F.relu(self.decoder_fc1(zc))
        h = F.relu(self.decoder_fc2(h))
        x_recon = self.decoder_out(h)
        return x_recon

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode [x, c], reparameterize, decode [z, c].

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, d).
        c : torch.Tensor
            Condition (one-hot encoded), shape (batch_size, n_cond).

        Returns
        -------
        x_recon : torch.Tensor
            Reconstructed input, shape (batch_size, d).
        mu : torch.Tensor
            Latent distribution mean, shape (batch_size, z_dim).
        log_var : torch.Tensor
            Latent distribution log-variance, shape (batch_size, z_dim).
        """
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, c)
        return x_recon, mu, log_var

    def sample_conditional(self, n: int, cond: torch.Tensor) -> torch.Tensor:
        """
        Generate n samples from the conditional prior p(x|c).

        Parameters
        ----------
        n : int
            Number of samples to generate.
        cond : torch.Tensor
            Condition (one-hot encoded), shape (n_cond,) or (1, n_cond).

        Returns
        -------
        samples : torch.Tensor
            Generated samples, shape (n, d).
        """
        device = next(self.parameters()).device

        # Ensure cond is the right shape
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)  # (1, n_cond)
        # Replicate condition for all n samples
        cond_batch = cond.repeat(n, 1).to(device)

        # Sample z from prior p(z) = N(0, I)
        z = torch.randn(n, self.z_dim, device=device)

        # Decode [z, c] to x
        with torch.no_grad():
            samples = self.decode(z, cond_batch)
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

        ELBO = E_q[log p(x|z,c)] - KL(q(z|x,c) || p(z))
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

        # KL divergence: KL(q(z|x,c) || N(0, I))
        # = 0.5 * sum_j (1 + log_var_j - μ_j^2 - exp(log_var_j))
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

        loss = recon_loss + beta * kl_loss
        return loss
