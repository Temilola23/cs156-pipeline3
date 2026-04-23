"""
Monte Carlo Dropout Bayesian Neural Network (Gal & Ghahramani 2016).

A standard 2-layer MLP with dropout kept active at inference time to
approximate Bayesian uncertainty via stochastic forward passes.

Architecture:
  Linear(d → h) → ReLU → Dropout(p) → Linear(h → h) → ReLU → Dropout(p) → Linear(h → 1)

Predictive distribution:
  p(y|x) ≈ (1/T) Σ_t f(x; W_t),  where W_t ~ dropout mask
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MCDropoutBNN(nn.Module):
    """
    Monte Carlo Dropout Bayesian Neural Network for regression.

    Parameters
    ----------
    d : int
        Input feature dimension.
    h : int, default=32
        Hidden layer dimension (both layers use same h).
    p : float, default=0.2
        Dropout probability. Applied to all ReLU activations.
    """

    def __init__(self, d: int, h: int = 32, p: float = 0.2):
        super().__init__()
        self.d = d
        self.h = h
        self.p = p

        # 2-layer MLP with dropout
        self.fc1 = nn.Linear(d, h)
        self.dropout1 = nn.Dropout(p=p)
        self.fc2 = nn.Linear(h, h)
        self.dropout2 = nn.Dropout(p=p)
        self.fc3 = nn.Linear(h, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dropout active (behavior depends on train/eval mode).

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (batch_size, d).

        Returns
        -------
        y : torch.Tensor
            Predicted ratings, shape (batch_size, 1).
        """
        h = F.relu(self.fc1(x))
        h = self.dropout1(h)
        h = F.relu(self.fc2(h))
        h = self.dropout2(h)
        y = self.fc3(h)
        return y

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        T: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate predictive mean and std via MC dropout (T stochastic forward passes).

        Strategy:
          1. Set model to train mode (to keep dropout active)
          2. For t = 1..T:
             - Forward pass on X → predictions (batch, 1)
          3. Compute mean: μ(x) = (1/T) Σ f(x; W_t)
          4. Compute var:  σ²(x) = (1/T) Σ f(x; W_t)² - μ(x)²
          5. Return mean, std = √σ²

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, d). Will be converted to torch.
        T : int, default=50
            Number of stochastic forward passes.

        Returns
        -------
        mean : np.ndarray
            Predicted mean, shape (n_samples,).
        std : np.ndarray
            Predicted std, shape (n_samples,).
        """
        # Convert to tensor
        X_tensor = torch.from_numpy(X).float()
        if next(self.parameters()).is_cuda:
            X_tensor = X_tensor.cuda()

        # Collect T forward passes
        predictions = []
        self.train()  # Keep dropout active!

        with torch.no_grad():
            for _ in range(T):
                y_t = self.forward(X_tensor)  # shape: (n_samples, 1)
                predictions.append(y_t.cpu().numpy())

        # Stack predictions: (T, n_samples, 1)
        predictions = np.concatenate(predictions, axis=1)  # (n_samples, T)

        # Compute mean and variance across T passes
        mean = predictions.mean(axis=1)  # (n_samples,)
        var = predictions.var(axis=1)    # (n_samples,)
        std = np.sqrt(var)               # (n_samples,)

        return mean, std
