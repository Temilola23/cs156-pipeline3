"""
Monte Carlo Dropout Bayesian Neural Network (Gal & Ghahramani 2016).

A flexible MLP with dropout kept active at inference time to
approximate Bayesian uncertainty via stochastic forward passes.

Default Architecture (backward compatible):
  Linear(d → 32) → ReLU → Dropout(p) → Linear(32 → 32) → ReLU → Dropout(p) → Linear(32 → 1)

Custom Architecture (via hidden_layers list):
  Linear(d → h1) → ReLU → Dropout(p) → Linear(h1 → h2) → ReLU → Dropout(p) → ... → Linear(hN → 1)

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
        (Deprecated) Hidden layer dimension for 2-layer MLP.
        Ignored if hidden_layers is specified.
    hidden_layers : list[int], optional
        List of hidden layer dimensions. If provided, overrides h.
        Default [32] creates a 2-layer network (backward compatible).
    p : float, default=0.2
        Dropout probability. Applied to all ReLU activations.
    """

    def __init__(self, d: int, h: int = 32, hidden_layers: list[int] | None = None, p: float = 0.2):
        super().__init__()
        self.d = d
        self.p = p

        # Support both old (h=) and new (hidden_layers=) API
        if hidden_layers is None:
            hidden_layers = [h]

        self.hidden_layers = hidden_layers
        self.h = hidden_layers[0]  # For backward compat, expose first hidden layer

        # Build sequential network: input → [hidden layers with ReLU+Dropout] → output
        layers = []
        prev_dim = d

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=p))
            prev_dim = hidden_dim

        # Output layer (no dropout on output)
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

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
        return self.net(x)

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
