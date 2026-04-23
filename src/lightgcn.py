#!/usr/bin/env python3
"""
LightGCN: From-scratch 4-layer Graph Convolutional Network for recommendation.

Reference: https://arxiv.org/abs/2002.02126 (He et al., 2020)

Architecture:
- Input: user & item embeddings E_u ∈ ℝ^(n_u × d), E_v ∈ ℝ^(n_v × d)
- Bipartite adjacency: A (user-item interactions, symmetrically normalized)
- Propagation: E^(k+1) = A_hat @ E^(k), no weights, no activations
- Output: mean-pool over K+1 layers (0, 1, ..., K)
- Scoring: s(u, v) = <E_u_final, E_v_final>
- Loss: BPR (Bayesian Personalized Ranking)

Key insight: Graph message passing without learnable weights or nonlinearities,
only structural signal from the bipartite graph.
"""
import torch
import torch.nn as nn
from typing import Tuple


class LightGCN(nn.Module):
    """
    LightGCN model for bipartite user-item graphs.

    Parameters
    ----------
    n_users : int
        Number of unique users
    n_items : int
        Number of unique items
    d : int, default=64
        Embedding dimension
    n_layers : int, default=4
        Number of propagation layers (K in the paper)
    """

    def __init__(self, n_users: int, n_items: int, d: int = 64, n_layers: int = 4):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.d = d
        self.n_layers = n_layers

        # Initialize user and item embeddings from N(0, 0.1)
        self.E_u = nn.Parameter(torch.empty(n_users, d))
        self.E_v = nn.Parameter(torch.empty(n_items, d))
        nn.init.normal_(self.E_u, mean=0.0, std=0.1)
        nn.init.normal_(self.E_v, mean=0.0, std=0.1)

    def normalize_adjacency(self, A: torch.sparse.FloatTensor) -> torch.sparse.FloatTensor:
        """
        Symmetrically normalize the adjacency matrix: A_hat = D^(-1/2) A D^(-1/2).

        Parameters
        ----------
        A : torch.sparse.FloatTensor
            Sparse COO tensor of shape (n_users + n_items, n_users + n_items)

        Returns
        -------
        A_hat : torch.sparse.FloatTensor
            Normalized sparse adjacency matrix
        """
        # Coalesce to handle duplicate indices
        A = A.coalesce()

        # Compute degree: D_i = sum_j A_ij
        indices = A.indices()
        values = A.values()
        size = A.size(0)

        # Degree calculation: sum values per row
        degree = torch.zeros(size, device=A.device, dtype=A.dtype)
        degree.scatter_add_(0, indices[0], values)

        # D^(-1/2): avoid division by zero
        degree_inv_sqrt = torch.where(
            degree > 0,
            1.0 / torch.sqrt(degree),
            torch.zeros_like(degree)
        )

        # D^(-1/2) @ A @ D^(-1/2)
        # For each edge (i, j) with value A_ij, replace with degree_inv_sqrt[i] * A_ij * degree_inv_sqrt[j]
        new_values = degree_inv_sqrt[indices[0]] * values * degree_inv_sqrt[indices[1]]

        A_hat = torch.sparse_coo_tensor(
            indices, new_values, size=(size, size), dtype=A.dtype, device=A.device
        )

        return A_hat

    def propagate(self, A_hat: torch.sparse.FloatTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward propagation: E^(k+1) = A_hat @ E^(k) for k = 0, 1, ..., n_layers.

        Returns the mean-pooled final embeddings across all layers.

        Parameters
        ----------
        A_hat : torch.sparse.FloatTensor
            Normalized sparse adjacency matrix of shape (n_users + n_items, n_users + n_items)

        Returns
        -------
        E_u_final : torch.Tensor
            Final user embeddings, shape (n_users, d)
        E_v_final : torch.Tensor
            Final item embeddings, shape (n_items, d)
        """
        # Stack embeddings into a single matrix: [E_u; E_v]
        E = torch.cat([self.E_u, self.E_v], dim=0)  # (n_users + n_items, d)

        # Store all layer embeddings for mean pooling
        E_layers = [E]

        # Propagate for n_layers steps
        for k in range(self.n_layers):
            # E^(k+1) = A_hat @ E^(k)
            # Sparse matrix multiplication
            E = torch.sparse.mm(A_hat, E)
            E_layers.append(E)

        # Mean pooling over all layers: E_final = mean(E^(0), E^(1), ..., E^(K))
        E_final = torch.stack(E_layers, dim=0).mean(dim=0)  # (n_users + n_items, d)

        # Split back into user and item embeddings
        E_u_final = E_final[: self.n_users, :]
        E_v_final = E_final[self.n_users :, :]

        return E_u_final, E_v_final

    def score(
        self, E_u_final: torch.Tensor, E_v_final: torch.Tensor, user_idx: int, item_idx: int
    ) -> torch.Tensor:
        """
        Compute the score s(u, v) = <E_u_final[u], E_v_final[v]>.

        Parameters
        ----------
        E_u_final : torch.Tensor
            Final user embeddings, shape (n_users, d)
        E_v_final : torch.Tensor
            Final item embeddings, shape (n_items, d)
        user_idx : int
            User index
        item_idx : int
            Item index

        Returns
        -------
        score : torch.Tensor
            Scalar dot product
        """
        return torch.dot(E_u_final[user_idx], E_v_final[item_idx])

    def bpr_loss(
        self,
        E_u_final: torch.Tensor,
        E_v_final: torch.Tensor,
        user_idx: int,
        pos_idx: int,
        neg_idx: int,
        weight_decay: float = 1e-4,
    ) -> torch.Tensor:
        """
        BPR (Bayesian Personalized Ranking) loss.

        L = -log(sigmoid(s(u, v+) - s(u, v-))) + λ (||E_u||^2 + ||E_v||^2)

        Parameters
        ----------
        E_u_final : torch.Tensor
            Final user embeddings, shape (n_users, d)
        E_v_final : torch.Tensor
            Final item embeddings, shape (n_items, d)
        user_idx : int
            User index
        pos_idx : int
            Positive item index
        neg_idx : int
            Negative item index
        weight_decay : float
            L2 regularization coefficient

        Returns
        -------
        loss : torch.Tensor
            Scalar BPR loss
        """
        # Compute scores
        s_pos = self.score(E_u_final, E_v_final, user_idx, pos_idx)
        s_neg = self.score(E_u_final, E_v_final, user_idx, neg_idx)

        # BPR objective: log sigmoid(s_pos - s_neg)
        score_diff = s_pos - s_neg
        bpr_loss = -torch.nn.functional.logsigmoid(score_diff)

        # L2 regularization on embeddings
        l2_loss = weight_decay * (
            torch.norm(E_u_final[user_idx]) ** 2 + torch.norm(E_v_final[pos_idx]) ** 2
            + torch.norm(E_v_final[neg_idx]) ** 2
        )

        return bpr_loss + l2_loss
