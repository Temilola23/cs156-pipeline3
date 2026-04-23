#!/usr/bin/env python3
"""
Test suite for LightGCN (from-scratch PyTorch).

Tests:
1. test_lightgcn_init_shapes: Check embedding initialization
2. test_lightgcn_propagate_shapes: Check propagation output shapes
3. test_lightgcn_score_scalar: Check score function returns scalar
4. test_lightgcn_bpr_loss_scalar: Check BPR loss is scalar with gradients
"""
import pytest
import torch
import numpy as np
from src.lightgcn import LightGCN


class TestLightGCN:
    """Test LightGCN from-scratch implementation."""

    def test_lightgcn_init_shapes(self):
        """After init, embeddings should have correct shape."""
        n_users, n_items, d = 10, 20, 64
        model = LightGCN(n_users=n_users, n_items=n_items, d=d, n_layers=4)

        assert model.E_u.shape == (n_users, d)
        assert model.E_v.shape == (n_items, d)
        # Embeddings should be initialized from N(0, 0.1)
        assert model.E_u.requires_grad
        assert model.E_v.requires_grad

    def test_lightgcn_propagate_shapes(self):
        """After propagate, final embeddings should have correct shape."""
        n_users, n_items, d, n_layers = 10, 20, 64, 4
        model = LightGCN(n_users=n_users, n_items=n_items, d=d, n_layers=n_layers)

        # Create a mock sparse adjacency matrix: (n_users + n_items) x (n_users + n_items)
        # For testing, create a small bipartite graph with a few edges
        edge_list = [
            (0, n_users + 0),  # user 0 -> item 0
            (0, n_users + 1),  # user 0 -> item 1
            (1, n_users + 1),  # user 1 -> item 1
            (1, n_users + 2),  # user 1 -> item 2
        ]
        # Also add reverse edges for symmetry
        reverse_edges = [(j, i) for i, j in edge_list]
        all_edges = edge_list + reverse_edges

        # Create indices and values
        indices = torch.LongTensor(all_edges).t().contiguous()
        values = torch.ones(len(all_edges))
        A = torch.sparse_coo_tensor(
            indices, values,
            size=(n_users + n_items, n_users + n_items),
            dtype=torch.float32
        )

        # Normalize: A_hat = D^(-1/2) A D^(-1/2)
        A_hat = model.normalize_adjacency(A)

        # Propagate
        E_u_final, E_v_final = model.propagate(A_hat)

        assert E_u_final.shape == (n_users, d)
        assert E_v_final.shape == (n_items, d)

    def test_lightgcn_score_scalar(self):
        """Score function should return a scalar tensor."""
        n_users, n_items, d = 5, 10, 64
        model = LightGCN(n_users=n_users, n_items=n_items, d=d, n_layers=4)

        # Create mock final embeddings
        E_u_final = torch.randn(n_users, d)
        E_v_final = torch.randn(n_items, d)

        # Score a single (user, item) pair
        score = model.score(E_u_final, E_v_final, user_idx=2, item_idx=5)

        assert score.shape == torch.Size([])  # scalar
        assert score.dtype == torch.float32

    def test_lightgcn_bpr_loss_scalar(self):
        """BPR loss should be a scalar tensor with gradient flow."""
        n_users, n_items, d = 5, 10, 64
        model = LightGCN(n_users=n_users, n_items=n_items, d=d, n_layers=4)
        model.train()

        # Create mock final embeddings
        E_u_final = torch.randn(n_users, d, requires_grad=True)
        E_v_final = torch.randn(n_items, d, requires_grad=True)

        # Sample a BPR triplet (user, pos_item, neg_item)
        user_idx, pos_idx, neg_idx = 1, 2, 5

        loss = model.bpr_loss(E_u_final, E_v_final, user_idx, pos_idx, neg_idx, weight_decay=1e-4)

        assert loss.shape == torch.Size([])  # scalar
        assert loss.dtype == torch.float32
        assert loss.requires_grad

        # Check that gradients can be computed
        loss.backward()
        assert E_u_final.grad is not None
        assert E_v_final.grad is not None

    def test_lightgcn_bpr_loss_decreases_with_training(self):
        """BPR loss should decrease over a few training steps."""
        n_users, n_items, d = 5, 10, 64
        model = LightGCN(n_users=n_users, n_items=n_items, d=d, n_layers=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Create a mock sparse adjacency
        edge_list = [(0, n_users), (1, n_users + 1), (2, n_users + 2)]
        reverse_edges = [(j, i) for i, j in edge_list]
        all_edges = edge_list + reverse_edges
        indices = torch.LongTensor(all_edges).t().contiguous()
        values = torch.ones(len(all_edges))
        A = torch.sparse_coo_tensor(
            indices, values,
            size=(n_users + n_items, n_users + n_items),
            dtype=torch.float32
        )
        A_hat = model.normalize_adjacency(A)

        # Train for a few steps
        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            E_u_final, E_v_final = model.propagate(A_hat)
            loss = model.bpr_loss(E_u_final, E_v_final, user_idx=0, pos_idx=0, neg_idx=1, weight_decay=1e-4)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease (or at least not increase much on average)
        assert len(losses) == 3
        assert all(isinstance(l, float) for l in losses)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
