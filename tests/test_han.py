#!/usr/bin/env python3
"""
Test suite for HAN (Heterogeneous Attention Network, from-scratch PyTorch).

Tests:
1. test_han_forward_shapes: Check forward pass produces correct shapes
2. test_han_attention_sums_to_one: Check attention weights sum to 1 (softmax sanity)
"""
import pytest
import torch
import numpy as np
from src.han import HAN


class TestHAN:
    """Test HAN from-scratch implementation."""

    def test_han_forward_shapes(self):
        """Forward pass should return (n_movies, h) embeddings and (n_movies, 2) logits."""
        n_movies = 10
        n_genres = 5
        h = 32
        n_classes = 2
        movie_feat_dim = 5
        
        model = HAN(movie_feat_dim=movie_feat_dim, n_genres=n_genres, h=h, n_classes=n_classes)
        
        # Create sample movie features
        movie_feat = torch.randn(n_movies, movie_feat_dim)
        
        # Create dummy adjacency matrices (dense)
        # mgm_adj: n_movies x n_movies (co-genre graph)
        mgm_adj = torch.randint(0, 2, (n_movies, n_movies)).float()
        # mg_adj: n_movies x n_genres (movie-genre bipartite, optional)
        mg_adj = torch.randint(0, 2, (n_movies, n_genres)).float()
        
        embeddings, logits = model(movie_feat, mgm_adj, mg_adj)
        
        assert embeddings.shape == (n_movies, h), f"Expected embeddings shape {(n_movies, h)}, got {embeddings.shape}"
        assert logits.shape == (n_movies, n_classes), f"Expected logits shape {(n_movies, n_classes)}, got {logits.shape}"

    def test_han_attention_weights_sum_to_one(self):
        """Node-level attention weights (alpha) should sum to 1 for each node."""
        n_movies = 8
        n_genres = 4
        h = 32
        n_classes = 2
        movie_feat_dim = 5
        
        model = HAN(movie_feat_dim=movie_feat_dim, n_genres=n_genres, h=h, n_classes=n_classes)
        
        movie_feat = torch.randn(n_movies, movie_feat_dim)
        mgm_adj = torch.randint(0, 2, (n_movies, n_movies)).float()
        mg_adj = torch.randint(0, 2, (n_movies, n_genres)).float()
        
        # Call forward to trigger attention computation
        embeddings, logits = model(movie_feat, mgm_adj, mg_adj)
        
        # Access attention weights from model (should be stored during forward)
        # For each meta-path, alpha should sum to ~1 per row
        if hasattr(model, 'alpha_last'):
            alpha = model.alpha_last  # (n_movies, n_movies) for MGM path
            row_sums = alpha.sum(dim=1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
                f"Attention rows should sum to 1, got: {row_sums}"

    def test_han_semantic_attention_sums_to_one(self):
        """Semantic-level attention weights (beta) over meta-paths should sum to 1."""
        n_movies = 8
        n_genres = 4
        h = 32
        n_classes = 2
        movie_feat_dim = 5
        
        model = HAN(movie_feat_dim=movie_feat_dim, n_genres=n_genres, h=h, n_classes=n_classes)
        
        movie_feat = torch.randn(n_movies, movie_feat_dim)
        mgm_adj = torch.randint(0, 2, (n_movies, n_movies)).float()
        mg_adj = torch.randint(0, 2, (n_movies, n_genres)).float()
        
        embeddings, logits = model(movie_feat, mgm_adj, mg_adj)
        
        if hasattr(model, 'beta_last'):
            beta = model.beta_last  # shape: (n_meta_paths,)
            total = beta.sum()
            assert torch.allclose(total, torch.tensor(1.0), atol=1e-5), \
                f"Semantic attention weights should sum to 1, got: {total}"
