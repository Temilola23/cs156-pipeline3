"""
Heterogeneous Attention Network (HAN) from-scratch PyTorch implementation.

Graph schema:
  - Node types: movie, genre
  - Edges: movie-genre (m-g), genre-movie (g-m), movie-genre-movie (m-g-m, MGM meta-path)
  
Meta-paths considered:
  - MGM: movie → genre → movie (co-genre relation)
  - M: movie (self-attention on features, optional second view)
  
Node-level attention (per meta-path Φ):
  e_ij^Φ = LeakyReLU(a_Φ^T [W_Φ h_i || W_Φ h_j])
  α_ij^Φ = softmax_j(e_ij^Φ)
  z_i^Φ = σ(Σ_j α_ij^Φ W_Φ h_j)
  
Semantic-level attention (over meta-paths):
  w_Φ = (1/|V|) Σ_i q^T tanh(W z_i^Φ + b)
  β_Φ = softmax(w_Φ) across Φ
  z_i = Σ_Φ β_Φ z_i^Φ
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class HANNodeAttention(nn.Module):
    """Node-level attention for a single meta-path."""
    
    def __init__(self, in_dim: int, h: int):
        super().__init__()
        self.W = nn.Linear(in_dim, h)
        self.a = nn.Parameter(torch.randn(1, h))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute node-level attention and aggregate.
        
        Args:
            h: (n_nodes, in_dim) - node features
            adj: (n_nodes, n_nodes) - adjacency matrix (binary or weighted)
        
        Returns:
            z: (n_nodes, h) - aggregated embeddings
            alpha: (n_nodes, n_nodes) - attention weights
        """
        n = h.shape[0]
        # Transform features
        Wh = self.W(h)  # (n, h)
        
        # Compute attention logits: e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        # Broadcasting trick: reshape to compute all pairs
        a_input = Wh.unsqueeze(0) + Wh.unsqueeze(1)  # (n, n, h): h_i + h_j for all pairs
        e = self.leaky_relu(torch.matmul(a_input, self.a.t()))  # (n, n, 1)
        e = e.squeeze(-1)  # (n, n)
        
        # Mask by adjacency: only attend to neighbors (or all if adj is fully connected)
        # e[i, j] = -inf if adj[i, j] == 0
        e = e.masked_fill(adj == 0, float('-inf'))
        
        # Softmax over neighbors
        alpha = torch.softmax(e, dim=1)  # (n, n)
        # Handle NaN from -inf (nodes with no neighbors): replace with 0
        alpha = torch.nan_to_num(alpha, nan=0.0)
        
        # Aggregate: z = σ(Σ_j α_ij W h_j)
        z = torch.sigmoid(torch.matmul(alpha, Wh))  # (n, h)
        
        return z, alpha


class HANSemanticAttention(nn.Module):
    """Semantic-level attention over meta-paths."""
    
    def __init__(self, h: int, n_meta_paths: int = 2):
        super().__init__()
        self.q = nn.Parameter(torch.randn(h))
        self.W = nn.Linear(h, h)
        self.b = nn.Parameter(torch.randn(h))
        self.n_meta_paths = n_meta_paths
        nn.init.xavier_uniform_(self.q.unsqueeze(0))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.b)
    
    def forward(self, z_list: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute semantic-level attention and aggregate.
        
        Args:
            z_list: list of (n_nodes, h) tensors, one per meta-path
        
        Returns:
            z: (n_nodes, h) - final aggregated embeddings
            beta: (n_meta_paths,) - semantic attention weights
        """
        n_nodes = z_list[0].shape[0]
        n_paths = len(z_list)
        
        # Compute semantic importance per meta-path
        # w_Φ = (1/|V|) Σ_i q^T tanh(W z_i^Φ + b)
        w = []
        for z in z_list:
            # z: (n_nodes, h)
            tanh_input = self.W(z) + self.b  # (n_nodes, h)
            tanh_out = torch.tanh(tanh_input)  # (n_nodes, h)
            # q^T tanh(...): (n_nodes,)
            w_phi = torch.matmul(tanh_out, self.q)  # (n_nodes,)
            w_phi_mean = w_phi.mean()  # scalar
            w.append(w_phi_mean)
        
        w = torch.stack(w)  # (n_paths,)
        
        # Softmax over meta-paths
        beta = torch.softmax(w, dim=0)  # (n_paths,)
        
        # Aggregate: z = Σ_Φ β_Φ z^Φ
        z_final = torch.zeros_like(z_list[0])
        for i, z in enumerate(z_list):
            z_final = z_final + beta[i] * z
        
        return z_final, beta


class HAN(nn.Module):
    """
    Heterogeneous Attention Network from scratch.
    
    Predicts binary classification on movie nodes based on movie features and genre graph.
    """
    
    def __init__(self, movie_feat_dim: int, n_genres: int, h: int = 32, n_classes: int = 2):
        super().__init__()
        self.h = h
        self.movie_feat_dim = movie_feat_dim
        self.n_genres = n_genres
        self.n_classes = n_classes
        
        # Genre embeddings (learnable)
        self.genre_embed = nn.Embedding(n_genres, h)
        
        # Node-level attention for MGM meta-path
        self.mgm_attention = HANNodeAttention(movie_feat_dim, h)
        
        # Node-level attention for M meta-path (self-attention on features)
        self.m_attention = HANNodeAttention(movie_feat_dim, h)
        
        # Semantic-level attention over 2 meta-paths
        self.semantic_attention = HANSemanticAttention(h, n_meta_paths=2)
        
        # Classification head
        self.classifier = nn.Linear(h, n_classes)
        
        # Store intermediate values for inspection
        self.alpha_last = None
        self.beta_last = None
    
    def forward(self, movie_feat: torch.Tensor, mgm_adj: torch.Tensor, mg_adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            movie_feat: (n_movies, movie_feat_dim) - movie features
            mgm_adj: (n_movies, n_movies) - movie-genre-movie adjacency (co-genre graph)
            mg_adj: (n_movies, n_genres) - movie-genre bipartite adjacency
        
        Returns:
            embeddings: (n_movies, h) - final movie embeddings
            logits: (n_movies, n_classes) - classification logits
        """
        n_movies = movie_feat.shape[0]
        
        # Meta-path 1: MGM (movie-genre-movie)
        z_mgm, alpha_mgm = self.mgm_attention(movie_feat, mgm_adj)
        self.alpha_last = alpha_mgm
        
        # Meta-path 2: M (self, using movie features with self-adjacency)
        # Create self-adjacency (fully connected for self-attention)
        self_adj = torch.ones(n_movies, n_movies, device=movie_feat.device)
        z_m, _ = self.m_attention(movie_feat, self_adj)
        
        # Semantic-level attention
        z_final, beta = self.semantic_attention([z_mgm, z_m])
        self.beta_last = beta
        
        # Classification
        logits = self.classifier(z_final)
        
        return z_final, logits
