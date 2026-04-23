# Derivation: LightGCN Propagation

## Background

LightGCN (He et al., 2020) is a simplified Graph Convolutional Network for recommendation.
It removes nonlinearities and learnable feature transformations that complicate GNNs,
keeping only the graph structure's implicit collaborative signal.

## Notation

- $n_u$ = number of users
- $n_v$ = number of items (variants call this "products" or "vertices")
- $d$ = embedding dimension (typically 64)
- $K$ = number of propagation layers (typically 4)
- $\mathcal{G} = (\mathcal{U} \cup \mathcal{V}, \mathcal{E})$ = bipartite user-item interaction graph
- $A \in \mathbb{R}^{(n_u + n_v) \times (n_u + n_v)}$ = adjacency matrix (block form below)
- $\tilde{A} = D^{-1/2} A D^{-1/2}$ = symmetrically normalized adjacency
- $E^{(k)} \in \mathbb{R}^{(n_u + n_v) \times d}$ = stacked embeddings at layer $k$

## Adjacency Structure

Stack users and items into a single vector:
$$\begin{bmatrix}
E_u^{(k)} \\
E_v^{(k)}
\end{bmatrix}$$

where $E_u^{(k)} \in \mathbb{R}^{n_u \times d}$ and $E_v^{(k)} \in \mathbb{R}^{n_v \times d}$.

The bipartite adjacency is:
$$A = \begin{bmatrix}
0 & A_{uv} \\
A_{vu} & 0
\end{bmatrix}$$

where $A_{uv}$ is the $n_u \times n_v$ user-item interaction matrix (1 if $(u,v) \in \mathcal{E}$, 0 else).
By symmetry, $A_{vu} = A_{uv}^\top$.

Degree matrix: $D = \text{diag}(d_1, d_2, \ldots, d_{n_u + n_v})$ where $d_i = \sum_j A_{ij}$.

## Normalization: $\tilde{A} = D^{-1/2} A D^{-1/2}$

Each edge $(i, j)$ is re-weighted:
$$\tilde{A}_{ij} = \frac{A_{ij}}{\sqrt{d_i d_j}}$$

**Why?** This normalization is the symmetric form of Laplacian normalization from spectral graph theory.
It:
1. Accounts for graph irregularity (high-degree nodes would dominate without scaling)
2. Stabilizes eigenvalues near $\lambda \in [-1, 1]$, aiding iterative propagation
3. Preserves the spectral properties needed for effective message-passing

## Propagation: $E^{(k+1)} = \tilde{A} E^{(k)}$

Message-passing without weights or nonlinearities:
$$E^{(k+1)} = \tilde{A} E^{(k)}$$

Expanding for user embedding at layer $k+1$:
$$E_u^{(k+1)} = \tilde{A}_{u,:} E^{(k)} = \sum_{i=1}^{n_u + n_v} \tilde{A}_{u,i} e_i^{(k)}$$

This decomposes:
- User-to-user: $\sum_{u' \neq u} \tilde{A}_{u,u'} e_{u'}^{(k)}$ (almost always 0 for bipartite)
- User-to-item: $\sum_{v=1}^{n_v} \tilde{A}_{u, n_u+v} e_v^{(k)}$ (nonzero from interactions)

Specifically, if user $u$ interacted with items $v_1, \ldots, v_m$:
$$E_u^{(k+1)} = \sum_{j=1}^{m} \frac{1}{\sqrt{d_u d_{n_u+v_j}}} E_v^{(k)}_j$$

i.e., average the embeddings of interacted items, scaled by the inverse-squared-degree factor.

## Final Embedding: Mean Pooling

After $K$ propagation steps, we have embeddings at layers $0, 1, \ldots, K$.
The final embedding averages all layers:
$$E_u^* = \frac{1}{K+1} \sum_{k=0}^{K} E_u^{(k)}$$

**Intuition:** Early layers ($k=0,1$) are local neighborhoods (1-hop, 2-hop collaborators).
Later layers ($k=3,4$) aggregate further out. Mean-pooling captures multi-hop structure.

## Scoring and Loss

**Scoring:** Inner product of final embeddings:
$$\hat{s}(u, v) = E_u^* \cdot E_v^*$$

**BPR Loss:** For each triplet (user $u$, positive item $v^+$, negative item $v^-$):
$$L = -\log(\sigma(\hat{s}(u, v^+) - \hat{s}(u, v^-))) + \lambda \left( \|E_u^*\|^2 + \|E_{v^+}^*\|^2 + \|E_{v^-}^*\|^2 \right)$$

where:
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid
- The first term is the ranking loss: maximize gap between positive and negative
- The second term is $L_2$ regularization ($\lambda \approx 10^{-4}$)

## Key Insight: No Learnable Parameters

Unlike standard GCN, LightGCN has **no weight matrices** on edges or nodes.
All learnable parameters are the initial embeddings $E_u^{(0)}, E_v^{(0)}$.
The graph structure itself (via $\tilde{A}$) drives learning—purely relational.

This gives:
- Simplicity: fewer parameters ↔ less overfitting
- Efficiency: only matrix-vector multiplications (no dense weights)
- Interpretability: collaborative signal is explicit (neighborhood averaging)
