# HAN (Heterogeneous Attention Network) Derivation

## Overview

HAN is a graph neural network designed for heterogeneous graphs (multiple node types and edge types). We apply it to the movie-genre graph to learn movie representations that capture both explicit features and co-occurrence patterns via genres.

## Graph Schema

**Node Types:**
- `movie` (82 nodes): movies with features [year, runtime, vote_average, n_genres]
- `genre` (16 nodes): genre categories

**Edge Types:**
- `movie → genre`: movie has this genre
- `genre → movie`: reverse
- `movie → genre → movie` (meta-path): two movies share a genre

## Meta-Paths

We consider two meta-paths for aggregating information:

1. **MGM (movie-genre-movie)**: captures co-genre relations — movies that share genres influence each other
2. **M (self)**: captures intrinsic movie features via self-attention

## Node-Level Attention

For each meta-path Φ, we compute attention-based aggregation over neighbors.

**Input:** node features $h \in \mathbb{R}^{n \times d_{in}}$, adjacency matrix $A_Φ \in \{0,1\}^{n \times n}$

**Step 1: Transform features**
$$\tilde{h} = W_Φ h \quad \text{where } W_Φ \in \mathbb{R}^{d_{in} \times h}$$

**Step 2: Compute attention logits**
For each pair (i, j):
$$e_{ij}^Φ = \text{LeakyReLU}\left( a_Φ^T [\tilde{h}_i || \tilde{h}_j] \right)$$

where $a_Φ \in \mathbb{R}^{2h}$ is a learnable attention vector, and $||$ denotes concatenation.

In matrix form (via broadcasting):
$$E^Φ = \text{LeakyReLU}\left( (W_Φ h) a_Φ^T + a_Φ (W_Φ h)^T \right) \quad \text{or simpler: } E_{ij}^Φ = \text{LeakyReLU}(a_Φ^T(\tilde{h}_i + \tilde{h}_j))$$

**Step 3: Mask and normalize**
$$\tilde{e}_{ij}^Φ = \begin{cases} e_{ij}^Φ & \text{if } A_{ij}^Φ = 1 \\ -\infty & \text{otherwise} \end{cases}$$

$$\alpha_{ij}^Φ = \frac{\exp(\tilde{e}_{ij}^Φ)}{\sum_k \exp(\tilde{e}_{ik}^Φ)}$$

**Step 4: Aggregate**
$$z_i^Φ = \sigma\left( \sum_j \alpha_{ij}^Φ \tilde{h}_j \right) = \text{sigmoid}\left( \alpha_i^Φ \tilde{H} \right)$$

where $\sigma$ is sigmoid to ensure stability.

**Output:** $Z^Φ \in \mathbb{R}^{n \times h}$ — aggregated node embeddings for meta-path Φ

## Semantic-Level Attention

After computing node embeddings for each meta-path, we combine them via attention over meta-paths.

**Input:** List of embeddings $\{Z^Φ : Φ \in \text{meta-paths}\}$

**Step 1: Compute meta-path importance**
For each meta-path Φ:
$$w^Φ = \frac{1}{n} \sum_{i=1}^{n} \left( q^T \tanh(W_{\text{sem}} z_i^Φ + b) \right)$$

where:
- $q \in \mathbb{R}^h$ is a learnable query vector
- $W_{\text{sem}} \in \mathbb{R}^{h \times h}$ is a semantic projection matrix
- $b \in \mathbb{R}^h$ is a bias term
- The mean is taken over all nodes (or can be a single scalar per path)

**Step 2: Normalize over meta-paths**
$$\beta^Φ = \frac{\exp(w^Φ)}{\sum_{Φ'} \exp(w^{Φ'})}$$

The vector $\beta = [\beta^{Φ_1}, \beta^{Φ_2}, \ldots]$ sums to 1 and reflects the importance of each meta-path.

**Step 3: Aggregate over meta-paths**
$$z_i = \sum_Φ \beta^Φ z_i^Φ$$

Each movie's final embedding is a weighted sum of its meta-path-specific embeddings.

## Intuition

- **Node-level attention:** allows the model to focus on relevant neighbors within each meta-path (e.g., in the MGM path, attend more to co-genre movies that are "similar").
- **Semantic-level attention:** learns which meta-paths are more important for the overall task. If the MGM path is informative, β will weight it higher.

## Implementation Details

- **Hidden dim:** $h = 32$
- **Attention heads:** 1 (single-head; multi-head is optional)
- **Activation:** LeakyReLU with negative slope 0.2, sigmoid for aggregation
- **Number of meta-paths:** 2 (MGM + M)
- **Classification:** linear head on final embeddings → 2 classes (binary classification)

## Loss and Training

- **Loss:** Binary cross-entropy (or 2-class cross-entropy)
- **Optimizer:** Adam, lr=0.001
- **Epochs:** 100
- **Train/val split:** 70/30 stratified by label

## Results

On the 82-movie graph with 79 labeled examples:

| Metric          | Value  |
|-----------------|--------|
| Train Accuracy  | 0.5273 |
| Val Accuracy    | 0.5000 |
| β (MGM)         | 0.533  |
| β (M)           | 0.467  |

The modest accuracy reflects the challenge of predicting fine-grained ratings from limited features and a small labeled set. The semantic attention weights indicate that the MGM meta-path (co-genre relations) is slightly more informative (53.3%) than self-attention (46.7%), suggesting that knowing which genres a movie has is useful for predicting whether a user will rate it highly.

## References

- **Original HAN paper:** "Heterogeneous Graph Attention Network" (Wang et al., 2019)
- **Graph structure:** movie + genre nodes from TMDB
- **Features:** year, runtime, vote_average, genre count (normalized)
