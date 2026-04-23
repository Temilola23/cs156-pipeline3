# Pipeline 3: SOTA Recommender Architectures Research Report

**Date:** April 16, 2026
**Focus:** Two-Tower / Dual-Encoder, Sequential, Graph-Based, Contrastive, and Generative Recommender Systems
**Target:** Predict/follow Temilola's MovieLens ratings + generate "fan class" recommendations (e.g., GoT fans → Knight of the Seven Kingdoms)

---

## Executive Summary

This report surveys state-of-the-art recommender architectures designed to address Pipeline 1's "Madam Web embarrassment" (naive content similarity failure) and Pipeline 2's N=162 ceiling. The architectures below span modern ML paradigms:

1. **Retrieval-scale:** Two-Tower dual encoders, graph convolutions, random walk embeddings
2. **Ranking-precise:** Sequential transformers (SASRec, BERT4Rec), attention mechanisms, contrastive losses
3. **Generative:** Variational autoencoders, diffusion-style models, LLM-as-judge for open-ended rating inference
4. **Fan-class synthesis:** User clustering → prototype recommendation → generative expansion

**Key insight:** No single method will impress the rubric alone. Narrative coherence demands:
- **One deep novel method** (with first-principles math + from-scratch code)
- **Apples-to-apples baselines** (P1 → P2 → P3 comparison table)
- **Mathematical grounding** (derivations, loss landscapes, attention visualizations)
- **Data augmentation strategy** (MovieLens + TMDB + synthetic data for N boost)

---

## 1. Two-Tower / Dual-Encoder Recommenders

### 1.1 Architecture Overview

**Two-Tower models** (also called dual-encoder models) consist of two independent neural networks:
- **User Tower:** Encodes user profile features, history, and context into a dense embedding
- **Item Tower:** Encodes movie features (genres, cast, plot, poster, metadata) into a shared embedding space
- **Retrieval:** At inference, compute dot-product or cosine similarity between user and item embeddings to rank candidates

**Why this impresses the rubric:**
- Scalable to large candidate sets (millions of movies)
- Naturally integrates multimodal features (text, images)
- Production-grade architecture (used by Meta, Amazon, Snap)
- Mathematical elegance: shared embedding space, contrastive optimization

### 1.2 Mathematical Foundations

**Dot-product similarity:**
$$\text{score}(u, i) = \mathbf{u}^T \mathbf{v}_i$$

where $\mathbf{u} \in \mathbb{R}^d$ is the user embedding and $\mathbf{v}_i \in \mathbb{R}^d$ is the item embedding.

**Loss function (InfoNCE-style contrastive):**
$$\mathcal{L} = -\log \frac{\exp(\text{score}(u, i^+) / \tau)}{\sum_{j \in \{i^+\} \cup \mathcal{N}^-} \exp(\text{score}(u, j) / \tau)}$$

where $i^+$ is a positive item, $\mathcal{N}^-$ is a set of negative samples, and $\tau$ is temperature.

**Tower architectures:**
- Dense features: feed through MLP
- Sparse categorical features: learnable embeddings + dense projection
- Multimodal: concatenate vision (ResNet) + text (sentence-transformers) embeddings before final MLP

### 1.3 Implementation

**Library options:**
- TensorFlow `tf.keras` (TF-Recommenders)
- PyTorch with custom embedding layers
- HuggingFace Transformers for text encoder
- OpenAI CLIP for text-image alignment

**From-scratch feasibility:** Moderate (5–7 days)
- Write user tower (MLP over user features)
- Write item tower (MLP over item features + image encoder)
- Implement contrastive loss (InfoNCE)
- Negative sampling strategy (in-batch negatives, hard negatives)

### 1.4 Apples-to-Apples Baseline

Compare against:
- **P1 baseline:** TF-IDF + cosine (content-only)
- **P2 baseline:** SVR on sentence embeddings
- **P3 novel:** Two-Tower with both text + visual features, trained end-to-end with contrastive loss

### 1.5 7-Day Effort Estimate

- Day 1: Data preprocessing (merge MovieLens + TMDB features, create train/val/test splits)
- Days 2–3: Build user and item towers (dense projections, test shape compatibility)
- Days 4–5: Implement contrastive loss, negative sampling, training loop
- Days 6–7: Evaluation, visualization of learned embeddings (t-SNE), comparison tables

### 1.6 Key References

- [Red Hat: Understanding the two-tower model](https://developers.redhat.com/articles/2026/01/26/understanding-recommender-systems-two-tower-model)
- [Google Cloud: TensorFlow two towers architecture](https://cloud.google.com/blog/products/ai-machine-learning/scaling-deep-retrieval-tensorflow-two-towers-architecture)
- [Hopsworks: Two-Tower Embedding Model](https://www.hopsworks.ai/dictionary/two-tower-embedding-model)

---

## 2. Sequential Recommenders

### 2.1 Overview

Sequential models predict **the next movie a user will rate highly** given their history. Three major classes:

1. **SASRec** (Self-Attention Sequential Recommendation)
2. **BERT4Rec** (Bidirectional Encoder for Sequential Rec)
3. **Caser** (CNN-based sequential)
4. **GRU4Rec** (RNN baseline)

### 2.2 SASRec (Self-Attentive Sequential)

**Why it impresses the rubric:**
- Transformer architecture (not covered in class)
- Captures long-range temporal dependencies
- Unidirectional attention (left-to-right) models realistic sequential prediction
- Mathematical depth: scaled dot-product attention, position encoding

**Mathematical foundation:**

Multi-head self-attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $Q = X W_Q$, $K = X W_K$, $V = X W_V$, and $X$ is the sequence of item embeddings.

Position encoding (absolute):
$$\text{PE}(t, 2i) = \sin(t / 10000^{2i/d})$$
$$\text{PE}(t, 2i+1) = \cos(t / 10000^{2i/d})$$

**Network:**
- Embed each item in user's history
- Pass through multiple transformer blocks (multihead attention + FFN)
- At each position $t$, predict all items in positions $t+1, \ldots, T$ using only positions $\leq t$ (causal masking)
- Output: logits for next-item ranking

**Loss:** Cross-entropy on next item:
$$\mathcal{L} = -\sum_t \log P(\text{next item at } t | \text{history}_{<t})$$

### 2.3 BERT4Rec (Bidirectional Encoder)

**Key differences from SASRec:**
- Bi-directional attention (can attend to future items)
- Cloze task training: randomly mask items, predict masked items using context from both directions
- Often stronger on datasets with moderate sequence length

**Training objective:**
$$\mathcal{L} = -\sum_{i \in \text{masked}} \log P(\text{item}_i | \text{context} \setminus \text{item}_i)$$

### 2.4 Caser (CNN-based Sequential)

**Why it impresses:**
- Novel use of CNN for sequences (not typical RNN/Transformer)
- Captures both short-term (vertical conv) and long-term (horizontal conv) patterns
- Lightweight relative to transformers

**Architecture:**
- Represent last $L$ items as $L \times d$ embedding matrix
- Apply vertical filters (height=1, 3, 5, ...) → point-level patterns
- Apply horizontal filters (varying heights) → segment-level patterns
- Concatenate conv outputs + user embedding → fully-connected output

### 2.5 Implementation

**Library options:**
- Transformers4Rec (NVIDIA Merlin) — production-grade
- Recbole — includes SASRec, BERT4Rec, Caser implementations
- From-scratch: PyTorch (moderate, 6–8 days)

**From-scratch steps:**
1. Build item embedding layer
2. Implement multi-head self-attention blocks (or CNN blocks for Caser)
3. Add causal masking / Cloze task logic
4. Training loop with warm-up scheduler
5. Evaluation: NDCG@10, Recall@20, HitRate

### 2.6 Apples-to-Apples Baseline

- **P1/P2:** No temporal modeling
- **P3 baseline (simple):** Collaborative filtering with temporal order (last-N items weighted)
- **P3 novel:** SASRec or BERT4Rec with position encoding and attention visualization

### 2.7 7-Day Effort Estimate

- Day 1: Preprocess sequences (sort by date watched, handle sparse users)
- Days 2–3: Implement attention or CNN blocks
- Days 4–5: Training, hyperparameter tuning (num layers, attention heads, learning rate)
- Days 6–7: Evaluation metrics, attention visualization, comparison

### 2.8 Key References

- [BERT4Rec: Sequential Recommendation with Bidirectional Encoder (arxiv)](https://arxiv.org/pdf/1904.06690)
- [Deep Self-Attention for Sequential Recommendation (SASRec paper)](https://ksiresearch.org/seke/seke21paper/paper035.pdf)
- [Caser: Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding](https://jiaxit.github.io/resources/wsdm18caser.pdf)
- [Shaped: Beyond Static Preferences — Sequential Models in Recommendation Systems](https://www.shaped.ai/blog/beyond-static-preferences-understanding-sequential-models-in-recommendation-systems-n-gram-sasrec-bert4rec-beyond)
- [Transformers4Rec Documentation (NVIDIA Merlin)](https://nvidia-merlin.github.io/Transformers4Rec/stable/why_transformers4rec.html)

---

## 3. Graph Neural Network Recommenders

### 3.1 Overview

Graph-based recommenders model users, movies, and side information (actors, directors, genres) as nodes, with edges representing interactions (user-rated-movie) or relations (movie-has-actor). GNNs propagate embeddings across the graph to capture higher-order collaborative signals.

**Why this impresses:**
- Novel to class (GNNs not covered)
- Naturally models rich heterogeneous relationships
- Mathematically grounded in spectral and spatial graph theory
- Production scale: Pinterest (PinSAGE), Amazon, etc.

### 3.2 LightGCN (Lightweight Graph Convolution)

**Motivation:** Simplify NGCF by removing non-linearities; observation: linear aggregation is often sufficient.

**Layer-wise propagation:**
$$\mathbf{e}_u^{(k+1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}} \mathbf{e}_i^{(k)}$$

where $\mathcal{N}_u$ is the set of movies rated by user $u$, and $\mathbf{e}_i^{(k)}$ is the item embedding at layer $k$.

**Final embedding:**
$$\mathbf{e}_u^* = \frac{1}{K+1} \sum_{k=0}^{K} \mathbf{e}_u^{(k)}$$

Averaging layer outputs stabilizes training and mitigates overfitting.

**Loss (BPR pairwise):**
$$\mathcal{L} = -\sum_{(u,i,j)} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj})$$

where $\hat{y}_{ui} = \mathbf{e}_u^T \mathbf{e}_i$ is the predicted score.

### 3.3 NGCF (Neural Graph Collaborative Filtering)

**Precursor to LightGCN; includes non-linear activations:**
$$\mathbf{e}_u^{(k+1)} = \sigma\left(W_1^{(k)} \sum_{i \in \mathcal{N}_u} \mathbf{e}_i^{(k)} + W_2^{(k)} \mathbf{e}_u^{(k)}\right)$$

More expressive but slower and more prone to overfitting than LightGCN.

### 3.4 PinSAGE (Pinterest GNN for Recommendations)

**Key innovation:** Random walk sampling + aggregation

1. Sample neighbors via k-hop random walks (biased by visit frequency)
2. Aggregate neighbor features via LSTM
3. Output: item embeddings suitable for large-scale retrieval

**Why it works:** Accounts for item popularity and semantic similarity via random walk patterns.

### 3.5 Heterogeneous Graphs: HAN & HetGNN

**HAN (Heterogeneous Graph Attention Network):**

Models movies as heterogeneous graphs with nodes: User, Movie, Actor, Director, Genre.

Meta-path level attention aggregates different node types:
$$\mathbf{e}_u^{(\text{Actor})} = \text{Attention}_{\text{Actor}}(\{\mathbf{e}_{a_1}, \ldots, \mathbf{e}_{a_m}\})$$

Semantic-level attention combines meta-paths:
$$\mathbf{e}_u^* = \sum_{\text{meta-path} \in \mathcal{P}} \alpha_{\text{meta-path}} \mathbf{e}_u^{(\text{meta-path})}$$

where $\alpha_{\text{meta-path}}$ is learned via another attention layer.

**HetGNN:**

For each node, encodes attribute interactions of heterogeneous neighbors:
$$\mathbf{e}_u^{(\text{type } t)} = \text{LSTM}(\{\text{features of neighbors of type } t\})$$

Aggregates across types:
$$\mathbf{e}_u^* = \text{FC}([\mathbf{e}_u^{(1)}, \ldots, \mathbf{e}_u^{(T)}])$$

### 3.6 Graph Embedding Baselines: DeepWalk & node2vec

**DeepWalk:**
- Generate random walks on user-item graph
- Treat walks as sentences; apply Word2Vec (skip-gram) to learn node embeddings
- Simple, interpretable, works well as unsupervised pretraining

**node2vec:**
- Biased random walks (parameters $p$, $q$ control exploration vs. exploitation)
- Return parameter $p$: likelihood of immediately revisiting a node
- In-out parameter $q$: control breadth-first vs. depth-first exploration
- Often outperforms DeepWalk; more expensive to compute

### 3.7 Implementation

**Library options:**
- PyTorch Geometric (DGL, PyG) — standard
- Recbole — includes LightGCN, NGCF
- Gensim — DeepWalk, node2vec
- From-scratch: Moderate (8–10 days for GCN layers + sampling)

**From-scratch steps:**
1. Build user-item-movie bipartite graph (NetworkX or DGL)
2. Implement graph convolution (sparse matrix multiplication)
3. Positive/negative sampling strategy (importance sampling)
4. Mini-batch training with graph sampling
5. Evaluation on held-out interactions

### 3.8 Apples-to-Apples Baseline

- **P1/P2:** No graph structure
- **P3 baseline (simple):** Weighted average of neighbors (kNN cosine)
- **P3 novel:** LightGCN or HAN with multimodal item features + heterogeneous metadata

### 3.9 7-Day Effort Estimate

- Day 1: Build interaction graph, preprocess heterogeneous metadata
- Days 2–3: Implement graph sampling, mini-batch construction
- Days 4–5: GCN layers, aggregation, loss function
- Days 6–7: Training, node embedding visualization, comparison

### 3.10 Key References

- [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation (arxiv)](https://arxiv.org/pdf/2002.02126)
- [NGCF: Neural Graph Collaborative Filtering](https://arxiv.org/pdf/1905.08969)
- [PinSAGE: Graph Convolutional Neural Networks for Web-Scale Recommender Systems](https://arxiv.org/pdf/1806.01261)
- [HAN: Heterogeneous Graph Attention Network (arxiv)](https://arxiv.org/pdf/1903.07293)
- [HetGNN: Heterogeneous Graph Neural Network (KDD 2019)](https://arxiv.org/pdf/2008.05033)
- [DeepWalk and node2vec: Graph Embeddings (Towards Data Science)](https://towardsdatascience.com/exploring-graph-embeddings-deepwalk-and-node2vec-ee12c4c0d26d/)
- [Nature 2025: A novel recommender system using light graph convolutional network](https://www.nature.com/articles/s41598-025-99949-y)

---

## 4. Contrastive & Self-Supervised Learning

### 4.1 Overview

Contrastive methods learn embeddings by pulling positive pairs close and pushing negative pairs apart. Applied to recommendation: learn user/item embeddings such that users give high ratings to nearby items.

### 4.2 InfoNCE Loss (Noise Contrastive Estimation)

**Principle:** Maximize likelihood of positive sample relative to negatives.

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(\mathbf{u}, \mathbf{i}^+) / \tau)}{\exp(\text{sim}(\mathbf{u}, \mathbf{i}^+) / \tau) + \sum_{j \in \mathcal{N}^-} \exp(\text{sim}(\mathbf{u}, \mathbf{i}_j) / \tau)}$$

where $\tau$ is temperature (controls sharpness of the distribution), $\text{sim}(\cdot, \cdot)$ is cosine similarity or dot product, and $\mathcal{N}^-$ is a set of negative samples.

**Gradient flow:**
- Pushes $\mathbf{u}$ and $\mathbf{i}^+$ closer
- Pushes $\mathbf{u}$ away from all $\mathbf{i}_j^-$
- Temperature $\tau$ controls steepness: small $\tau$ → sharp decision boundary, large $\tau$ → soft

### 4.3 SimCLR for Movies

**Intuition:** Generate augmented views of a movie (e.g., crop poster, perturb features), learn embeddings such that different augmentations of the same movie cluster together.

**Two augmentation strategies:**
1. **Visual:** Random crops, color jitter, Gaussian blur on poster images
2. **Textual:** Synonym replacement, sentence reordering in plot summaries

**Training:**
- For each movie, create two augmented versions
- Pass through encoder $f(\cdot)$ → embeddings $\mathbf{z}_1, \mathbf{z}_2$
- Maximize $\text{sim}(\mathbf{z}_1, \mathbf{z}_2)$ while minimizing $\text{sim}(\mathbf{z}_1, \mathbf{z}_j)$ for other movies $j$

$$\mathcal{L}_{\text{SimCLR}} = -\log \frac{\exp(\text{sim}(\mathbf{z}_1, \mathbf{z}_2) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{k \neq i} \exp(\text{sim}(\mathbf{z}_1, \mathbf{z}_k) / \tau)}$$

where the sum is over all augmented views in the batch.

### 4.4 CLIP-Style Multi-Modal Contrastive Learning

**Goal:** Align text embeddings with image embeddings for movies.

**Architecture:**
- Text encoder: Transformer (BERT, GPT-style)
- Image encoder: Vision Transformer or ResNet
- Both map to shared embedding space of dimension $d$

**Training:**
- For each movie, encode plot summary (text) and poster (image)
- Maximize similarity between text and image of the same movie
- Minimize similarity with negative samples (other movies)

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(\mathbf{t}, \mathbf{v}) / \tau)}{\sum_{j=1}^N \exp(\text{sim}(\mathbf{t}, \mathbf{v}_j) / \tau)} - \log \frac{\exp(\text{sim}(\mathbf{v}, \mathbf{t}) / \tau)}{\sum_{j=1}^N \exp(\text{sim}(\mathbf{v}, \mathbf{t}_j) / \tau)}$$

### 4.5 S3-Rec & CL4SRec: Contrastive Learning for Sequences

**S3-Rec (Self-Supervised Sequential Recommendation):**

Derives self-supervision signals from data correlations (attributes, items, segments, sequences):
- Attribute-level: movies with similar genres should have nearby embeddings
- Item-level: movies co-rated by users should be close
- Segment-level: movies in the same "rating block" correlate
- Sequence-level: temporal proximity matters

Uses InfoNCE loss to maximize mutual information across these signals.

**CL4SRec (Contrastive Learning for Sequential Recommendation):**

Augments user sequences via:
- Random item cropping (remove a contiguous subsequence)
- Random item masking (set to a special token)
- Random item reordering

Maximizes consistency between original and augmented sequences:
$$\mathcal{L}_{\text{CL4SRec}} = -\log \frac{\exp(\text{sim}(\mathbf{u}_{\text{orig}}, \mathbf{u}_{\text{aug}}) / \tau)}{\sum_{k} \exp(\text{sim}(\mathbf{u}_{\text{orig}}, \mathbf{u}_k) / \tau)}$$

where $\mathbf{u}_{\text{orig}}$ and $\mathbf{u}_{\text{aug}}$ are embeddings of the original and augmented sequences.

### 4.6 Triplet Loss for Metric Learning

**Principle:** Learn a distance metric such that user $u$ rates movie $i$ higher than movie $j$.

**Triplet:** (anchor user $u$, positive item $i$ ∈ rated, negative item $j$ ∉ rated or lower-rated)

$$\mathcal{L}_{\text{triplet}} = \max(0, \alpha + d(\mathbf{u}, \mathbf{i}_j) - d(\mathbf{u}, \mathbf{i}))$$

where $\alpha$ is margin and $d(\cdot, \cdot)$ is Euclidean or cosine distance.

**Hard triplet mining:** Select triplets where negatives are closest to anchor (hardest to separate) → faster convergence.

### 4.7 Implementation

**Library options:**
- PyTorch + custom loss functions
- Sentence-Transformers (for text contrastive pre-training)
- OpenAI CLIP (for image-text alignment)
- HuggingFace (for transformer-based encoders)

**From-scratch feasibility:** High (4–6 days)
- Build dual encoders (text + image)
- Implement InfoNCE / SimCLR loss
- Negative sampling strategy (in-batch, hard negatives)
- Visualization: embedding space via t-SNE/UMAP

### 4.8 Apples-to-Apples Baseline

- **P1/P2:** No contrastive pre-training
- **P3 baseline:** Train embeddings via simple cosine similarity on fixed features
- **P3 novel:** CLIP-style or S3-Rec contrastive learning with learned augmentations

### 4.9 7-Day Effort Estimate

- Day 1: Data augmentation pipeline (image crops, text perturbations)
- Days 2–3: Dual encoder architectures (text + image)
- Days 4–5: InfoNCE / SimCLR loss, negative sampling
- Days 6–7: Pre-training, fine-tuning on user ratings, evaluation

### 4.10 Key References

- [Contrastive Representation Learning (Lil'Log)](https://lilianweng.github.io/posts/2021-05-31-contrastive/)
- [SimCLR: Tutorial on Contrastive Learning](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html)
- [InfoNCE Loss: Normalized Temperature Scaled Cross-Entropy (Medium)](https://medium.com/self-supervised-learning/nt-xent-loss-normalized-temperature-scaled-cross-entropy-loss-ea5a1ede7c40)
- [CLIP: Connecting Text and Images (OpenAI)](https://openai.com/index/clip/)
- [S³-Rec: Self-Supervised Learning for Sequential Recommendation (arxiv)](https://arxiv.org/abs/2008.07873)
- [CL4SRec: Contrastive Learning for Sequential Recommendation (arxiv)](https://arxiv.org/pdf/2404.10936)
- [Triplet Loss for Metric Learning (Wikipedia)](https://en.wikipedia.org/wiki/Triplet_loss)

---

## 5. Generative & Variational Models

### 5.1 Overview

Generative models learn a distribution over items conditioned on user history. Can generate uncertainty estimates, hallucinate candidate movies, or produce ranking distributions.

### 5.2 RecVAE (Variational Autoencoder for Recommendations)

**Why this impresses:**
- Generative framework (not seen in class)
- Handles implicit feedback naturally (user watched movie = 1, unobserved = 0)
- Probabilistic interpretation: uncertainty in ratings
- Mathematically grounded in variational inference

**Model:**

Input: user's binary rating vector $\mathbf{x}_u \in \{0,1\}^M$ (one-hot for each movie).

**Encoder (inference network):**
$$q(\mathbf{z} | \mathbf{x}_u) = \mathcal{N}(\mu_u, \sigma_u^2)$$

where $\mu_u = \text{MLP}_\mu(\mathbf{x}_u)$ and $\log \sigma_u^2 = \text{MLP}_\sigma(\mathbf{x}_u)$.

**Latent prior:**
$$p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$$

**Decoder (generative network):**
$$p(\mathbf{x}_u | \mathbf{z}) = \text{Multinomial}(\text{softmax}(\text{MLP}_{\text{dec}}(\mathbf{z})))$$

**ELBO (Evidence Lower Bound):**
$$\mathcal{L} = -\mathbb{E}_{q(\mathbf{z}|\mathbf{x}_u)}[\log p(\mathbf{x}_u|\mathbf{z})] + \beta \cdot \text{KL}(q(\mathbf{z}|\mathbf{x}_u) \| p(\mathbf{z}))$$

where $\beta$ is an annealing weight (starts low, increases during training to prevent posterior collapse).

**RecVAE innovations:**
1. Composite prior: mixture of Gaussians with learned means (capture user archetypes)
2. Annealing $\beta$: prevent latent code collapse
3. New stopping criterion: validation NDCG plateau

**At test time:** Sample $\mathbf{z} \sim q(\mathbf{z}|\mathbf{x}_u)$ and decode to get predicted rating distribution.

### 5.3 DLRM (Facebook's Deep Learning Recommendation Model)

**Architecture:** Handle dense + sparse features

- **Sparse features:** Categorical inputs (genre tags, actor IDs) → embeddings
- **Dense features:** Numeric inputs (runtime, budget, imdb_score) → MLP
- **Interaction:** Compute pairwise dot-products between embedding pairs
- **Output MLP:** Concatenate interaction features → final score

**Sparse embeddings to dense features interaction:**
$$\mathbf{z} = [\mathbf{x}_{\text{dense}}, \text{FM-interaction}(\{\mathbf{e}_{\text{sparse}_1}, \ldots, \mathbf{e}_{\text{sparse}_k}\})]$$

where FM-interaction computes all pairwise dot products:
$$\text{FM}(\mathbf{E}) = [\mathbf{e}_1 \cdot \mathbf{e}_2, \mathbf{e}_1 \cdot \mathbf{e}_3, \ldots, \mathbf{e}_{k-1} \cdot \mathbf{e}_k]$$

**Why it works:**
- Factorization machines reduce dimensionality of interactions
- Sparse feature handling is production-grade
- Scales to large catalogs (billions of items)

### 5.4 Bayesian Personalized Ranking (BPR)

**Motivation:** Only implicit feedback available (watched/not watched). No explicit ratings to regress. Optimize AUC instead.

**Bradley-Terry model foundation:**

Given user $u$ and two items $i$ (rated), $j$ (unobserved):
$$P(\text{user } u \text{ prefers } i \text{ to } j) = \frac{\exp(\hat{y}_{ui})}{exp(\hat{y}_{ui}) + \exp(\hat{y}_{uj})} = \sigma(\hat{y}_{ui} - \hat{y}_{uj})$$

where $\hat{y}_{ui} = \mathbf{u}^T \mathbf{v}_i$ is predicted score.

**BPR optimization criterion:**
$$\mathcal{L} = -\sum_{(u,i,j) \in D_s} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \lambda ||\theta||^2$$

where $D_s$ is the set of sampled triplets (user, positive, negative) and $\theta$ are model parameters.

**Gradient descent with bootstrap sampling:**
- Randomly sample triplets from training set
- Update embeddings via stochastic gradient descent
- Intuition: optimize ranking order, not absolute scores

### 5.5 LLM-as-Judge Recommender

**Wild-card approach (see Section 6.3 below).**

Use a frontier LLM (e.g., Claude 3.5 Sonnet) to generate rating predictions for Temilola given:
- Her historical ratings
- Movie metadata (plot, cast, genre, reviews)
- Zero-shot or few-shot prompting

**Example prompt:**
> "Temilola has rated [list of 20 movies + her ratings].
> Now, given that she enjoyed [highly-rated movie], predict her rating (1-5)
> for [candidate movie: plot summary, cast, genre].
> Justify your prediction based on her taste patterns."

**Advantages:**
- No training required (zero-shot)
- Natural language reasoning (interpretable)
- Can leverage LLM's world knowledge
- Can generate qualitative explanations

**Disadvantages:**
- Computationally expensive (API calls)
- May not align with her actual preferences
- Can be gamed (model may over-rely on clichés)

### 5.6 Implementation

**Library options:**
- PyTorch (VAE framework)
- Recbole (includes RecVAE)
- DLRM open-source (PyTorch/Caffe2)
- LLM APIs (OpenAI, Anthropic, local LLaMA)

**From-scratch feasibility:** Moderate to High (8–10 days for RecVAE or DLRM)

### 5.7 Apples-to-Apples Baseline

- **P1/P2:** Regression on point estimates
- **P3 baseline:** NeuMF (neural collaborative filtering) — simpler generative approach
- **P3 novel:** RecVAE with composite prior + annealing, or DLRM with sparse+dense interaction

### 5.8 7-Day Effort Estimate (RecVAE)

- Day 1: Prepare implicit feedback (binary rating matrix)
- Days 2–3: Encoder/decoder MLPs, reparameterization trick
- Days 4–5: ELBO loss, $\beta$ annealing schedule, training loop
- Days 6–7: Evaluation (NDCG, Recall), sampling at test time, visualization

### 5.9 Key References

- [RecVAE: A New Variational Autoencoder for Top-N Recommendations (arxiv)](https://arxiv.org/abs/1912.11160)
- [Variational Autoencoders for Collaborative Filtering (Netflix)](https://arxiv.org/pdf/1802.05814)
- [DLRM: Deep Learning Recommendation Model for Personalization (Facebook)](https://arxiv.org/pdf/1906.00091)
- [BPR: Bayesian Personalized Ranking from Implicit Feedback (arxiv)](https://arxiv.org/pdf/1205.2618)
- [Memory Assisted LLM for Personalized Recommendation System (arxiv)](https://arxiv.org/html/2505.03824v1)
- [Predicting Movie Hits Before They Happen with LLMs (UMAP '25)](https://arxiv.org/pdf/2505.02693)

---

## 6. Apples-to-Apples Baseline Comparison Table

| Method | Class Covered? | Novel Factor | Output Type | N Scale | Math Depth | From-Scratch Effort | Library Maturity |
|--------|---|---|---|---|---|---|---|
| **P1: TF-IDF + KMeans** | Yes | — | Cluster | Low | Low | Easy | High |
| **P1: OLS / Ridge** | Yes | — | Point estimate | Low | Low | Easy | High |
| **P2: SVR on embeddings** | Yes | — | Point estimate | Low | Low | Easy | High |
| **P2: Autoencoder (vanilla)** | Likely | Slight | Reconstruction | Low | Medium | Medium | High |
| **P3: Two-Tower dual-encoder** | No | High | Ranking/retrieval | **High** | **High** | Medium | High |
| **P3: SASRec** | No | High | Sequential prediction | **High** | **High** | Medium | High |
| **P3: LightGCN** | No | High | Collaborative signal | **High** | **High** | Medium | High |
| **P3: CLIP-style contrastive** | No | High | Joint embedding | **High** | **High** | Medium | Medium |
| **P3: RecVAE** | No | High | Distribution/ranking | **High** | **High** | High | Medium |
| **P3: BPR** | No | Medium | Ranking | Medium | **High** | Low | High |
| **P3: LLM-as-Judge** | No | **Wild** | Rating + explanation | **Very High** | Low (but novel) | Low (API) | Very High |

---

## 7. Fan-Class Generation: Clustering + Synthesis

### 7.1 The Idea

Temilola's wild idea: "fans of GoT tend to like Knight of the Seven Kingdoms... have a model generate the ideal movie for each subset of fans."

**Pipeline:**
1. **User clustering:** Partition MovieLens users into archetypes (e.g., "fantasy enthusiasts," "sci-fi purists," "romance fans")
2. **Archetype profiling:** For each cluster, compute centroid rating vector + typical rating patterns
3. **Movie generation:** Given a cluster centroid, generate or recommend the "ideal" movie for that archetype
4. **Validation:** Check whether generated movie aligns with actual preferences of that cluster

### 7.2 Clustering Algorithms

**K-means on user rating vectors:**
- Normalize ratings to [0,1]
- Compute centroids in rating space
- Silhouette / Davies-Bouldin index for optimal K

**Non-Negative Matrix Factorization (NMF):**
- Decompose user-movie rating matrix as $R = WH$
- $W$: user-archetype membership (soft clustering)
- $H$: archetype-movie preferences
- Interpretable: archetypes correspond to latent factors

**Bayesian clustering (Dirichlet Process Mixture Model):**
- Non-parametric: number of clusters inferred from data
- Produces soft cluster assignments (probabilities)

### 7.3 Archetype Centroid Representation

For cluster $c$:
$$\mathbf{c}_c = \frac{1}{|C_c|} \sum_{u \in C_c} \mathbf{r}_u$$

where $\mathbf{r}_u \in \mathbb{R}^M$ is user $u$'s rating vector (N=1682 movies in MovieLens 1M, for example).

**Alternatively, in latent space** (from NMF or VAE):
$$\mathbf{z}_c = \frac{1}{|C_c|} \sum_{u \in C_c} \mathbf{z}_u$$

### 7.4 Movie Generation / Recommendation

**Option A: Retrieval**
Given archetype centroid $\mathbf{c}_c$, rank movies by cosine similarity:
$$\text{score}(i) = \frac{\mathbf{c}_c \cdot \mathbf{v}_i}{\|\mathbf{c}_c\| \|\mathbf{v}_i\|}$$

Output top-10 movies as "ideal for cluster $c$."

**Option B: Generative (Advanced)**
If using a VAE-based model with continuous latent space:
- Encode archetype centroid $\mathbf{c}_c$ to latent $\mathbf{z}_c$
- Decode to get predicted rating distribution over all movies
- Sample or greedy-select top movie

**Option C: Synthetic Generation (Wildcard)**
Use a text-generation LLM to create a hypothetical movie description for the archetype:
- Prompt: "Design the ideal movie for fans of [archetype profile: top-rated genres, directors, themes]. Describe the movie in 1-2 sentences."
- Embed the generated description
- Find most-similar real movie in TMDB

### 7.5 Implementation

**Libraries:**
- scikit-learn: KMeans, NMF
- scipy: hierarchical clustering
- PyTorch: VAE-based clustering
- Gensim: topic modeling (alternative archetype discovery)

**7-Day effort:**
- Days 1–2: Preprocess MovieLens, normalize ratings, handle sparsity
- Days 3–4: Clustering algorithm, hyperparameter tuning (K, regularization)
- Days 5–6: Centroid representation, archetype profiling (genres, directors, themes per cluster)
- Day 7: Movie retrieval / generation, visualization (t-SNE of archetypes + top movies)

### 7.6 Validation

**Metric 1: Intra-cluster homogeneity**
- Users in same cluster rate movies similarly
- Silhouette score, Davies-Bouldin index

**Metric 2: Generated movie alignment**
- Have Temilola manually verify: "Would a GoT fan really like [generated movie]?"
- Compute overlap with ground-truth high-rated movies in cluster

### 7.7 Key References

- [Unsupervised Classification: Building a Movie Recommender with Clustering (Towards Data Science)](https://towardsdatascience.com/unsupervised-classification-project-building-a-movie-recommender-with-clustering-analysis-and-4bab0738efe6/)
- [MovieLens clustering analysis (GitHub)](https://github.com/SalmaHisham/Analysis-of-the-MovieLen-dataset)

---

## 8. Wild Card: LLM-as-Judge Recommender

### 8.1 Motivation

Pipeline 1 had the "Madam Web embarrassment." A frontier LLM has semantic understanding that simple embeddings lack.

**Hypothesis:** A large language model, given Temilola's historical ratings and a movie's metadata, can predict her rating better than a learned model (especially with small N=162).

### 8.2 Architecture

1. **Context window:** Temilola's 20 highest-rated and 10 lowest-rated movies (with her ratings)
2. **Candidate movie:** Plot summary, cast, director, genres, IMDB score
3. **Prompt:**
   > "Based on Temilola's movie taste, predict her rating (1-5 stars) for the following movie.
   >
   > Her highly-rated movies:
   > - [Movie 1]: 5 stars. Why she might have liked it: [extract theme]
   > - [Movie 2]: 5 stars. Why she might have liked it: [extract theme]
   > ...
   >
   > Her low-rated movies:
   > - [Movie A]: 1 star. Why she disliked it: [extract theme]
   > ...
   >
   > Candidate: [Movie title, plot, cast, director, genres]
   >
   > Prediction: [rating] stars. Reasoning: [short explanation]"

4. **Output:** Parse predicted rating from LLM response

### 8.3 Variants

**Zero-shot:** Use prompt as above, no fine-tuning

**Few-shot:** Include 5–10 examples of (context, movie, rating) to guide the LLM

**Fine-tuned:** Fine-tune a smaller LM (e.g., Mistral 7B) on Temilola's historical data (but ethical concerns: personal data)

**Ensemble:** Ask multiple LLMs, average their predictions

### 8.4 Evaluation

**Metrics:**
- MAE (mean absolute error) on held-out test set
- Spearman correlation with true ratings
- Qualitative: read explanations, assess coherence

**Comparison:**
- vs. Two-Tower (learned, data-efficient)
- vs. SASRec (learns sequential patterns)
- vs. LightGCN (learned graph signal)

### 8.5 Advantages & Disadvantages

**Advantages:**
- Zero/few-shot learning: leverage LLM world knowledge
- Interpretable: generates natural-language explanations
- Robust to small N: doesn't overfit like neural models might
- **Novel to class:** Prompting-based ML is not covered

**Disadvantages:**
- Expensive: API cost per prediction
- Slow: LLM inference is slower than forward pass
- Black-box: interpretations could be hallucinated
- Alignment risk: LLM might not capture Temilola's true taste

### 8.6 Implementation

**Libraries:**
- `openai` (ChatGPT, GPT-4)
- `anthropic` (Claude)
- `llama-cpp-python` (local models like Mistral)
- `langchain` (prompt engineering framework)

**Cost estimate:** ~1000 API calls × $0.001–$0.01 per call = $1–$10 (modest)

**7-Day effort:**
- Day 1: Gather Temilola's rated movies, extract themes
- Days 2–3: Prompt engineering (iterate on prompt template)
- Days 4–5: API calls for all test movies, parse responses, handle errors
- Days 6–7: Evaluation, error analysis, comparison with neural baselines

### 8.7 Key References

- [LLM-as-a-Judge: A Complete Guide (Evidently AI)](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)
- [Enhancing Rating Prediction with Off-the-Shelf LLMs Using In-Context User Reviews (arxiv)](https://arxiv.org/html/2510.00449)
- [Memory Assisted LLM for Personalized Recommendation System (arxiv)](https://arxiv.org/html/2505.03824v1)
- [Predicting Movie Hits Before They Happen with LLMs (UMAP '25)](https://arxiv.org/pdf/2505.02693)

---

## 9. Recommended Pipeline 3 Narrative Arc

### Phase 1: Foundation & Baselines
1. **Intro:** "Pipeline 1 recommended Madam Web. Pipeline 2 hit the N=162 ceiling. Pipeline 3: break through with modern recommender architectures."
2. **Data:** MovieLens 1M + augmented TMDB features (cast, crew, genres, plots)
3. **Preprocessing:** Build interaction matrix, feature engineering (embeddings, heterogeneous metadata)
4. **Analysis:** EDA on user/item distributions, sparsity, temporal patterns

### Phase 2: Novel Method Deep Dive
Choose **one as primary** (the novel, math-heavy implementation):

**Recommendation:** **LightGCN + Heterogeneous Metadata (HAN-style)**
- Combines graph collab. signal + semantic metadata
- Novel to class (GNNs)
- First-principles: spectral graph theory, neighborhood aggregation, attention
- From-scratch: implementable in 7 days
- Naturally handles "fan classes" via cluster-level embeddings

**OR: SASRec + CLIP-style pretraining**
- Sequential + contrastive learning
- Novel to class (Transformers, contrastive learning)
- Captures temporal user preference evolution + multimodal semantics
- Slightly more engineering-heavy

**OR: RecVAE + Archetype synthesis**
- Generative + clustering for fan classes
- Novel to class (VAE, Bayesian inference)
- Probabilistic interpretation, natural for "generate ideal movie"

### Phase 3: Apples-to-Apples Comparison
- **Baseline 1 (P1 benchmark):** Content cosine (TF-IDF features)
- **Baseline 2 (P2 benchmark):** SVR on sentence-transformers
- **Baseline 3 (simple ML):** NeuMF (neural CF, class-adjacent)
- **Baseline 4 (strong ML):** LightGCN (lightweight, interpretable)
- **Novel method:** [Your chosen method from Phase 2]
- **Wild card:** LLM-as-Judge (zero-shot)

**Comparison table:** Metrics (MAE, NDCG@10, Recall@20, training time), math complexity, implementation effort

### Phase 4: Results & Visualization
- **Embedding space:** t-SNE of learned user/item embeddings
- **Attention maps:** If using transformers (SASRec), visualize which past items matter for prediction
- **Fan classes:** Show discovered user archetypes, top movies per archetype
- **Error analysis:** Movies the model predicted poorly (new Madam Web embarrassments?)
- **Qualitative:** Show LLM-as-Judge explanations for 5–10 test cases

### Phase 5: Executive Summary & Discussion
- Did we beat P2's best?
- Which architecture shines (speed, accuracy, interpretability)?
- Limitations: still N=162 or did data augmentation help?
- Future work: fine-tuning on larger datasets, real-time feedback, multi-user recommendations

---

## 10. Data Augmentation Strategy

To break the N=162 ceiling:

### 10.1 MovieLens + TMDB Merge
- Download MovieLens 1M (1 million ratings, ~6K movies)
- Enrich with TMDB API: cast, directors, genres, runtime, budget, release year, plot summaries, poster URLs
- Filter to Temilola's watched movies (162) + additional popular movies she hasn't rated

### 10.2 Synthetic Data
**Option 1: User simulation**
- Cluster MovieLens users (or use Temilola's data)
- For each cluster, generate synthetic users with similar rating patterns
- Add Gaussian noise to simulate mood/context variation

**Option 2: Feature augmentation**
- Use sentence-transformers to embed plot summaries
- Use CLIP to embed posters
- Generate additional "virtual" movie embeddings via interpolation or mixing

### 10.3 Additional Data Sources
- **IMDB datasets:** Actor/director graphs, plot keywords
- **Rotten Tomatoes:** Critic scores (proxy for quality)
- **Box Office Mojo:** Budget, revenue (proxy for popularity)

### 10.4 Train/Val/Test Split
- Training: 80% of Temilola's 162 + synthetic data
- Validation: 10% of Temilola's 162 (held-out for hyperparameter tuning)
- Test: 10% of Temilola's 162 (final evaluation, never seen by model)
- External test (wild card): LLM-as-Judge predicts on test set independently

---

## 11. Final Recommendations for Pipeline 3

### Must-Have
1. **Novel architecture:** Pick one (Two-Tower, SASRec, LightGCN, RecVAE, CLIP, or BPR)
2. **From-scratch implementation:** Show the math, code, and results side-by-side with library version
3. **Apples-to-apples comparison:** Table of P1 + P2 + P3 methods
4. **First-principles derivations:** Loss functions, attention, graph convolutions, or variational inference
5. **Visualizations:** Embeddings, attention maps, fan classes, error analysis

### Should-Have
1. **Data augmentation:** Merge MovieLens 1M + TMDB, show N boost
2. **Fan-class discovery:** Cluster users, profile archetypes, recommend for each
3. **Sequential modeling:** Capture "Date Watched" temporal signal
4. **Multi-modal learning:** Text (plots, reviews) + images (posters)
5. **Uncertainty quantification:** VAE, Bayesian, or ensemble (show confidence intervals)

### Nice-to-Have (Wild Cards)
1. **LLM-as-Judge:** Zero-shot rating prediction with explanations
2. **Generative movie creation:** Use LLM to synthesize ideal movie for archetype, then find nearest real movie
3. **Interactive visualization:** Plotly/Streamlit app where Temilola can explore recommendations and archetypes
4. **Transfer learning:** Fine-tune sentence-transformers or vision models on movie domain
5. **Reinforcement learning:** Bandit algorithm for adaptive movie suggestions

### Avoid (Already Covered)
- Plain content-based similarity (P1)
- Regression on fixed features (P2)
- Vanilla neural networks without novel loss/architecture
- Over-relying on one library without from-scratch implementation

---

## 12. References & Resources

### Papers
- [Two-Tower Models](#12-references--resources) (Red Hat, Google Cloud, Hopsworks)
- [SASRec](https://arxiv.org/pdf/1808.09781)
- [BERT4Rec](https://arxiv.org/pdf/1904.06690)
- [Caser](https://jiaxit.github.io/resources/wsdm18caser.pdf)
- [LightGCN](https://arxiv.org/pdf/2002.02126)
- [NGCF](https://arxiv.org/pdf/1905.08969)
- [PinSAGE](https://arxiv.org/pdf/1806.01261)
- [HAN](https://arxiv.org/pdf/1903.07293)
- [RecVAE](https://arxiv.org/abs/1912.11160)
- [DLRM](https://arxiv.org/pdf/1906.00091)
- [BPR](https://arxiv.org/pdf/1205.2618)
- [InfoNCE](https://medium.com/self-supervised-learning/nt-xent-loss-normalized-temperature-scaled-cross-entropy-loss-ea5a1ede7c40)
- [SimCLR](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html)
- [S3-Rec](https://arxiv.org/abs/2008.07873)
- [LLM4Rec Survey](https://www.mdpi.com/1999-5903/17/6/252)

### Libraries & Frameworks
- **PyTorch Geometric (DGL):** Graph neural networks
- **Recbole:** Recommendation system benchmarks (SASRec, BERT4Rec, LightGCN, RecVAE, etc.)
- **Transformers4Rec (NVIDIA Merlin):** Sequential recommendation at scale
- **Sentence-Transformers:** Text embedding models (all-MiniLM-L6-v2, etc.)
- **OpenAI CLIP:** Multimodal embeddings
- **Scikit-learn:** Clustering, NeuMF (via pipeline)
- **LangChain:** LLM-as-Judge prompting
- **NetworkX / Pyvis:** Graph visualization

### Datasets
- **MovieLens 1M / 25M:** User ratings, demographics, item metadata
- **TMDB / IMDB:** Rich movie metadata (cast, crew, genres, budgets, plots)
- **Letterboxd API:** Temilola's personal watch history (already have)

### Recommended Tools for Visualization
- **Plotly / Altair:** Interactive embeddings, recommendation lists
- **Matplotlib / Seaborn:** Heatmaps (attention), loss curves, metrics over time
- **t-SNE / UMAP:** Embedding space visualization
- **Graphviz / Pyvis:** Heterogeneous graph structure

---

## 13. Success Criteria for Pipeline 3

| Rubric LO | 4 (Solid) | 5 (Mindblowing) | Evidence |
|---|---|---|---|
| **MLCode** | Working library code | From-scratch + library comparison | Two implementations (from-scratch + recbole/PyG) side-by-side, both verified |
| **MLExplanation** | Clear with notation | Intuitive + analytical, every choice motivated | Detailed loss functions, diagrams (architecture, attention, graph), intuitive prose |
| **MLFlexibility** | Solid | Beyond class + meta-knowledge | Novel method (GNN/seq/contrastive), transfer to other domains (e.g., product recs), ablations |
| **MLMath** | Typeset equations | First-principles derivations | InfoNCE from KL divergence, attention from softmax mechanics, graph convolution from spectral theory, VAE from ELBO |
| **#datavis** | Adequate | Stunning | t-SNE embeddings, attention heatmaps, fan-class radars, error distributions, LLM explanations |
| **#algorithms** | Correct | Clever + novel | Novel architecture (not in class), novel loss, novel sampling strategy, novel data fusion |
| **#professionalism** | Polished notebook | Polished PDF, narrative flow, humor | No bullet points, smooth transitions, funny moments (e.g., "fixing Madam Web"), cohesive arc |

---

**This report is ready for action. Pick your novel method, implement from scratch, and make Prof. Watson's jaw drop.**
