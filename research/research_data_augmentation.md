# Pipeline 3 Research: Data Augmentation, Public Datasets, & Statistical Methods

**Agent:** Claude Haiku Research Agent
**Date:** April 16, 2026
**Focus:** Breaking the N=162 ceiling via data augmentation, public datasets, and novel statistical methods
**Goal:** Deliver 3–5 ranked concrete methods, deep math hooks, 7-day feasibility estimates, and wild-card ideas

---

## Executive Summary

Your existing dataset (162 personal Letterboxd ratings + TMDB metadata) hits a hard wall: you're limited to learning patterns from 162 labeled examples in a high-dimensional space. This document catalogs **public datasets, augmentation strategies, and causal inference methods** to:

1. **Merge rich external rating distributions** (MovieLens 25M, IMDB, TMDB) with your personal ratings
2. **Apply statistical/ML augmentation** (GenMatch, propensity scoring, SMOTE, CTGAN, TVAE) to synthetically expand your feature space
3. **Use transfer learning and warm-start approaches** to position you as a "known user" in MovieLens, enabling collaborative filtering insights
4. **Correct for selection bias** (MNAR: missing-not-at-random ratings — you watched only movies you *probably* liked)

**Expected outcome:** 1,000–10,000 augmented training examples, richer feature space, causal understanding of rating drivers, and ammunition for Kalman filters, VAEs, counterfactual modeling, or graph neural networks in Pipeline 3's novel method.

---

## Part 1: Public Datasets to Merge In

### 1.1 IMDB Non-Commercial Datasets

**What it is:**
Free, daily-updated TSV files from [datasets.imdbws.com](https://datasets.imdbws.com/) containing the backbone of movie metadata.

**Files & Schema:**

- **title.basics.tsv.gz**: `tconst, titleType, primaryTitle, originalTitle, isAdult, startYear, endYear, runtimeMinutes, genres`
  - ~9.5M rows; ~460 MB gzipped
  - Maps directly to TMDB via title matching; weak but usable

- **title.ratings.tsv.gz**: `tconst, averageRating (0–10, weighted), numVotes`
  - ~1.4M rows with ratings; average ≈ 6.7
  - **CRITICAL:** This is a selection-biased sample (only rated movies)

- **title.crew.tsv.gz**: `tconst, directors, writers`
  - Link to personnel; enables director/writer preference learning

- **title.principals.tsv.gz**: `tconst, ordering, nconst, category (actor, director, etc.), job, characters`
  - Cast info; useful for "favorite actor" embeddings

- **name.basics.tsv.gz**: `nconst, primaryName, birthYear, deathYear, primaryProfession`
  - 12M+ people; enables personnel popularity scoring

- **title.akas.tsv.gz**: `titleId, ordering, title, region, language, types, attributes, isOriginalTitle`
  - Multi-region titles; helps with regional bias detection

**Why it helps the rubric:**
- Adds **~1.4M rated movies** as external reference class
- Enables **feature engineering**: director count, writer count, cast size, personnel popularity
- Correcting for IMDB's selection bias teaches **causal inference** (MNAR lesson)
- Cross-study validation: Does IMDB rating predict your rating?

**Math hooks for derivations:**
- Selection bias correction: logistic propensity $p(rated | features)$, then inverse-weighting
- Cross-dataset covariate balancing: $E[X | rated, IMDB] = E[X | rated, Temilola]$ (genetic matching goal)

**Library + from-scratch:**
- **From-scratch:** Plain pandas + SQL joins; build propensity model in numpy
- **Library:** `dask` for parallel TSV loading, `polars` for speed

**7-day effort:** 2 days (ETL, EDA), 2 days (ID mapping + dedup), 1 day (bias audit)
**Implementation:** `wget` TSV, parse with pandas, fuzzy-match titles via `fuzzywuzzy`, cross-validate rating prediction

---

### 1.2 MovieLens 25M

**What it is:**
The benchmark collaborative filtering dataset: [grouplens.org/datasets/movielens/25m](https://grouplens.org/datasets/movielens/25m/)

**Schema:**
- **ratings.csv**: `userId, movieId, rating (0.5–5 in 0.5 steps), timestamp`
  - 25M ratings from 162K users on 62K movies
  - Average density: ~1% (very sparse)

- **movies.csv**: `movieId, title, genres`

- **tags.csv**: `userId, movieId, tag, timestamp`
  - 1.1M tags from 15K users; 1,129 tag vocabulary

- **links.csv**: `movieId, imdbId, tmdbId`
  - **GOLD:** Maps ML movies to IMDB/TMDB tconst

**Why it helps the rubric:**
- **Direct warm-start path:** Find your K nearest neighbors (taste-alike users) on MovieLens using your 162 ratings as an anchor. If you have 50 overlap movies with ML, compute cosine similarity in rating space, retrieve top-50 users, pseudo-label their unrated movies.
- Addresses N=162 ceiling without synthetic data — you inherit 25M sparse examples
- Tag vocabulary (1,129 dimensions) enriches semantic feature space
- Exact ID mapping means zero fuzzy-matching overhead

**Math hooks:**
- User similarity: $sim(u, v) = \frac{\sum_i (r_{u,i} - \bar{r}_u)(r_{v,i} - \bar{r}_v)}{\sqrt{\sum_i (r_{u,i} - \bar{r}_u)^2} \sqrt{\sum_i (r_{v,i} - \bar{r}_v)^2}}$ (Pearson)
- Pseudo-labeling: $\hat{r}_{u,j} = \bar{r}_u + \sum_{v \in kNN(u)} w_{u,v} (r_{v,j} - \bar{r}_v)$ where $w_{u,v} \propto sim(u, v)$

**Library + from-scratch:**
- **From-scratch:** Dense/sparse matrix ops in numpy; implement cosine similarity and kNN manually
- **Library:** `surprise` (SVD, kNN), `implicit` (ALS), `cornac` (hybrid)

**7-day effort:** 1 day (download + audit), 2 days (warm-start embedding), 2 days (pseudo-labeling + validation), 1 day (denoising pseudo-labels)

**Implementation:**
1. Download MovieLens 25M (~4 GB)
2. Match your movies to movieId via title + year
3. Compute Pearson similarity to all 162K users (sparse matrix trick)
4. Retain top-50 similar users
5. Pseudo-label top-1000 movies (by frequency in your taste neighborhood)
6. Upweight pseudo-labels with confidence = (agreement among top-k neighbors) × (neighbor similarity)

---

### 1.3 TMDB Advanced Endpoints

**What it is:**
Beyond basic metadata, TMDB's API offers rich supplemental endpoints.

**Key endpoints for augmentation:**

- **GET /movie/{id}/recommendations**: Similar movies hand-tuned by TMDB's algorithm
  - Returns up to 20 per movie; enables implicit similarity graph

- **GET /movie/{id}/similar**: Movies with similar keywords/genres
  - Different algorithm, slightly noisier

- **GET /movie/{id}/keywords**: Genre-free semantic tags (1,000+ vocab)
  - Example: `sci-fi-adventure, time-travel, dystopian`

- **GET /movie/{id}/watch/providers**: Streaming availability by region
  - Feature: Is it on Netflix/theatrical/premium? Proxy for reach/quality

- **GET /movie/{id}/reviews**: TMDB user reviews (text + rating)
  - ~50–500 reviews per popular movie; sentiment-augment features

- **GET /movie/{id}/videos**: Trailers, teasers, clips
  - Can extract YouTube video metadata (duration, views, dislikes via external API)

- **GET /movie/{id}/credits**: Full cast + crew with popularity scores
  - Cast popularity (from social media) as proxy for actor prestige

**Why it helps the rubric:**
- Recommendations endpoint gives you a **small-world neighbor graph** of 62K MovieLens movies → enables graph-based embeddings (GraphSAGE cold-start warm-up)
- Keywords are dense, semantic features (vs. genre one-hots)
- Review sentiment can be used as **external rating proxy** to calibrate your labels
- Watch provider data → genre + region bias understanding

**Math hooks:**
- Recommendation graph: Adjacency matrix $A \in \mathbb{R}^{n \times n}$ where $A_{ij} = $ "recommended" indicator; spectral clustering or PageRank for centrality
- Sentiment transfer: Treat review sentiment as noisy label; use as auxiliary task in multi-task learning

**Library + from-scratch:**
- **From-scratch:** Use `requests` + JSON parsing; manual graph construction with networkx
- **Library:** `tmdb-api` (Python wrapper), `networkx` (graph)

**7-day effort:** 2 days (API design + rate-limit handling), 2 days (parallel download 10K movies), 2 days (graph EDA + embedding)

**Implementation:**
1. Batch-download recommendations + keywords for all 62K MovieLens movies (requires ~310K API calls; use 40 req/s, 7.75K seconds ≈ 2.15 hrs with parallelization)
2. Build directed recommendation graph, compute PageRank
3. Extract keywords as multi-hot vectors; TF-IDF score them
4. Combine with TMDB genres → richer feature matrix

---

### 1.4 Letterboxd Public Reviews (Ethical Scraping)

**What it is:**
Letterboxd ([letterboxd.com](https://letterboxd.com/)) is the Netflix of movie tracking: ~7M users, ~500M ratings, public profiles.

**Official & scraping options:**

- **Official API Beta** ([letterboxd.com/api-beta](https://letterboxd.com/api-beta/)): Limited public endpoints; requires OAuth
- **Apify scrapers** ([apify.com](https://apify.com/)): Pre-built scrapers return structured JSON (username, rating, date, review text, likes)
- **DIY scraping with PRAW-like tools**: Possible but rate-limited; use caution

**What you can extract:**
- Movie ratings (0.5–5 stars) from public profiles
- Review text (sentiment-encodable)
- Watchlist, to-watch lists (indicator of interest)
- Watch dates (temporal signal)
- Likes on reviews (crowd-sourced review quality)

**Why it helps the rubric:**
- **Community ground truth:** ~500M ratings provide statistical reference. Do Letterboxd ratings correlate with yours? With MovieLens? With IMDB?
- **Text augmentation:** Review text → TF-IDF, BERT embeddings, sentiment → additional features
- **Temporal dynamics:** Watch date + rating drift → can model "rewatchability" or seasonal bias
- **Ethical angle:** Public-profile-only + aggregation + no re-identification = defensible

**Math hooks:**
- Cross-platform bias: Fit a simple linear transform $r_{Letterboxd} = \alpha + \beta \cdot r_{Temilola} + \epsilon$; examine residuals
- Review helpfulness regression: $P(\text{helpful} | \text{review text, rating}) \sim \sigma(\theta^T x)$

**Library + from-scratch:**
- **From-scratch:** `requests` + BeautifulSoup; throttle to 1–2 req/sec
- **Library:** Apify (paid but reliable; $5–50 per 1K profiles), `selenium` (headless browser)

**7-day effort:** 1 day (ethics audit + legal review), 2 days (scraper setup), 2 days (data pipeline + dedup), 1 day (EDA + bias audit)

**Implementation (ethical bounds):**
1. Scrape only **public** profiles (consent = public)
2. Aggregate at **dataset level only** (no individual re-identification)
3. Maximum: 1,000 random public profiles (~50K ratings) to avoid excessive load
4. Use official Apify tool (supports data export compliance)

---

### 1.5 CMU Movie Summary Corpus + Wikipedia

**What it is:**
[cs.cmu.edu/~ark/personas](http://www.cs.cmu.edu/~ark/personas/): 42,306 movie plot summaries extracted from Wikipedia, with character-level NLP annotations.

**Contents:**
- **movie.metadata.tsv**: Movie ID, name, release date, box office, genres, runtime
- **plot_summaries.txt**: Wikipedia plot text (avg 200–500 words)
- **character.metadata.tsv**: Character names, persona embeddings (from paper), actor links
- **NLP preprocessed:** Stanford CoreNLP tags, parses, NER, coreference

**Why it helps the rubric:**
- **Text features:** Summaries → TF-IDF, BERT embeddings (sentence-transformers), GPT-2 encoded sentiment trajectory
- **Character diversity:** Count of named characters, gender distribution → proxy for ensemble cast quality
- **Content-based filtering** bridge: If you only have 162 personal ratings, plot similarity can help predict cross-movie transfer
- **Generative angle:** Can use summaries as auxiliary input to VAE or diffusion model for "synthetic Temilola-like movie generation"

**Math hooks:**
- Summary embedding: $\mathbf{s}_m = \frac{1}{T} \sum_{t=1}^T \text{BERT}(w_t)$ (mean pooling), then $\cos(\mathbf{s}_i, \mathbf{s}_j)$ for content similarity
- Character statistics: $(n_{chars}, \% female, avg name length) \to$ regression features

**Library + from-scratch:**
- **From-scratch:** Tokenization + TF-IDF by hand with numpy
- **Library:** `sentence-transformers`, `gensim` (TF-IDF), HuggingFace `transformers` (BERT)

**7-day effort:** 1 day (download), 2 days (embedding + EDA), 2 days (similarity matrix construction), 1 day (auxiliary feature engineering)

---

### 1.6 Rotten Tomatoes Critic Scores

**Why it helps:**
RT Tomatometer (critic consensus) and audience score are independent signals. Combining IMDB, MovieLens, Letterboxd, and RT ratings gives you a **meta-review ensemble** → strong training signal for predicting Temilola's rating from "what the world thinks."

**Implementation:** Scrape RT via [rottentomatoes.com](https://rottentomatoes.com/) or use third-party API (RapidAPI has RT aggregators); fuzzy-match by title to your dataset.

**7-day effort:** 0.5 days (scraper setup), 0.5 days (matching + dedup)

---

### 1.7 Hugging Face Movie Review Datasets

**What it is:**
Pre-packaged sentiment datasets on [huggingface.co/datasets](https://huggingface.co/datasets):

- **stanfordnlp/imdb**: 25K positive, 25K negative reviews; binary labels
- **cornell-movie-review-data/rotten_tomatoes**: 5.3K positive, 5.3K negative RT reviews
- **ajaykarthick/imdb-movie-reviews**: 50K reviews with 1–10 star labels

**Why it helps:**
- Transfer learning: Fine-tune RoBERTa or BERT on movie reviews; use as feature extractor for your summaries
- Sentiment as auxiliary feature: Average review sentiment → rating proxy
- Zero-shot transfer: Even without fine-tuning, a pre-trained sentiment model extracts weak labels

**7-day effort:** 0.5 days (download), 1 day (fine-tuning if desired), 1 day (feature extraction)

---

## Part 2: Data Augmentation Techniques (Statistics & ML)

### 2.1 Genetic Matching (GenMatch) for Selection Bias Correction

**What it is:**
GenMatch (Diamond & Sekhon, 2013) uses a **genetic algorithm** to optimize covariate weights for matching, solving this problem:

> You rated movies you thought you'd like. This selection bias breaks standard regression. GenMatch finds a weighting that makes "rated movies" indistinguishable from "unrated but similar movies" in covariate space.

**How it works (intuition):**
1. Treat "whether Temilola rated it" as a binary treatment $T \in \{0, 1\}$
2. Features: IMDB rating, runtime, genres, year, cast size, director prestige, etc.
3. Goal: Find weights $w = (w_1, \ldots, w_p)$ on each feature such that the **weighted Mahalanobis distance** between rated and unrated movies is minimized → covariate balance
4. Use genetic algorithm (GENOUD) to optimize; return a weight matrix $W$ for nearest-neighbor matching

**Mathematical form:**
$$\text{balance score} = \sum_k \rho_k |t_k^{(treated)} - t_k^{(control)}|$$
where $\rho_k$ is the genetic-algorithm-chosen weight on feature $k$, and $t_k$ is the standardized feature. GenMatch minimizes this.

**Why it helps the rubric:**
- **Causal inference angle:** GenMatch teaches selection bias + covariate balancing (not covered in class)
- **Addresses MNAR:** Only-watched-likely-good problem is a textbook selection bias case
- **Novel method:** Genetic matching ≠ class material (linear methods, trees, NN)
- **Math derivation:** Can hand-derive the weighting optimization objective

**Library + from-scratch:**
- **R (gold standard):** `Matching::GenMatch()` (Sekhon's own package)
- **Python:** `pymatchit` (MatchIt port) or roll-your-own GENOUD wrapper (`pygenoud` exists but unmaintained)
- **From-scratch:** Implement GENOUD genetic algorithm + Mahalanobis distance in pure Python (feasible; ~500 lines)

**7-day effort:**
- Library version: 1 day (call `pymatchit` or R via `rpy2`)
- From-scratch: 3 days (genetic algorithm + distance metric), 2 days (validation), 1 day (reporting)

**Class baseline for comparison:** OLS regression on all features vs. matched subset (balance check)

**Implementation roadmap:**
1. Create feature matrix $X$ of IMDB/TMDB attributes for all 62K ML movies
2. Create treatment vector $T$ = 1 if in your 162, 0 if in IMDB but not rated by you
3. Run GenMatch (via pymatchit or R) to find weights $W$
4. Match each unrated movie to 1–3 rated movies with similar (weighted) features
5. Transfer-learn: Assume unrated matched movies have similar rating as their matched rated counterpart
6. Generate "synthetic" 1,000–5,000 pseudo-labeled training examples

---

### 2.2 Propensity Score Matching & Inverse Probability Weighting (IPW)

**What it is:**
Simpler than GenMatch; more direct approach to selection bias.

**How it works:**
1. Estimate propensity score: $\hat{p}(x) = P(T=1 | X=x)$ via logistic regression
   $$\hat{p}(x) = \frac{1}{1 + e^{-(\beta_0 + \sum_j \beta_j x_j)}}$$

2. **Matching:** Keep only unrated movies with $\hat{p}(x)$ close to rated movies (overlap region). Match rated $\to$ unrated 1:1.

3. **IPW:** Reweight training examples: $w_i = \frac{T_i}{\hat{p}(x_i)} + \frac{1 - T_i}{1 - \hat{p}(x_i)}$
   - High-propensity (likely-rated) movies get downweighted; rare, unlikely-to-be-rated movies get upweighted
   - Fit model on weighted data; this corrects bias asymptotically

**Why it helps the rubric:**
- **Causal inference:** IPW is canonical counterfactual estimation technique (Rubin causal model)
- **Teachable:** Clear, hand-derivable math from first principles
- **Simpler than GenMatch:** Good apples-to-apples comparison

**Math hooks:**
- Propensity: Logistic regression objective $-[T \log \hat{p} + (1-T) \log(1-\hat{p})]$
- IPW weighting: Proof that $E_{\text{weighted}}[Y] \to E[Y | T=1, X \text{ balanced}]$ (unbiased under overlap)

**Library + from-scratch:**
- **Library:** `statsmodels.api`, `causalml`, `DoWhy`
- **From-scratch:** Implement logistic regression, compute weights, re-fit OLS

**7-day effort:** 1 day (propensity model), 1 day (matching + overlap check), 1 day (IPW weighting), 1 day (validation)

**Class baseline:** Simple OLS on all features vs. OLS on matched/weighted subset

---

### 2.3 SMOTE / ADASYN for Rating Distribution Imbalance

**What it is:**
Your personal ratings are imbalanced: skewed to high ratings (you watch movies you think you'll like). SMOTE/ADASYN generate synthetic minority examples.

**How it works (SMOTE):**
1. For each minority example (low rating), find $k$ nearest neighbors
2. Randomly select one neighbor; create synthetic point at random position on line segment between them
   $$\mathbf{x}_{synthetic} = \mathbf{x}_i + \lambda (\mathbf{x}_{neighbor} - \mathbf{x}_i), \quad \lambda \sim U(0, 1)$$

**ADASYN variant:** Adaptive; generates more synthetic examples in hard-to-learn regions (near decision boundary)

**Why it helps the rubric:**
- **Pragmatic augmentation:** Balances your rating distribution without causal modeling
- **ML engineering:** Addresses class imbalance (ratings 1–3 vs. 4–5)
- **Library-friendly:** Scikit-learn has this built-in; also teachable from scratch
- **Data augmentation method not covered in class**

**Math hooks:**
- Geometric mean sampling: $d(\mathbf{x}_i, \mathbf{x}_{neighbor})$ in feature space; linear interpolation preserves manifold structure
- ADASYN density weighting: $r_i = $ (number of majority neighbors) / $k$; samples $\sim r_i$

**Library + from-scratch:**
- **Library:** `imbalanced-learn` (SMOTE, ADASYN, BorderlineSMOTE)
- **From-scratch:** kNN + interpolation (~100 lines)

**7-day effort:** 0.5 days (library call), 1 day (hyperparameter tuning), 1 day (from-scratch), 1 day (validation)

**Implementation:**
1. Train kNN on your 162-dimensional feature space (TMDB + text embeddings)
2. Apply SMOTE with k=5, random_state=42, oversample minority (rating ≤ 3) to 50% of majority
3. Generate 500–1,000 synthetic examples
4. Train model on synthetic + original; evaluate on held-out real ratings

---

### 2.4 CTGAN, TVAE, TabDDPM for Tabular Synthetic Data

**What it is:**
Deep generative models that learn the joint distribution of tabular features (continuous + categorical) and generate realistic synthetic rows.

**CTGAN (Conditional GAN):**
- Generator: $G(\mathbf{z}) \to \hat{\mathbf{x}}$, where $\mathbf{z} \sim \mathcal{N}(0, I)$
- Discriminator: $D(\mathbf{x}) \to P(\text{real})$
- Loss: Wasserstein GAN objective with gradient penalty
- Handles mixed data types via mode-specific normalization for continuous features

**TVAE (Tabular VAE):**
- Encoder: $\mathbf{x} \to q(\mathbf{z} | \mathbf{x})$
- Decoder: $\mathbf{z} \to p(\mathbf{x} | \mathbf{z})$
- Loss: Reconstruction + KL divergence
- Variational inference explicitly models uncertainty; sometimes more stable than CTGAN

**TabDDPM (Diffusion-based):**
- Recent SOTA (ICML 2023); iteratively denoise Gaussian noise → realistic tabular data
- More stable training, fewer mode-collapse issues
- Slower sampling (~1–10s per record) but higher quality

**Why it helps the rubric:**
- **Generative modeling:** Creates novel movie-like features; not purely interpolating like SMOTE
- **Complex distributions:** Learns non-Gaussian, multimodal patterns (e.g., runtime is bimodal: short films vs. long epics)
- **Novel to class:** No generative models (except autoencoders, which are unsupervised) covered in-class
- **Scalability:** Can generate 10K–100K synthetic examples

**Math hooks:**
- CTGAN loss: Wasserstein distance $W(\mathbb{P}_{data}, \mathbb{P}_{generator})$, gradient penalty $\lambda \mathbb{E}_{\hat{\mathbf{x}}}[(\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1)^2]$
- TVAE ELBO: $\mathcal{L} = \mathbb{E}_{q}[-\log p(\mathbf{x} | \mathbf{z})] + \text{KL}(q(\mathbf{z} | \mathbf{x}) \| p(\mathbf{z}))$
- Diffusion: Reverse SDE $d\mathbf{x} = -\frac{1}{2} \beta(t) \mathbf{x} dt + \sqrt{\beta(t)} d\mathbf{w}$ iterated backwards

**Library + from-scratch:**
- **Library:** `sdv` (Synthetic Data Vault) with CTGAN, TVAE, TabDDPM; TensorFlow/PyTorch backends
- **From-scratch:** CTGAN is ~1000 lines (Wasserstein GAN + mode handling); TabDDPM is ~500 lines (diffusion SDE solver)

**7-day effort:**
- Library: 2 days (train + hyperparameter sweep), 1 day (quality eval)
- From-scratch TVAE: 4 days (VAE + categorical decoding), 2 days (eval)

**Implementation:**
1. Install `sdv`: `pip install sdv`
2. Create tabular data: 162 rows × (TMDB features + embeddings) columns
3. Train CTGAN: `ctgan = CTGAN(epochs=300, batch_size=32); ctgan.fit(X_train)`
4. Generate 5,000 synthetic examples: `X_synthetic = ctgan.sample(5000)`
5. Validate: (a) Statistical similarity (Kolmogorov-Smirnov test, Wasserstein distance), (b) ML-based validator (can discriminator tell real from fake?)

**Quality metrics:**
- **Statistical fidelity:** $\text{KS} = \max_x |CDF_{real}(x) - CDF_{synthetic}(x)|$ (should be <0.05)
- **Discriminative score:** Train classifier on real/synthetic labels; should achieve ~50% accuracy
- **Downstream task:** Train rating predictor on real + synthetic; compare to real-only baseline

---

### 2.5 Mixup / CutMix for Embedding Augmentation

**What it is:**
Interpolate in embedding space between two training examples to create virtual examples.

**Mixup in embeddings:**
1. Sample $\lambda \sim \text{Beta}(\alpha, \alpha)$ (e.g., $\alpha = 0.2$)
2. Pick two examples: $(\mathbf{e}_i, y_i)$ and $(\mathbf{e}_j, y_j)$ (embeddings + labels)
3. Create synthetic: $\mathbf{e}' = \lambda \mathbf{e}_i + (1 - \lambda) \mathbf{e}_j$, $y' = \lambda y_i + (1 - \lambda) y_j$
4. Train on mixed examples

**CutMix variant:** Instead of linear interpolation, randomly select a fraction of embedding dimensions from one example, rest from the other (analog of image cutout).

**Why it helps the rubric:**
- **Data augmentation from embeddings:** Doesn't require generating new raw features (harder)
- **Regularization:** Mixup acts as implicit smoothness constraint; reduces overfitting
- **Interpretable:** Can visualize what virtual movies look like in embedding space
- **From-scratch math:** Beta distribution, linear interpolation

**Math hooks:**
- Mixup loss: $\mathcal{L}(\lambda \mathbf{e}_i + (1-\lambda) \mathbf{e}_j, \lambda y_i + (1-\lambda) y_j)$
- Regularization effect: Forces model to assign similar predictions to nearby embeddings
- Connection to adversarial robustness: Mixup-trained models are more robust

**Library + from-scratch:**
- **Library:** `torchvision.transforms.v2.MixUp`, `torch_geometric` (for graph embeddings)
- **From-scratch:** 50 lines of numpy

**7-day effort:** 1 day (library call), 1 day (from-scratch + validation)

**Implementation:**
1. Use your existing TMDB + sentence-transformer embeddings (162 movies × 384 dims)
2. For each training epoch, randomly select pairs; apply Mixup
3. Generate 500–1000 virtual examples
4. Train regression model (rating prediction) on original + mixed
5. Compare to baseline (original only)

**Evaluation:**
- Does Mixup reduce test RMSE? (should improve by 5–15%)
- Are virtual examples meaningfully in-between originals? (measure via nearest-neighbor distance)

---

### 2.6 Counterfactual Data Augmentation (CDA)

**What it is:**
Generate synthetic training data that simulates *counterfactual* (alternative) scenarios: "What if this movie had a different director, or was released in a different year?"

**How it works:**
1. Learn a causal model of rating: $Y = f(D, \mathbf{X})$ where $D$ = director, $\mathbf{X}$ = other features
2. Intervene: Set $D' = $ alternative director; keep $\mathbf{X}$ fixed
3. Use causal model (or simulator) to predict $Y'$ under intervention
4. Add $(\mathbf{X}, D', Y')$ to training set as synthetic example

**Example:** You rated *Oppenheimer* (Nolan) 9/10. Counterfactual: "What if Spielberg directed it?" Predict rating under intervention.

**Why it helps the rubric:**
- **Causal inference + generative modeling:** Combines two "novel" angles
- **Interpretability:** Synthetic examples directly answer "what if" questions
- **Debiasing:** Can remove confounder bias by conditioning out unmeasured confounders
- **Ambitious:** Only a few papers on CDA in recommender systems; shows deep thinking

**Math hooks:**
- Causal graph: $D \to Y$, $\mathbf{X} \to Y$; possibly latent confounder $U \to D, U \to Y$
- Do-operator: $P(Y | do(D=d), X=x) = \int P(Y | D, X, U) P(U | X) dU$ (requires causal identification)
- Counterfactual prediction: Fit simulator $\hat{f}: (X, D', U) \to \hat{Y}$, sample $U$ from posterior

**Library + from-scratch:**
- **Library:** `dowhy`, `causalml`, `pymc3` (for Bayesian causal modeling)
- **From-scratch:** Build causal DAG, fit structural equation model, sample counterfactuals

**7-day effort:** 2 days (causal assumption specification), 2 days (model fitting), 2 days (counterfactual generation + validation)

---

### 2.7 Inverse Propensity Weighting (IPW) for MNAR Ratings

**What it is:**
Addresses the "missing not at random" problem: You *didn't* rate movies you thought you'd dislike. This is not random missingness; it's selective.

**How it works:**
1. Model missingness: $P(\text{rated} = 1 | \text{features}, \text{true rating})$
2. Upweight rare, high-quality unrated movies (hard to observe because you'd have hated them... or loved them)
3. Inverse-weight correction: $\hat{Y} = \frac{1}{n} \sum_i \frac{Y_i \cdot T_i}{P(T_i = 1 | X_i, Y_i)}$

**Why it helps:**
- **Addresses MNAR directly:** You watched only movies you *probably* liked → huge selection bias
- **Statistical rigor:** Makes assumptions explicit (ignorability, overlap); can be tested
- **Theoretical foundation:** Doubly robust estimation theory (Robins, Kennedy)

**Math hooks:**
- Propensity: $P(T | X, Y) = \frac{1}{1 + e^{-(\alpha + \beta X + \gamma Y)}}$
- Doubly robust estimator: $\hat{\tau} = \frac{1}{n} \sum_i \left[ \frac{T_i Y_i}{\hat{p}_i} - \frac{T_i (m(X_i) - Y_i)}{\hat{p}_i} + m(X_i) \right]$
  where $m(X) = \mathbb{E}[Y | X]$ (doubly robust: works if either $\hat{p}$ or $m$ is correct)

**7-day effort:** 2 days (propensity model), 1 day (weighting), 1 day (doubly robust estimator), 1 day (sensitivity analysis)

---

### 2.8 Wild Card: "Synthetic Temilola Twin" via K-Nearest Taste Neighbors on MovieLens

**The idea:**
Instead of just pseudo-labeling 1,000 movies, construct a **synthetic "Temilola twin" user** on MovieLens:

1. **Anchor:** Match your 162 Letterboxd ratings to 50–100 MovieLens movies (title overlap).
2. **Find taste neighbors:** On MovieLens, find your K=50 nearest neighbors (Pearson similarity in rating vector).
3. **Blend the twins:** Create a synthetic user profile as a weighted average of your ratings + your neighbors' ratings:
   $$r_{twin}(m) = \alpha \cdot r_{you}(m) + (1-\alpha) \cdot \frac{1}{K} \sum_{k=1}^K r_{neighbor_k}(m)$$
   Set $\alpha = 0.6$ (you are 60% signal; neighbors are 40% noise but capture collaborative patterns).

4. **Generate massive rating distribution:** Sample 10,000 unrated movies; for each, predict $\hat{r}_{twin}(m)$ via the blend. This is your "augmented self."

5. **Use twin to pseudo-label:** For your model, treat twin ratings as weak labels with confidence proportional to neighbor agreement.

**Why it's wild:**
- **Theoretically grounded:** Collaborative filtering on MovieLens is the gold standard; your twin inherits 25M examples
- **Practical:** Generates 10K examples from first principles (not synthetic generative model)
- **Interpretable:** Can analyze which taste profiles align with yours (e.g., "you cluster with film-noir enthusiasts")
- **Novel angle:** Not SMOTE, not GenMatch, not VAE; this is a hybrid cold-start solution specific to your problem
- **Validation ready:** Compare twin-predicted ratings to actual Temilola ratings on held-out test set

**Math hooks:**
- User similarity: Cosine/Pearson in 62K-dimensional movie space (sparse)
- Weighted blend: Shrinkage estimator; relates to Bayesian regression with data-dependent prior
- Confidence calibration: Agreement among K neighbors = signal strength

**7-day effort:** 2 days (MovieLens matching), 2 days (kNN + similarity), 1 day (blending + sampling), 1 day (validation)

**Implementation:**
```python
# Pseudocode
movielens_data = load_movielens_25m()
your_ml_ratings = match_letterboxd_to_ml(your_ratings, movielens_data)
neighbors = knn_users(your_ml_ratings, movielens_data, k=50)
twin_profile = 0.6 * your_ml_ratings + 0.4 * neighbors.mean(axis=0)
pseudo_labels = twin_profile.sample(10000)
pseudo_labels = denoise_by_neighbor_agreement(pseudo_labels, neighbors)
X_augmented = concatenate([your_features, pseudo_features])
y_augmented = concatenate([your_ratings, pseudo_labels])
model = train_on_augmented(X_augmented, y_augmented)
```

---

## Part 3: Merging Strategy & Feature Unification

### 3.1 Cross-Dataset ID Mapping

**Challenge:** IMDB uses `tconst` (tt1234567), MovieLens uses `movieId`, TMDB uses `id`. Your Letterboxd is manually curated. Matching is fuzzy.

**Solution:**
1. **Use MovieLens links.csv:** Has `imdbId, tmdbId` already; maps 62K movies across all three sources
2. **Fuzzy title match (fallback):** For Letterboxd, use `fuzzywuzzy` with title + year:
   ```python
   from fuzzywuzzy import fuzz
   score = fuzz.token_sort_ratio(lbxd_title, ml_title)  # threshold ~85
   ```
3. **Validation:** Manual spot-checks on 20–30 ambiguous titles (e.g., "The Ring" 2002 vs. 1998)

**7-day effort:** 1 day (links.csv + dedup), 1 day (fuzzy matching), 1 day (manual validation)

---

### 3.2 Feature Unification

**Challenge:** Different sources have different feature representations.

**Strategy:**

| Feature Type | IMDB | TMDB | MovieLens | Letterboxd |
|---|---|---|---|---|
| Release year | startYear | release_date | (in title) | date_watched |
| Runtime | runtimeMinutes | runtime | – | – |
| Genres | genres (pipe-sep) | genres | genres (pipe-sep) | tags |
| Cast/crew | principals, name.basics | credits | – | – |
| Popularity | numVotes | popularity score | – | (likes on reviews) |
| External rating | averageRating | vote_average | (user ratings are features) | (user ratings are features) |
| Keywords | – | keywords | tags (1,129 dim) | – |

**Unified representation:**
1. **Categorical:** One-hot encode genres (expand to union of all), director name (top-100 by frequency), cast (top-50 actors)
2. **Numerical:** Year, runtime, log(votes), log(popularity), cast size, director/writer count
3. **Embeddings:** BERT on plot summaries (CMU corpus); sentence-transformers on overview text (TMDB)
4. **Sparse:** TF-IDF on keywords/tags

**Result:** Each movie → ~500–1000 dimensional feature vector (sparse + dense)

---

### 3.3 Transfer Learning: Warm-Start User Embedding

**Idea:** Position yourself as a "known user" on MovieLens to inherit collaborative information.

**Method:**
1. Train a matrix factorization model (e.g., SVD, ALS, NMF) on all 25M MovieLens ratings:
   $$r_{u,i} \approx \mathbf{u}_u^T \mathbf{v}_i$$
   where $\mathbf{u}_u$ ∈ ℝ^K is user latent factor, $\mathbf{v}_i$ ∈ ℝ^K is item latent factor (K=50–100).

2. Use your 50–100 matched ML ratings to fit your user embedding:
   $$\mathbf{u}_{Temilola} = \arg\min_{\mathbf{u}} \sum_i (r_i - \mathbf{u}^T \mathbf{v}_i)^2 + \lambda \|\mathbf{u}\|^2$$

3. For unrated movies, use $\hat{r}_{Temilola}(m) = \mathbf{u}_{Temilola}^T \mathbf{v}_m + \text{neighborhood bias}$

4. This warm-start embedding transfers to your own movie rating model as:
   - **Feature:** Concatenate $\mathbf{u}_{Temilola}$ with TMDB/IMDB features
   - **Initialization:** Initialize your model's user embedding layer with $\mathbf{u}_{Temilola}$

**Why it helps:**
- Captures global taste patterns (collaborative signal) + local content signal (features)
- Reduces cold-start risk
- Elegant bridge between MovieLens and your personal dataset

**7-day effort:** 2 days (train ML matrix factorization), 1 day (warm-start your embedding), 2 days (integration + validation)

---

## Part 4: Statistical Rigor & Validation

### 4.1 Train/Val/Test Splits with Leakage Control

**Challenge:** Augmented data creates risk of leakage (synthetic examples leak information about test set).

**Solution:**
1. **Original data (162 movies):** Randomly split into 80/20 (130 train, 32 test). Lock test set immediately.
2. **Synthetic data:** Generate from training features *only*. Never touch test features.
3. **Cross-validation:** Use K-fold on training set (K=5) to tune hyperparameters. Use heldout validation set (20% of train = ~26 movies) for early stopping.
4. **Final evaluation:** On test set (32 real, unseen movies).

**Leakage audit:**
- Ensure no test movie is in IMDB neighbor set when matching
- If using MovieLens pseudo-labels, exclude test movies from pseudo-label generation

---

### 4.2 Cross-Modal Validation: Does IMDB/ML Rating Predict Yours?

**Hypothesis:** "If IMDB highly rates a movie, Temilola also highly rates it."

**Test:**
1. Extract IMDB/MovieLens ratings for your 162 movies
2. Fit simple linear model: $r_{Temilola} = \beta_0 + \beta_1 r_{IMDB} + \epsilon$
3. Report R², correlation, RMSE
4. Interpret residuals: What movies do you rate differently from IMDB? (taste profile signature)

**Expected:** Moderate correlation (0.4–0.6) due to individual taste, signaling that external ratings contain signal but aren't deterministic.

---

### 4.3 Bootstrap Confidence Intervals

**Method:**
1. Resample training set with replacement (B=1000 times)
2. Fit model on each resample
3. Evaluate on original test set
4. Report 95% CI of test RMSE: $[\hat{\text{RMSE}}_{0.025}, \hat{\text{RMSE}}_{0.975}]$

**Interpretation:** Uncertainty in model performance due to training set variability.

---

### 4.4 Missing Data Mechanism Analysis

**Questions:**
1. Is missingness random (MCAR), random given observables (MAR), or selective (MNAR)?
2. Does propensity score $P(\text{rated} | \text{features})$ vary widely? (High variance → strong MNAR)

**Test:**
- Fit logistic regression: $P(rated) = \sigma(\mathbf{w}^T \mathbf{x})$
- Visualize propensity score distribution
- Compute overlap region: % of unrated movies with $0.2 < \hat{p} < 0.8$ (should be >50% for valid inference)
- Sensitivity analysis: How much would estimates change if MNAR assumption violated?

---

## Part 5: Implementation Roadmap (7-Day Timeline)

### Day 1: Data collection & EDA
- Download IMDB datasets (~2 hrs), MovieLens (~1 hr), CMU corpus (~0.5 hrs)
- Load + basic EDA; identify missing values, outliers
- Match MovieLens to IMDB/TMDB via links.csv

### Days 2–3: Feature engineering & unification
- Fuzzy-match Letterboxd to MovieLens (manual spot-checks)
- Construct unified feature matrix (genres, runtime, year, cast, embeddings)
- Handle categorical encoding, normalization

### Days 4–5: Augmentation methods
- GenMatch or propensity score matching: Select 500–1000 pseudo-labeled examples
- SMOTE on 162 ratings: Generate 500 synthetic examples
- CTGAN or TVAE: Train generative model, sample 5000 synthetic examples
- Mixup: Interpolate 500 examples in embedding space

### Day 6: Integration & validation
- Combine all augmented sets (2,500–10,000 total synthetic examples)
- Split train/val/test (control leakage)
- Cross-modal validation (IMDB vs. your ratings)
- Bootstrap CIs on test performance

### Day 7: Reporting & interpretation
- Ablation study: Contribute of each augmentation method
- Summary statistics (n, feature dimensions, class balance before/after)
- Visualizations (propensity score overlap, synthetic vs. real distributions)

---

## Part 6: Summary Table

| Method | Description | Math Complexity | Library | From-Scratch | 7-Day Effort | Rubric Value |
|---|---|---|---|---|---|---|
| **GenMatch** | Genetic algorithm covariate balancing for selection bias | High (genetic algorithm + Mahalanobis) | `pymatchit` or R | Feasible (~500 lines) | 3 days lib, 5 days scratch | 5/5 (causal + novel) |
| **Propensity Score Matching** | Logistic regression + matching to balance covariates | Medium (logistic regression) | `statsmodels`, `causalml` | Easy (~200 lines) | 1.5 days | 4/5 (causal, simpler) |
| **SMOTE/ADASYN** | Synthetic minority oversampling via kNN interpolation | Low-Medium | `imbalanced-learn` | Easy (~100 lines) | 1 day | 3.5/5 (standard but works) |
| **CTGAN** | Conditional GAN for tabular synthetic data | High (Wasserstein GAN) | `sdv` | Feasible (~1000 lines) | 2 days lib, 5 days scratch | 4.5/5 (generative, novel) |
| **TVAE** | Variational autoencoder for tabular data | High (VAE + ELBO) | `sdv` | Feasible (~600 lines) | 2 days lib, 4 days scratch | 4.5/5 (generative, VAE variant) |
| **Mixup** | Linear interpolation in embedding space | Low | `torchvision`, `torch-geometric` | Easy (~50 lines) | 1 day | 3/5 (standard, solid) |
| **Counterfactual Augmentation** | Simulate "what-if" scenarios via causal model | Very High (causal inference) | `dowhy`, `pymc3` | Hard (~1000 lines) | 4 days | 5/5 (ambitious, novel, math-heavy) |
| **IPW for MNAR** | Inverse propensity weighting for missing-not-at-random | Medium-High (doubly robust) | `causalml`, `DoWhy` | Medium (~300 lines) | 2 days | 4.5/5 (rigorous, addresses bias) |
| **Synthetic Twin (Wild Card)** | k-NN taste neighbors + blend for pseudo-user | Medium (collaborative filtering) | `surprise`, `implicit` | Medium (~400 lines) | 3 days | 4/5 (creative, interpretable) |

---

## Part 7: Recommended Ranking for Pipeline 3

**Tier 1 (Must Include):**
1. **Synthetic Twin** (wild card): Engaging narrative ("meet Temilola's taste neighbor"), practical, good visuals
2. **GenMatch**: Heavy math + causal inference angle; shows rigor

**Tier 2 (Should Include if time):**
3. **TVAE**: Generative modeling (novel to class); easier than CTGAN
4. **MovieLens warm-start embedding**: Elegant transfer learning bridge

**Tier 3 (Nice to have):**
5. SMOTE / Propensity Score Matching (either, not both; select one for simplicity)
6. Mixup (lightweight, good visualization)

---

## Part 8: References & URLs

### Datasets
- [IMDB Datasets (datasets.imdbws.com)](https://datasets.imdbws.com/)
- [MovieLens 25M (grouplens.org)](https://grouplens.org/datasets/movielens/25m/)
- [TMDB API (developer.themoviedb.org)](https://developer.themoviedb.org/)
- [CMU Movie Summary Corpus (cs.cmu.edu)](http://www.cs.cmu.edu/~ark/personas/)
- [Letterboxd API (letterboxd.com/api-beta)](https://letterboxd.com/api-beta/)
- [Hugging Face Datasets (huggingface.co/datasets)](https://huggingface.co/datasets)

### GenMatch & Propensity Scoring
- [GenMatch R Documentation (sekhon.berkeley.edu)](http://sekhon.berkeley.edu/matching/GenMatch.html)
- [MatchIt Genetic Matching (kosukeimai.github.io)](https://kosukeimai.github.io/MatchIt/reference/method_genetic.html)
- [pymatchit-causal PyPI](https://pypi.org/project/pymatchit-causal/)
- [Inverse Probability Weighting Overview (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8757413/)

### Synthetic Data & Augmentation
- [CTGAN GitHub (github.com/sdv-dev)](https://github.com/sdv-dev/CTGAN)
- [TabDDPM (yandex-research GitHub)](https://github.com/yandex-research/tab-ddpm)
- [SMOTE/ADASYN Overview (analyticsvidhya.com)](https://www.analyticsvidhya.com/blog/2020/10/overcoming-class-imbalance-using-smote-techniques/)
- [Mixup & CutMix (torchvision docs)](https://docs.pytorch.org/vision/stable/auto_examples/transforms/plot_cutmix_mixup.html)
- [Counterfactual Data Augmentation Survey (arxiv.org)](https://arxiv.org/html/2405.18917v1)

### Transfer Learning & Recommendations
- [Deep Transfer Learning for Cold-Start (plosone.org)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0273486)
- [Collaborative Filtering Overview (keras.io)](https://keras.io/examples/structured_data/collaborative_filtering_movielens/)
- [kNN User Similarity (analyticsvidhya.com)](https://www.analyticsvidhya.com/blog/2026/02/collaborative-filtering-recommendation-system/)

### Missing Data & MNAR
- [Doubly Robust Joint Learning for MNAR (mlr.press)](http://proceedings.mlr.press/v97/wang19n/wang19n.pdf)
- [MNAR Concepts (stefvanbuuren.name)](https://stefvanbuuren.name/fimd/sec-MCAR.html)

### Reddit & Letterboxd Scraping
- [Reddit Scraping with PRAW (dev.to)](https://dev.to/agenthustler/how-to-scrape-reddit-in-2026-subreddits-posts-comments-via-python-4el5)
- [Letterboxd Scraping Tools (apify.com)](https://apify.com/logiover/letterboxd-film-review-scraper/api/mcp)

---

## Closing Thoughts

**Your N=162 ceiling is real, but breakable:**

1. **MovieLens warm-start + synthetic twin** gives you 10K pseudo-labeled examples (2–3 week timescale)
2. **GenMatch or IPW** corrects selection bias rigorously (publishable-quality causal inference)
3. **TVAE or CTGAN** generates novel synthetic movies (opens generative pipeline for your wild ideas: "generate ideal movie for Temilola")
4. **Combining all three** yields a coherent narrative: bias-corrected augmentation + cold-start warm-up + generative modeling

**For the rubric:**
- Novel methods: GenMatch (causal matching), TVAE (generative), counterfactual augmentation (causal inference)
- Math-heavy: Propensity scoring, genetic algorithms, VAE ELBO, diffusion SDEs
- From-scratch: Feasible for GenMatch, TVAE, and synthetic twin
- Visuals: Propensity score overlap plots, augmented vs. real feature distributions, taste-neighbor embeddings, latent space traversals

This should position Pipeline 3 as "mind-blowing" — rigorous, mathematically dense, and genuinely novel.

---

*Generated by Claude Haiku Research Agent, April 16, 2026*
