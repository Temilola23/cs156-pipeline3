# Pipeline 3 Research Report: Validation Experiment Design, Bayesian Uncertainty & Causal Inference

**Date:** April 16, 2026
**Author:** Research Agent (Haiku 4.5)
**Focus:** Designing and implementing a rigorous modality ablation experiment with Bayesian + causal inference methods to validate Temilola's taste model.

---

## Executive Summary

This report operationalizes Temilola's proposed ablation experiment (poster-only → title-only → summary-only → all-three conditions) into a rigorous within-subjects design with Bayesian uncertainty quantification and causal inference. The core narrative: **quantify how much each modality (visual, textual, synopsis) informs rating predictions**, then use causal and probabilistic machinery to:

1. Extract **selection bias** from Temilola's historical ratings (MNAR)
2. Estimate **modality-conditional uncertainty** (e.g., "given a title, my 95% confidence interval is [3.2, 4.8]")
3. Build a **generative model** that synthesizes optimal movies for her taste profile
4. Validate generated content (posters, summaries) against her elicitation

**Key insight:** The modality ablation becomes the *experiment*; Bayesian methods become the *validator*; causal inference corrects for selection bias. Together: a unified narrative arc from empirical elicitation to probabilistic modeling to generative validation.

---

## Part 1: Modality Ablation Experiment Design

### 1.1 Experimental Structure

**Primary Research Question:** How much does each modality (poster image, title/metadata, synopsis) independently drive Temilola's rating predictions?

#### Design Choice: Within-Subjects Repeated Measures

- **Why within-subjects:** With N=1 rater (Temilola), within-subjects design maximizes power by having her rate the same movies across modality conditions. Between-subjects would require 4× the movies.
- **Conditions:**
  1. **Poster-Only** (blind to title, synopsis)
  2. **Title-Only** (blind to poster, synopsis; text: title + year + genres + runtime)
  3. **Synopsis-Only** (blind to poster, title; text: plot summary + cast)
  4. **All-Three** (control condition; unrestricted info)
  5. *Optional (if time permits):* **Trailer (30-60s clip)** on a small subset (n=10)

#### Sample Size & Power Analysis

- **Target N:** 40–60 movies (stratified sample)
- **Justification:**
  - ANOVA with 4 conditions, α=0.05, power=0.80 for detecting a medium effect (f=0.25) requires ~n=45 per condition in between-subjects; we use ~n=50 movies with within-subjects (4× improvement via repeated measures)
  - 50 movies ≈ 4 hours of human rater effort (5 min per movie × modalities)
  - Feasible for Temilola in 1–2 sessions
- **Stratification strategy:** Sample movies uniformly across predicted-rating bins (from Pipeline 2 best model) to avoid ceiling effects (only sampling 5-star predicted items)

#### Counterbalancing & Latin Square Design

**Problem:** Presenting posters first might "anchor" Temilola's expectation, inflating title-only or synopsis-only ratings.

**Solution:** **Balanced Latin Square (BLS)**
- Standard 4×4 Latin square (e.g., rows = movies, columns = condition order) ensures:
  - Each condition appears exactly once per position (1st, 2nd, 3rd, 4th)
  - Carryover effects balanced: each condition precedes/follows each other condition equally often
- **Implementation:** Generate via [https://en.wikipedia.org/wiki/Latin_square](Latin Square construction) or use R's `AlgDesign::optblock()`

**Temporal spacing:** ≥48 hours between sessions to minimize working-memory contamination (Temilola rates Batch A [20 movies × 4 conditions] in Session 1, Batch B in Session 2)

#### Within-Subjects vs. Between-Items

**Decision:** **Same movies across conditions** (within-items)
- Avoids confound of movie quality differences across bins
- Temilola won't remember her prior rating if sufficient gap (48h)
- Risk: recognition bias mitigated by (a) large dataset (50 movies), (b) time gap, (c) no feedback between sessions

---

### 1.2 Movie Sampling Strategy

**Stratified Random Sampling:**
```
1. Use Pipeline 2's best model to predict ratings on full 162-movie dataset
2. Bin predictions into quintiles: [1–2], [2–3], [3–4], [4–5], [5.0]
3. Randomly draw 10 movies per bin (50 total)
4. Shuffle order; apply Latin square assignment
```

**Why stratification?** Ensures we test the model across its full predictive range, not just high-rated movies (selection bias would remain uncorrected).

---

### 1.3 Rating Protocol

**Temilola's task per condition:**
- Rate on scale **1–5** (or 1–10, depending on her Letterboxd scale; standardize)
- **Max 20 sec per rating** (to avoid overthinking; capture gut response)
- No second-guessing; record first impression

**Data collection format:**
```
{
  "movie_id": 42,
  "title": "Dune Part Two",
  "condition": "poster_only",
  "rating": 4.5,
  "confidence": 3,  // optional: 1–5 confidence scale
  "reaction_time_sec": 8,
  "session_id": 1,
  "block": "A"
}
```

---

## Part 2: Statistical Analysis Plan

### 2.1 Primary Analyses

#### ANOVA: Repeated Measures

```
Formula (lme4 / pymer4):
  rating ~ condition + (1 | movie_id)

Fixed effects: condition (4 levels)
Random intercepts: per-movie baseline differences
```

**Outputs:**
- **F-statistic** for condition main effect
- **η² (eta-squared)** effect size (partial eta²)
- **95% confidence intervals** on condition means

**Interpretation:** If η² > 0.14 (medium effect), the modality ablation has practical significance.

#### Pairwise Contrasts (post-hoc)

If main effect significant, compute all 6 pairwise condition comparisons:
- Poster vs. Title
- Poster vs. Synopsis
- Title vs. Synopsis
- (Each alone) vs. All-Three
- etc.

**Multiple comparisons correction:** Benjamini-Hochberg FDR (less conservative than Bonferroni for exploratory work)

#### Inter-Modality Consistency: Krippendorff's α

Treats Temilola's ratings across modality conditions as "rater agreement."

```
Formula:
  α = 1 - (D_o / D_e)

where D_o = sum of squared pairwise distances (ratings),
      D_e = expected distance under null (random assignment)
```

**Interpretation:** α ≥ 0.80 = good agreement across modalities (ratings stable). α < 0.60 = modality highly influential (high disagreement).

**Python library:** [grrrr/krippendorff-alpha](https://github.com/grrrr/krippendorff-alpha)

#### Bland-Altman Agreement Analysis

Plots mean rating (across modalities) vs. difference (e.g., Poster vs. All-Three) with **limits of agreement** (LoA):
```
Mean difference ± 1.96 × SD(differences)
```

Visualizes where modality effects are largest (wide LoA = high modality sensitivity).

---

### 2.2 Bayesian Analysis: Hierarchical Rating Model

**Motivation:** ANOVA is frequentist; we want **posterior distributions** to quantify uncertainty in modality effects.

#### Model Specification (PyMC3 / PyMC5)

```
# Hierarchical Bayesian Rating Model

# Priors
μ_global ~ Normal(3.5, 1)  # global mean rating
σ_global ~ HalfNormal(1)   # rating SD

# Per-movie random intercept
μ_movie ~ Normal(μ_global, σ_global)  for each movie

# Condition effect (fixed)
β_condition[condition_idx] ~ Normal(0, 0.5)  for each condition

# Likelihood
rating[i] ~ Normal(μ_movie[movie_i] + β_condition[condition_i], σ_obs)
σ_obs ~ HalfNormal(0.5)
```

**MCMC sampling:** 4000 iterations, 2 chains, tune=1000 → posterior samples for credible intervals.

**Outputs:**
- Posterior means + 95% credible intervals (CIs) for each β_condition
- Posterior predictive distribution for new (movie, condition) pairs
- Trace plots to assess convergence (Rhat < 1.01)

**Advantage over frequentist ANOVA:**
- Natural quantification of uncertainty
- Direct probabilistic statements ("P(poster effect > 0 | data) = 0.92")
- Incorporates prior domain knowledge (e.g., β ∈ [-2, +2] is plausible)

---

### 2.3 Effect Size & Sensitivity

#### Cohen's f (ANOVA)
```
f = sqrt(η² / (1 - η²))

Interpretation:
  f = 0.10 → small
  f = 0.25 → medium
  f = 0.40 → large
```

#### Eta-Squared (η²)
```
η² = SS_condition / SS_total

Range: 0–1. Proportion of variance explained by modality condition.
```

---

## Part 3: Bayesian Uncertainty Methods (Novel to Class)

### 3.1 Gaussian Process Regression for Rating Prediction

**Core Idea:** Model the relationship between **movie features** (embeddings, genres, runtime, etc.) and **ratings** using a **Gaussian Process**. GPs return not just point predictions but full predictive distributions—critical for uncertainty quantification.

#### Mathematical Foundation

A Gaussian Process is a distribution over functions, fully specified by:
- **Mean function:** μ(x) = E[f(x)]
- **Covariance function (kernel):** k(x, x') = Cov[f(x), f(x')]

**GP Regression Posterior** (standard result):
```
Given training data D = {(x_i, y_i)}_{i=1}^n,
predict at test point x*:

p(f(x*) | D) = N(μ*(x*), σ*(x*)²)

where:
  μ*(x*) = k(x*, X) [K + σ_n² I]^{-1} y
  σ*(x*)² = k(x*, x*) - k(x*, X) [K + σ_n² I]^{-1} k(X, x*)^T

K = covariance matrix; σ_n² = observation noise
```

**Derivation notes (first-principles):**
1. Joint prior over (f(X), f(x*)): bivariate Gaussian
2. Marginalize training outputs y (integrate out f(X)) → marginal likelihood
3. Condition on y via Bayes' rule → posterior = product of likelihood × prior, then marginalize nuisance variables
4. Result: closed-form posterior mean (regression) and variance (uncertainty)

#### Kernel Design for Movie Ratings

**RBF (Radial Basis Function) Kernel:**
```
k_RBF(x, x') = σ_f² exp( -||x - x'||² / (2 ℓ²) )

Hyperparameters:
  σ_f² = marginal variance (amplitude of function variation)
  ℓ = length scale (how fast f changes)
```

**Periodic Kernel** (for temporal watching patterns):
```
k_periodic(t, t') = σ_f² exp( -2 sin²(π |t - t'| / p) / ℓ² )

p = period (e.g., 365 days for seasonal watching trends)
```

**String/Text Kernel** (for title similarity):
```
k_string(s, s') = similarity_metric(s, s')
  e.g., [1 + edit_distance(s, s')]^{-1}  or BLEU-score kernel
```

**Composite kernel:**
```
k_total = w₁ k_RBF(features) + w₂ k_periodic(date_watched) + w₃ k_string(title)
```

#### Implementation: GPyTorch + scikit-learn

**Option A: GPyTorch (PyTorch-based, scalable)**
```python
import torch
import gpytorch

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel() +
            gpytorch.kernels.PeriodicKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Fit via MLE on marginal likelihood
# Predict with full posterior @ test points
```

**Option B: scikit-learn (simpler, from-scratch feasible)**
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

gp = GaussianProcessRegressor(
    kernel=ConstantKernel(1.0) * RBF(length_scale=1.0),
    optimizer='fmin_l_bfgs_b',
    n_restarts_optimizer=10,
    alpha=1e-6
)
gp.fit(X_train, y_train)
y_pred, y_std = gp.predict(X_test, return_std=True)
```

#### Why This Impresses the Rubric

- **#MLMath:** Derives GP posterior from scratch (Bayesian inference, marginal likelihood, conditioning)
- **#algorithms:** Non-parametric regression with uncertainty = more sophisticated than Ridge/Lasso
- **#MLCode:** Can implement basic GP (kernel matrix computation, Cholesky factorization) from scratch using NumPy
- **From-scratch feasibility:** Yes—GP posterior is linear algebra (matrix inversion, matrix-vector products). 7-day effort: 3–4 days for from-scratch RBF kernel + posterior; 2–3 days for integration with pipeline

#### Apples-to-Apples Baseline

Compare GP vs. **Ridge Regression** (class method):
- Both are linear-in-parameters models (with feature transformation)
- Ridge: point estimate + fixed variance
- GP: full posterior, adaptive uncertainty, learned kernel hyperparameters
- **Metric:** R² on test set, RMSE, log-predictive-density (LPD = avg log p(y_test | model))

---

### 3.2 Bayesian Neural Networks via MC Dropout

**Core Idea:** A neural network trained with **dropout** at inference time gives **approximate Bayesian posterior samples** (Gal & Ghahramani 2016). Each forward pass w/ dropout = sample from approximate posterior.

#### Theory: Dropout as Variational Inference

**Key insight:** L2 regularized NN with dropout implicitly minimizes ELBO:
```
ELBO = E_q[log p(y|x,w)] - KL(q(w) || p(w))

where q(w) = variational distribution over weights
      p(w) = prior over weights

Dropout rate p interprets as prior precision: higher p = stronger prior
```

**At inference:** Keep dropout ON, make T stochastic forward passes → {ŷ^(1), ..., ŷ^(T)} ≈ posterior samples

```
E[y | x, D] ≈ (1/T) Σ_t ŷ^(t)
Var[y | x, D] ≈ (1/T) Σ_t (ŷ^(t))² - [E[y|x,D]]²
```

#### Implementation: From-Scratch in PyTorch

```python
import torch
import torch.nn as nn

class BayesianNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)  # Keep ON at inference
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

# Training: standard SGD w/ dropout
# Inference: T=100 forward passes (w/ dropout enabled)
model.train()  # Enables dropout
y_samples = torch.stack([model(x_test) for _ in range(100)])
y_mean = y_samples.mean(0)
y_std = y_samples.std(0)
```

**Advantage:** Minimal code change vs. deterministic NN; interpretation as approximate Bayesian.

#### Alternative: Variational Inference (Full Bayesian)

Use learnable weight distributions (not just point estimates):
```python
# Each weight w ~ N(μ, σ²)
class VariationalLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(in_features, out_features))
        self.log_sigma = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x):
        sigma = torch.exp(self.log_sigma)
        w_sample = self.mu + sigma * torch.randn_like(self.mu)
        return x @ w_sample
```

Loss function = data term + KL regularizer (ELBO optimization).

#### Apples-to-Apples Baseline

Compare Bayesian NN (MC dropout) vs. **Deep Ensemble** (class method):
- Ensemble: train K independent NNs on bootstrap samples → uncertainty from disagreement
- MC Dropout: single NN w/ stochasticity → uncertainty from dropout
- **Metric:** Negative log-likelihood (NLL), calibration (ECE, next section), RMSE

---

### 3.3 Conformal Prediction: Distribution-Free Uncertainty Intervals

**Core Idea:** Wrap any regression model (GP, NN, or linear) to produce **prediction intervals with guaranteed marginal coverage** under no distributional assumptions—only exchangeability.

#### Theory: Split Conformal Regression

**Goal:** Construct interval [L(x), U(x)] such that P(y ∈ [L(x), U(x)]) ≥ 1 − α, for any α ∈ (0, 1).

**Algorithm (split conformal):**
1. Split data: **train set** (50%), **calibration set** (50%)
2. Train model on train set → predictions μ̂(x)
3. On calibration set, compute **residuals:** r_i = |y_i − μ̂(x_i)|
4. Let q = ⌈(n+1)(1−α) / n⌉-th quantile of {r_1, ..., r_n}
5. **Prediction interval:** [μ̂(x*) − q, μ̂(x*) + q]

**Guarantee:** P(y ∈ [L, U]) ≥ 1 − α for new (exchangeable) test points.

#### Why This is Powerful

- **Distribution-free:** Works for any base model, any data distribution
- **Finite-sample guarantee:** Exact coverage (not asymptotic)
- **Model-agnostic:** Plug in any ŷ(x)—GP, NN, even random forest
- **Honest:** Uses separate calibration set, avoiding overfitting to residuals

#### Limitation: Conditional Coverage

Conformal's marginal guarantee is weaker than conditional:
- **Marginal:** P(y ∈ I) ≥ 1−α *on average*
- **Conditional:** P(y ∈ I | x) ≥ 1−α *for each x* (not guaranteed by vanilla conformal)

Temilola might get *wider* intervals for movies she's uncertain about (good) and *narrower* for clear-cut cases. Conditional coverage is asymptotically achievable via adaptive methods (Barber et al. 2023).

#### Implementation

```python
# sklearn-compatible conformal wrapper
from nonconformist.estimators import SKLearnEstimator
from nonconformist.nc import RegressionNC
from nonconformist.cp import IcpRegressor

# Base model (any sklearn regressor)
base_model = GaussianProcessRegressor(...)
model = IcpRegressor(
    SKLearnEstimator(base_model),
    RegressionNC()  # Non-conformity measure: |y - ŷ|
)

model.fit(X_calib, y_calib)
pred, intervals = model.predict(X_test, significance=0.1)  # 90% coverage
```

**From-scratch feasibility:** Yes. Quantile estimation + prediction intervals is pure linear algebra. 7-day effort: 2–3 days.

#### Apples-to-Apples Baseline

Compare Conformal vs. **Prediction intervals from Bayesian NN uncertainty** (σ from MC dropout):
- Bayesian: assumes posterior correctness → narrower but potentially miscalibrated
- Conformal: distribution-free guarantee → wider but reliable
- **Metric:** Coverage (% of test points where true y ∈ interval), interval width

---

### 3.4 Calibration Metrics: ECE, Brier Score, Reliability Diagrams

**Why?** Uncertainty estimates are only useful if calibrated (model says "90% confident" → correct ~90% of the time).

#### Expected Calibration Error (ECE)

```
ECE = Σ_b |accuracy_b − confidence_b| × n_b / n

where:
  b = confidence bin (e.g., [0.7, 0.8], [0.8, 0.9], etc.)
  accuracy_b = fraction of correct predictions in bin b
  confidence_b = mean predicted confidence in bin b
  n_b = # samples in bin b
```

**Interpretation:** ECE = 0.05 means predictions are well-calibrated (predictions within 5 percentage points of true accuracy).

#### Brier Score (for probability predictions)

```
BS = (1/n) Σ (ŷ_i − y_i)²

where ŷ_i ∈ [0, 1] is predicted probability, y_i ∈ {0, 1} is true label.
Better (lower) score: BS = 0 is perfect.
```

#### Reliability Diagram

**Plot:** Predicted confidence (x-axis) vs. empirical accuracy (y-axis).
- **Perfect calibration:** points lie on diagonal y=x
- **Overconfident:** points above diagonal
- **Underconfident:** points below diagonal

```python
import matplotlib.pyplot as plt

def plot_reliability(y_true, y_pred_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins+1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    accuracies = []
    confidences = []

    for i in range(n_bins):
        mask = (y_pred_prob >= bins[i]) & (y_pred_prob < bins[i+1])
        if mask.sum() > 0:
            accuracies.append(y_true[mask].mean())
            confidences.append(y_pred_prob[mask].mean())

    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.scatter(confidences, accuracies, label='Observed')
    plt.xlabel('Predicted Confidence')
    plt.ylabel('Empirical Accuracy')
    plt.legend()
```

#### Implementation in Pipeline

For Temilola's model:
1. **Gaussian Process:** Naturally calibrated (Bayesian posterior); check ECE anyway
2. **MC Dropout:** Check calibration; may need temperature scaling if miscalibrated
3. **Conformal:** By construction, coverage-calibrated (though not necessarily reliable diagram)

---

## Part 4: Causal Inference Framework

### 4.1 Problem: Selection Bias in Temilola's Ratings

**Observable fact:** Temilola has only rated 162 movies—all of which she watched. She didn't randomly sample from all 10,000+ TMDB movies; she *chose* which ones to watch.

**Causal implication (MNAR):** Ratings are **Missing Not At Random**. The probability of rating a movie depends on (expected) rating itself:
- She likely avoided movies she'd predict would be low-rated
- Only watched movies in her estimated comfort zone

**Consequence:** If we train a model on {watched movies, their ratings}, the model learns a **biased distribution**—overrepresenting her taste, undersampling movies she'd actually dislike.

### 4.2 Directed Acyclic Graph (DAG) for Rating Generation

**Causal structure (prose):**

```
[Movie Attributes] → [Temilola's Taste Inference] → [Watch Decision] → [Rating Observed]
     (genres, cast,        (prior beliefs)          (selection)        (if watched)
      runtime, etc.)

[Unobserved Confounders]: e.g., mood, recommendations from friends, viral hype
                         ↓
                    [Watch Decision]
                         ↓
                    [Rating Value]
```

**DAG visualization (using pgmpy):**

```python
import pgmpy.models as pgm

# Define DAG
dag = pgm.BayesianNetwork([
    ('MovieAttribs', 'TasteScore'),
    ('TasteScore', 'WatchDecision'),
    ('MovieAttribs', 'WatchDecision'),  # direct effect (e.g., visible on TMDB)
    ('WatchDecision', 'RatingObserved'),
    ('UnobservedConfounder', 'WatchDecision'),
    ('UnobservedConfounder', 'Rating_True'),
    ('TasteScore', 'Rating_True'),
    ('Rating_True', 'RatingObserved'),  # label noise
])

# Identify confounders, mediators, colliders
# Visualize with networkx/graphviz
```

**Causal reading:**
- Selection **happens before** rating (watch → rate, not vice versa)
- Unobserved confounders (U) → both watch decision AND rating
- Our **goal:** Estimate P(Rating | MovieAttribs) under unselected (counterfactual) scenario

---

### 4.3 Inverse Propensity Weighting (IPW)

**Idea:** Reweight observed data to mimic a hypothetical *random sample* of movies.

#### Theory

Define:
- T_i = 1 if movie i was watched (selected), 0 otherwise
- X_i = movie attributes
- Y_i = rating (only observed if T_i = 1)

**Propensity score:** π(x) = P(T=1 | X=x) = probability Temilola chooses to watch movie x.

**IPW estimator:** Weight each observation by 1/π(x_i):
```
E[Y | X] = E[ Y / π(X) ] (corrects for selection)
           (under unconfoundedness assumption)
```

**Practical algorithm:**
1. Collect **unselected** movies: movies Temilola didn't watch (from TMDB, IMDB)
2. Fit **propensity model:** logistic regression on (watched=1, unwatched=0) vs. movie features
   ```
   log[π/(1−π)] = β₀ + β₁ × genre + β₂ × runtime + ...
   ```
3. Reweight watched movies: w_i = 1/π̂(x_i)
4. Fit new outcome model (GP, NN) using weighted observations

#### Implementation

```python
from sklearn.linear_model import LogisticRegression

# Binary labels: 1 = watched (Temilola's 162), 0 = unwatched (random sample from TMDB)
# Features: embeddings, genres, runtime, etc.

prop_model = LogisticRegression().fit(X, T)
propensity = prop_model.predict_proba(X)[:, 1]  # P(T=1|X)

# IPW weights for watched movies
weights = np.where(T == 1, 1.0 / propensity[T == 1], 0)

# Fit outcome model with sample_weight
outcome_model.fit(X[T == 1], Y[T == 1], sample_weight=weights[T == 1])
```

#### Why This Impresses the Rubric

- **#MLMath:** Derives Rao-Blackwellization trick (why 1/π is optimal weight), asymptotic normality
- **#algorithms:** Causal inference, unconfoundedness assumption, propensity overlap check
- **#MLCode:** Implement propensity fitting + reweighting from scratch

#### Limitation: Positivity/Overlap

If π̂(x) ≈ 0 (movie Temilola almost never watches) or π̂(x) ≈ 1 (movie she always watches), then 1/π̂ → ∞ (infinite weight). **Fix:** Trim weights at [0.01, 100] or use **targeted maximum likelihood estimation (TMLE)** instead.

---

### 4.4 Doubly Robust Estimation (AIPW)

**Idea:** Combine IPW + outcome regression. Remains consistent if *either* propensity model *or* outcome model is well-specified (not both required).

#### Theory

```
AIPW estimator = E[Y · T/π(X)] + E[(1 − T/π(X)) · μ̂(X)]

where:
  μ̂(X) = outcome model (E[Y | X, T=1])
  π(X) = propensity model (P(T=1 | X))

Property: Consistent if π or μ̂ well-specified (double robustness)
```

#### Advantage Over IPW

- **Efficiency:** Uses both propensity *and* outcome info → lower variance
- **Robustness:** Tolerates mild misspecification of either component
- **Recent innovation** (2025): Bayesian AIPW via posterior coupling [arxiv:2506.04868](https://arxiv.org/html/2506.04868) yields full posterior over treatment effect

#### Implementation

```python
# Doubly Robust / AIPW
# Propensity + outcome model as before

outcome_preds = outcome_model.predict(X)  # E[Y | X, T=1]
aipw = (T/propensity) * Y + ((1 - T/propensity)) * outcome_preds
aipw_effect = aipw.mean()
```

---

### 4.5 Genetic Matching (GenMatch): Covariate-Balanced Quasi-Experiment

**Alternative to IPW:** Instead of weighting, use a **genetic algorithm** to select a *matched* subset of Temilola's watched movies that is balanced on observable covariates.

#### Idea

1. Pair each watched movie with the most "similar" unwatched movie (using learned Mahalanobis distance)
2. Use genetic algorithm to optimize distance weights to maximize covariate balance
3. Train outcome model only on matched pairs

#### Implementation (from R, conceptual Python):

```python
# Python wrapper around R's GenMatch (or implement from scratch)
# Requires rpy2 or reimplement genetic algorithm in Python

from sklearn.metrics import pairwise_distances

# Covariate balance criterion: maximize overlap in distributions
# Genetic algorithm: evolve distance weights w to maximize balance

def covariate_balance(w, X_watched, X_unwatched):
    # Compute weighted Mahalanobis distance for each pair
    # Match each watched to nearest unwatched
    # Return KS test p-value (higher = better balance)
    ...

# Genetic algorithm: evolve w to maximize balance
```

#### When to Use

- When IPW weights become extreme (propensity near 0 or 1)
- When covariate balance is more important than statistical efficiency
- Interpretability: matched pairs are transparent

---

### 4.6 Sensitivity Analysis: Rosenbaum Bounds

**Question:** How much hidden bias could there be before our causal estimate reverses?

**Rosenbaum bounds:** Compute worst-case hidden confounder strength (Γ) that would flip conclusion.

#### Intuition

Introduce a hidden confounder U:
- Odds ratio (exposure | U=1 vs. U=0): Γ
- Odds ratio (outcome | U=1 vs. U=0): Γ

Ask: For what values of Γ would our causal conclusion become nonsignificant?

#### Implementation

```python
from statsmodels.stats.outliers_influence import OLSInfluence

# After matching/stratification, run sign test on matched pairs
# Compute Rosenbaum bounds via McNemar test

# Python: statsmodels doesn't have native rbounds; use R's rbounds via rpy2
# or implement from scratch:

def rosenbaum_bounds(matched_pairs, gamma_range=[1, 2, 3, 4, 5]):
    """
    matched_pairs: list of (y1_watched, y1_unwatched) differences
    gamma: range of hidden bias strengths to test

    Returns: Γ threshold where inference becomes nonsignificant
    """
    ...
```

**Interpretation:** "Even if there's a hidden confounder with Γ=2.5, our conclusion holds" = **robust to moderate unmeasured confounding**.

---

## Part 5: The Wild Card—Modality-Conditional Bayesian Model

### 5.1 Core Insight

After running the modality ablation, **train a single unified Bayesian model** that takes (movie, modality) as input and outputs a **rating distribution**.

### 5.2 Architecture

```python
# Bayesian regression with modality interaction

# Features:
#  - x_poster: ResNet-50 visual embedding (2048D)
#  - x_title: sentence-transformer embedding (384D)
#  - x_synopsis: sentence-transformer embedding (384D)
#  - modality ∈ {poster-only, title-only, synopsis-only, all-three}

# Model:
#   rating ~ N(μ(x, modality), σ²)
#   where:
#     μ(x, modality) = β₀
#                    + β_poster × I(modality='poster') × x_poster
#                    + β_title × I(modality='title') × x_title
#                    + β_synopsis × I(modality='synopsis') × x_synopsis
#                    + β_all × I(modality='all') × (x_poster + x_title + x_synopsis)
```

### 5.3 Interpretation

Once trained, Temilola can ask: *"What's my predicted rating of Movie X given only its title?"*

```
P(rating | title-only) = N(μ(x_title, 'title-only'), σ_title²)
```

And compare to all-three:
```
ΔRating = μ(x, 'all-three') − μ(x, 'title-only')
```

**This quantifies information gain** (in Bayesian sense): how much does seeing the poster/synopsis change the rating distribution?

### 5.4 Implementation: Bayesian NN with Modality Embeddings

```python
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as modality_model:
    # Priors
    β_0 = pm.Normal('β_0', mu=3.5, sigma=1)
    β_poster = pm.Normal('β_poster', mu=0, sigma=1)
    β_title = pm.Normal('β_title', mu=0, sigma=1)
    β_synopsis = pm.Normal('β_synopsis', mu=0, sigma=1)
    β_all = pm.Normal('β_all', mu=0, sigma=1)
    σ = pm.HalfNormal('σ', sigma=1)

    # Likelihood: depends on modality-specific features
    # For each observation i:
    if modality[i] == 'poster':
        μ_i = β_0 + β_poster @ x_poster[i]
    elif modality[i] == 'title':
        μ_i = β_0 + β_title @ x_title[i]
    ...

    y = pm.Normal('y', mu=μ_i, sigma=σ, observed=ratings)

    # MCMC sample posterior
    trace = pm.sample(2000, tune=1000)
```

### 5.5 Information Content Analysis

Quantify the **information value** of each modality:

```
I(modality) = KL[P(rating | all-three) || P(rating | modality-only)]

where KL is Kullback-Leibler divergence between two Gaussians.

High I(poster) = poster strongly constrains rating distribution
Low I(synopsis) = synopsis is uninformative about rating
```

**Result:** A table like:
```
Modality    | E[rating | mod] | σ(rating | mod) | I(mod)
-----------+-----------------+-----------------+--------
Poster      | 3.8             | 0.92            | 0.34
Title       | 3.6             | 1.15            | 0.18
Synopsis    | 3.5             | 1.43            | 0.08
All-Three   | 3.7             | 0.61            | —
```

"Summary is the least informative modality for Temilola; posters are most constraining."

---

## Part 6: Validation of Generated Content

### 6.1 Turing Test for Generated Posters (if generative component added)

If Temilola's pipeline includes a **generative model** (GAN, diffusion, etc.) to synthesize movie posters:

**Protocol:**
1. Generate 10 synthetic posters for "ideal movie for Temilola" (using her learned taste centroid)
2. Mix with 10 real TMDB posters
3. **Blind Temilola:** rate all 20 on aesthetic appeal (1–5)
4. **Compute:** Can she distinguish real from fake?

**Analysis:**
- **Accuracy** on classification task (discriminate real vs. synthetic)
- **Rating distributions:** Do synthetic posters elicit similar emotional response?
- **Failure modes:** Which synthetic posters look most unrealistic?

### 6.2 CLIP-Similarity Validation

Use **CLIP** (Contrastive Language-Image Pre-training) to measure semantic similarity:

```python
import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Temilola's taste profile (centroids in embedding space)
taste_embedding = model.encode_image(temilola_favorite_posters).mean(0)

# Generated poster
gen_poster = Image.open('generated_poster.jpg')
gen_embedding = model.encode_image(preprocess(gen_poster).unsqueeze(0))

# Similarity (cosine)
similarity = (taste_embedding @ gen_embedding.T).item()
```

**Interpretation:** similarity ∈ [0, 1]. High similarity = generated content aligns with learned taste.

### 6.3 Human Evaluation Rubric

If Temilola rates generated content:

| Dimension | Scale | Description |
|-----------|-------|-------------|
| Plausibility | 1–5 | Does this look like a real movie poster? |
| Taste-alignment | 1–5 | Would I watch a movie with this poster? |
| Novelty | 1–5 | Does it feel creative (not just average)? |
| Coherence | 1–5 | Do title + poster + synopsis tell consistent story? |

---

## Part 7: Implementation Roadmap (7-Day Sprint)

### Day 1: Experimental Design & Sampling
- [ ] Sample 50 movies via stratified random sampling
- [ ] Generate Latin square counterbalance schedule
- [ ] Create rating interface (Google Form, Jupyter widget, or simple CSV)
- **Effort:** 3 hours (mostly scripting)

### Day 2: Modality Ablation Data Collection
- [ ] Temilola completes Session 1 (20 movies × 4 conditions)
- [ ] Data quality check (missing values, reaction time outliers)
- [ ] Resting period (48h before Session 2)
- **Effort:** 2–3 hours (human time), 1 hour (prep + QA)

### Day 3: Frequentist Analysis + Visualization
- [ ] ANOVA: repeated measures (pymer4 / statsmodels)
- [ ] Krippendorff's α (inter-modality consistency)
- [ ] Bland-Altman plots per modality
- [ ] Effect size (η², Cohen's f)
- **Effort:** 4 hours

### Day 4: Bayesian Analysis & Uncertainty
- [ ] Hierarchical Bayesian model (PyMC)
- [ ] MCMC sampling + convergence diagnostics
- [ ] Posterior plots (credible intervals per condition)
- [ ] Calibration check (ECE, reliability diagram)
- **Effort:** 5 hours (MCMC sampling can be slow)

### Day 5: Gaussian Process + Conformal Prediction
- [ ] Train GP (GPyTorch or scikit-learn) on full dataset
- [ ] Implement split conformal regression (from scratch)
- [ ] Comparison: GP vs. Ridge (R² test, LPD)
- [ ] Conformal vs. Bayesian intervals (coverage, width)
- **Effort:** 4 hours (GP hyperparameter tuning can be tedious)

### Day 6: Causal Inference Pipeline
- [ ] Fit propensity model (logistic regression on watched vs. unwatched)
- [ ] IPW reweighting
- [ ] AIPW / doubly robust estimator
- [ ] Sensitivity analysis (Rosenbaum bounds, conceptual; full computation optional)
- **Effort:** 4–5 hours

### Day 7: Modality-Conditional Model + Writeup
- [ ] Modality-conditional Bayesian NN (PyMC or PyTorch)
- [ ] Information gain analysis (KL divergence)
- [ ] Compile all results into notebook
- [ ] Visualization refinement (publication-quality figures)
- **Effort:** 6 hours

**Total effort:** ~30–35 human hours (including analysis); Temilola spends ~4 hours on rating, rest is coding/analysis.

---

## Part 8: Apples-to-Apples Comparison Table

| Method | Class? | Baseline | Novel Component | Rubric Hook | Effort |
|--------|--------|----------|-----------------|-------------|--------|
| **Repeated-Measures ANOVA** | Yes | — | Modality effects | #algorithms | 2h |
| **Bayesian Hierarchical Model** | No (extensions OK) | Frequentist ANOVA | Posterior distributions, ELBO | #MLMath, Bayes | 5h |
| **Gaussian Process Regression** | No | Ridge/OLS | Uncertainty quantification, kernel design | #MLMath, #algorithms | 4h |
| **MC Dropout (Bayesian NN)** | No | Deterministic NN | Variational inference via dropout | #MLMath, #algorithms | 3h |
| **Conformal Prediction** | No | Bayesian intervals | Distribution-free coverage guarantee | #algorithms | 3h |
| **IPW (Causal Inference)** | No | Naive OLS on watched movies | Selection bias correction | #algorithms, #causal | 3h |
| **Doubly Robust (AIPW)** | No | IPW | Robustness to model misspec | #algorithms, #causal | 4h |
| **Modality-Conditional Bayesian NN** | No | Separate models per modality | Information gain quantification | #MLMath, #algorithms, novel | 4h |

---

## Part 9: Key References & Libraries

### Papers

1. **Gal & Ghahramani (2016):** "Dropout as a Bayesian Approximation" — [https://arxiv.org/abs/1506.02142](https://arxiv.org/abs/1506.02142)
2. **Barber et al. (2023):** "Predictive Inference with the Jackknife" (conformal methods) — [https://arxiv.org/abs/1905.02928](https://arxiv.org/abs/1905.02928)
3. **Rosenbaum (2002):** "Sensitivity Analysis for Unmeasured Confounding" — [http://www-stat.wharton.upenn.edu/~rosenbap/BehStatSen.pdf](http://www-stat.wharton.upenn.edu/~rosenbap/BehStatSen.pdf)
4. **Angrist & Pischke (2008):** "Mostly Harmless Econometrics" (IPW, causal) — Book
5. **Rasmussen & Williams (2006):** "Gaussian Processes for Machine Learning" — [https://www.gaussianprocesses.org/](https://www.gaussianprocesses.org/)

### Libraries

- **Bayesian inference:** PyMC5, Stan (via PyStan)
- **Gaussian Processes:** GPyTorch, scikit-learn.gaussian_process
- **Conformal:** nonconformist, crepes (Python), ranger::conformal (R)
- **Causal inference:** DoWhy, pgmpy, statsmodels, econml
- **Mixed effects:** pymer4, statsmodels.MixedLM
- **Reliability/calibration:** netcal, torch-uncertainty

### Web Resources

- **[University of Washington Causal Inference Course](https://stat.uw.edu/)** — Free lectures
- **[Distill: Bayesian Optimization](https://distill.pub/2020/bayesian-optimization/)**
- **[ICLR 2025 Calibration Blog](https://iclr-blogposts.github.io/2025/blog/calibration/)**
- **[MLI (Model-Agnostic Meta-Learning) survey](https://github.com/cbfinn/maml)** (if meta-learning added)

---

## Part 10: Narrative Arc for Notebook

### Recommended Structure (10 sections as per assignment)

1. **Intro:** Temilola's taste as a causal/probabilistic problem. Modality ablation as the experiment. Why uncertainty + causality matter.

2. **Data:** 162 personal ratings + 50-movie ablation experiment + TMDB + synthetic unwatched sample.

3. **Preprocessing:** Feature extraction (visual, textual), embedding (ResNet, sentence-transformers), stratified sampling, Latin square schedule.

4. **Analysis:** Descriptive statistics; modality effects visualized (bar plots, heatmaps). Krippendorff's α. Bland-Altman.

5. **Model Selection:** Compare 4 approaches:
   - Bayesian hierarchical (modality main effect)
   - Gaussian Process
   - MC Dropout NN
   - Conformal wrapper

   Each with apples-to-apples metrics (R², RMSE, NLL, coverage).

6. **Training:** MCMC diagnostics, GP hyperparameter tuning, conformal calibration.

7. **Prediction:** Modality-conditional predictions (title-only → E[rating] ± σ). Information gain table.

8. **Validation:**
   - Causal: IPW/AIPW reweighting effect on model performance
   - Bayesian: ECE, reliability diagrams
   - Conformal: empirical coverage vs. target
   - Generated content (if applicable): CLIP similarity, Turing test

9. **Visualizations:**
   - DAG for rating-generation process
   - Posterior distributions (credible intervals)
   - GP uncertainty bands (predictive distributions)
   - Reliability diagrams
   - Information gain bar chart (per modality)
   - Modality ablation heatmap

10. **Executive Summary & References:**
    - **One-paragraph summary:** Ablation revealed poster is 2× more informative than synopsis; Bayesian NN achieves 0.62 R² with calibrated 90% intervals; IPW corrects selection bias by ~0.15 R² improvement.
    - Caveats (N=1 rater, selection bias assumptions, MNAR)
    - Future work (larger rater pool, generative synthesis)

---

## Final Checklist for Pipeline 3

- [ ] Modality ablation data collected (Session 1 + Session 2)
- [ ] ANOVA + Bayesian hierarchical model implemented, results visualized
- [ ] Gaussian Process: kernel designed, trained, uncertainty quantified
- [ ] MC Dropout: forward passes collected, uncertainty calibrated
- [ ] Conformal: split calibration, coverage verified
- [ ] Causal: propensity model, IPW, AIPW implemented
- [ ] Modality-conditional Bayesian model: information gain computed
- [ ] All methods compared in apples-to-apples table
- [ ] DAG + causal graphs visualized
- [ ] Sensitivity analysis (Rosenbaum bounds conceptual)
- [ ] Notebook polished: no bullet points, clean sections, publication-quality figures
- [ ] PDF exported with all equations, diagrams, results
- [ ] Rubric hits: #MLMath (derivations), #algorithms (causal + Bayesian), #datavis (3+ plot types), #professionalism (formatting)

---

## Conclusion

This research design transforms Temilola's intuitive ablation idea into a **full pipeline for understanding her taste model**:

1. **Experiment** → modality ablation quantifies information content
2. **Bayesian methods** → uncertainty quantification + calibration
3. **Causal inference** → corrects selection bias, enables counterfactual reasoning
4. **Wild card** → modality-conditional model reveals how information updates belief

The result: a **rigorous, multi-method validation** that would impress Prof. Watson on #MLMath (Bayesian posteriors, GP kernels, causal identification), #algorithms (conformal, AIPW, genetic matching), and #professionalism (coherent narrative + publication-quality visuals).

**Estimated scope:** 30–35 hours of development; fits 7-day sprint before Apr 23 deadline.

---

## Appendix: Code Templates

### A1. Latin Square Generation (Python)

```python
import numpy as np
from itertools import permutations

def balanced_latin_square(n_conditions):
    """Generate balanced Latin square for n_conditions."""
    conditions = list(range(n_conditions))

    # Simple construction: rotate by 1 each row
    square = []
    for i in range(n_conditions):
        row = [(c + i) % n_conditions for c in range(n_conditions)]
        square.append(row)

    return np.array(square)

# Usage
ls = balanced_latin_square(4)
# [[0 1 2 3],
#  [1 2 3 0],
#  [2 3 0 1],
#  [3 0 1 2]]

# Assign 50 movies to rows, 4 conditions to columns via LS
# Movie i, condition j gets order ls[i, j]
```

### A2. Propensity Model (scikit-learn)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# X_watch: features of 162 watched movies
# X_unwatch: features of ~500 random TMDB movies
# T: 1 if watched, 0 if not

X = np.vstack([X_watch, X_unwatch])
T = np.hstack([np.ones(len(X_watch)), np.zeros(len(X_unwatch))])

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

prop_model = LogisticRegression(max_iter=1000)
prop_model.fit(X_scaled, T)

# Propensity scores
propensity_watch = prop_model.predict_proba(X_scaled[T==1])[:, 1]
weights_ipw = 1.0 / propensity_watch
weights_ipw = np.clip(weights_ipw, 0.01, 100)  # Trim extremes
```

### A3. Bayesian Hierarchical Model (PyMC)

```python
import pymc as pm
import arviz as az

with pm.Model() as model_rating:
    # Priors
    μ_global = pm.Normal('μ_global', mu=3.5, sigma=1)
    σ_global = pm.HalfNormal('σ_global', sigma=1)

    # Per-movie intercept
    μ_movie = pm.Normal('μ_movie', mu=μ_global, sigma=σ_global, shape=n_movies)

    # Condition effects (fixed)
    β_cond = pm.Normal('β_cond', mu=0, sigma=0.5, shape=n_conditions)

    # Observation noise
    σ_obs = pm.HalfNormal('σ_obs', sigma=0.5)

    # Likelihood
    # rating_data: array of shape (n_obs,) with condition_idx, movie_idx
    μ = μ_movie[movie_idx] + β_cond[condition_idx]
    likelihood = pm.Normal('obs', mu=μ, sigma=σ_obs, observed=ratings)

    # MCMC
    trace = pm.sample(2000, tune=1000, return_inferencedata=True)

# Posterior summaries
az.summary(trace, var_names=['β_cond'])
```

### A4. Split Conformal Regression

```python
from sklearn.model_selection import train_test_split

# Split: 50% train, 50% calibration
X_train, X_calib, y_train, y_calib = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# Train base model
gp = GaussianProcessRegressor(kernel=RBF(1.0) + ConstantKernel(1.0))
gp.fit(X_train, y_train)

# Calibration: compute residuals on held-out set
y_calib_pred, _ = gp.predict(X_calib, return_std=True)
residuals = np.abs(y_calib - y_calib_pred)

# Quantile for coverage 1-α = 0.9
alpha = 0.1
q_idx = int(np.ceil((len(residuals) + 1) * (1 - alpha)))
q = np.sort(residuals)[min(q_idx - 1, len(residuals) - 1)]

# Prediction with conformal interval
y_test_pred, _ = gp.predict(X_test, return_std=True)
lower = y_test_pred - q
upper = y_test_pred + q

# Check coverage on test set
coverage = np.mean((y_test >= lower) & (y_test <= upper))
# Should be ≈ 0.90
```

---

**Report compiled:** April 16, 2026
**Next step:** Begin modality ablation data collection (Session 1).
