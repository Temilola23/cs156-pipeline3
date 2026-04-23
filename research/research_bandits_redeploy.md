# Pipeline 3: Contextual Bandits & Thompson Sampling — Decision Under Uncertainty

> **Executive summary:** The probabilistic models in Pipeline 3 (GP, conformal prediction, MC-Dropout) generate posterior distributions over expected ratings. Thompson sampling converts these posteriors into a decision policy: "Which next movie should Temilola watch to maximize discovery?" This layer closes the feedback loop: experiment → posterior → bandit decision → new rating → posterior update. It recasts Pipeline 3 from passive prediction into active learning + exploration-exploitation.

---

## 1. The narrative fit (why bandits belong in Pipeline 3)

### The problem bandits solve

By Act II, we have:
- **Gaussian Process** with posterior mean μ(x) and variance σ²(x) over expected ratings for unwatched movies.
- **Conformal prediction** wrapping the GP with distribution-free 90% intervals.
- **MC-Dropout Bayesian NN** as a second uncertainty engine.

But the notebook asks: *Given these posteriors, which next movie should Temilola watch?*

The naive answer: "Watch the movie with highest predicted μ(x)." This is **myopic greedy**—it ignores the value of *information*. If a movie has high μ but low σ, watching it won't teach us much. If a movie has uncertain μ but could unlock a new cluster of taste, it's worth exploring.

**Bandits solve this** by formalizing the exploration-exploitation tradeoff: sample from the posterior, pick the action (movie) with highest sampled reward under that sample.

### Where it slots into the narrative

**Before Act III (generation):** We have a probabilistic model of Temilola's taste. Thompson sampling lets us *actively choose the next experiment condition* in Act I's modality-ablation design: should we test poster-only or title-only next, given what we've learned?

**After Act II (uncertainty quantification):** Instead of just showing uncertainty bands, we use them to decide: "Your taste is uncertain between Sci-Fi and Romance. Watch this movie (located at the cluster boundary) to resolve that ambiguity."

**Bridge to Act III (generation):** Once we've used Thompson sampling to actively gather high-value ratings, the posterior is tighter, and Act III's CVAE sampling from the high-rating region is more confident.

---

## 2. The five bandit approaches (ranked by fit to this project)

### 2.1 Thompson Sampling (TS) on GP posterior — **CORE APPROACH**

**Description:**
Thompson sampling is the simplest form of posterior-sampling-based decision-making. At each step:
1. Sample a reward function from the posterior: θ ∼ p(θ | observed data)
2. Greedily optimize under that sample: a* = argmax_{a} μ(a; θ)
3. Observe reward r(a*), update posterior
4. Repeat

For a GP regressor, the posterior is a Gaussian, so sampling θ means sampling function-values at candidate movie embeddings. This gives an "optimistic" decision rule: you're not gambling, you're committing to one plausible view of the reward landscape.

**Math hook:**
$$p(\theta | D) \propto p(D | \theta) p(\theta)$$
Sample $\tilde{\theta} \sim \mathcal{N}(\mu, \Sigma)$ where $\mu, \Sigma$ come from GP posterior.
$$a^* = \arg\max_a \tilde{\mu}(a; \tilde{\theta})$$

**Why it's perfect for Pipeline 3:**
- We're *already building a GP*. TS uses its posterior directly.
- From-scratch implementation: ~30 lines (Cholesky sample, find argmax over grid).
- Plays beautifully with conformal intervals: Thompson sample from conformal region.
- Connects to Act I's experiment design: use TS to pick which modality condition to test next.

**Library options:**
- `gpytorch` + `botorch` (BoTorch has TS utilities)
- From scratch: numpy + scipy.linalg.cholesky (see code skeleton below)

**From-scratch feasibility:** ⭐⭐⭐⭐⭐ (easiest bandit variant)

**Implementation skeleton:**
```python
def thompson_sample_gp_posterior(gp_mean, gp_cov, n_samples=10):
    """Sample reward functions from GP posterior."""
    L = np.linalg.cholesky(gp_cov + 1e-6 * np.eye(len(gp_cov)))  # Cholesky decomposition
    z = np.random.randn(len(gp_mean), n_samples)
    samples = gp_mean[:, None] + L @ z  # Each column is a function sample
    return samples

def bandit_action_ts(candidate_movies_embeddings, gp_mean, gp_cov, n_ts_samples=10):
    """Pick next movie using Thompson sampling."""
    # Predict mean & cov for candidates
    mu_cand, cov_cand = gp.predict(candidate_movies_embeddings)

    # Sample posterior
    samples = thompson_sample_gp_posterior(mu_cand, cov_cand, n_samples=n_ts_samples)

    # Greedy under each sample, return consensus
    best_actions = np.argmax(samples, axis=0)
    best_idx = np.bincount(best_actions).argmax()  # Mode
    return best_idx, mu_cand[best_idx], np.sqrt(cov_cand[best_idx, best_idx])
```

**Apples-to-apples baseline:** ε-greedy (pick top μ, else random) vs Thompson sampling (posterior-driven).

**Regret bound:**
Thompson sampling achieves Bayesian regret:
$$\mathbb{E}[R_T] = O(d \sqrt{T \log T})$$
where $d$ is feature dimension, $T$ is horizon. Matches information-theoretic lower bounds (Russo & Van Roy 2018).

---

### 2.2 LinUCB (Linear Upper Confidence Bound) on embedding features — **CONTEXTUAL APPROACH**

**Description:**
Unlike Thompson sampling (which requires Bayesian posterior), LinUCB works in the *frequentist* framework with confidence sets. It maintains:
- Linear model: $\hat{r}_t(a) = \mathbf{x}_a^T \hat{\boldsymbol{\theta}}_t$
- Uncertainty ellipsoid: $||\hat{\boldsymbol{\theta}}_t - \boldsymbol{\theta}^*||_{A_t} \leq c_t$ (with high probability)
- Pick action: $a^* = \arg\max_a (\hat{r}_t(a) + c_t \||\mathbf{x}_a||_{A_t^{-1}})$

The UCB term $c_t \||\mathbf{x}_a||_{A_t^{-1}}$ is the *optimism under uncertainty*: actions with high feature-norm in low-confidence directions get a bonus.

**Math hook:**
$$\text{UCB}_t(a) = \hat{r}_t(a) + \alpha_t \|\mathbf{x}_a\|_{A_t^{-1}}$$
where $A_t = \lambda I + \sum_{s=1}^{t-1} \mathbf{x}_{a_s} \mathbf{x}_{a_s}^T$ and $\alpha_t = O(\sqrt{d \log t})$ (confidence radius).

**Why it fits:**
- Works with *deterministic embeddings* (sentence-transformers, ResNet) from P2.
- More direct than Thompson sampling: no Bayesian inference needed.
- Natural for multimodal ablation: context includes {poster-encoding, title-encoding, synopsis-encoding}, and LinUCB learns their importance.

**Library options:**
- `contextualbandits` package (has LinUCB + batch context)
- `vowpal_wabbit` (industrial-strength, but steep learning curve)
- From scratch: numpy (matrix inversion, Gram matrix updates)

**From-scratch feasibility:** ⭐⭐⭐⭐ (medium; matrix operations required)

**Apples-to-apples baseline:** LinUCB vs ε-greedy on embedding features.

**Regret bound:**
$$\mathbb{E}[R_T] = O(d \sqrt{T \log T})$$
Matches Thompson sampling asymptotically, but often tighter constants in practice.

---

### 2.3 Batch Active Learning via Information Gain — **EXPERIMENT DESIGN APPROACH**

**Description:**
Instead of picking *one* next movie, use information-theoretic criteria to pick a *batch* of k movies whose ratings would maximally reduce posterior uncertainty. This is "batch Thompson sampling" or "batch uncertainty sampling."

Two variants:
1. **Max entropy reduction:** Pick movies that maximize expected reduction in H(Z | D ∪ {new movies}).
2. **KL divergence:** Pick movies that maximize E[KL(p(Z | D ∪ {(a,r)}) || p(Z | D))].

For a GP, this becomes: *pick the k movies with highest marginal variance σ²(x)*, or with highest "expected improvement over current best."

**Math hook:**
$$a^*_{\text{batch}} = \arg\max_{S \subseteq \mathcal{A}, |S|=k} \mathbb{E}_{r \sim p(r|D)}[H(Z | D \cup \{(a, r) : a \in S\})]$$

Approximation: greedily select movies with max variance (myopic information gain).

**Why it fits Act I:**
Act I is the modality-ablation experiment. Instead of sequentially rating movies, we *design the full batch upfront*: which 20 movies (across all 4 conditions) will most reduce uncertainty about modality importance? Information gain answers this.

**Implementation:**
```python
def batch_active_learning(gp_mean, gp_var, candidate_movies, k=20):
    """Pick k movies with highest marginal variance."""
    variances = gp_var[candidate_movies]
    top_k_idx = np.argsort(variances)[-k:]
    return top_k_idx, variances[top_k_idx]
```

**From-scratch feasibility:** ⭐⭐⭐⭐⭐ (trivial if GP variance is available)

**Apples-to-apples baseline:** Information-gain design vs uniform random design (Act I as-is).

---

### 2.4 Contextual Restless Bandit with Temporal Drift — **SEQUENCE APPROACH**

**Description:**
Real taste is *non-stationary*. The posterior over Temilola's taste evolves over time (seasons change, he discovers new genres, movie-watching fatigue waxes/wanes). A "restless" bandit is one where the reward function drifts *even without the agent acting*.

Model:
- Latent taste state: $z_t = z_{t-1} + \text{noise}$
- Reward: $r_t(a) = \mathbf{x}_a^T z_t + \epsilon_t$

This is a **Hidden State Tracking with Bandits** problem: you must simultaneously estimate the drifting state and make decisions.

The Kalman filter (already in Act II) *is* a restless bandit solver. At each timestep:
1. Kalman-predict: forecast $z_{t+1}$
2. Thompson sample from p(z_t | D_{1:t})
3. Pick action: $a_t^* = \arg\max_a \mathbb{E}_{z_t}[r_t(a)]$
4. Observe $r_t(a_t^*)$
5. Kalman-update: incorporate $r_t$ into posterior on $z_t$

**Why it fits:**
- Act II already builds a Kalman filter on Date Watched sequence.
- Restless bandit *upgrades* that filter: instead of just smoothing ratings, we're using it to *decide which movie to watch next*.
- Captures real phenomenon: "I'm getting tired of sci-fi, the algorithm should detect my drift and recommend romance."

**Math hook:**
Kalman innovation: $\tilde{r}_t = r_t - H z_t$ (prediction error). Use this to drive the bandit decision:
$$a_t^* = \arg\max_a \mathbb{E}_{z_t|D_{1:t}}[\mathbf{x}_a^T z_t]$$

**From-scratch feasibility:** ⭐⭐⭐ (requires integrating Kalman + Thompson; medium complexity)

**Apples-to-apples baseline:** Restless bandit vs static Thompson sampling (ignore date-watched drift).

---

### 2.5 Neural Bandits (Riquelme et al., 2018) — **DEEP LEARNING APPROACH**

**Description:**
For high-dimensional contexts (images, long text), a linear model is too restrictive. A neural bandit uses a deep network to map context → reward, and maintains a posterior over the weights via MC-Dropout or ensemble.

Framework:
1. Train a NN: $\hat{r}_\theta(x) = f_\theta(x)$
2. Maintain an ensemble or dropout posterior: $\{\theta^{(1)}, ..., \theta^{(B)}\}$
3. At decision time, sample: $\tilde{\theta} \sim p(\theta | D)$ (via ensemble or dropout)
4. Pick: $a^* = \arg\max_a f_{\tilde{\theta}}(a)$

This is "Thompson sampling with a deep posterior."

**Why it might be overkill for Pipeline 3:**
- We're already using a GP (shallow, interpretable) for uncertainty.
- Data: N=162 + augmented ≈ 10K, which is small for a NN.
- But if we train CVAE for generation (Act III), we can *reuse its encoder* as the context map in the neural bandit.

**When to use:**
- If Act III's CVAE shows interesting structure in the latent space, a neural bandit could help us navigate that space.
- Apples-to-apples: neural bandit on CVAE latents vs GP on raw embeddings.

**From-scratch feasibility:** ⭐⭐ (requires MC-Dropout training & uncertainty calibration; harder)

---

## 3. Math deep-dives

### 3.1 Thompson Sampling: Posterior sampling derivation

**Setup:** Temilola watches movies; we model his rating as:
$$r(x) \sim \mathcal{N}(f(x), \sigma_{\text{noise}}^2)$$

We place a GP prior on $f$: $f \sim \mathcal{GP}(\text{mean}=0, \text{kernel}=K)$.

**Posterior** (Bayes rule + marginalization over $f$):
$$p(f | D) = \frac{p(D | f) p(f)}{p(D)}$$

For a GP, the posterior is also a GP:
$$f | D \sim \mathcal{GP}(\mu_{\text{post}}, K_{\text{post}})$$

where:
$$\mu_{\text{post}}(x) = K(x, X) [K(X, X) + \sigma^2 I]^{-1} \mathbf{y}$$
$$K_{\text{post}}(x, x') = K(x, x') - K(x, X) [K(X, X) + \sigma^2 I]^{-1} K(X, x')$$

**Thompson sampling:**
Sample a function from the posterior: $\tilde{f} \sim p(f | D)$.
Pick the action that looks best under $\tilde{f}$:
$$a^* = \arg\max_a \tilde{f}(x_a)$$

To sample $\tilde{f}$, we can evaluate it on a finite set of candidate movies:
$$\tilde{\mathbf{f}} = [\tilde{f}(x_1), ..., \tilde{f}(x_k)]^T \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{K})$$

where $\boldsymbol{\mu}$ and $\mathbf{K}$ are the posterior mean and kernel evaluated on candidates.

**Sampling algorithm:**
```
1. Compute Cholesky decomposition: L L^T = K_post(X_cand, X_cand)
2. Sample z ~ N(0, I)
3. Set tilde{f} = mu + L @ z
4. Return argmax tilde{f}
```

**Bayesian regret bound:**
Thompson sampling incurs expected regret:
$$\mathbb{E}[R_T] = O(d \sqrt{T \log T} \log(1/\delta))$$

This is tight for Gaussian bandits (Russo & Van Roy, 2018).

---

### 3.2 LinUCB: Optimism under uncertainty

**Setup:** Linear contextual bandit. At step $t$, you receive context $x_t \in \mathbb{R}^d$, must pick action $a_t \in \{1, ..., k\}$ (movie), observe reward $r_t = x_t^T \theta^* + \eta_t$ where $\eta_t$ is i.i.d. noise.

**Confidence set construction:**
Define the design matrix: $A_t = \lambda I + \sum_{s=1}^{t-1} x_{a_s} x_{a_s}^T$

By standard online-learning concentration, with high probability:
$$\|\hat{\theta}_t - \theta^*\|_{A_t} \leq c_t$$

where $\hat{\theta}_t = A_t^{-1} \sum_{s=1}^{t-1} x_{a_s} r_s$ and $c_t = O(\sqrt{d \log t})$.

**Decision rule (Optimism in the Face of Uncertainty):**
$$a_t^* = \arg\max_{a} \left[ \hat{\theta}_t^T x_a + c_t \|x_a\|_{A_t^{-1}} \right]$$

The first term is the *plug-in estimate*; the second is the *exploration bonus*.

**Intuition:**
- High $\|x_a\|_{A_t^{-1}}$: feature direction is under-explored, so the upper confidence bound is loose. Explore!
- Low $\|x_a\|_{A_t^{-1}}$: feature direction is well-estimated. Exploit.

**Regret bound:**
$$\mathbb{E}[R_T] = \tilde{O}(d \sqrt{T})$$

(Abbasi-Yadkori et al., 2011)

---

### 3.3 Information gain & batch active learning

**Problem:** Given data $D$, which batch $S$ of $k$ movies should we label next to maximally reduce uncertainty?

**Information gain:**
$$IG(S) = H(\Theta | D) - \mathbb{E}_{r \sim p(r|D)}[H(\Theta | D \cup \{(x_s, r_s) : s \in S\})]$$

where $\Theta$ represents the unknown parameters of interest (e.g., movie ratings for unwatched movies).

**For a GP**, the posterior is Gaussian, so:
$$H(\Theta | D) = \frac{1}{2} \log \det(2\pi e K_{\text{post}})$$

Greedy approximation: pick each movie with **maximum marginal variance** under the current posterior.

$$x_{t+1}^* = \arg\max_x \sigma_{\text{post}}^2(x)$$

This is a *myopic* approximation but often works well in practice.

**Apples-to-apples:** Batch design with information gain vs uniform random batch (Act I baseline).

---

## 4. Implementation recipes

### Recipe 1: From-scratch Thompson Sampling (30 lines)

```python
import numpy as np
from scipy.linalg import cholesky
from scipy.spatial.distance import cdist

class ThompsonSamplingGP:
    def __init__(self, length_scale=1.0, sigma_noise=0.1):
        self.X = None
        self.y = None
        self.length_scale = length_scale
        self.sigma_noise = sigma_noise

    def rbf_kernel(self, X1, X2):
        """RBF kernel: exp(-||x - x'||^2 / (2 * length_scale^2))"""
        sqdist = cdist(X1, X2, 'sqeuclidean')
        return np.exp(-sqdist / (2 * self.length_scale**2))

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test, return_std=True):
        """GP posterior mean and variance."""
        K_XX = self.rbf_kernel(self.X, self.X)
        K_test_X = self.rbf_kernel(X_test, self.X)
        K_test_test = self.rbf_kernel(X_test, X_test)

        L = cholesky(K_XX + self.sigma_noise**2 * np.eye(len(self.X)), lower=True)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))

        mu = K_test_X @ alpha
        v = np.linalg.solve(L, K_test_X.T)
        cov = K_test_test - v.T @ v

        return mu, np.sqrt(np.diag(cov) + 1e-6) if return_std else np.diag(cov)

    def thompson_sample(self, X_test, n_samples=10):
        """Sample from posterior and return best action."""
        mu, std = self.predict(X_test, return_std=True)
        cov = np.diag(std**2)  # Approximate as diagonal

        # Add small jitter for numerical stability
        cov += 1e-6 * np.eye(len(X_test))

        # Cholesky + sample
        L = cholesky(cov, lower=True)
        samples = mu[:, None] + L @ np.random.randn(len(X_test), n_samples)

        # Best action under each sample
        best_actions = np.argmax(samples, axis=0)

        # Return consensus best action
        best_idx = np.bincount(best_actions).argmax()
        return best_idx, mu[best_idx]

# Usage
gp = ThompsonSamplingGP(length_scale=1.0)
gp.fit(X_train, y_train)
best_movie_idx, predicted_rating = gp.thompson_sample(X_candidates, n_samples=100)
```

### Recipe 2: LinUCB with sklearn (library integration)

```python
from contextualbandits.online import LinUCB

# X shape: (n_movies, d_features)
# actions: movie indices
# rewards: ratings (0-5)

bandit = LinUCB(alpha=0.5)  # Exploration parameter

for movie_idx, features, rating in dataset:
    # Predict & pick action
    predicted_reward, _, _ = bandit.predict(features.reshape(1, -1))

    # Observe reward and update
    bandit.partial_fit(features.reshape(1, -1), action=movie_idx, reward=rating)
```

### Recipe 3: Information Gain for Batch Design

```python
def batch_active_learning_gp(gp_model, X_candidates, batch_size=10):
    """Select batch using variance-based information gain."""
    _, var = gp_model.predict(X_candidates, return_std=True)

    # Greedy: pick k highest-variance movies
    top_k_idx = np.argsort(var)[-batch_size:]
    return top_k_idx

# For Act I: design the 50-movie × 4-condition experiment
stratified_50 = stratified_sample(dataset, strata_var='genre', n=50)
X_cand = embed_movies(stratified_50)  # sentence-transformers

batch = batch_active_learning_gp(gp, X_cand, batch_size=50)
experiment_design = expand_to_conditions(dataset[batch], conditions=['poster-only', 'title-only', 'synopsis-only', 'all'])
```

---

## 5. Library ecosystem

| Library | Purpose | From-scratch alternative | When to use |
|---|---|---|---|
| `gpytorch` | GPU-accelerated GPs | scipy.linalg + numpy | Fast exact GPs; optional |
| `botorch` | BO + Thompson sampling | TS from scratch (Recipe 1) | If already using PyTorch |
| `contextualbandits` | LinUCB + batch bandits | numpy matrix updates | Full contextual bandit pipeline |
| `vowpal_wabbit` | Industrial-strength VW | N/A | Production systems (overkill here) |
| `mabwiser` | Simple MAB policies | From scratch (trivial for ε-greedy) | Quick baseline |
| `scikit-optimize` | Bayesian optimization | BoTorch / TS | Design-of-experiments variant |

**Recommendation for Pipeline 3:**
- **Core:** From-scratch Thompson sampling (Recipe 1) + GP from existing P2 code.
- **Optional library:** `botorch` if we integrate Act III's CVAE and want BO to design the next ablation experiment.

---

## 6. Narrative slots for bandits in Pipeline 3

### Option A: Act I.5 (Experiment Design)

**Slot:** Between data description and manual ablation study.

*Section: "Designing the modality-ablation experiment via information gain."*

We have 162 movies. Which 50 should we test in the 4-condition ablation? Don't pick randomly—use information gain:
1. Fit a quick GP on current embeddings + ratings.
2. Use batch active learning (max variance) to select 50 movies most likely to *disambiguate* the modality effects.
3. Latin-square design, run ablation.
4. Show that info-gain design found more variable movies than random design (visualization: variance distribution).

**Deliverable:** Information gain heatmap, comparison table (random vs IG batch).

### Option B: Act II.5 (Active Learning via Bandits)

**Slot:** After conformal prediction, before Kalman filter.

*Section: "Using posterior uncertainty to guide the next movie choice."*

After fitting GP + conformal on the initial 162, ask: which unwatched movies should we rate *next* to maximally improve our model?

1. Fit Thompson sampling on candidate unwatched movies.
2. Simulate bandit decisions: pick top-10 movies via TS vs top-10 by mean prediction.
3. Show: Thompson sampling picks a *diverse* set (high-value + high-uncertainty), whereas mean-greedy picks only high-value.
4. Optionally: Temilola rates those top-10, observe if Thompson's picks led to higher ratings / more discovery.

**Deliverable:** Action selection scatter plot (μ vs σ, highlighting TS picks), reward curves (TS learning vs greedy).

### Option C: Act II.7 (Restless Bandits + Kalman)

**Slot:** Integrate bandits into the Kalman filter section.

*Section: "Tracking evolving taste: restless bandits meet Kalman filtering."*

The Kalman filter estimates $z_t$ (latent taste state at time $t$). Use Thompson sampling *at each time step* to recommend the movie that would most reduce posterior entropy over $z_t$.

1. Kalman predict: $\hat{z}_{t|t-1}$, $\Sigma_{t|t-1}$
2. Thompson sample from p(z_t | D_{1:t-1}): pick a plausible taste state
3. For each candidate movie $x$, predict expected rating under sampled $z_t$
4. Pick the movie with highest sampled reward
5. Observe rating, Kalman update
6. Repeat

**Deliverable:** Time-series plot: Kalman trajectory + bandit recommendations overlaid, showing when taste drifts and algorithm pivots.

---

## 7. Apples-to-apples baselines

### Comparison 1: Thompson Sampling vs ε-Greedy

| Method | Rule | Exploration | Convergence |
|---|---|---|---|
| ε-Greedy | Pick best μ, else random | Fixed ε (10%) | Linear regret |
| Thompson Sampling | Sample from posterior, pick best under sample | Adaptive (posterior-driven) | O(√T log T) regret |

**Experiment:**
- Fit GP on initial 162 ratings.
- Candidate pool: 1000 unwatched movies.
- Simulate 50 rounds of each method (no actual new ratings; use GP simulator).
- Plot cumulative reward: TS should dominate ε-greedy.

**Code:**
```python
def simulate_bandits(gp, candidates, n_rounds=50):
    gp_true = copy.deepcopy(gp)  # Simulator
    X_available = candidates.copy()

    rewards_ts = []
    rewards_greedy = []

    for _ in range(n_rounds):
        # Thompson sampling
        idx_ts, _ = gp.thompson_sample(X_available, n_samples=50)
        r_ts = gp_true.predict(X_available[[idx_ts]])[0][0]
        rewards_ts.append(r_ts)

        # ε-greedy
        if np.random.rand() < 0.1:
            idx_greedy = np.random.choice(len(X_available))
        else:
            idx_greedy = np.argmax(gp.predict(X_available)[0])
        r_greedy = gp_true.predict(X_available[[idx_greedy]])[0][0]
        rewards_greedy.append(r_greedy)

        # Update (remove from available)
        X_available = np.delete(X_available, [idx_ts, idx_greedy], axis=0)

    return np.cumsum(rewards_ts), np.cumsum(rewards_greedy)
```

### Comparison 2: Restless Bandit vs Static Bandit

| Method | Temporal model | Captures drift |
|---|---|---|
| Static Thompson | Single posterior over taste | No; assumes θ fixed |
| Restless (Kalman + TS) | State-space z_t; Kalman tracks drift | Yes; adapts as taste evolves |

**Experiment:**
- Fit both to the 162 + time sequence
- Predict ratings for the next 20 movies (chronologically)
- Restless should have lower RMSE on *recent* movies (post-drift)
- Static should overfit to historical preference

---

## 8. Wild card: Bayesian optimization for experiment design

**Idea:** Instead of "which next movie," ask "which next 4-condition ablation should we run to maximally resolve uncertainty about modality importance?"

**Setup:**
- We've done Session 1 of Act I (20 movies × 4 conditions rated).
- Modality importance: γ_poster, γ_title, γ_synopsis (learned from Session 1 data).
- Posterior: p(γ | observed Session 1 data) is Gaussian ≈ N(μ_γ, Σ_γ).

**Question:** Which next 20 movies should we test (what genres, what combination of features) to *tighten the posterior on γ most efficiently*?

**Method:**
1. Surrogate model: Bayesian NN that predicts rating given (movie features, modality flags, γ).
2. Acquisition function: Expected Information Gain (EIG) = E_{r ~ ε} [KL(p(γ | D ∪ {(x,r)}) || p(γ|D))]
3. Optimize the acquisition over the space of possible 20-movie batches (or use greedy hill-climbing).
4. The BO returns the "most informative" Session 2 design.

**Why it's wild:**
- You're not just deciding *which* movie to watch; you're designing the *experiment* to inform the design of *future* experiments.
- Connects to Active Learning literature (Houlsby et al., ICML 2011).

**From-scratch feasibility:** ⭐⭐ (requires BO implementation; use `scikit-optimize` as shortcut)

**Code skeleton:**
```python
from skopt import gp_minimize
from skopt.space import Real

def expected_info_gain(session_2_design, session_1_data, gp_surrogate):
    """Predict EIG for a proposed Session 2 design."""
    # Simulate future ratings under the design
    predicted_ratings = gp_surrogate.predict(session_2_design)[0]

    # Compute posterior on γ after observing these ratings
    new_data = np.vstack([session_1_data, session_2_design])
    new_gamma_posterior = fit_modality_model(new_data)

    # KL divergence between old and new posterior
    kl = kl_divergence(old_gamma_posterior, new_gamma_posterior)
    return -kl  # Negative because we're minimizing

# Optimize session 2 design
best_design = gp_minimize(
    lambda x: expected_info_gain(x, session_1_data, gp_surrogate),
    space=[Real(0, 1) for _ in range(20*d_features)],
    n_calls=100,
    random_state=42
)
```

---

## 9. References

1. **Russo, D., & Van Roy, B.** (2018). Learning to Optimize via Posterior Sampling. *Mathematics of Operations Research*, 44(4), 1440–1465. [[PDF](https://arxiv.org/abs/1805.09563)]
   - Foundational Thompson sampling regret bounds. *Must-read.*

2. **Abbasi-Yadkori, Y., Pál, D., & Szepesvári, C.** (2011). Improved Algorithms for Linear Stochastic Bandits. *NeurIPS*. [[PDF](https://arxiv.org/abs/1402.6028)]
   - LinUCB theory and regret. *Essential for confidence-set approach.*

3. **Gal, Y., & Ghahramani, Z.** (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. *ICML*. [[PDF](http://proceedings.mlr.press/v48/gal16.pdf)]
   - MC-Dropout for neural bandits.

4. **Riquelme, C., Tucker, G., & Snoek, J.** (2018). Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling. *ICML*. [[PDF](https://arxiv.org/abs/1802.09127)]
   - Neural bandits in practice. Nice empirical comparisons.

5. **Slivkins, A.** (2019). *Introduction to Multi-Armed Bandits*. [[Book link](https://arxiv.org/abs/1904.07272)]
   - Accessible textbook covering bandits, contextual bandits, and non-stationary settings. *Excellent reference.*

6. **Sutton, R. B., & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. Chapter 2 (Multi-Armed Bandits).
   - Classic intro to bandits + ε-greedy. *Context.*

7. **Houlsby, N., Huszár, F., Ghahramani, Z., & Lengyel, M.** (2011). Bayesian Active Learning for Classification and Preference Learning. [[arXiv](https://arxiv.org/abs/1112.5745)]
   - Information gain for experiment design. *For the wild-card section.*

8. **BoTorch documentation:** https://botorch.org/
9. **contextualbandits package:** https://github.com/david-cortes/contextualbandits

---

## 10. Summary: Why bandits fit Pipeline 3's arc

| Act | Layer | Bandit role |
|---|---|---|
| **Act I** | Experiment design | Information gain picks the 50 movies + 4-condition batch most likely to disambiguate modality importance |
| **Act II** | Posterior-driven decisions | Thompson sampling on GP posterior answers "which unwatched movie should we rate next?" Feeds Act III generation |
| **Act II–III** | Adaptive learning | Restless bandit + Kalman filter capture taste drift; generative model responds to evolving posterior |
| **Act III** | Validation loop | Blind-rate the generated movie; use bandit framework to design the follow-up experiments |

**The closed loop:**
Experiment (Act I) → Posterior (Act II) → **Bandit decision** → New rating → Updated posterior → Generation (Act III) → Validation → **Bandit decides next experiment**

This is the **minimal, elegant** way to add bandits to Pipeline 3 without derailing the narrative. It's not a separate method; it's the *decision-making layer* that consumes the uncertainty from the probabilistic stack and feeds the generative stack.

---

**Estimated implementation time:** 2–3 days for core (Thompson sampling recipe + information gain batch design + comparison table). 1 additional day per "nice-to-have" (LinUCB, restless bandit, BO for experiment design).
