# Derivation 14: Bayesian Hierarchical One-Way ANOVA (PyMC)

**Task 1.12**
**Date**: 2026-04-23
**Purpose**: Estimate how much of the variance in Temilola's 324 movie ratings is explained by **modality** (all / metadata / poster / synopsis) vs. movie-level variation.

---

## 1. Generative Model

We model the observed ratings $y_{ij}$ (rating for movie $i$ under modality $j$) as a hierarchical structure:

$$\begin{align}
\mu &\sim \mathcal{N}(3.5, 1.5) \quad \text{(grand mean, rating scale 1--5)} \\
\sigma_{\text{modality}} &\sim \text{HalfNormal}(0.5) \quad \text{(between-modality SD)} \\
a_m &\sim \mathcal{N}(0, \sigma_{\text{modality}}), \quad m \in \{0, 1, 2, 3\} \quad \text{(modality offsets)} \\
\sigma_{\text{movie}} &\sim \text{HalfNormal}(0.5) \quad \text{(between-movie SD)} \\
b_i &\sim \mathcal{N}(0, \sigma_{\text{movie}}), \quad i \in \{1, \ldots, 81\} \quad \text{(movie effects)} \\
\sigma_{\text{obs}} &\sim \text{HalfNormal}(0.5) \quad \text{(observation noise)} \\
y_{ij} &\sim \mathcal{N}(\mu + a_{m(j)} + b_i, \sigma_{\text{obs}})
\end{align}$$

### Interpretation

- **$\mu$**: Grand mean rating (expected rating when all effects are zero, after centering).
- **$a_m$**: Offset for modality $m$ (how much does modality shift ratings relative to grand mean?).
- **$b_i$**: Per-movie random effect (movies vary in baseline rating quality).
- **$\sigma_{\text{modality}}$**: Controls the magnitude of modality-to-modality variation.
- **$\sigma_{\text{movie}}$**: Controls the magnitude of movie-to-movie variation.
- **$\sigma_{\text{obs}}$**: Unexplained noise (residual variance).

---

## 2. Why Bayesian Hierarchical (Partial Pooling)?

### Classic Fixed-Effects ANOVA Problem
If we estimated modality effects separately (fixed-effects ANOVA):
- Each modality mean gets a point estimate.
- Small sample groups are unstable (high variance).
- No borrowing of strength across modalities.

### Hierarchical Solution (Partial Pooling)
By placing $a_m \sim \mathcal{N}(0, \sigma_{\text{modality}})$:
- **If** modalities are similar (small $\sigma_{\text{modality}}$), all estimates shrink toward 0.
- **If** modalities are different (large $\sigma_{\text{modality}}$), estimates are less pulled.
- The data *learns* $\sigma_{\text{modality}}$ from the ensemble of modality means.
- Smaller groups are automatically regularized; larger groups less so.

**Benefit**: Stable, interpretable estimates that account for uncertainty in the group-level variance.

---

## 3. Non-Centered Parameterization

We use a **non-centered** (or **hierarchical centered**) parameterization:

$$\begin{align}
a_m &= \sigma_{\text{modality}} \cdot a_m^{\text{raw}} \quad \text{where} \quad a_m^{\text{raw}} \sim \mathcal{N}(0, 1) \\
b_i &= \sigma_{\text{movie}} \cdot b_i^{\text{raw}} \quad \text{where} \quad b_i^{\text{raw}} \sim \mathcal{N}(0, 1)
\end{align}$$

### Why Non-Centered?

The centered parameterization
$$a_m \sim \mathcal{N}(0, \sigma_{\text{modality}})$$
creates **funnel geometry** in the posterior: when $\sigma_{\text{modality}}$ is small, the individual $a_m$ are tightly constrained, causing high correlations and slow mixing in NUTS.

The non-centered form decorrelates the hierarchical parameters, making NUTS sampler more efficient:
- $a_m^{\text{raw}}$ has a fixed $\mathcal{N}(0,1)$ prior (no funnel).
- $a_m$ is determined by $\sigma_{\text{modality}}$ and $a_m^{\text{raw}}$.
- The sampler can update the variance parameter and the raw parameters more independently.

---

## 4. Prior Choices

### $\mu \sim \mathcal{N}(3.5, 1.5)$
- **Center**: 3.5 is the midpoint of the 1--5 rating scale.
- **SD**: 1.5 allows reasonable spread while staying on-scale.

### $\sigma_{\text{modality}}, \sigma_{\text{movie}}, \sigma_{\text{obs}} \sim \text{HalfNormal}(0.5)$
- **HalfNormal**: Non-negative support (variance must be ≥ 0).
- **Scale 0.5**: For a 1--5 scale, $\sigma \approx 2$ is plausible maximum. HalfNormal(0.5) puts ~95% of probability below 1, accommodating realistic variation while strongly penalizing large variances.
- **Weakly informative**: Does not impose a hard upper bound, but gently regularizes against unrealistic values.

---

## 5. Data & Encoding

### Input
- **File**: `data/modality_ratings.jsonl`
- **Rows**: 324 (Temilola's ratings)
- **Columns**: `tmdb_id, rating, modality` (plus metadata joined from `movies_meta.json`)

### Encoding

```python
# Modality → integer
modality_map = {'all': 0, 'metadata': 1, 'poster': 2, 'synopsis': 3}

# TMDB ID → movie index (0 ... 80)
# 81 unique movies across the 324 ratings
```

---

## 6. Inference

### Sampling Strategy
- **Algorithm**: NUTS (No-U-Turn Sampler) with dual averaging.
- **Chains**: 2 chains.
- **Iterations per chain**: 1000 tune + 1000 draw = 2000 total.
- **Target acceptance**: 0.95.
- **Total samples**: 2 × 1000 = 2000 post-warmup draws.

### Convergence Diagnostics
- **$\hat{R}$ (Gelman-Rubin Statistic)**: Should be < 1.01.
  - Compares within-chain and between-chain variance.
  - $\hat{R} \approx 1.0$ → chains are mixing well.
- **Effective Sample Size (ESS)**: Accounts for autocorrelation; should be >> 400 for reliable estimates.
- **Divergences**: Gradients that flip sign, indicating sampler trouble. 0 is ideal; <10 is acceptable.

---

## 7. Results Interpretation

### Posterior Estimates
Each parameter's posterior is summarized by:
- **Mean**: Point estimate.
- **SD**: Posterior standard deviation.
- **94% HDI**: Highest Density Interval (Bayesian credible interval).

### Key Questions Answered

1. **How much does modality matter?**
   Compare $\sigma_{\text{modality}}$ (modality variance) to $\sigma_{\text{movie}}$ (movie variance):
   - If $\sigma_{\text{modality}} \ll \sigma_{\text{movie}}$, then movies matter far more than modality.
   - If $\sigma_{\text{modality}}$ is comparable, modality is a significant factor.

2. **Which modality is "best"?**
   Look at the posterior offsets $a_m$:
   - Positive → ratings higher than grand mean for that modality.
   - Negative → ratings lower.
   - If all 94% HDIs overlap zero, modality effects are negligible.

3. **How much noise?**
   $\sigma_{\text{obs}}$ is the residual SD unexplained by the model.

---

## 8. Variance Decomposition

Total variance in ratings can be decomposed as:

$$\text{Var}(y) \approx \sigma_{\text{modality}}^2 + \sigma_{\text{movie}}^2 + \sigma_{\text{obs}}^2$$

(Approximate because the effects are additive on the mean, not the variance.)

**Interpreting the plot**:
- **Tall $\sigma_{\text{movie}}$ bar**: Movies (narrative quality, genre, etc.) drive most variation.
- **Tall $\sigma_{\text{modality}}$ bar**: Modality (visual vs. textual cues) is a major factor.
- **Tall $\sigma_{\text{obs}}$ bar**: Residual noise dominates; model does not fit well.

---

## 9. Assumptions & Limitations

1. **Gaussian likelihood**: Assumes ratings are approximately normally distributed. For bounded ordinal data, a more robust approach (e.g., ordinal logit) could be explored.

2. **Additive effects**: Modality and movie effects are added linearly. No interaction terms (e.g., "Does modality X work especially well for movie Y?").

3. **Homoscedastic**: All observations have the same noise SD $\sigma_{\text{obs}}$. Could relax with group-level heteroskedasticity.

4. **2 chains**: The current run uses 2 chains. Best practice is 4+ chains for robust $\hat{R}$ diagnostics. With 2 chains, rely on ESS and visual inspection.

---

## 10. References

- **Gelman et al. (2013)**: *Bayesian Data Analysis*, 3rd ed. — Hierarchical modeling, partial pooling.
- **Betancourt & Girolami (2013)**: Hamiltonian Monte Carlo and the geometry of high-dimensional spaces (funnels).
- **PyMC Docs** (v5.27): Non-centered parameterization, NUTS sampler.
- **ArviZ Docs** (v0.23): Summary statistics, diagnostics, visualization.

---

## Appendix: Code Snippet (Non-Centered)

```python
with pm.Model() as model:
    mu = pm.Normal('mu', mu=3.5, sigma=1.5)
    sigma_modality = pm.HalfNormal('sigma_modality', sigma=0.5)
    sigma_movie = pm.HalfNormal('sigma_movie', sigma=0.5)
    sigma_obs = pm.HalfNormal('sigma_obs', sigma=0.5)

    # Non-centered: raw parameters
    a_m_raw = pm.Normal('a_m_raw', mu=0, sigma=1.0, shape=4)
    a_m = pm.Deterministic('a_m', sigma_modality * a_m_raw)

    b_i_raw = pm.Normal('b_i_raw', mu=0, sigma=1.0, shape=81)
    b_i = pm.Deterministic('b_i', sigma_movie * b_i_raw)

    # Likelihood
    mu_ij = mu + a_m[modality_idx] + b_i[movie_idx]
    y = pm.Normal('y', mu=mu_ij, sigma=sigma_obs, observed=ratings)

    idata = pm.sample(draws=1000, tune=1000, chains=2, target_accept=0.95)
```

This ensures efficient sampling on hierarchical structure while remaining interpretable.
