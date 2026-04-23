# Derivation: IPW and AIPW Estimators for Average Treatment Effect

## Potential Outcomes Framework

**Setup:** Binary treatment $T \in \{0, 1\}$, outcome $Y \in \mathbb{R}$, covariates $X \in \mathbb{R}^p$.

For each unit $i$, define potential outcomes:
- $Y_i(1)$ = outcome under treatment
- $Y_i(0)$ = outcome under control

Observed outcome: $Y_i = T_i Y_i(1) + (1 - T_i) Y_i(0)$

### Parameter of Interest
**Average Treatment Effect (ATE):**
$$\tau = \mathbb{E}[Y(1) - Y(0)]$$

## Key Assumption: Unconfoundedness (Conditional Independence)

Treatment assignment is independent of potential outcomes given observed covariates:
$$T \perp\!\!\!\perp (Y(0), Y(1)) \mid X$$

This implies:
- **Propensity score:** $e(x) = P(T=1 \mid X=x)$ fully captures confounding
- **No hidden bias:** All confounders are measured in $X$

## IPW Estimator (Inverse Probability Weighting)

### Derivation

Under unconfoundedness, we can reweight by inverse propensity scores to create pseudo-populations where treatment is independent of $X$:

$$\begin{align}
\tau_{IPW} &= \mathbb{E}\left[\frac{T \cdot Y}{e(X)}\right] - \mathbb{E}\left[\frac{(1-T) \cdot Y}{1 - e(X)}\right] \\
&= \mathbb{E}[Y(1)] - \mathbb{E}[Y(0)]
\end{align}$$

**Intuition:** 
- Units with $T=1$ are weighted by $1/e(X)$ (more weight if rare to be treated given X)
- Units with $T=0$ are weighted by $1/(1-e(X))$ (more weight if rare to be untreated given X)
- Reweighting balances the covariate distributions between groups

### Sample Estimator

$$\hat{\tau}_{IPW} = \frac{1}{n} \sum_{i=1}^n \left[ \frac{T_i Y_i}{\hat{e}(X_i)} - \frac{(1-T_i) Y_i}{1 - \hat{e}(X_i)} \right]$$

where $\hat{e}(X_i)$ is the estimated propensity score.

### Variance

IPW is sensitive to:
- **Positivity violation:** If $e(x) \approx 0$ or $e(x) \approx 1$, weights explode
- **Propensity misspecification:** If logistic regression is misspecified

## AIPW Estimator (Augmented IPW, Doubly Robust)

### Motivation

AIPW adds an outcome model $\mu_t(x) = \mathbb{E}[Y | T=t, X=x]$ to reduce variance:

**Key property:** Consistent if *either* the propensity model OR outcome model is correct (not both required).

### Derivation

Define residuals:
$$R_t(x) = Y - \mu_t(X)$$

Under unconfoundedness:
$$\begin{align}
\tau &= \mathbb{E}[\mu_1(X) - \mu_0(X)] \\
&\quad + \mathbb{E}[T \cdot R_1(X) / e(X)] - \mathbb{E}[(1-T) \cdot R_0(X) / (1 - e(X))]
\end{align}$$

The first term uses the outcome model directly. The second/third terms use residual weighting (IPW on residuals).

### Sample Estimator

$$\hat{\tau}_{AIPW} = \frac{1}{n} \sum_{i=1}^n \left[ 
\hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) 
+ \frac{T_i (\hat{Y}_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)} 
- \frac{(1-T_i) (\hat{Y}_i - \hat{\mu}_0(X_i))}{1 - \hat{e}(X_i)} 
\right]$$

### Double Robustness Property

**Theorem:** If $\hat{e}(x) \to e(x)$ or $\hat{\mu}_t(x) \to \mu_t(x)$ (or both), then $\hat{\tau}_{AIPW} \to \tau$.

**Proof sketch:** The IPW residual terms vanish asymptotically if either model is correct.

### Efficiency

When the outcome model is well-specified:
$$\text{Var}(\hat{\tau}_{AIPW}) \leq \text{Var}(\hat{\tau}_{IPW})$$

AIPW gains efficiency from outcome information while maintaining robustness to propensity misspecification.

## Implementation Details

### Propensity Score Estimation

Fit logistic regression via gradient descent on negative log-likelihood:
$$\ell(w, b) = -\frac{1}{n} \sum_i [T_i \log(\sigma(x_i^\top w + b)) + (1-T_i) \log(1 - \sigma(x_i^\top w + b))]$$

where $\sigma(z) = 1/(1 + e^{-z})$ is the sigmoid.

Gradient step: $w \leftarrow w - \alpha \nabla_w \ell$

**Numerical stability:**
- Clip logits to $[-500, 500]$ to prevent overflow
- Clip propensity scores to $[0.01, 0.99]$ for IPW to avoid extreme weights

### Outcome Model Estimation

Fit two separate ridge-regularized OLS models:
$$\hat{\mu}_t(x) = \arg\min_w \left\| Y - X w \|^2_2 + \lambda \|w\|^2_2 \right\|$$

solved via normal equations with $\lambda = 10^{-3}$.

## Interpretation: Real Data Results

On Temilola's 324 movie ratings comparing synopsis vs metadata presentation:

**Sample:** 162 observations (81 synopsis, 81 metadata)  
**Outcome:** Rating (0–10)  
**Covariates:** Year, runtime, vote average, # genres (standardized)

**Key finding:** 
- Naive difference in means: 0.006 points
- IPW ATE: 0.004 ± 0.406 [95% CI: -0.77, 0.80]
- AIPW ATE: 0.017 ± 0.112 [95% CI: -0.20, 0.23]

**Interpretation:** No significant causal effect detected. The synopsis modality vs metadata modality does not substantially affect ratings after adjusting for movie characteristics.

**Overlap:** Perfect overlap detected (propensity scores all ~0.5), suggesting treatment is well-balanced and no extrapolation required.

## References

1. Rotnitzky, A., & Robins, J. M. (1995). "Semiparametric efficiency bounds." Sankhya, 57, 5–21.
2. Kennedy, E. H. (2016). "Semiparametric doubly robust estimation." Handbook of Causal Inference, 2, 1–22.
3. Angrist, J. D., & Pischke, J. S. (2008). Mostly Harmless Econometrics: An Empiricist's Companion. Princeton University Press.
