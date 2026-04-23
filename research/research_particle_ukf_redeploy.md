# Research Report: Particle Filters & Unscented Kalman Filters for Nonlinear Taste Dynamics
## Pipeline 3 — Nonlinear Extensions of the Kalman+RTS Arc

**Date:** April 16, 2026
**Context:** ACT II of the Mega-Pitch frames Temilola's Date-Watched sequence as a 1D state-space problem. Linear Kalman+RTS is the classical baseline. But what if taste evolution is **nonlinear**? This report rehabilitates particle filters, UKF, and EKF as the natural next steps in model complexity—strictly more general than Kalman, with direct apples-to-apples comparisons on smoothed RMSE.

**Key insight:** The MEGA_PITCH marked these as "out-of-scope" because Kalman+RTS alone hits the 5-rubric targets. But including them adds a crucial **model-comparison ladder**: HMM (discrete, class baseline) → Linear Kalman+RTS (continuous, first-principles) → UKF (smooth nonlinearity) → Particle Filter (arbitrary nonlinearity), each strictly more expressive. This ladder itself is a compelling narrative arc *within* ACT II, and the from-scratch implementations are tractable in 7 days.

---

## Executive Summary

Personal taste is unlikely to be perfectly linear. A nonlinear state-space model might capture:
- **Mood shocks:** sudden genre shifts (not gradual drift)
- **Saturation effects:** after many drama watches, rating propensity for drama drops (S-curve)
- **Regime-dependent dynamics:** state evolution differs by watch context (e.g., binge-mode propensity increases faster)

This report surveys five approaches to nonlinear filtering, each with math hooks for from-scratch derivation, each paired with a class-covered baseline, each measurable on the same metric (smoothed RMSE on Date-Watched holdout).

**The core recommendation:** Implement **Unscented Kalman Filter (UKF)** as the headline nonlinear method (elegant sigma-point math, ~120 LOC from-scratch), backed by **Extended Kalman Filter (EKF)** as Jacobian baseline, and optionally **Particle Filter (SIR)** as the non-parametric wild card. All three sit naturally in ACT II's sequence narrative and directly extend Kalman's pedagogical arc.

---

## Part 1: Why Nonlinear Filtering Matters Here

### The Linear Kalman Assumption

Linear Kalman filter assumes:
$$x_t = A x_{t-1} + w_t, \quad y_t = C x_t + v_t, \quad w_t \sim N(0, Q), \quad v_t \sim N(0, R)$$

This assumes **Temilola's latent rating propensity evolves linearly** (constant drift/decay) and observations are **linear projections** of that latent state.

In reality:
- **Nonlinear dynamics:** taste propensity might follow logistic growth (saturation) or oscillate (recovery cycles)
- **Nonlinear observations:** mapping latent propensity to observed 1–5 rating might be S-shaped (sigmoid), not linear
- **Discrete shifts:** mood can jump (watching a "Game of Thrones" premiere resets expectations), not just drift
- **Interaction effects:** state evolution might depend on current movie features (context-dependent)

### Why Test Nonlinearity?

From rubric perspective:
- **MLFlexibility:** Testing multiple model architectures shows meta-knowledge ("when is each right?")
- **MLMath:** Nonlinear filters require Jacobians (EKF), sigma points (UKF), or particle representations (PF)—more sophisticated linear algebra
- **Apples-to-apples:** Comparing smoothed RMSE across the ladder (HMM → Kalman → EKF → UKF → PF) directly proves which assumption is justified
- **Domain sense:** Shows the student thought about whether Temilola's taste is truly linear

---

## Part 2: Five Nonlinear Filtering Approaches

### Approach 1: Extended Kalman Filter (EKF) — *Jacobian Baseline*

#### What it is

Relaxes linearity by **locally linearizing** around current state estimate:
$$x_t = f(x_{t-1}) + w_t, \quad y_t = h(x_t) + v_t$$

where $f$ and $h$ are arbitrary smooth functions.

#### Math Hook (First-Principles)

**Prediction step:**
- Nonlinear propagation: $\hat{x}_{t|t-1} = f(\hat{x}_{t-1|t-1})$
- Linearize $f$ around posterior mean: $F_t = \nabla_x f(\hat{x}_{t-1|t-1})$ (Jacobian)
- Covariance prediction: $P_{t|t-1} = F_t P_{t-1|t-1} F_t^T + Q$ (using linear approximation)

**Update step:**
- Predicted observation: $\hat{y}_{t|t-1} = h(\hat{x}_{t|t-1})$
- Linearize $h$: $H_t = \nabla_x h(\hat{x}_{t|t-1})$ (Jacobian)
- Kalman gain: $K_t = P_{t|t-1} H_t^T (H_t P_{t|t-1} H_t^T + R)^{-1}$ (unchanged formula)
- State update: $\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t (y_t - h(\hat{x}_{t|t-1}))$
- Covariance update: $P_{t|t} = (I - K_t H_t) P_{t|t-1}$

**Key trade-off:** First-order Taylor error accumulates; if $f$ and $h$ are highly curved, linearization mismatch worsens uncertainty estimates (covariance becomes optimistic).

#### Application to Watch History

Example nonlinear models:
1. **Saturation dynamics:** $x_t = A x_{t-1} + b \tanh(c x_{t-1}) + w_t$ (after many drama watches, propensity saturates)
2. **Sigmoid observation:** $y_t = \sigma(C x_t + d) + v_t$ (map continuous propensity to [0,1] interval, then scale to 1–5 rating)
3. **Binge recovery:** $x_t = (1 - \lambda) x_{t-1} + \lambda x_{\text{mean}} + w_t$ where $\lambda$ depends on time gap since last watch (slow recovery after drought)

#### Why It Impresses the Rubric

- **Math density:** Jacobian computation (numerical or symbolic), Taylor expansion, linearization error bounds
- **Comparison:** Direct side-by-side with linear Kalman (same data, same metric) shows when linearity breaks down
- **Interpretability:** If EKF beats Kalman significantly, nonlinear model is justified; if not, Occam's razor favors linear
- **Visualization:** Predicted uncertainty bands shrink in EKF vs. Kalman when curvature is high → shows the difference

#### Implementation Feasibility

**From-scratch:** ~80–100 lines (numerical Jacobian via finite differences or symbolic via JAX)
```python
# Pseudocode
def ekf_predict(x_hat, P, f, f_jacobian, Q, dt):
    x_pred = f(x_hat, dt)  # Nonlinear dynamics
    F = f_jacobian(x_hat, dt)  # Jacobian
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred

def ekf_update(x_pred, P_pred, y, h, h_jacobian, R):
    y_pred = h(x_pred)
    H = h_jacobian(x_pred)
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_upd = x_pred + K @ (y - y_pred)
    P_upd = (np.eye(len(x_pred)) - K @ H) @ P_pred
    return x_upd, P_upd
```

**Library option:** `filterpy.KalmanFilter` can be wrapped with custom `f` and `h`; or write from scratch in numpy/JAX.

**7-day effort:** LOW-MEDIUM
- Day 1: Understand Jacobian computation, Taylor approximation
- Days 2–3: Implement EKF from scratch, test on synthetic data
- Day 4: RTS backward pass for EKF (smoother)
- Day 5: Compare to linear Kalman on watch data

---

### Approach 2: Unscented Kalman Filter (UKF) — *Sigma-Point Magic*

#### What it is

Avoids Jacobians entirely. Instead, use **deterministic sampling** to represent the posterior: generate a **cloud of "sigma points"** that capture the mean and covariance exactly (for Gaussians). Transform sigma points through nonlinear $f$ and $h$, recover moments from transformed samples.

**The key insight:** Nonlinear transformation of a Gaussian mixture can be approximated via nonlinear transformation of deterministically chosen samples, rather than via Taylor expansion.

#### Math Hook (First-Principles)

**Sigma-point generation:**
Generate $2n+1$ sigma points around the posterior mean $\hat{x}_{t-1|t-1}$ with covariance $P_{t-1|t-1}$ (where $n$ = state dimension):

$$\mathcal{X}_0 = \hat{x}_{t-1|t-1}$$
$$\mathcal{X}_i = \hat{x}_{t-1|t-1} + \sqrt{(n + \lambda)} (\text{chol}(P_{t-1|t-1}))_i, \quad i = 1, \ldots, n$$
$$\mathcal{X}_{n+i} = \hat{x}_{t-1|t-1} - \sqrt{(n + \lambda)} (\text{chol}(P_{t-1|t-1}))_i, \quad i = 1, \ldots, n$$

where $\lambda = \alpha^2 (\kappa + n) - n$ (tuning parameter, default $\alpha = 10^{-3}$, $\kappa = 0$ or $3-n$).

**Weights for mean/covariance recovery:**
$$w_0^{(m)} = \frac{\lambda}{\lambda + n}, \quad w_i^{(m)} = w_i^{(c)} = \frac{1}{2(n+\lambda)}, \quad i \geq 1$$

(Weights sum to 1 for mean, and the weighted sample covariance recovers the true covariance for Gaussians.)

**Prediction step:**
1. Propagate sigma points through $f$: $\mathcal{X}_i^{(pred)} = f(\mathcal{X}_i)$
2. Recover predicted mean: $\hat{x}_{t|t-1} = \sum_i w_i^{(m)} \mathcal{X}_i^{(pred)}$
3. Recover predicted covariance: $P_{t|t-1} = \sum_i w_i^{(c)} (\mathcal{X}_i^{(pred)} - \hat{x}_{t|t-1}) (\mathcal{X}_i^{(pred)} - \hat{x}_{t|t-1})^T + Q$

**Update step:**
1. Transform predicted sigma points through observation model: $\mathcal{Y}_i = h(\mathcal{X}_i^{(pred)})$
2. Recover predicted observation: $\hat{y}_{t|t-1} = \sum_i w_i^{(m)} \mathcal{Y}_i$
3. Recover observation covariance: $S_t = \sum_i w_i^{(c)} (\mathcal{Y}_i - \hat{y}_{t|t-1}) (\mathcal{Y}_i - \hat{y}_{t|t-1})^T + R$
4. Cross-covariance: $P_{xy} = \sum_i w_i^{(c)} (\mathcal{X}_i^{(pred)} - \hat{x}_{t|t-1}) (\mathcal{Y}_i - \hat{y}_{t|t-1})^T$
5. Kalman gain: $K_t = P_{xy} S_t^{-1}$ (same form as linear Kalman)
6. State update: $\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t (y_t - \hat{y}_{t|t-1})$
7. Covariance update: $P_{t|t} = P_{t|t-1} - K_t S_t K_t^T$

**Theoretical advantage:** UKF is **3rd-order accurate** for Gaussian posteriors (captures mean, covariance, and some 3rd moments exactly), vs. EKF's 1st-order. In practice, UKF often gives better uncertainty estimates without Jacobian computation.

#### Application to Watch History

Same nonlinear models as EKF:
- Sigmoid observation: $y_t = 5 \cdot \sigma(C x_t)$ (rating ∈ [1,5])
- Saturation dynamics: $x_t = A x_{t-1} + b \tanh(c x_{t-1}) + w_t$
- Neural net black-box: $y_t = \text{NNet}(x_t) + v_t$ (UKF doesn't require Jacobians, so any differentiable or non-differentiable $h$ works)

#### Why It Impresses the Rubric

- **Algorithmic elegance:** Deterministic sampling instead of calculus—more intuitive than Jacobians
- **Math-heavy:** Sigma-point weights, Cholesky decomposition, weighted moment recovery from first principles
- **Comparison paper:** EKF vs. UKF on same nonlinear data, same metric, shows accuracy gains
- **Visualization:** Sigma-point cloud evolution (2D projection) shows how nonlinear $f$ spreads the samples
- **Black-box compatibility:** Can use neural net observation models without deriving Jacobians

#### Implementation Feasibility

**From-scratch:** ~120–150 lines
```python
# Pseudocode
def sigma_points(x_hat, P, n, alpha, kappa):
    lambda_ = alpha**2 * (kappa + n) - n
    sqrt_term = np.linalg.cholesky((n + lambda_) * P)
    X = np.hstack([x_hat[:, None],
                    x_hat[:, None] + sqrt_term,
                    x_hat[:, None] - sqrt_term])
    w_m = np.hstack([lambda_ / (n + lambda_),
                     0.5 * np.ones(2*n) / (n + lambda_)])
    w_c = w_m.copy()
    return X, w_m, w_c

def ukf_predict(x_hat, P, X, w_m, w_c, f, Q, dt):
    X_pred = np.array([f(X[:, i], dt) for i in range(X.shape[1])]).T
    x_pred = X_pred @ w_m
    P_pred = ((X_pred - x_pred[:, None]) * np.sqrt(w_c)) @ (X_pred - x_pred[:, None]).T + Q
    return x_pred, P_pred, X_pred

def ukf_update(x_pred, P_pred, X_pred, y, w_m, w_c, h, R):
    Y = np.array([h(X_pred[:, i]) for i in range(X_pred.shape[1])]).T
    y_pred = Y @ w_m
    S = ((Y - y_pred[:, None]) * np.sqrt(w_c)) @ (Y - y_pred[:, None]).T + R
    P_xy = ((X_pred - x_pred[:, None]) * np.sqrt(w_c)) @ (Y - y_pred[:, None]).T
    K = P_xy @ np.linalg.inv(S)
    x_upd = x_pred + K @ (y - y_pred)
    P_upd = P_pred - K @ S @ K.T
    return x_upd, P_upd
```

**Library option:** `filterpy.UnscentedKalmanFilter` (reference implementation, easy to adapt).

**7-day effort:** MEDIUM
- Day 1: Understand sigma-point construction, weights, Cholesky
- Days 2–3: Implement UKF from scratch
- Day 4: RTS backward pass for UKF (extended to sigma points)
- Day 5: Visualize sigma clouds, compare to EKF
- Day 6: Comparison table (Kalman vs. EKF vs. UKF on nonlinear model)

---

### Approach 3: Particle Filter (Sequential Importance Resampling) — *Non-Parametric Gold Standard*

#### What it is

Represents the posterior not as a Gaussian, but as a **discrete cloud of particles** (weighted samples). Each particle is a hypothetical state trajectory; weights updated as observations arrive. Handles arbitrary nonlinear/non-Gaussian models.

**Core idea:** Replace probability density $p(x_t | y_{1:t})$ with empirical distribution $\frac{1}{N} \sum_{i=1}^N \delta(x_t - x_t^{(i)})$ (or weighted: $\sum_i w_i^{(i)} \delta(\cdot)$).

#### Math Hook (First-Principles)

**Sequential Importance Sampling (SIS):**

At time $t-1$, we have weighted particles $\{x_{t-1}^{(i)}, w_{t-1}^{(i)}\}_{i=1}^N$ approximating $p(x_{t-1} | y_{1:t-1})$.

At time $t$, observe $y_t$. **Importance sampling update:**
$$w_t^{(i)} \propto w_{t-1}^{(i)} \frac{p(y_t | x_t^{(i)}) p(x_t^{(i)} | x_{t-1}^{(i)})}{q(x_t^{(i)} | x_{t-1}^{(i)}, y_t)}$$

where $q(\cdot)$ is the **proposal distribution** (usually $q = p(x_t | x_{t-1})$, the prior).

If $q = p(\cdot | x_{t-1})$:
$$w_t^{(i)} \propto w_{t-1}^{(i)} p(y_t | x_t^{(i)})$$

This is the **bootstrap particle filter:** propose from prior, weight by likelihood.

**Resampling (Systematic Resampling):**

After many time steps, weights become concentrated (few particles carry most weight; others are negligible). **Resampling** draws $N$ new particles with replacement from $\{x_t^{(i)}\}$ with probabilities $w_t^{(i)}$, then reset all weights to $1/N$.

**Effective Sample Size (ESS):**
$$\text{ESS}_t = \frac{1}{\sum_i (w_t^{(i)})^2}$$

Resample if $\text{ESS}_t < N_{\text{thresh}}$ (e.g., 0.5N).

**Posterior approximation:**
$$p(x_t | y_{1:t}) \approx \sum_{i=1}^N w_t^{(i)} \delta(x_t - x_t^{(i)})$$

**Credible intervals:** Empirical quantiles of particles.

#### Application to Watch History

**Particle representation:** Each particle is a hypothetical latent rating propensity trajectory $(x_1^{(i)}, \ldots, x_T^{(i)})$.

**Why particles help:**
1. **Non-Gaussian posteriors:** If mood regime is multi-modal ("either user is in romance binge OR action binge, unclear which"), particles naturally represent both modes
2. **Discrete+continuous mix:** Can model discrete mood switches + continuous propensity drift simultaneously
3. **Extreme nonlinearity:** Can handle highly nonlinear observation models (e.g., rating given propensity via a neural net)

**Example dynamics:**
- Prior: $x_t = x_{t-1} + w_t$ (random walk on [1,5] propensity scale)
- Likelihood: $p(y_t | x_t^{(i)}) = N(y_t; \sigma(x_t^{(i)}), R)$ (sigmoid-mapped propensity + Gaussian noise)

#### Why It Impresses the Rubric

- **Non-parametric theory:** Minimal distributional assumptions; elegant connection to Bayesian inference
- **Math-rich:** Importance sampling ratio, resampling correctness proof (exchangeability argument), ESS criterion
- **Flexibility:** Can handle multimodal posteriors, discrete-continuous hybrids, arbitrary observation models
- **Visualization:** Particle cloud animated over time (2D projection) is compelling; shows how particles collapse/diverge
- **Comparison:** Particle filter vs. Kalman/EKF/UKF under highly nonlinear model highlights strengths/weaknesses of each

#### Implementation Feasibility

**From-scratch:** ~150–200 lines
```python
# Pseudocode
def pf_predict(particles, weights, f, Q, dt, N):
    """Predict via bootstrap: resample, propagate."""
    # Resample if ESS is low
    ess = 1.0 / np.sum(weights**2)
    if ess < 0.5 * N:
        idx = np.random.choice(N, N, p=weights)
        particles = particles[idx, :]
        weights = np.ones(N) / N

    # Propagate via dynamics
    noise = np.random.randn(N, particles.shape[1]) @ np.linalg.cholesky(Q)
    particles_pred = np.array([f(particles[i, :], dt) for i in range(N)]) + noise
    return particles_pred, weights

def pf_update(particles, weights, y, h, R, N):
    """Update weights via likelihood."""
    likelihoods = np.array([likelihood_gaussian(y, h(particles[i, :]), R) for i in range(N)])
    weights_new = weights * likelihoods
    weights_new /= np.sum(weights_new)  # Normalize
    return particles, weights_new

def likelihood_gaussian(y, y_pred, R):
    """p(y | y_pred) under Gaussian noise."""
    return np.exp(-0.5 * ((y - y_pred)**2).sum() / R)
```

**Library option:** `particles` package (Chopin), `filterpy` basic SIR, or `TensorFlow Probability`.

**7-day effort:** MEDIUM-HIGH
- Days 1–2: Understand importance sampling, resampling algorithms, ESS criterion
- Days 3–4: Implement SIR particle filter from scratch
- Day 5: Add RTS-like backward smoothing (FFBSi = Forward Filter Backward Simulator)
- Day 6: Compare to Kalman/EKF/UKF on data with multimodal posterior

---

### Approach 4: Rao-Blackwellized Particle Filter (RBPF) — *Wild Card: Hybrid Linear+Nonlinear*

#### What it is

Splits state into two parts: $x_t = (x_t^{(\text{lin})}, x_t^{(\text{nonlin})})$

- **Linear substate** $x_t^{(\text{lin})}$ evolves linearly conditioned on nonlinear substate
- **Nonlinear substate** $x_t^{(\text{nonlin})}$ is sampled via particles

For each particle (nonlinear state), the linear substate is integrated out analytically via Kalman filter. Result: fewer particles needed than standard PF.

#### Math Hook

Given particle $x_t^{(\text{nonlin}), (i)}$:
1. Kalman filter on $x_t^{(\text{lin})} | x_t^{(\text{nonlin}), (i)}$ (conditional Kalman)
2. Weight particles by marginal likelihood after integrating out linear state
3. Much better efficiency than full particle filter

#### Application to Watch History

**Example decomposition:**
- **Nonlinear substate:** Discrete mood regime (drama/action/horror, sampled)
- **Linear substate:** Continuous propensity drift within regime (Kalman-filtered)

**7-day effort:** HIGH (requires understanding both Kalman and particles, trickier to implement)

**Recommendation:** **Skip for 7-day crunch; mention as future work.** It's a wild card, not essential.

---

### Approach 5: Particle Smoother (FFBSi) — *Backward Refinement*

#### What it is

Extends particle filter with a **backward pass** (like RTS smoother for Kalman). Forward filter generates particles; backward simulator samples indices from weighted particles at time $T$, traces backward to time 1, producing smoothed particle trajectories.

#### Math Hook

**Forward:** Run particle filter as usual, store particle sets $\{x_t^{(i)}\}$ and weights $\{w_t^{(i)}\}$ for all $t$.

**Backward (Fixed-Interval Back-Simulation, FFBSi):**
1. At time $T$, sample trajectory index from final weights: $I_T \sim \text{Cat}(w_T^{(i)})$
2. For $t = T-1, \ldots, 1$:
   - Compute relative weights: $\tilde{w}_t^{(i)} = w_t^{(i)} p(x_{t+1}^{(I_{t+1})} | x_t^{(i)})$
   - Sample backward index: $I_t \sim \text{Cat}(\tilde{w}_t^{(i)})$
   - Trace backward

Result: Single smoothed trajectory sampled from $p(x_{1:T} | y_{1:T})$ (joint posterior over entire sequence).

#### Why It Matters

- **Smoothed estimates** (use all data) rather than filtered (causal only)
- Natural for visualization: one representative "likely" trajectory through latent space
- Multiple independent samples give credible intervals on smoothed states

#### Implementation Feasibility

**From-scratch:** ~60–80 lines (append to forward filter)

**7-day effort:** LOW-MEDIUM (implement after forward PF)

---

## Part 3: Apples-to-Apples Comparison Framework

### Metric: Smoothed RMSE on Hold-Out Data

For each method (HMM, Kalman, EKF, UKF, PF), fit on first T−10 watch events, evaluate smoothed state estimates on final 10 watches:

$$\text{Smoothed RMSE} = \sqrt{\frac{1}{10} \sum_{t=T-10}^T (\hat{x}_{t|T} - y_t)^2}$$

where $\hat{x}_{t|T}$ is the smoothed latent state (using all $T$ observations).

### Comparison Table Structure

| **Method** | **Class-Covered?** | **From-Scratch?** | **Nonlinear?** | **Smoothed RMSE** | **Filtered RMSE** | **Uncertainty Calibration** | **Runtime (ms)** |
|---|---|---|---|---|---|---|---|
| HMM Baseline | ✅ | (Likely not) | Discrete | 0.XX | 0.XX | Entropy of posteriors | ~50 |
| Kalman + RTS | ❌ | ✅ | ❌ | 0.XX | 0.XX | Covariance (Gaussian) | ~10 |
| EKF + RTS | ❌ | ✅ | ✅ (Jacobian) | 0.XX | 0.XX | Covariance (approx) | ~30 |
| UKF + RTS | ❌ | ✅ | ✅ (Sigma) | 0.XX | 0.XX | Covariance (approx) | ~100 |
| Particle Filter | ❌ | ✅ | ✅ (Arbitrary) | 0.XX | 0.XX | Empirical percentiles | ~200 |

**Expected outcome:** Kalman dominates on linear data; EKF/UKF edge it out if true nonlinearity; PF wins under multimodal posteriors or extreme nonlinearity (but with more compute).

---

## Part 4: Implementation Roadmap (7-Day Gate)

### Day 1: Baseline & Data Setup
- Assemble Date-Watched sequence as (time, observed-rating) pairs
- Fit HMM baseline (class method) for apples-to-apples anchor
- Define nonlinear dynamics $f$ and observation model $h$ (sigmoid, saturation, etc.)

### Days 2–3: Linear Kalman + RTS (if not already done in main arc)
- Derive forward Kalman filter (prediction + update)
- Derive RTS smoother (backward recursion)
- Implement from scratch in numpy (~80 lines total)
- Test on synthetic data, visualize filtered vs. smoothed trajectories

### Days 3–4: Extended Kalman Filter
- Understand Jacobian computation (numerical finite-difference)
- Implement EKF prediction/update with Jacobians (~80 lines)
- Implement EKF + RTS smoother (~40 lines)
- Compare smoothed RMSE vs. Kalman

### Days 4–5: Unscented Kalman Filter
- Understand sigma-point construction, Cholesky decomposition, weighted moments
- Implement UKF prediction/update (~120 lines)
- Implement UKF + RTS smoother (~60 lines)
- Compare to EKF: accuracy, runtime, interpretability

### Day 6: Particle Filter (Optional; can skip for time)
- Understand importance sampling, resampling, ESS criterion
- Implement SIR particle filter (~150 lines)
- Implement FFBSi backward smoother (~60 lines)
- Compare to deterministic filters

### Day 7: Comparison & Visualization
- Evaluation: smoothed RMSE, filtered RMSE, uncertainty calibration, runtime
- Build comparison table
- Visualizations:
  - Filtered vs. smoothed state trajectories (Kalman vs. EKF vs. UKF vs. PF)
  - Uncertainty bands (Kalman covariance, sigma clouds, particle clouds)
  - Attention/weights heatmap (if particles used)
  - Nonlinear dynamics (curve $f$ vs. linearization $F$)
- Narrative: "When does nonlinearity matter?"

---

## Part 5: From-Scratch Code Skeleton

### Nonlinear Dynamics (Example)

```python
import numpy as np

# Dynamics: logistic growth with recovery cycle
def dynamics_f(x, dt, a=0.05, b=1.0, c=2.0):
    """x_{t+1} = a*x_t + b*tanh(c*x_t) + noise"""
    return a * x + b * np.tanh(c * x)

# Observation: sigmoid mapping to [1,5] rating scale
def observation_h(x, scale=1.0):
    """y = 3 + 2*sigmoid(scale*x) (maps to roughly [1,5])"""
    return 3.0 + 2.0 * (1.0 / (1.0 + np.exp(-scale * x)))

# Jacobian of dynamics (for EKF)
def jacobian_f(x, dt, a=0.05, b=1.0, c=2.0):
    """dF/dx at point x"""
    return a + b * c * (1.0 - np.tanh(c * x)**2)

# Jacobian of observation (for EKF)
def jacobian_h(x, scale=1.0):
    """dH/dx at point x"""
    sig = 1.0 / (1.0 + np.exp(-scale * x))
    return 2.0 * scale * sig * (1.0 - sig)
```

### EKF Core

```python
def ekf_step(x_hat, P, y, dt, Q, R):
    """One step: predict + update."""
    # Predict
    x_pred = dynamics_f(x_hat, dt)
    F = jacobian_f(x_hat, dt)
    P_pred = F * P * F + Q  # Scalar case

    # Update
    y_pred = observation_h(x_pred)
    H = jacobian_h(x_pred)
    S = H * P_pred * H + R
    K = P_pred * H / S
    x_upd = x_pred + K * (y - y_pred)
    P_upd = (1.0 - K * H) * P_pred

    return x_upd, P_upd
```

### UKF Core

```python
def sigma_points_1d(x_hat, P, alpha=0.001, kappa=0, beta=2.0):
    """Generate sigma points for 1D state."""
    n = 1
    lambda_ = alpha**2 * (kappa + n) - n
    c = np.sqrt((n + lambda_) * P)

    X = np.array([x_hat, x_hat + c, x_hat - c])
    w_m = np.array([lambda_ / (n + lambda_), 0.5 / (n + lambda_), 0.5 / (n + lambda_)])
    w_c = w_m.copy()
    w_c[0] += 1.0 - alpha**2 + beta

    return X, w_m, w_c

def ukf_step(x_hat, P, X, w_m, w_c, y, dt, Q, R):
    """One UKF step."""
    # Predict
    X_pred = np.array([dynamics_f(X[i], dt) for i in range(len(X))])
    x_pred = np.sum(w_m * X_pred)
    P_pred = np.sum(w_c[:, None] * (X_pred - x_pred)**2) + Q

    # Update
    Y = np.array([observation_h(X_pred[i]) for i in range(len(X_pred))])
    y_pred = np.sum(w_m * Y)
    S = np.sum(w_c[:, None] * (Y - y_pred)**2) + R
    P_xy = np.sum(w_c[:, None] * (X_pred - x_pred) * (Y - y_pred))
    K = P_xy / S
    x_upd = x_pred + K * (y - y_pred)
    P_upd = P_pred - K**2 * S

    return x_upd, P_upd
```

---

## Part 6: Visualization & Narrative

### Key Figures

1. **State Trajectories:** Overlay filtered and smoothed state estimates for Kalman, EKF, UKF, PF on same subplot grid (one per method)
2. **Uncertainty Bands:** Kalman 1σ/2σ bands vs. EKF vs. UKF vs. empirical percentiles (PF)
3. **Nonlinearity Illustration:** Plot true dynamics curve $f$ vs. local linearization $F$ at representative points
4. **Convergence:** Smoothed RMSE as number of particles increases (for PF)
5. **Runtime Scaling:** Log-log plot of runtime vs. state dimension or particle count
6. **Posterior Geometry:** 2D projection of joint posterior (e.g., first two principal components of smoother covariance)

### Narrative Arc

> "We assumed Temilola's taste evolved linearly. Let's test that. We fit five models of increasing complexity: HMM (discrete baseline from class), Linear Kalman (fundamental continuous model), EKF (smooth nonlinearity via Jacobians), UKF (smart sigma-point approximation), and Particle Filter (arbitrary nonlinearity). On smoothed RMSE, [INSERT RESULT]. This suggests [INSERT INTERPRETATION: "taste is effectively linear" OR "nonlinear model is justified" OR "multimodal posteriors reveal regime uncertainty"]. Visualization [REFERENCE FIGURE] shows where each method diverges—revealing what structure the data actually has."

---

## Part 7: References & Library Cross-Checks

### Classical Filtering (Theory)

- **Kalman, 1960:** "A New Approach to Linear Filtering and Prediction Problems" (*ASME Trans.*)
- **Rauch, Tung, Striebel, 1965:** "Maximum likelihood estimates of linear dynamic systems" (*AIAA J.*)
- **Bar-Shalom, Li, Kirubarajan, 2001:** *Estimation with Applications to Tracking and Navigation* (standard reference)
- **Särkkä, 2013:** *Bayesian Filtering and Smoothing* (excellent lecture notes, available online)

### Nonlinear Filtering

- **Extended Kalman Filter:** Introduced in Kalman & Bucy (1961), formalized in Bar-Shalom et al.
- **Unscented Kalman Filter:** Julier & Uhlmann (1997), "A New Method for the Nonlinear Transformation of Means and Covariances in Filters and Estimators" (*IEEE Trans.*)
- **Unscented Transform:** Julier & Uhlmann (2004), "Unscented Filtering and Nonlinear Estimation" (*IEEE Proc.*)

### Particle Filtering

- **Sequential Importance Sampling (SIR):** Gordon, Salmond, Smith (1993), "Novel Approach to Nonlinear/Non-Gaussian Bayesian State Estimation" (*IEE Proc.*)
- **Particle Filter Smoothing (FFBSi):** Doucet, Johansen (2009), "A Tutorial on Particle Filtering and Smoothing" (*Handbook of Nonlinear Filtering*)
- **Effective Sample Size:** Liu & Chen (1995), "Blind Deconvolution via Sequential Imputations" (*JASA*)

### Libraries

- **filterpy** (Rob Moore): MIT-licensed, clean implementations of Kalman, EKF, UKF, RTS
  ```
  pip install filterpy
  from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter, RtsSmootherND
  ```
- **particles** (Nicolas Chopin): Advanced SMC methods, nice API
  ```
  pip install particles
  from particles import PythonRV, FeynmanKac, SMCsampler
  ```
- **TensorFlow Probability:** `tfp.distributions`, `tfp.experimental.mcmc` for particle methods
- **PyMC:** Bayesian inference; can set up state-space models via `pm.math.scan`

---

## Part 8: Why This Fits the Mega-Pitch Arc

### ACT II: "Model" (Uncertainty & Sequence)

The Mega-Pitch's ACT II frames the sequence modeling problem as:

> "Frame his ratings as a 1D state-space: latent taste $z_t$ evolves slowly, observed rating $y_t = z_t + \text{noise}$. Fit **Kalman filter + RTS smoother** (predict-update + backward recursion), plus a **modern apples-to-apples comparator** (SASRec self-attention sequential recommender)."

**The problem:** This assumes linearity. The Mega-Pitch says particle filters/UKF are "explicitly out-of-scope."

**The reframe:** They're actually essential to the narrative. Here's why:

1. **Pedagogical arc:** HMM (discrete, class material) → Kalman (continuous, from-scratch) → EKF (smooth nonlinear) → UKF (smart sigma points) → PF (non-parametric). Each is a natural generalization.
2. **Rubric alignment:** Showing that you tested nonlinearity (and measured it) is **MLFlexibility** = 5. "Did you think about whether your assumptions are justified?"
3. **Apples-to-apples is mandatory.** Testing Kalman alone doesn't prove it's right. Testing Kalman vs. UKF on the same hold-out data does.
4. **Math density:** EKF Jacobians, UKF sigma points, PF importance sampling are all "first-principles derivations" the rubric demands.
5. **Visualization gold:** Particle clouds and sigma-point spreads are stunning PDFs.

### Narrative Revision

**OLD (Mega-Pitch):**
> "Sequence: Date Watched is a 1D state-space. Fit Kalman filter + RTS smoother."

**NEW (This Report):**
> "Sequence: Is Temilola's taste linear? We test a model ladder. Baseline is HMM (discrete, class material). We fit Linear Kalman + RTS (from-scratch, assumes Gaussian linearity). Then we ask: **What if taste is nonlinear?** We implement EKF (Jacobian-based), UKF (sigma-point), and Particle Filter (non-parametric), all on the same hold-out data, measured by smoothed RMSE. Result: [Kalman sufficient / nonlinear helps]. Visualization shows where each assumption breaks."

This **strengthens** ACT II without adding much timeline (UKF implementation is tractable, ~120 lines).

---

## Conclusion & Recommendation

### Core Recommendation for 7-Day Pipeline 3

**Must-Have:**
1. **Linear Kalman + RTS Smoother** (from-scratch, ~80 lines, 2 days)
2. **Unscented Kalman Filter + RTS** (from-scratch, ~120 lines, 2 days)

**Nice-to-Have (if ahead of schedule):**
3. **Extended Kalman Filter** (from-scratch, ~80 lines, 1.5 days) — simpler than UKF; if doing Jacobians, might as well
4. **Particle Filter SIR + FFBSi** (from-scratch, ~200 lines, 2 days) — wild card; shows non-parametric thinking

**Out-of-Scope (too heavy for 7 days):**
- Rao-Blackwellized PF (hybrid linear+nonlinear; trickier)
- Full Neural ODE comparator (belongs in architecture stream, not state-space)

### Why UKF is the Headline

- **Elegant math:** Sigma points are intuitive (deterministic sampling, not Taylor expansion)
- **Practical:** Often beats EKF in accuracy; no Jacobian derivation needed
- **Rubric-aligned:** First-principles derivation of weights, Cholesky, weighted recovery
- **Visualization-friendly:** Sigma-point clouds evolving over time is visually compelling in a PDF
- **From-scratch is feasible:** ~120 lines, no AD framework required

### Comparison Table (For Executive Summary)

| **Method** | **Novelty** | **Math Density** | **7-Day Effort** | **Rubric Win** |
|---|---|---|---|---|
| Kalman + RTS | ⭐⭐⭐ | ⭐⭐⭐ | LOW | Closed-form, first-principles |
| EKF | ⭐⭐ | ⭐⭐⭐ | MEDIUM | Jacobian calculus, linearization intuition |
| UKF | ⭐⭐⭐ | ⭐⭐⭐ | MEDIUM | Sigma-point math, no Jacobians, elegant |
| Particle Filter | ⭐⭐⭐ | ⭐⭐⭐ | MEDIUM-HIGH | Non-parametric Bayes, importance sampling, resampling |

**Final verdict:** Include **Kalman + RTS** and **UKF + RTS** in ACT II; visualize all three (Kalman, EKF, UKF, optionally PF) on a comparison grid; conclude with "UKF balances theoretical elegance and practical accuracy."

---

**End of Report**
