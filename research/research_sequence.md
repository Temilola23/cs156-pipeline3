# Research Report: Sequence & State-Space Modeling for Personal Watch History
## Pipeline 3 — Temporal Structure & Predictive Dynamics

**Date:** April 16, 2026
**Context:** 162 dated watch events from Letterboxd + TMDB. Temporal structure has been completely ignored in P1 & P2. Assignment requires novel method(s) not covered in class, with heavy emphasis on first-principles derivations and apples-to-apples baselines.

---

## Executive Summary

Personal watch history is fundamentally a **time-indexed sequence of discrete events** with:
- **162 events** (small N, but rich structure: date + movie features + user rating)
- **Irregular temporal cadence** (days/weeks between watches)
- **Latent "mood regimes"** (binge patterns, genre shifts, rating drift)
- **Two prediction tasks:** (1) Next genre/rating given history, (2) Detect taste-shift changepoints

This report surveys **8 classical state-space methods** and **6 modern neural sequence models**, pinpointing which are most novel-to-class and implement-able in 7 days on a 162-event dataset. **Wild-card hybrid ideas** included.

**Rubric alignment:** Every method paired with class baseline, first-principles math derivations provided, from-scratch implementation feasibility assessed, and apples-to-apples comparison sketched.

---

## Part 1: Classical State-Space Models

These models explicitly model latent state evolution + observation process. All are **extensions of HMM** (likely covered around session 20) or **Bayesian filtering theory**—making them novel yet grounded.

### 1. Hidden Markov Model (HMM) — *Baseline*, Likely Covered

**What it is:**
Discrete latent states (e.g., "action mood," "drama mood," "binge mode") with Markovian transitions. Observable: genre + rating. Inference via forward-backward algorithm → marginal posteriors at each time step.

**Math hook (first-principles):**
- Forward pass: $\alpha_t(z) = P(z_t=z | y_{1:t}) \propto \sum_{z'} P(y_t|z) P(z|z') \alpha_{t-1}(z')$
- Backward pass: $\beta_t(z) = P(y_{t+1:T}|z_t=z) = \sum_{z'} P(y_{t+1}|z') P(z'|z) \beta_{t+1}(z')$
- Smoothed posterior: $\gamma_t(z) \propto \alpha_t(z) \beta_t(z)$
- **Why novel:** Extending HMM to *heterogeneous mood regimes* (K regimes with custom emission dists per genre) is novel-to-class.

**Why it impresses rubric:**
- Apples-to-apples baseline: naive rating predictor vs. HMM forward-filtering
- Derivation: full forward-backward + Viterbi + Baum-Welch EM from Bayes
- Visualization: state sequence, transition heatmap, regime-colored watch timeline

**Library + from-scratch:**
- Library: `hmmlearn` (sklearn-adjacent), `pomegranate` (flexible)
- From-scratch: Forward-backward fully differentiable in PyTorch (~100 lines)
- Feasibility: **High** — trivial for N=162

**7-day effort: LOW** (library baseline in 1 day; from-scratch + analysis in 2–3 days)

**Class baseline:** Standard discrete HMM from class materials vs. custom watch-history HMM

---

### 2. Kalman Filter (Linear Gaussian State-Space) — *Novel*, Fundamental

**What it is:**
Assumes linear dynamics + Gaussian noise. Estimates continuous-valued latent state (e.g., user's "latent rating propensity") given noisy observations (actual ratings).

**Model structure:**
- State: $x_t = A x_{t-1} + w_t$, $w_t \sim N(0, Q)$
- Observation: $y_t = C x_t + v_t$, $v_t \sim N(0, R)$

**Math hook (first-principles):**
- **Prediction step:** $\hat{x}_{t|t-1} = A \hat{x}_{t-1|t-1}$, $P_{t|t-1} = A P_{t-1|t-1} A^T + Q$
- **Kalman gain:** $K_t = P_{t|t-1} C^T (C P_{t|t-1} C^T + R)^{-1}$
- **Update step:** $\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t (y_t - C \hat{x}_{t|t-1})$
- **Posterior covariance:** $P_{t|t} = (I - K_t C) P_{t|t-1}$
- **Derivation:** From Bayes rule on joint Gaussian + marginalizing = optimal MMSE estimator

**Why it impresses rubric:**
- **First-principles:** Full derivation from Bayes + Gaussians + linear algebra
- **Math density:** Gain matrix, information filter, Joseph form covariance
- **Extensions obvious:** EKF, UKF, particle filter all follow from this
- **Visualization:** Filtered state trajectory + prediction intervals (2σ bands)

**Application to watch history:**
- Latent state = "latent user rating tendancy" (continuous [1–5] scale)
- Observations = actual ratings (noisy due to mood/context)
- Forecasting: next predicted rating ± confidence band

**Library + from-scratch:**
- Library: `filterpy` (MIT-licensed, simple), `statsmodels` (for state-space models)
- From-scratch: Kalman filter = ~50 lines NumPy; RTS smoother = ~30 more
- Feasibility: **Very high** — closed-form updates, no iterative optimization

**7-day effort: LOW** (library 1 day; from-scratch derivation + code 2 days; analysis 1 day)

**Class baseline:** OLS regression (P1) vs. Kalman filter + RTS smoother

---

### 3. Extended Kalman Filter (EKF) — *Novel*, Nonlinear Dynamics

**What it is:**
Relaxes linearity. Models nonlinear dynamics $x_t = f(x_{t-1}) + w_t$ and observations $y_t = h(x_t) + v_t$ via **local Taylor linearization** around current state estimate.

**Math hook:**
- Jacobians: $F_t = \nabla_x f(\hat{x}_{t-1|t-1})$, $H_t = \nabla_x h(\hat{x}_{t|t-1})$
- Prediction: Use nonlinear $f$ directly; approx posterior as Gaussian with Jacobian-linearized covariance
- Update: Use Jacobian $H_t$ in Kalman gain
- **Trade-off:** Higher fidelity than KF, but curvature mismatch → worse uncertainty estimates than UKF

**Application:**
- Nonlinear state model: $x_t = a x_{t-1} + b \sin(x_{t-1}) + w_t$ (captures binge-recovery cycles)
- Rating observation: $y_t = \sigma(C x_t) \in (0,1)$ (logistic mapping of latent propensity to [0,1])

**Why it impresses rubric:**
- More realistic than KF (nonlinear rating dynamics)
- Tangible math: Jacobians, Taylor expansion, comparison with UKF

**Library + from-scratch:**
- Library: `filterpy.KalmanFilter` with custom nonlinear dynamics
- From-scratch: ~80 lines (Jacobian numerics + KF loop)
- Feasibility: **High** — standard calculus

**7-day effort: MEDIUM** (understand EKF vs. KF theory: 1 day; implementation: 2 days; debugging: 1 day; viz: 1 day)

**Class baseline:** Kalman vs. EKF: nonlinear rating dynamics reveal EKF edge

---

### 4. Unscented Kalman Filter (UKF) — *Novel*, Sigma-Point Approximation

**What it is:**
Avoids Jacobians via **sigma-point sampling.** Represents posterior as weighted sample cloud; nonlinear $f$ and $h$ applied to samples; posterior moments recovered from transformed samples.

**Why it matters:**
- More accurate than EKF for moderately nonlinear systems (proven empirically)
- No need to compute Jacobians (attractive for black-box neural dynamics)
- "Unscented transformation" is elegant Bayesian idea

**Math hook:**
- Generate $2n+1$ sigma points from $N(\hat{x}_{t-1|t-1}, P_{t-1|t-1})$ using Cholesky + weights
- Propagate through nonlinear $f$: $\mathcal{X}^{(pred)}_i = f(\mathcal{X}_i)$
- Recover mean/covariance from weighted samples: $\hat{x}_{t|t-1} = \sum w^{(m)}_i \mathcal{X}^{(pred)}_i$, etc.
- Apply same to observation $h(\cdot)$ in update step
- **Theoretical guarantee:** 3rd-order accurate for Gaussian posteriors (vs. 1st-order EKF)

**Application:**
- Handles nonlinear rating model without Jacobians
- Can use arbitrary neural net as $h$ (black-box observation model)

**Why it impresses rubric:**
- Algorithmic novelty: statistical approximation instead of Taylor
- Math: weighted resampling, Cholesky factorization, error bounds
- Comparison paper ready: EKF vs. UKF on same nonlinear dynamics

**Library + from-scratch:**
- Library: `filterpy.UnscentedKalmanFilter` (reference implementation)
- From-scratch: ~120 lines (sigma generation, weighted moments)
- Feasibility: **Medium** — requires Cholesky, weighted stats; not hard conceptually

**7-day effort: MEDIUM** (theory: 1 day; implement: 2 days; comparison EKF vs. UKF: 2 days)

**Class baseline:** EKF vs. UKF: accuracy under highly nonlinear rating model

---

### 5. Rauch–Tung–Striebel (RTS) Smoother — *Novel*, Backward Pass Optimization

**What it is:**
Two-pass algorithm: forward Kalman filter, then **backward recursion** using future information to refine all past state estimates (fixed-interval smoothing).

**Why it matters:**
- Filtered estimates are causal (only past data); RTS smoothed estimates use all T observations
- Significantly better state uncertainty quantification
- Detects "regime shifts" more cleanly

**Math hook:**
- Forward pass: Standard Kalman filter → collect $\{\hat{x}_{t|t}, P_{t|t}\}_{t=1}^T$
- Backward pass (t = T-1, ..., 1):
  $$
  J_t = P_{t|t} A^T P_{t+1|t}^{-1}
  $$
  $$
  \hat{x}_{t|T} = \hat{x}_{t|t} + J_t (\hat{x}_{t+1|T} - A\hat{x}_{t|t})
  $$
  $$
  P_{t|T} = P_{t|t} + J_t (P_{t+1|T} - P_{t+1|t}) J_t^T
  $$
- **Interpretation:** $J_t$ is "influence" of future on present; smoother state is convex combo of filtered + future-influenced

**Application:**
- Smoothed latent rating propensity over full watch history
- Uncertainty quantification on mood regimes
- Better changepoint detection (smoother estimates → sharper transitions)

**Why it impresses rubric:**
- Relatively unknown to undergrads (not standard ML course coverage)
- Clean math: backward recursion from dynamic programming
- Practical: smoothed estimates beat filtered for time-series visualization

**Library + from-scratch:**
- Library: `filterpy.RTS` or `statsmodels`
- From-scratch: ~40 lines after forward Kalman
- Feasibility: **High** — just matrix operations

**7-day effort: LOW-MEDIUM** (1 day conceptual + code; 1 day visualization; 1 day error analysis)

**Class baseline:** Filtered (Kalman) vs. smoothed (RTS) state estimates side-by-side

---

### 6. Particle Filter (Sequential Monte Carlo) — *Novel*, Nonparametric Bayes

**What it is:**
Represents posterior as **discrete particle cloud** rather than Gaussian. Each particle is a hypothetical state trajectory; weights updated as observations arrive. No closure assumptions (nonlinear, non-Gaussian OK).

**Why it matters:**
- Handles arbitrary nonlinear/non-Gaussian models
- Intuitive: discrete samples approximate posterior
- Can model discrete latent states + continuous latent dynamics simultaneously

**Math hook:**
- Particle representation: $\{x^{(i)}_t, w^{(i)}_t\}_{i=1}^N$ where $w^{(i)}_t \propto w^{(i)}_{t-1} \cdot p(y_t | x^{(i)}_t)$
- Resampling (multinomial, stratified, systematic) when effective sample size drops
- Sequential importance resampling (SIR):
  $$
  w^{(i)}_t \propto w^{(i)}_{t-1} \frac{p(y_t | x^{(i)}_t) p(x^{(i)}_t | x^{(i)}_{t-1})}{q(x^{(i)}_t | x^{(i)}_{t-1}, y_t)}
  $$
- **Posterior:** $p(x_t | y_{1:t}) \approx \sum_{i=1}^N w^{(i)}_t \delta(x_t - x^{(i)}_t)$
- **Credible intervals:** Empirical percentiles from particles

**Application:**
- Discrete mood modes (action/drama/horror) + continuous latent rating propensity
- Resampling prevents "particle degeneracy" (all weight on one trajectory)
- Natural for multi-modal posteriors (e.g., "user in binge mode OR sampling mode")

**Why it impresses rubric:**
- Non-parametric → mathematically elegant (minimal assumptions)
- First-principles: importance sampling + resampling from Bayes
- Comparison: particle filter vs. Kalman under nonlinear model shows when NP wins
- Visualization: particle cloud evolution over time

**Library + from-scratch:**
- Library: `particles` package (Chopin, 2025), `filterpy` basic SIR, `TensorFlow Probability`
- From-scratch: ~150 lines (initialization, resampling, weight update)
- Feasibility: **Medium** — resampling tricks needed; gradient-free so no AD required

**7-day effort: MEDIUM-HIGH** (theory: 1 day; implement basic SIR: 2 days; optimize resampling: 1 day; visualize particles: 1 day)

**Class baseline:** Kalman vs. Particle Filter on non-Gaussian rating model

---

### 7. Hidden Semi-Markov Model (HSMM) — *Novel*, Explicit Duration Distributions

**What it is:**
HMM with explicit **state duration distribution** instead of implicit geometric (exponential). Each latent state has a sojourn time drawn from Poisson, Gamma, or custom distribution.

**Why it matters:**
- "Action binge" lasts 3–5 movies on average (explicit structure)
- Standard HMM forces exponential duration → unrealistic
- Matches domain intuition: mood regimes have characteristic duration

**Math hook:**
- HMM: $P(\text{duration} = d | \text{state}) = (1-p)^{d-1} p$ (geometric → memoryless)
- HSMM: $P(\text{duration} = d | \text{state}) = \text{Poisson}(d; \lambda)$ or $\text{Gamma}(d; \alpha, \beta)$ (has memory)
- Likelihood: $p(y_{1:T} | \text{states, durations}) = \prod_t p(y_t | z_t) p(\text{duration}_t | z_t) p(z_t | z_{t-1})$
- Inference: forward-backward adapted to explicitly enumerate duration paths
- Viterbi: modified to track (state, duration) pairs

**Application:**
- Latent states: "drama binge," "sci-fi sampling," "rewatch mode"
- Duration model: Poisson $\lambda_k$ per state $k$ (e.g., drama avg 4 days)
- Forecast: next state + likely duration → when taste will shift

**Why it impresses rubric:**
- Domain-aware extension of HMM (not generic black-box)
- Adds explicit temporal structure missing from standard HMM
- Inference slightly harder (more novel than HMM)
- Visualization: duration histograms per regime

**Library + from-scratch:**
- Library: `hmmlearn` does NOT support; `LaMa` R package (not Python). Custom needed.
- From-scratch: ~200 lines (modified forward-backward + Viterbi)
- Feasibility: **Medium** — HMM code as template, extend to durations

**7-day effort: MEDIUM** (adapt HMM code: 2 days; Poisson/Gamma setup: 1 day; inference: 2 days)

**Class baseline:** HMM vs. HSMM: does duration model improve log-likelihood?

---

### 8. Switching State-Space Models (SSSM) / Regime-Switching Models — *Wild Card*, Markov Regimes

**What it is:**
Combines HMM (discrete regime) + Kalman Filter (continuous state per regime). At each time step, regime is drawn from Markov chain; given regime, state evolves linearly (Kalman-style).

**Why it's wild:**
- Captures "user is in X mood regime; within that regime, rating propensity drifts linearly"
- Two-level hierarchy: high-level discrete switches + low-level smooth dynamics
- Example: "Drama Regime" (A-matrix favors drama, high baseline rating) vs. "Thriller Regime"

**Math hook:**
- Regime chain: $z_t \in \{1, \ldots, K\}$ with transition $P(z_t | z_{t-1})$
- Regime-specific Kalman: $x_t = A_{z_t} x_{t-1} + w_t$, $y_t = C_{z_t} x_t + v_t$
- Likelihood: $p(y_{1:T}) = \sum_{\text{regimes}} P(\text{regimes}) \prod_t p(y_t | x_t, z_t) p(x_t | x_{t-1}, z_t)$
- Inference: Kim filter (forward) + backward EM for parameters
- **Interpretation:** Kalman per regime, HMM chains the regimes

**Application:**
- K=3 regimes: "binge," "sampling," "rewatch"
- Each has own Kalman dynamics (binge drifts ↑ rating, sampling stable, rewatch ↓)
- Forecast: regime sequence + state trajectory = full prediction w/ mode uncertainty

**Why it impresses rubric:**
- Combines two major model classes (HMM + Kalman)
- Interpretable: regimes have semantic meaning
- Advanced: Kim filter inference not standard undergrad material
- Comparison: HMM vs. SSSM shows gains from continuous state

**Library + from-scratch:**
- Library: `statsmodels.tsa.regime_switching` (Markov Switching), but limited customization
- From-scratch: ~300 lines (Kim filter forward + EM)
- Feasibility: **Medium-High** — builds on Kalman + HMM; EM is iterative

**7-day effort: HIGH** (theory: 2 days; Kim forward: 2 days; EM: 2 days; heavy lift)

**Class baseline:** HMM vs. Kalman vs. SSSM: three-way comparison on regime detection

---

## Part 2: Modern Neural Sequence Models

These methods integrate deep learning with sequence structure. Most are **RNN/Transformer extensions** (RNNs likely covered in class; Transformers maybe not in full detail).

### 1. LSTM / GRU — *Baseline*, Likely Covered

**What it is:**
RNN variants with gating mechanisms to handle vanishing gradient. LSTM has input/forget/output gates; GRU is simpler (reset/update gates).

**Why it's a baseline:**
- Covered in class (session 15–17?)
- Natural comparison: vanilla RNN (bad gradient flow) vs. LSTM (learn long-range dependencies)

**Application to watch history:**
- Sequence of movie embeddings (genre, runtime, rating) → LSTM → predict next rating
- Hidden state captures "current mood"

**7-day effort: LOW** (implement from PyTorch; train; eval: 2 days)

**Class baseline:** Vanilla RNN vs. LSTM gradient flow; LSTM as apples-to-apples for other methods

---

### 2. GRU4Rec — *Novel-to-Class*, Sequential Recommender Specific

**What it is:**
GRU adapted for **session-based recommendation**: sequence of items (movies) → predict next item. Uses ranking loss (BPR or cross-entropy), not regression.

**Why novel:**
- Seq2Seq framing for recommendations (not covered in class)
- Custom loss function (BPR-max): optimize ranking, not ratings
- Handles session-level (sparse) sequences

**Math hook:**
- $y_t = \text{softmax}(W_h h_t)$ where $h_t$ is GRU hidden state at time $t$
- Loss: BPR (pairwise ranking) or negative log-likelihood
  $$
  \mathcal{L} = -\sum_t \log \sigma(\text{score}_{\text{true}} - \text{score}_{\text{neg}})
  $$
  where negatives sampled from non-observed movies
- Training trick: GPU-efficient batching with in-batch negatives

**Application to watch history:**
- Session = "viewing streak" (movies watched within 7 days)
- Items = movies; labels = implicit (watch yes/no) or explicit (rating)
- Forecast: given history, top-k movies for next watch

**Why it impresses rubric:**
- Sequence-to-sequence applied to recommendations
- Custom ranking loss (not standard ML 101)
- Comparison: classification (next genre) vs. ranking (next movie)
- Visualization: attention weights (if using attention variant)

**Library + from-scratch:**
- Library: `hidasib/GRU4Rec_PyTorch_Official` (official), Transformers4Rec (NVIDIA)
- From-scratch: ~200 lines (GRU + BPR loss + negative sampling)
- Feasibility: **High** — GRU is standard, loss is new but simple

**7-day effort: LOW-MEDIUM** (understand BPR: 1 day; implement: 2 days; tune: 1 day)

**Class baseline:** Multinomial classification (next genre) vs. GRU4Rec ranking (next movie)

---

### 3. Transformer / Self-Attention Decoder (SASRec) — *Novel-to-Class*, Parallelizable

**What it is:**
Applies multi-head self-attention to sequence of items. **SASRec** (Self-Attentive Sequential Recommendation): Transformer encoder masked to avoid future leakage, predicts next item per position.

**Why novel:**
- Transformer likely not fully covered in class
- Parallelizable (all timesteps at once, vs. LSTM sequential)
- Interpretable: attention weights show which past movies influenced next

**Math hook:**
- Multi-head attention: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$
- Self-attention in Transformer: item embeddings serve as Q, K, V
- Causal mask: attend only to positions $\leq t$ (no future leakage)
- Positional encoding: $\text{PE}(t, 2i) = \sin(t / 10000^{2i/d})$, etc.
- Per-position loss: $\mathcal{L}_t = -\log P(\text{next item} | \text{context via attention})$

**Application:**
- Embed movies (using TMDB poster features or learned)
- Transformer attends to history → context vector at each position
- Head interpretability: one head might track "same genre," another "rating progression"

**Why it impresses rubric:**
- **Parallelizable:** trains faster than LSTM
- **Interpretable:** attention visualization (saliency maps)
- **Math-dense:** positional encodings, causal masking, multi-head derivation
- **Comparison:** LSTM (sequential, hard to interpret) vs. Transformer (parallel, interpretable)

**Library + from-scratch:**
- Library: PyTorch `nn.Transformer`, `pytorch-forecasting.TemporalFusionTransformer`
- From-scratch: ~300 lines (multi-head attention, positional encoding, causal mask)
- Feasibility: **Medium** — linear algebra for attention, but standard

**7-day effort: MEDIUM** (implement attention: 2 days; add positional encoding: 1 day; train + eval: 2 days)

**Class baseline:** LSTM vs. SASRec: speed, interpretability, accuracy

---

### 4. Mamba / S4 / S5 — *Novel-to-Class*, Structured State-Space Sequences

**What it is:**
Recent (2024–2025) neural sequence models that **parameterize continuous-time state-space equations** with structured recurrence. Claim to beat Transformers on long sequences with **linear time complexity**.

**Key models:**
- **S4 (Structured State Space sequence model):** Parameterize A-matrix (state dynamics) with learnable structure; use FFT-based convolution for efficient training
- **S5:** Simplified S4; removes FFTs, uses diagonal A-matrix; equally efficient, easier to implement
- **Mamba:** Adds data-dependent selection to S4 states (selective SSM); gates state dimension per timestep based on input

**Why novel:**
- Bridges classical Kalman/state-space theory with neural networks
- **Kalman connection:** $x_t = A x_{t-1} + B u_t$ is a classical SSM; S4/Mamba learn $A, B, C$ end-to-end
- Linear time: no quadratic attention cost like Transformers

**Math hook (S4):**
- State-space model: $\dot{x}(t) = A x(t) + B u(t)$, $y(t) = C x(t) + D u(t)$ (continuous-time)
- Discretization (bilinear): $x_k = \bar{A} x_{k-1} + \bar{B} u_k$
- **Key trick:** Parameterize A via **diagonal + rank-2 perturbation** (DPLR form) → A has special structure → FFT-based convolution O(N log N)
- Training: unroll as a convolution; inference: recurrent (classic Kalman)
- **Mamba extension:** $x_t = (A_t \odot x_{t-1}) + (B_t \odot u_t)$ where $A_t, B_t$ depend on input (selective)

**Application to watch history:**
- Continuous-time latent dynamics: user's rating propensity evolves smoothly between discrete watches
- Structured A-matrix: eigenvectors of A capture periodic patterns (e.g., weekly watching cycles, seasonal)
- Mamba selection: amplify state dimensions relevant to current movie features (self-attention-like without quadratic cost)

**Why it impresses rubric:**
- **Explicitly Kalman-inspired:** cite the rubric's own hint ("Kalman filter is extension of HMM")
- **First-principles:** state-space ODE, discretization, structured parameterization
- **SOTA comparison:** S4/Mamba beat Transformers on some benchmarks (language modeling, time series)
- **Novel to class:** Definitely not covered (post-2023 architecture)
- **Visualization:** latent state trajectory (like Kalman filter smoothed state), eigenvalue spectrum of A

**Library + from-scratch:**
- Library: `jax-ml/mamba` (official JAX), `state-spaces/mamba` (PyTorch), `albanie/s4` (S4 implementation)
- From-scratch S4: ~500 lines (state-space discretization, DPLR parameterization, FFT convolution, bilinear transform)
- From-scratch S5: ~300 lines (simpler diagonal A, skip FFT tricks)
- Feasibility: **Medium-High** — requires understanding of continuous → discrete conversion, FFT (for S4)

**7-day effort: HIGH for S4 from-scratch; MEDIUM for library S4/Mamba + analysis**

**Best path:** Use library Mamba/S5, customize A-matrix initialization for time-series domain (diagonal dominance favors stable dynamics), compare to Transformer.

**Class baseline:** Transformer vs. Mamba: time, memory, and interpretability

---

### 5. Temporal Fusion Transformer (TFT) — *Novel-to-Class*, Multi-Horizon Forecasting + Intervals

**What it is:**
Transformer-based architecture designed for **multi-horizon time series forecasting** with **quantile predictions** (confidence intervals) and **variable-length temporal context**.

**Why novel:**
- Forecasting-specific architecture (not just language models)
- Generates prediction intervals (not point forecasts alone)
- Interprets via "attention over temporal features"

**Architecture:**
- **Encoder (LSTM):** encodes past observations
- **Decoder (Transformer):** attends to encoder over future horizon
- **Quantile head:** outputs quantile forecasts (0.1, 0.5, 0.9) → prediction intervals
- **Gating layers:** feature importance via learned gates

**Math hook:**
- Decoder produces representation $z_t$ via multi-head attention over encoder states
- Quantile regression head: $\hat{q}_\tau(t) = f_\tau(z_t)$ for each quantile $\tau$
- Loss: quantile loss $\mathcal{L}_\tau = \sum |\tau - \mathbb{1}(y_t < \hat{q}_\tau)| |y_t - \hat{q}_\tau|$
- **Gating:** Learned gates per feature dim → interpretability (which features matter?)

**Application to watch history:**
- Encoder: past ratings + genres + timestamps
- Decoder: forecast rating for next watch (and confidence interval)
- Multi-horizon: predict next 1, 2, 3 watches ahead
- Intervals: "92% chance next watch will be 7+ rated"

**Why it impresses rubric:**
- **Uncertainty quantification:** not just $\hat{y}_t$, but $P(y_t | \text{history})$ approximated by quantiles
- **Interpretable:** gating weights reveal feature importance per timestep
- **Math:** quantile regression, multi-head attention, Transformer decoder specifics
- **Practical:** forecast intervals matter for recommendations ("confidence in next genre")

**Library + from-scratch:**
- Library: `pytorch-forecasting.TemporalFusionTransformer`, `darts`
- From-scratch: ~400 lines (LSTM encoder, masked Transformer decoder, quantile head, gating)
- Feasibility: **Medium** — combines LSTM + Transformer + new loss; all standard components

**7-day effort: MEDIUM-HIGH** (set up: 1 day; understand TFT: 1 day; implement: 2 days; tune + eval: 2 days)

**Class baseline:** Point forecast (vanilla Transformer) vs. TFT with intervals

---

### 6. Variational Recurrent Neural Network (VRNN) — *Wild Card*, Latent Variable Sequences

**What it is:**
Combines **VAE** (latent variable model) with **RNN** (sequential). At each timestep, a latent variable $z_t$ is drawn conditioned on RNN hidden state $h_{t-1}$ and current observation; RNN then evolves based on $(z_t, y_t)$.

**Why wild-card:**
- Less known than LSTM/Transformer; explicitly stochastic
- Models **multi-modal conditional distributions** (e.g., "given past watches, next rating could be 5 OR 2 depending on hidden mood")
- Generative: can sample diverse next-watch sequences

**Architecture:**
- VAE at each step: $p(z_t | h_{t-1})$, $p(y_t | z_t, h_{t-1})$
- RNN: $h_t = \text{RNN}(h_{t-1}, y_t, z_t)$
- Training: ELBO per timestep
  $$
  \mathcal{L}_t = \mathbb{E}_{q(z_t|h_{t-1}, y_t)}[-\log p(y_t | z_t, h_{t-1})] + KL(q || p(\cdot | h_{t-1}))
  $$

**Application:**
- $z_t$ encodes "unobserved context" (mood, external factors affecting rating)
- Forward: filter forward to get posterior over next $z$, sample → diverse next-watch predictions
- Generative: sample $z \sim p(\cdot | h)$, then $\hat{y} \sim p(\cdot | z, h)$ → synthetic watch sequence

**Why it impresses rubric:**
- **Generative modeling:** not just predict, but sample diverse plausible futures
- **Uncertainty via distribution:** posterior over latents = model uncertainty
- **Math-dense:** VAE ELBO, KL divergence, hierarchical graphical model
- **Interpretability:** latent $z$ can be analyzed (e.g., mood dimensions)

**Library + from-scratch:**
- Library: None standard; custom PyTorch needed
- From-scratch: ~400 lines (VAE per-step encoder/decoder, RNN integration, ELBO training)
- Feasibility: **Medium** — VAE + RNN both doable; interleaving is design choice

**7-day effort: MEDIUM-HIGH** (understand VRNN: 1 day; implement: 3 days; tune variational loss: 2 days)

**Class baseline:** Deterministic RNN vs. VRNN: do latent variables help uncertainty quantification?

---

## Part 3: Specialized Temporal Methods

### 1. Change-Point Detection (PELT / BOCPD) — *Novel*, Taste-Shift Detection

**What it is:**
Algorithms to detect **abrupt shifts** in time series statistics. PELT = pruned exact linear time (offline, O(n)); BOCPD = Bayesian online (online, posterior at each step).

**PELT:**
- Solves: $\arg\min_{\{c_k\}} \sum_{i=1}^K \mathcal{L}(\text{seg}_i) + \beta K$ where $\mathcal{L}$ is loss per segment, $\beta$ is penalty
- Pruning: remove candidate breakpoints that can't be optimal (dynamic programming)
- Complexity: $O(n)$ unlike brute-force $O(n^2)$

**BOCPD:**
- Posterior: $P(\text{changepoint at } t | y_{1:t})$
- Online: update at each new observation
- Generative model: segments have different parameters $\theta_k$; within-segment likelihood from prior (Normal-Gamma conjugate, Poisson, etc.)
- Interpretation: "run length" = time since last changepoint

**Application to watch history:**
- Detect genre shifts (e.g., "user switches from drama to horror")
- Detect rating trend breaks (e.g., "suddenly rating lower")
- Semantic: "taste regime changed at watch #47" → actionable

**Why it impresses rubric:**
- Interpretable outputs (exact changepoint times)
- Unsupervised anomaly detection angle
- Math: dynamic programming (PELT) or Bayesian filtering (BOCPD)

**Library + from-scratch:**
- Library: `ruptures` (PELT + others), `bocd` (BOCPD)
- From-scratch PELT: ~150 lines (DP + pruning)
- From-scratch BOCPD: ~100 lines (beta-binomial / normal-gamma updates)
- Feasibility: **High** — standard algorithms

**7-day effort: LOW-MEDIUM** (library: 1 day; from-scratch one: 2 days; analysis: 1 day)

**Application to Pipeline 3:** Segment watch history by changepoints → each segment is a separate "regime" → fit HMM/Kalman per segment

---

### 2. Hawkes Process / Self-Exciting Point Process — *Wild-Card*, Binge Behavior Modeling

**What it is:**
Stochastic process where past events **increase probability of future events** (self-exciting). Intensity function:
$$
\lambda(t) = \mu + \sum_{t_i < t} \alpha \exp(-\beta(t - t_i))
$$
- $\mu$: baseline rate (spontaneous watches)
- $\alpha \exp(-\beta \Delta t)$: "excitement" from previous watch decays exponentially

**Why novel:**
- Captures binge behavior (one watch → higher chance of next within hours/days)
- Point process inference (not standard regression)
- Generative: can sample realistic watch times

**Neural Hawkes:**
- Replace $\alpha \exp(-\beta \Delta t)$ with neural network parameterization
- LSTM-based: hidden state of RNN encodes cumulative effect of past events
- Transformer-based: self-attention weights act as adaptive kernel

**Application to watch history:**
- Times: "how long between watches?" (not just dates, but inter-event gaps)
- Hawkes captures: "watched movie at 8pm → 65% chance watch again by midnight"
- Generative: sample realistic binge patterns

**Why it impresses rubric:**
- **Point process inference:** mathematically rich (likelihood is integral of intensity)
- **Generative modeling:** sample synthetic sequences
- **Unexpected angle:** capture temporal dependencies beyond Markov assumption
- **Comparison:** Poisson (baseline) vs. Hawkes (self-exciting) log-likelihood

**Library + from-scratch:**
- Library: `hawkeslib` (fast parameter estimation), `HawkesPyLib` (Ogata thinning)
- From-scratch: ~250 lines (intensity computation, MLE via BFGS, likelihood evaluation)
- Feasibility: **Medium** — likelihood evaluation tricky (integral of intensity function); numerical optimization needed

**7-day effort: MEDIUM-HIGH** (theory: 1.5 days; implement likelihood: 2 days; MLE optimization: 1.5 days)

**Class baseline:** Poisson (constant rate) vs. Hawkes (self-exciting)

---

### 3. Survival Analysis / Hazard Functions — *Novel*, "Time Until Next Watch" Prediction

**What it is:**
Models time-to-event (e.g., "how many days until next watch?") via hazard function $h(t) = \lambda(t)$ (instantaneous risk).

**Models:**
- **Kaplan-Meier:** nonparametric survival curve (proportion of "no watch yet")
- **Cox Proportional Hazards:** $h(t | x) = h_0(t) \exp(\beta^T x)$ (baseline hazard × covariate modulation)
- **Accelerated Failure Time (AFT):** $\log T = \alpha + \beta^T x + \sigma \epsilon$ (log-time regression)

**Application:**
- Event: next watch (target variable = days until next watch)
- Covariates: previous rating, genre, user rating history
- Question: "Given I just watched horror, how many days until next watch?" → survival curve

**Why it impresses rubric:**
- Handles censoring (user data ends, watch never happens)
- Interpretable: hazard ratios tell causal-ish stories
- Math: partial likelihood (Cox), Kaplan-Meier KM curve derivation

**Library + from-scratch:**
- Library: `scikit-survival`, `lifelines`
- From-scratch Cox: ~200 lines (partial likelihood, gradient descent, risk sets)
- Feasibility: **High** — standard stats methods

**7-day effort: LOW-MEDIUM** (understand Cox/KM: 1 day; implement: 2 days; eval: 1 day)

**Class baseline:** Naive "average days until next watch" vs. Cox regression with features

---

### 4. Neural ODE — *Wild-Card*, Continuous-Time Latent Dynamics

**What it is:**
Treats hidden state evolution as solution to an ODE: $\frac{dx}{dt} = f_\theta(x(t), t)$ (black-box network).

**Why novel:**
- Continuous-time system (not discrete LSTM steps)
- Irregular sampling friendly: observe at any times, integrator handles interpolation
- Generative: solve ODE backward to generate trajectories

**Math hook:**
- Latent ODE: $h(t) = \text{ODESolver}(\frac{dh}{dt} = f_\theta(h, t), h(0) = h_0)$
- Observation: $y_t = g(h(t_t))$ (project latent state to rating space)
- Training: adjoint sensitivity method (backprop through ODE solver)
- **Irregular sampling:** naturally handles non-uniform timesteps

**Application:**
- Latent state = continuous user rating propensity function $r(t)$
- Between watches: $r(t)$ evolves via learned ODE
- Observation: rating $y_t = r(t_t) + \text{noise}$ at watch times
- Forecast: integrate ODE to predict $r(t_{\text{future}})$

**Why it impresses rubric:**
- **Continuous-time modeling:** fundamentally different from discrete RNN/Transformer
- **Irregular sampling:** watch history has irregular gaps; Neural ODE handles naturally
- **Math-dense:** ODE theory, adjoint method (backprop through numerical integration)
- **Comparison:** discrete Kalman Filter (continuous state, discrete obs) vs. Neural ODE (continuous evolution via neural network)

**Library + from-scratch:**
- Library: `torchdiffeq` (PyTorch ODE solver)
- From-scratch ODE solver: ~100 lines (RK4 / RK45 Runge-Kutta), but use library
- Training backbone: ~300 lines (ODE as layer, adjoint method)
- Feasibility: **Medium** — library handles hard parts; conceptual learning curve

**7-day effort: MEDIUM-HIGH** (understand ODE theory: 2 days; integrate `torchdiffeq`: 2 days; train + eval: 2 days)

**Class baseline:** Discrete Kalman vs. Neural ODE: continuous-time expressiveness

---

## Part 4: Application-Specific Methods

### Prophet (Decomposition + Changepoint) — Smart Hybrid

**What it is:**
Facebook's time-series forecasting library combining **trend** (piecewise-linear), **seasonality** (Fourier), and **holidays** in additive/multiplicative form.

**Why consider:**
- Built-in changepoint detection (automatic or manual)
- Interpretable decomposition (handy for narrative)
- Fast (Stan-based Bayesian inference, but optimized)

**Application:**
- Decompose rating time series into trend + seasonality
- Detect rating trend shifts (changepoints)
- Forecast next watch rating with uncertainty bands

**Why it impresses rubric:**
- Practical (production-proven)
- Interpretable outputs (trend plot, seasonal plot, forecast bands)
- Not ML-novel, but useful for narrative ("user's rating trend shifted at watch #47")

**Library + from-scratch:**
- Library: `fbprophet` / `prophet`
- From-scratch: replicating Prophet is ~500 lines (piecewise-linear trend, Fourier seasonality, Stan sampling logic)
- Feasibility: **High** (use library)

**7-day effort: LOW** (library fit + analyze: 1 day; compare to Kalman: 1 day)

**Class baseline:** Simple ARIMA vs. Prophet: interpretability vs. accuracy

---

## Wild-Card Synthesis: Neural HSMM or Diffusion Generative Sequence Model

### Option 1: Neural HSMM — Probabilistic Regime Switching

**Hybrid of:** HSMM (explicit durations) + neural network (flexible transition/emission)

**Idea:**
- Latent states (moods): "binge drama," "sampling action," "rewatch"
- Each state has learned duration distribution (parameterized by small MLP)
- Transitions: neural network (context-dependent, not just Markov matrix)
- Emission: neural network maps mood → rating distribution

**7-day feasibility:** **MEDIUM-HIGH** (adapt HSMM code ~200 lines to use MLPs for transitions/durations/emissions; tricky but doable)

**Why it impresses:** Combines classical (duration) + neural (flexibility) elegantly

---

### Option 2: Score-Based Diffusion for Watch Sequence Generation

**Wild idea:**
Train a **diffusion model** to learn the distribution of rating sequences, then:
1. Sample diverse synthetic "plausible next watches"
2. Condition on user history (classifier guidance or LoRA fine-tune)
3. Generate movie recommendations with confidence

**Math hook:**
- Forward: progressively add noise to rating sequence
- Reverse: learn score function $\nabla \log p(y_t | y_{1:t-1})$
- Sampling: solve reverse SDE to denoise → sample synthetic sequences

**7-day feasibility:** **HIGH** (generative, but standard diffusion code available; custom: tricky)

**Pros:** Generative angle (novel to Temilola's prior work), can generate diverse futures
**Cons:** Training can be finicky; may not beat simpler methods on small N=162

---

## Summary: Ranked Methods for Pipeline 3

| **Rank** | **Method** | **Novelty** | **Math Density** | **7-Day Effort** | **From-Scratch?** | **Key Advantage** |
|---|---|---|---|---|---|---|
| **1** | Mamba / S4 | ⭐⭐⭐ | ⭐⭐⭐ | MEDIUM | Library + analysis | Kalman-inspired, SOTA, interpretable |
| **2** | Kalman + RTS | ⭐⭐⭐ | ⭐⭐⭐ | LOW | YES | First-principles, closed-form, uncertainty |
| **3** | GRU4Rec | ⭐⭐ | ⭐⭐ | LOW-MEDIUM | YES | Sequence-specific, ranking loss novel |
| **4** | Particle Filter | ⭐⭐⭐ | ⭐⭐⭐ | MEDIUM | YES | Non-parametric, flexible, interpretable |
| **5** | HSMM | ⭐⭐⭐ | ⭐⭐ | MEDIUM | YES | Domain-aware duration, extends HMM |
| **6** | Hawkes Process | ⭐⭐⭐ | ⭐⭐⭐ | MEDIUM | YES | Self-exciting, generative, binge behavior |
| **7** | TFT | ⭐⭐ | ⭐⭐ | MEDIUM-HIGH | Library | Intervals, interpretable, domain-specific |
| **8** | Change-Point Detection | ⭐⭐ | ⭐⭐ | LOW-MEDIUM | YES/Library | Regime shifts, unsupervised |
| **9** | Neural ODE | ⭐⭐⭐ | ⭐⭐⭐ | MEDIUM-HIGH | Library | Continuous-time, irregular sampling |
| **10** | VRNN | ⭐⭐⭐ | ⭐⭐⭐ | MEDIUM-HIGH | YES | Stochastic sequences, generative |
| **Wildcard** | Diffusion Seq Gen | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | HIGH | Library | Novel generative angle |

---

## Recommended Core Pipeline 3 Structure

### Narrative Arc:
1. **Data preprocessing:** Temporal features (inter-watch gaps, day-of-week, weekday/weekend, season, "days since last genre")
2. **Baseline (apples-to-apples):** Standard HMM from class
3. **Novel method 1:** Kalman Filter + RTS Smoother (math-heavy, from-scratch)
4. **Novel method 2:** Mamba or GRU4Rec (SOTA, library + customization)
5. **Specialized method:** PELT changepoint detection (regime analysis)
6. **Synthesis:** Combine Kalman + changepoint → regime-aware smoothing
7. **Generative (wildcard):** VRNN or Neural Hawkes (sample synthetic watches)
8. **Comparison table:** HMM vs. Kalman vs. SSSM (structured SSM) vs. Transformer vs. Mamba

### Time Budget (7 days):
- **Day 1:** Data engineering (inter-event gaps, feature construction)
- **Days 2–3:** HMM baseline + Kalman from-scratch (derivations + code)
- **Days 4–5:** Mamba/SASRec library setup + custom decoder tuning
- **Day 6:** PELT changepoint detection + regime analysis
- **Day 7:** Viz (state trajectories, attention heatmaps, comparison table), narrative polish

### Math Deliverables:
1. **Kalman forward-backward derivation** (full Bayes rules)
2. **RTS backward recursion** (dynamic programming interpretation)
3. **Mamba/S4 state-space discretization** (continuous→discrete)
4. **Attention mechanism** (softmax, scaling, causal masking for Transformer/SASRec)
5. **PELT dynamic programming** (cost function, pruning rules)

### Visualizations:
- Filtered vs. smoothed latent state (Kalman)
- Attention heatmap (SASRec over history)
- Regime changepoints on timeline
- Prediction intervals (Kalman ±2σ bands)
- Latent state phase portrait (2D projection)
- Mamba eigenvalue spectrum (dynamics structure)

---

## Key References & Libraries

### Classical Methods:
- [State Space Models and the Kalman Filter (QuantStart)](https://www.quantstart.com/articles/State-Space-Models-and-the-Kalman-Filter/)
- [HMM Forward-Backward Algorithm (Wikipedia)](https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm)
- [Kalman & RTS Smoother (Särkkä lecture notes)](https://users.aalto.fi/~ssarkka/course_k2011/pdf/handout7.pdf)
- `filterpy` library (MIT, Rob Moore)

### State-Space Sequences:
- [From S4 to Mamba Survey (2025)](https://arxiv.org/abs/2503.18970)
- [Mamba: Linear-Time Sequences](https://arxiv.org/abs/2312.00752)
- [Mamba GitHub (official)](https://github.com/state-spaces/mamba)
- [S4 & S5 implementations](https://github.com/state-spaces/)

### Sequential Recommenders:
- [GRU4Rec PyTorch Official](https://github.com/hidasib/GRU4Rec_PyTorch_Official)
- [SASRec Paper](https://ksiresearch.org/seke/seke21paper/paper035.pdf)
- [BERT4Rec Paper](https://arxiv.org/pdf/1904.06690)

### Specialized Methods:
- [Change-Point Detection: ruptures library](https://centre-borelli.github.io/ruptures-docs/)
- [PELT Algorithm](https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/)
- [Hawkes Process: hawkeslib](https://hawkeslib.readthedocs.io/)
- [Neural Hawkes Process](https://arxiv.org/abs/1612.09328)
- [HSMM (Murphy)](https://www.cs.ubc.ca/~murphyk/Papers/segment.pdf)
- [Survival Analysis: scikit-survival](https://scikit-survival.readthedocs.io/)
- [Prophet Forecasting](https://facebook.github.io/prophet/)

### Neural Sequences:
- [Temporal Fusion Transformer](https://arxiv.org/abs/1912.09363)
- [VRNN (Chung et al.)](https://arxiv.org/abs/1506.02216)
- [Neural ODE & adjoint method](https://arxiv.org/abs/1806.07522)
- `torchdiffeq` library (ODE solvers for PyTorch)

### Diffusion Models:
- [Score-Based Generative Modeling via SDEs](https://arxiv.org/abs/2011.13456)
- [Diffusion Models Overview (Song & Ermon)](https://yang-song.net/blog/2021/score/)

---

## Conclusion

**Best 3-method core for Pipeline 3:**

1. **Kalman Filter + RTS Smoother** (classical, from-scratch, first-principles math, apples-to-apples with HMM baseline)
2. **Mamba or SASRec** (SOTA, library-based, interpretable, aligns with rubric's Kalman/seq-to-seq hints)
3. **PELT Changepoint Detection** (unsupervised regime discovery, easy to implement, adds narrative dimension)

**Optional wildcard (if time permits):**
- Neural Hawkes Process (generative, self-exciting, captures binge behavior)
- or VRNN (stochastic sequences, diverse sampling)

This mix achieves:
✓ **Novelty:** All three are extensions of or distinct from class material
✓ **Math density:** Kalman derivations + RTS + Mamba theory + PELT DP
✓ **From-scratch:** Kalman + RTS (100% custom), PELT (100% custom), Mamba (library + analysis)
✓ **Apples-to-apples:** HMM baseline explicitly paired with each method
✓ **Narrative:** "Discover regimes → smooth within regime → forecast next watch"
✓ **Visualization:** State trajectories, changepoints, intervals, attention maps
✓ **Rubric alignment:** Addresses all LOs (MLCode, MLExplanation, MLFlexibility, MLMath)

---

**End of Report**
